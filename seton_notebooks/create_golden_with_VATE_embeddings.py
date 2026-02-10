#!/usr/bin/env python3
"""
Append VATE (trained Siamese) TEXT embeddings to an existing file that already contains IMAGE embeddings.

SAFE VERSION:
- Preserves all existing columns exactly
- Appends fraud_txt_emb_* and real_txt_emb_* as float32
- Hard-fails on ANY column collision
- No index reset, no reordering, no dtype pollution

Example:
python seton_notebooks/create_golden_with_vate_text_embeddings.py \
  --input text_to_image/Golden/golden_embeddings_train.parquet \
  --output text_to_image/Golden_and_Text/train_pairs_with_img_and_vate_txt_embs.parquet \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# VATE / your repo imports
from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.learning.siamese import SiameseModelPairs


# ---------------------------
# IO
# ---------------------------
def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


# ---------------------------
# Text normalization
# ---------------------------
def normalize_name(x: object, strip_com: bool) -> str:
    s = unicodedata.normalize("NFC", str(x))
    s = s.lstrip("-").strip()
    if strip_com:
        s = re.sub(r"\.com$", "", s, flags=re.IGNORECASE)
    return s


def pick_device(override: Optional[str]) -> torch.device:
    if override:
        d = torch.device(override)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class nullcontext:
    def __enter__(self):  # noqa: D401
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------
# Checkpoint loading helpers
# ---------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats:
      - raw state_dict (mapping param_name -> tensor)
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model": ...}
    """
    if isinstance(ckpt, dict):
        # If it looks like a raw state_dict already (tensor values), return as-is.
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]

        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                if len(sd) > 0 and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd  # type: ignore[return-value]

    raise RuntimeError(
        "Unrecognized checkpoint format. Expected a state_dict or a dict containing "
        "'state_dict'/'model_state_dict'/'model'."
    )


# ---------------------------
# VATE text embedding extraction
# ---------------------------
def _project_text_features_if_needed(
    siamese_model: torch.nn.Module,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    If the Siamese model exposes an explicit projection module for text, apply it.
    Otherwise assume the returned tensor is already the final embedding.
    """
    # Common attribute names for a text projection head
    for attr in ("text_projection", "text_proj", "proj_text", "projection_text"):
        if hasattr(siamese_model, attr):
            mod = getattr(siamese_model, attr)
            if callable(mod):
                return mod(text_features)
    # Some implementations have a shared projection head, but that would be ambiguous.
    return text_features


@torch.no_grad()
def _encode_text_batch(
    texts: List[str],
    siamese_model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Attempts multiple safe pathways to obtain FINAL VATE text embeddings from the trained Siamese model.

    Preferred:
      1) siamese_model.encode_text(texts)
      2) siamese_model.get_text_embeddings(texts)
      3) siamese_model.backbone.encode_text(texts) (+ optional projection)

    Returns: (B, D) float tensor on CPU.
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    )

    # Path 1: encode_text on the Siamese model
    if hasattr(siamese_model, "encode_text") and callable(getattr(siamese_model, "encode_text")):
        with autocast_ctx:
            out = siamese_model.encode_text(texts)  # type: ignore[attr-defined]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("siamese_model.encode_text(...) did not return a torch.Tensor.")
        return out.float().detach().cpu()

    # Path 2: get_text_embeddings on the Siamese model
    if hasattr(siamese_model, "get_text_embeddings") and callable(getattr(siamese_model, "get_text_embeddings")):
        with autocast_ctx:
            out = siamese_model.get_text_embeddings(texts)  # type: ignore[attr-defined]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("siamese_model.get_text_embeddings(...) did not return a torch.Tensor.")
        return out.float().detach().cpu()

    # Path 3: backbone has encode_text (then apply projection if model exposes it)
    if hasattr(siamese_model, "backbone"):
        backbone = getattr(siamese_model, "backbone")
        if hasattr(backbone, "encode_text") and callable(getattr(backbone, "encode_text")):
            with autocast_ctx:
                feats = backbone.encode_text(texts)  # type: ignore[attr-defined]
            if not isinstance(feats, torch.Tensor):
                raise RuntimeError("siamese_model.backbone.encode_text(...) did not return a torch.Tensor.")
            with autocast_ctx:
                emb = _project_text_features_if_needed(siamese_model, feats)
            if not isinstance(emb, torch.Tensor):
                raise RuntimeError("Text projection did not return a torch.Tensor.")
            return emb.float().detach().cpu()

    raise RuntimeError(
        "Could not find a supported text-encoding method.\n"
        "Tried: siamese_model.encode_text, siamese_model.get_text_embeddings, "
        "siamese_model.backbone.encode_text.\n"
        "Inspect your SiameseModelPairs/backbone implementation and add a compatible method."
    )


@torch.no_grad()
def embed_unique_texts_vate(
    uniq_texts: List[str],
    siamese_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    do_l2_normalize: bool,
) -> np.ndarray:
    """
    Returns float32 embeddings of shape (N, D) from the TRAINED VATE Siamese model.
    """
    n = len(uniq_texts)
    if n == 0:
        raise ValueError("No texts to embed.")

    siamese_model.eval()

    use_amp = device.type == "cuda"
    embeddings_cpu: List[torch.Tensor] = []

    for start in tqdm(range(0, n, batch_size), desc="Embedding text (VATE)"):
        chunk = uniq_texts[start : start + batch_size]

        # The underlying model/backbone should handle tokenization internally.
        e_cpu = _encode_text_batch(chunk, siamese_model=siamese_model, device=device, use_amp=use_amp)

        if do_l2_normalize:
            e_cpu = F.normalize(e_cpu, dim=-1, eps=1e-8)

        embeddings_cpu.append(e_cpu)

    emb = torch.cat(embeddings_cpu, dim=0).numpy().astype(np.float32)
    return emb


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet with image embeddings (golden or raw)")
    ap.add_argument("--output", required=True, help="Output parquet with image + VATE text embeddings")

    # VATE-specific
    ap.add_argument(
        "--backbone",
        default="siglip",
        choices=["clip", "coca", "flava", "siglip"],
        help="Backbone type used by the saved Siamese model",
    )
    ap.add_argument(
        "--model-weights",
        required=True,
        help="Path to trained Siamese model weights (.pt) to produce VATE embeddings",
    )
    ap.add_argument("--embedding-dim", type=int, default=768, help="Siamese embedding_dim used in training")
    ap.add_argument("--projection-dim", type=int, default=768, help="Siamese projection_dim used in training")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--strip-com", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output}")

    df = load_table(args.input)

    if args.max_rows is not None:
        df = df.head(int(args.max_rows))

    # Required columns
    if "fraudulent_name" not in df.columns or "real_name" not in df.columns:
        raise RuntimeError("Input must contain 'fraudulent_name' and 'real_name' columns.")

    df = df.copy()
    df["fraudulent_name"] = df["fraudulent_name"].map(lambda x: normalize_name(x, args.strip_com))
    df["real_name"] = df["real_name"].map(lambda x: normalize_name(x, args.strip_com))

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # Build the exact backbone wrapper your VATE code uses
    tester = BaselineTester(model_type=args.backbone, batch_size=args.batch_size, device=str(device))
    backbone_module = tester.model_wrapper

    # Instantiate the Siamese model in the same configuration used during training
    siamese_model = SiameseModelPairs(
        embedding_dim=int(args.embedding_dim),
        projection_dim=int(args.projection_dim),
        backbone=backbone_module,
    ).to(device)

    ckpt = torch.load(args.model_weights, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()

    # Deduplicate text
    all_texts = pd.concat(
        [df["fraudulent_name"], df["real_name"]],
        ignore_index=True,
    ).astype(str)

    uniq_texts = pd.unique(all_texts).tolist()
    print(f"[INFO] unique text strings: {len(uniq_texts):,}")

    emb_mat = embed_unique_texts_vate(
        uniq_texts=uniq_texts,
        siamese_model=siamese_model,
        device=device,
        batch_size=args.batch_size,
        do_l2_normalize=(not args.no_normalize),
    )

    dim = emb_mat.shape[1]
    print(f"[INFO] VATE text embedding dim = {dim}")

    # Map text → embedding
    text_to_idx = {t: i for i, t in enumerate(uniq_texts)}
    fraud_idx = df["fraudulent_name"].map(text_to_idx).to_numpy()
    real_idx = df["real_name"].map(text_to_idx).to_numpy()

    fraud_embs = emb_mat[fraud_idx]
    real_embs = emb_mat[real_idx]

    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols = [f"real_txt_emb_{i}" for i in range(dim)]

    # HARD FAIL on collision
    for c in fraud_cols + real_cols:
        if c in df.columns:
            raise RuntimeError(f"Column collision detected: {c}")

    text_df = pd.DataFrame(
        np.hstack([fraud_embs, real_embs]),
        columns=fraud_cols + real_cols,
        dtype=np.float32,
    )

    out_df = pd.concat([df, text_df], axis=1)

    save_table(out_df, args.output)
    print(f"[INFO] wrote clean merged file → {args.output}")


if __name__ == "__main__":
    main()