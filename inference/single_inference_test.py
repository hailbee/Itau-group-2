#!/usr/bin/env python3
"""
Text -> VATE embedding -> Student projection -> Nearest real_name retrieval.

This script:
1) Takes an input query string.
2) Generates its VATE text embedding using the SAME pathway you used in
   create_golden_with_vate_text_embeddings.py:
      - BaselineTester(model_type=..., ...)
      - SiameseModelPairs(..., backbone=tester.model_wrapper)
      - siamese_model.encode_text / get_text_embeddings / backbone.encode_text (+ optional projection)
3) Loads best_model.pt (SiameseEmbeddingModel) and projects the VATE embedding into
   the student space.
4) Loads vate_test_student_only.parquet (or any parquet/csv) containing real-side
   student embeddings (default prefix: real_student_*) and a real-name column.
5) Finds the row with highest cosine similarity and prints the name (optionally top-k).

Example:
  python query_vate_student_retrieval.py \
    --query "Nike Air Force 1 white" \
    --parquet text_to_image/evaluation/vate_test_student_only.parquet \
    --model-path saved_models/best_model.pt \
    --vate-backbone siglip \
    --vate-model-weights weights/best_model_siglip_pair.pt \
    --topk 5
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from siamese import SiameseEmbeddingModel

# VATE / your repo imports (must match your embedding pipeline)
from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.learning.siamese import SiameseModelPairs


# ---------------------------
# Device
# ---------------------------
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


# ---------------------------
# Text normalization (matches your VATE script)
# ---------------------------
def normalize_name(x: object, strip_com: bool) -> str:
    s = unicodedata.normalize("NFC", str(x))
    s = s.lstrip("-").strip()
    if strip_com:
        s = re.sub(r"\.com$", "", s, flags=re.IGNORECASE)
    return s


# ---------------------------
# Checkpoint helper (matches your VATE script)
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
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # raw state_dict

        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                if len(sd) > 0 and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd

    raise RuntimeError(
        "Unrecognized checkpoint format. Expected a state_dict or a dict containing "
        "'state_dict'/'model_state_dict'/'model'."
    )


# ---------------------------
# Dataframe helpers
# ---------------------------
def _sorted_indexed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    indexed: List[Tuple[int, str]] = []
    for c in cols:
        suf = c[len(prefix):]
        if re.fullmatch(r"-?\d+", suf):
            indexed.append((int(suf), c))
    if not indexed:
        raise KeyError(f"No indexed columns found for prefix '{prefix}' (expected e.g. {prefix}0, {prefix}1, ...)")
    indexed.sort(key=lambda t: t[0])
    return [c for _, c in indexed]


def _mat_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_indexed_cols(df, prefix)
    return df[cols].to_numpy(dtype=np.float32, copy=False)


def _pick_name_column(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise KeyError(f"--name-col='{user_col}' not found in dataframe columns.")
        return user_col

    candidates = [
        "real_name",
        "real",
        "real_text",
        "real_str",
        "real_label",
        "real_id",
        "real_filename",
        "real_file",
        "real_path",
        "name",
        "text",
        "filename",
        "path",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lowered = {c: str(c).lower() for c in df.columns}
    for c, lc in lowered.items():
        if "real" in lc and ("name" in lc or "path" in lc or "file" in lc):
            return c

    return "__index__"


# ---------------------------
# VATE embedding extraction (copied in spirit from your script)
# ---------------------------
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _project_text_features_if_needed(
    siamese_model: torch.nn.Module,
    text_features: torch.Tensor,
) -> torch.Tensor:
    for attr in ("text_projection", "text_proj", "proj_text", "projection_text"):
        if hasattr(siamese_model, attr):
            mod = getattr(siamese_model, attr)
            if callable(mod):
                return mod(text_features)
    return text_features


@torch.no_grad()
def _encode_text_batch(
    texts: List[str],
    siamese_model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    )

    if hasattr(siamese_model, "encode_text") and callable(getattr(siamese_model, "encode_text")):
        with autocast_ctx:
            out = siamese_model.encode_text(texts)  # type: ignore[attr-defined]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("siamese_model.encode_text(...) did not return a torch.Tensor.")
        return out.float().detach().cpu()

    if hasattr(siamese_model, "get_text_embeddings") and callable(getattr(siamese_model, "get_text_embeddings")):
        with autocast_ctx:
            out = siamese_model.get_text_embeddings(texts)  # type: ignore[attr-defined]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("siamese_model.get_text_embeddings(...) did not return a torch.Tensor.")
        return out.float().detach().cpu()

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
def embed_text_with_vate(
    text: str,
    *,
    siamese_model: torch.nn.Module,
    device: torch.device,
    do_l2_normalize: bool,
) -> np.ndarray:
    use_amp = device.type == "cuda"
    emb_cpu = _encode_text_batch([text], siamese_model=siamese_model, device=device, use_amp=use_amp)  # (1, D) on CPU
    if do_l2_normalize:
        emb_cpu = F.normalize(emb_cpu, dim=-1, eps=1e-8)
    vec = emb_cpu[0].numpy().astype(np.float32, copy=False)
    return vec


def load_vate_siamese_model(
    *,
    backbone: str,
    model_weights: str,
    embedding_dim: int,
    projection_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.nn.Module:
    tester = BaselineTester(model_type=backbone, batch_size=batch_size, device=str(device))
    backbone_module = tester.model_wrapper

    siamese_model = SiameseModelPairs(
        embedding_dim=int(embedding_dim),
        projection_dim=int(projection_dim),
        backbone=backbone_module,
    ).to(device)

    ckpt = torch.load(model_weights, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    return siamese_model


# ---------------------------
# Student model loading (auto-infer hidden_dim)
# ---------------------------
def _load_student_state(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    return ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt


def _infer_hidden_dim_from_student_state(state: dict, text_dim: int, out_dim: int) -> int:
    candidates: List[int] = []
    for _, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[1]) == int(text_dim):
            candidates.append(int(v.shape[0]))
    if not candidates:
        raise RuntimeError(
            f"Could not infer hidden_dim from student checkpoint: no 2D weight with second dim == text_dim ({text_dim})."
        )
    hidden_dim = max(candidates)

    found_out = False
    for _, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[0]) == int(out_dim) and int(v.shape[1]) == int(hidden_dim):
            found_out = True
            break
    if not found_out:
        print(
            f"[WARN] Could not find a (out_dim, hidden_dim)=({out_dim}, {hidden_dim}) 2D weight in student checkpoint. "
            f"Proceeding anyway."
        )
    return hidden_dim


# ---------------------------
# Main
# ---------------------------
@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="VATE(text) -> student -> nearest real_name in vate_test_student_only.parquet")
    ap.add_argument("--query", type=str, default=None, help="Query string. If omitted, will read from stdin.")

    # retrieval dataset
    ap.add_argument("--parquet", type=str, default="vate_test_student_only.parquet", help="Parquet/CSV containing real student embeddings.")
    ap.add_argument("--real-student-prefix", type=str, default="real_student_", help="Prefix for real-side student vectors.")
    ap.add_argument("--name-col", type=str, default=None, help="Name column to display (default auto-detect).")

    # student model
    ap.add_argument("--model-path", type=str, default="saved_models/best_model.pt", help="Path to best_model.pt checkpoint.")

    # VATE embedding model
    ap.add_argument("--vate-backbone", type=str, default="siglip", choices=["clip", "coca", "flava", "siglip"])
    ap.add_argument("--vate-model-weights", type=str, default="weights/best_model_siglip_pair.pt", help="Path to trained VATE Siamese model weights.")
    ap.add_argument("--vate-embedding-dim", type=int, default=768)
    ap.add_argument("--vate-projection-dim", type=int, default=768)
    ap.add_argument("--vate-batch-size", type=int, default=256)
    ap.add_argument("--strip-com", action="store_true")
    ap.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization for VATE embeddings (default is normalize).")

    # runtime
    ap.add_argument("--topk", type=int, default=1, help="How many top matches to print.")
    ap.add_argument("--batch-size", type=int, default=8192, help="Batch size for cosine computation over the dataset.")
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    query = args.query
    if query is None:
        query = input("Enter query string: ").strip()
    if not query:
        raise ValueError("Empty query string.")

    query = normalize_name(query, strip_com=bool(args.strip_com))

    # -------------------------
    # Load retrieval dataset
    # -------------------------
    if args.parquet.lower().endswith(".csv"):
        df = pd.read_csv(args.parquet)
    else:
        df = pd.read_parquet(args.parquet)

    name_col = _pick_name_column(df, args.name_col)
    if name_col == "__index__":
        names = np.arange(len(df))
    else:
        names = df[name_col].astype(str).to_numpy()

    real_mat = _mat_from_prefix(df, args.real_student_prefix)  # (N, D)
    out_dim = int(real_mat.shape[1])
    n = int(real_mat.shape[0])

    print(f"[INFO] Loaded {n:,} real vectors | out_dim={out_dim} | prefix='{args.real_student_prefix}'")
    print(f"[INFO] Using name column: {name_col}")

    real_mat_t = torch.from_numpy(real_mat).to(device)
    real_mat_t = F.normalize(real_mat_t, dim=1)

    # -------------------------
    # Load VATE Siamese + embed query
    # -------------------------
    if not os.path.exists(args.vate_model_weights):
        raise FileNotFoundError(f"VATE weights not found: {args.vate_model_weights}")

    vate_model = load_vate_siamese_model(
        backbone=args.vate_backbone,
        model_weights=args.vate_model_weights,
        embedding_dim=args.vate_embedding_dim,
        projection_dim=args.vate_projection_dim,
        batch_size=args.vate_batch_size,
        device=device,
    )

    vate_vec = embed_text_with_vate(
        query,
        siamese_model=vate_model,
        device=device,
        do_l2_normalize=(not bool(args.no_normalize)),
    )
    text_dim = int(vate_vec.shape[0])
    print(f"[INFO] VATE text_dim={text_dim}")

    # -------------------------
    # Load student model + project query
    # -------------------------
    student_state = _load_student_state(args.model_path, device=device)
    hidden_dim = _infer_hidden_dim_from_student_state(student_state, text_dim=text_dim, out_dim=out_dim)
    print(f"[INFO] Inferred student hidden_dim={hidden_dim}")

    student_model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        image_dim=out_dim,
    ).to(device)

    _ = student_model.load_state_dict(student_state, strict=False)
    student_model.eval()

    q = torch.from_numpy(vate_vec[None, :]).to(device)  # (1, text_dim)

    # Model signature: model(fraud_txt, real_txt) -> (z_f, z_r)
    _, z_r = student_model(q, q)
    q_student = F.normalize(z_r, dim=1)  # (1, out_dim)

    # -------------------------
    # Cosine retrieval (chunked)
    # -------------------------
    topk = int(max(1, args.topk))
    bs = int(max(1, args.batch_size))

    best_scores = torch.full((topk,), -1e9, device=device)
    best_indices = torch.full((topk,), -1, dtype=torch.long, device=device)

    for start in range(0, n, bs):
        end = min(start + bs, n)
        chunk = real_mat_t[start:end]  # (C, D)
        scores = (chunk @ q_student[0])  # (C,) since normalized

        k = min(topk, scores.numel())
        vals, idxs = torch.topk(scores, k=k, largest=True)
        idxs = idxs + start

        merged_scores = torch.cat([best_scores, vals], dim=0)
        merged_indices = torch.cat([best_indices, idxs], dim=0)

        vals2, pos2 = torch.topk(merged_scores, k=topk, largest=True)
        best_scores = vals2
        best_indices = merged_indices[pos2]

    best_scores_cpu = best_scores.detach().float().cpu().numpy()
    best_indices_cpu = best_indices.detach().cpu().numpy()

    print("\n==============================")
    print(" TOP MATCHES")
    print("==============================")
    for rank, (idx, score) in enumerate(zip(best_indices_cpu, best_scores_cpu), start=1):
        if int(idx) < 0:
            continue
        name = names[int(idx)]
        print(f"{rank:>2}. idx={int(idx):>6} | cos={float(score):.6f} | name={name}")
    print("==============================\n")


if __name__ == "__main__":
    main()

"""
python inference/single_inference_test.py \
  --query "g3nb00k" \
  --parquet text_to_image/evaluation/vate_test_student_only.parquet \
  --model-path saved_models/best_model.pt \
  --vate-backbone siglip \
  --vate-model-weights weights/best_model_siglip_pair.pt \
  --topk 5
"""