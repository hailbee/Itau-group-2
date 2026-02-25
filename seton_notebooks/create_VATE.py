#!/usr/bin/env python3
"""
Create VATE text embeddings EXACTLY like Evaluator, write them into a parquet/CSV, and optionally save a VATE-only file.

Key properties (to match Evaluator / fix AUC):
- Loads SiameseModelPairs like main.py (evaluate_saved mode).
- Uses EmbeddingExtractor / SupConEmbeddingExtractor + batched_embedding (same as Evaluator).
- Computes cosine similarity with torch.nn.functional.cosine_similarity (same as Evaluator).
- Prints sanity ROC AUC (Evaluator-style).

I/O behavior:
- If --vate-only-output is provided, OVERWRITES it if it exists (always).
- By default, hard-fails on embedding column collisions.
- With --overwrite-cols, overwrites fraud_txt_emb_* / real_txt_emb_* if they already exist.

Example:
python3 seton_notebooks/create_VATE.py \
  --input ../Downloads/validate_pairs_with_siglip_embeddings.parquet \
  --vate-only-output ../Downloads/vate_validate.parquet \
  --vate-include-keys fraudulent_name real_name label \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256 \
  --overwrite-cols
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.learning.siamese import SiameseModelPairs
from utils.embeddings import EmbeddingExtractor, SupConEmbeddingExtractor, batched_embedding


# ---------------------------
# IO
# ---------------------------
def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # pandas overwrites by default
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


# ---------------------------
# Checkpoint helpers
# ---------------------------
def load_checkpoint_safely(path: str, map_location: torch.device) -> Any:
    # torch>=2.6 supports weights_only
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # main.py assumes raw state_dict; support common wrappers too.
    if isinstance(ckpt, dict):
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        for k in ("state_dict", "model_state_dict", "model", "model_state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                if len(sd) > 0 and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd
    raise RuntimeError("Unrecognized checkpoint format.")


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def load_state_dict_robust(model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        for p in ("module.", "model.", "net.", "encoder.", "siamese_model."):
            if any(k.startswith(p) for k in sd.keys()):
                model.load_state_dict(_strip_prefix(sd, p), strict=True)
                print(f"[INFO] Loaded after stripping prefix: {p!r}")
                return
        raise


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)

    ap.add_argument("--backbone", default="siglip", choices=["clip", "coca", "flava", "siglip"])
    ap.add_argument("--model-weights", required=True)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--projection-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)

    # Only if you trained with these modes; otherwise leave default (matches evaluate_saved)
    ap.add_argument("--model-type", default=None, help="use 'supcon' or 'infonce' to select SupConEmbeddingExtractor")

    # IMPORTANT: match evaluator behavior if you want identical AUC numbers
    ap.add_argument("--head1024", action="store_true", help="If set, embeds only df.head(1024) (like Evaluator)")

    # Column overwrite behavior
    ap.add_argument(
        "--overwrite-cols",
        action="store_true",
        help="If fraud_txt_emb_* / real_txt_emb_* already exist, overwrite them in-place instead of failing.",
    )

    # VATE-only output
    ap.add_argument(
        "--vate-only-output",
        default=None,
        help="If set, also write a second file containing ONLY the VATE text embedding columns (plus optional keys).",
    )
    ap.add_argument(
        "--vate-include-keys",
        nargs="*",
        default=[],
        help="Column names to keep in the VATE-only output (e.g., fraudulent_name real_name label).",
    )

    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df = load_table(args.input).copy()

    if args.head1024:
        df = df.head(1024).copy()
        print("[INFO] Using df.head(1024) to match Evaluator")

    # Require columns (Evaluator expects these)
    for col in ("fraudulent_name", "real_name", "label"):
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    # Build backbone wrapper exactly like main.py
    tester = BaselineTester(model_type=args.backbone, batch_size=args.batch_size, device=str(device))
    backbone_module = tester.model_wrapper

    # Build SiameseModelPairs exactly like main.py
    siamese_model = SiameseModelPairs(
        embedding_dim=int(args.embedding_dim),
        projection_dim=int(args.projection_dim),
        backbone=backbone_module,
    ).to(device)

    ckpt = load_checkpoint_safely(args.model_weights, map_location=device)
    sd = _extract_state_dict(ckpt)
    load_state_dict_robust(siamese_model, sd)
    siamese_model.eval()

    # EXACT extractor selection logic from Evaluator
    if args.model_type in ["supcon", "infonce"]:
        print("[INFO] USING SUPCON EMBEDDING EXTRACTOR (matches Evaluator)")
        extractor = SupConEmbeddingExtractor(siamese_model)
    else:
        print("[INFO] USING STANDARD EMBEDDING EXTRACTOR (matches Evaluator)")
        extractor = EmbeddingExtractor(siamese_model)

    fraud_names = df["fraudulent_name"].astype(str).tolist()
    real_names = df["real_name"].astype(str).tolist()

    # EXACT embedding calls from Evaluator
    fraud_embs = batched_embedding(extractor, fraud_names, args.batch_size)
    real_embs = batched_embedding(extractor, real_names, args.batch_size)

    if not isinstance(fraud_embs, torch.Tensor) or not isinstance(real_embs, torch.Tensor):
        raise RuntimeError("batched_embedding did not return torch.Tensor(s).")

    # Save raw embeddings (don’t renormalize here; cosine_similarity handles it like Evaluator)
    fraud_np = fraud_embs.detach().cpu().to(torch.float32).numpy()
    real_np = real_embs.detach().cpu().to(torch.float32).numpy()

    if fraud_np.shape != real_np.shape:
        raise RuntimeError(f"Embedding shape mismatch: fraud={fraud_np.shape}, real={real_np.shape}")

    dim = fraud_np.shape[1]
    print(f"[INFO] text emb dim={dim}")

    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols = [f"real_txt_emb_{i}" for i in range(dim)]
    embed_cols = fraud_cols + real_cols

    # Handle collisions
    collisions = [c for c in embed_cols if c in df.columns]
    if collisions and not args.overwrite_cols:
        raise RuntimeError(
            "Column collision(s) detected.\n"
            f"Collisions: {collisions[:10]}{' ...' if len(collisions) > 10 else ''}\n"
            "If you intend to overwrite these columns, pass --overwrite-cols."
        )

    if collisions and args.overwrite_cols:
        # Remove existing embedding columns so concat is clean and index-aligned
        df = df.drop(columns=collisions)

    # Build embedding dataframe with explicit index to prevent alignment bugs
    text_df = pd.DataFrame(
        np.hstack([fraud_np, real_np]),
        columns=embed_cols,
        index=df.index,
        dtype=np.float32,
    )

    out_df = pd.concat([df, text_df], axis=1)

    # Optional VATE-only output (OVERWRITES by default)
    if args.vate_only_output is not None:
        keep_keys: List[str] = []
        for k in args.vate_include_keys:
            if k not in out_df.columns:
                raise RuntimeError(f"Requested key column for VATE-only output not found: {k}")
            keep_keys.append(k)

        vate_only_cols = keep_keys + embed_cols
        vate_df = out_df.loc[:, vate_only_cols].copy()

        # Enforce float32 on embedding columns explicitly
        for c in embed_cols:
            vate_df[c] = vate_df[c].astype(np.float32, copy=False)

        save_table(vate_df, args.vate_only_output)
        print(f"[INFO] wrote VATE-only file → {args.vate_only_output}")

    # Sanity: print the AUC computed exactly like Evaluator (cosine sim + roc_curve/auc)
    with torch.no_grad():
        sims = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()

    y = out_df["label"].astype(float).to_numpy()

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, sims)
    print(f"[INFO] sanity ROC AUC (Evaluator-style): {auc(fpr, tpr):.4f}")


if __name__ == "__main__":
    main()
