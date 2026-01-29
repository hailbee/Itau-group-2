#!/usr/bin/env python3
"""
Append TEXT embeddings that match Evaluator.test_pairs exactly.

- Reads an existing "golden" parquet (already contains fraud_aligned_* / real_aligned_*).
- Loads the trained SiameseModelPairs exactly like main.py (evaluate_saved mode).
- Uses the same extractor + batched_embedding path as Evaluator.
- Appends fraud_txt_emb_* and real_txt_emb_* (float32), aligned by row order.

Example:
python seton_notebooks/create_golden_with_VATE_embeddings.py \
  --input text_to_image/Golden/golden_embeddings_test.parquet \
  --output text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256 \
  --device cuda
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

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
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backbone", default="siglip", choices=["clip", "coca", "flava", "siglip"])
    ap.add_argument("--model-weights", required=True)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--projection-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")

    # Only if you trained with these modes; otherwise leave default (matches evaluate_saved)
    ap.add_argument("--model-type", default=None, help="use 'supcon' or 'infonce' to select SupConEmbeddingExtractor")

    # IMPORTANT: match evaluator behavior if you want identical AUC numbers
    ap.add_argument("--head1024", action="store_true", help="If set, embeds only df.head(1024) (like Evaluator)")

    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df = load_table(args.input).copy()

    if args.head1024:
        df = df.head(1024).copy()
        print("[INFO] Using df.head(1024) to match Evaluator")

    # Require columns
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
    real_names  = df["real_name"].astype(str).tolist()

    # EXACT embedding calls from Evaluator
    fraud_embs = batched_embedding(extractor, fraud_names, args.batch_size)
    real_embs  = batched_embedding(extractor, real_names, args.batch_size)

    if not isinstance(fraud_embs, torch.Tensor) or not isinstance(real_embs, torch.Tensor):
        raise RuntimeError("batched_embedding did not return torch.Tensor(s).")

    # Save raw embeddings (don’t renormalize here; cosine_similarity handles it like Evaluator)
    fraud_np = fraud_embs.detach().cpu().to(torch.float32).numpy()
    real_np  = real_embs.detach().cpu().to(torch.float32).numpy()
    dim = fraud_np.shape[1]
    print(f"[INFO] text emb dim={dim}")

    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols  = [f"real_txt_emb_{i}" for i in range(dim)]
    for c in fraud_cols + real_cols:
        if c in df.columns:
            raise RuntimeError(f"Column collision detected: {c}")

    text_df = pd.DataFrame(
        np.hstack([fraud_np, real_np]),
        columns=fraud_cols + real_cols,
        index=df.index,           # prevents alignment bugs
        dtype=np.float32,
    )

    out_df = pd.concat([df, text_df], axis=1)
    save_table(out_df, args.output)
    print(f"[INFO] wrote → {args.output}")

    # Optional: print the AUC computed exactly like Evaluator (sanity)
    with torch.no_grad():
        sims = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()
    y = out_df["label"].astype(float).to_numpy()
    # quick AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, sims)
    print(f"[INFO] sanity ROC AUC (Evaluator-style): {auc(fpr,tpr):.4f}")


if __name__ == "__main__":
    main()
