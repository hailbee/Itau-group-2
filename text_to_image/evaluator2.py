#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from siamese import SiameseEmbeddingModel


# -------------------------
# Helpers
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def _mat(df: pd.DataFrame, prefix: str) -> torch.Tensor:
    cols = _sorted_prefixed_cols(df, prefix)
    return torch.tensor(df[cols].to_numpy(dtype=np.float32, copy=False))


def pick_device(device_override=None):
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate trained text→image embeddings using cosine similarity (ROC AUC)"
    )

    # files
    ap.add_argument("--test", required=True, help="Test parquet or csv")
    ap.add_argument(
        "--model-path",
        default="saved_models/best_model.pt",
        help="Path to trained model checkpoint",
    )

    # architecture (MUST MATCH TRAINING)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--out-dim", type=int, required=True)

    # prefixes
    ap.add_argument("--fraud-prefix", default="fraud_emb_")
    ap.add_argument("--real-prefix", default="real_emb_")
    ap.add_argument("--label-col", default="label")

    # runtime
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # -------------------------
    # Load data
    # -------------------------
    df = (
        pd.read_parquet(args.test)
        if args.test.endswith(".parquet")
        else pd.read_csv(args.test)
    )

    y = df[args.label_col].astype(int).to_numpy()

    fraud_txt = _mat(df, args.fraud_prefix)
    real_txt = _mat(df, args.real_prefix)

    text_dim = fraud_txt.shape[1]

    print(
        f"[INFO] text_dim={text_dim} | hidden_dim={args.hidden_dim} | out_dim={args.out_dim}"
    )

    # -------------------------
    # Load model
    # -------------------------
    ckpt = torch.load(args.model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    model = SiameseEmbeddingModel(
        embedding_dim=text_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    print("[INFO] Model loaded successfully")

    # -------------------------
    # Compute similarities
    # -------------------------
    sims = []
    bs = int(args.batch_size)

    for start in range(0, len(df), bs):
        end = min(start + bs, len(df))

        f = fraud_txt[start:end].to(device)
        r = real_txt[start:end].to(device)

        z_f, z_r = model(f, r)

        z_f = F.normalize(z_f, dim=1)
        z_r = F.normalize(z_r, dim=1)

        sims.append(F.cosine_similarity(z_f, z_r, dim=1).cpu())

    sims = torch.cat(sims).numpy()

    # -------------------------
    # Metrics
    # -------------------------
    roc_auc = float(roc_auc_score(y, sims))

    print("\n==============================")
    print(" STUDENT EMBEDDING EVALUATION")
    print("==============================")
    print(f"ROC AUC (cosine): {roc_auc:.6f}")
    print("==============================\n")


if __name__ == "__main__":
    main()

"""

CHANGE HIDDEN AND OUT IF NEEDED
python text_to_image/evaluator2.py \
  --test text_to_image/evaluation/vate_test.parquet \
  --hidden-dim 768 \
  --out-dim 768


"""