#!/usr/bin/env python3
"""
Append PRECOMPUTED VATE text embeddings to a golden parquet.

This version:
- DOES NOT load any model
- DOES NOT compute embeddings
- Simply loads a VATE parquet that already contains:
    fraud_emb_0 ... fraud_emb_{D-1}
    real_emb_0  ... real_emb_{D-1}
- Renames them to:
    fraud_txt_emb_*
    real_txt_emb_*
- Appends to the input dataframe by row order

Example:
python seton_notebooks/create_golden_with_VATE_embeddings_precomputed.py \
  --input text_to_image/Golden/golden_embeddings_test.parquet \
  --vate-parquet text_to_image/evaluation/vate_test.parquet \
  --output text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --overwrite
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


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
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vate-parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output}")

    # ---------------------------
    # Load data
    # ---------------------------
    df = load_table(args.input).copy()
    vate_df = load_table(args.vate_parquet).copy()

    if len(df) != len(vate_df):
        raise RuntimeError(
            f"Row mismatch: input={len(df)} vate={len(vate_df)}"
        )

    # ---------------------------
    # Find embedding columns
    # ---------------------------
    fraud_cols = [c for c in vate_df.columns if c.startswith("fraud_txt_emb_")]
    real_cols  = [c for c in vate_df.columns if c.startswith("real_txt_emb_")]

    if not fraud_cols or not real_cols:
        raise RuntimeError("VATE parquet missing fraud_emb_* or real_emb_* columns")

    fraud_cols = sorted(fraud_cols, key=lambda x: int(x.split("_")[-1]))
    real_cols  = sorted(real_cols,  key=lambda x: int(x.split("_")[-1]))

    dim = len(fraud_cols)
    print(f"[INFO] detected VATE embedding dim = {dim}")

    # ---------------------------
    # Rename to expected schema
    # ---------------------------
    rename_map = {}
    for i, c in enumerate(fraud_cols):
        rename_map[c] = f"fraud_txt_emb_{i}"
    for i, c in enumerate(real_cols):
        rename_map[c] = f"real_txt_emb_{i}"

    vate_df = vate_df.rename(columns=rename_map)

    new_fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    new_real_cols  = [f"real_txt_emb_{i}" for i in range(dim)]

    # ---------------------------
    # Safety: no column collisions
    # ---------------------------
    for c in new_fraud_cols + new_real_cols:
        if c in df.columns:
            raise RuntimeError(f"Column collision detected: {c}")

    # ---------------------------
    # Append embeddings
    # ---------------------------
    text_df = vate_df[new_fraud_cols + new_real_cols].astype(np.float32)
    text_df.index = df.index  # critical: preserve row alignment

    out_df = pd.concat([df, text_df], axis=1)

    save_table(out_df, args.output)
    print(f"[INFO] wrote → {args.output}")


if __name__ == "__main__":
    main()