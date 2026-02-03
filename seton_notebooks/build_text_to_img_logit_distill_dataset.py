#!/usr/bin/env python3
"""
Build dataset for LOGIT DISTILLATION (text → image similarity).

Input file MUST contain:
  fraud_txt_emb_0..D-1
  real_txt_emb_0..D-1
  fraud_aligned_0..K-1
  real_aligned_0..K-1

Output:
  fraud_txt_*
  real_txt_*
  teacher_cos
  teacher_logit
  (+ optional label / metadata)
"""

import argparse, os, re
import numpy as np
import pandas as pd


# -----------------------------
# helpers
# -----------------------------
def prefixed_cols(df, prefix):
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'")
    cols.sort(key=lambda c: int(c[len(prefix):]))
    return cols


def l2norm(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--keep-cols", default="")
    ap.add_argument("--teacher-scale", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    N = len(df)
    if N == 0:
        raise ValueError("Input file is empty")

    # ---- REQUIRED prefixes (NO GUESSING)
    fraud_txt_cols = prefixed_cols(df, "fraud_txt_emb_")
    real_txt_cols  = prefixed_cols(df, "real_txt_emb_")
    fraud_img_cols = prefixed_cols(df, "fraud_aligned_")
    real_img_cols  = prefixed_cols(df, "real_aligned_")

    # ---- load arrays
    fraud_txt = df[fraud_txt_cols].to_numpy(dtype=np.float32)
    real_txt  = df[real_txt_cols].to_numpy(dtype=np.float32)
    fraud_img = df[fraud_img_cols].to_numpy(dtype=np.float32)
    real_img  = df[real_img_cols].to_numpy(dtype=np.float32)

    if fraud_txt.shape != real_txt.shape:
        raise ValueError("fraud_txt and real_txt dims mismatch")
    if fraud_img.shape != real_img.shape:
        raise ValueError("fraud_img and real_img dims mismatch")

    D = fraud_txt.shape[1]
    K = fraud_img.shape[1]

    # ---- teacher cosine + logit
    fraud_img_n = l2norm(fraud_img)
    real_img_n  = l2norm(real_img)
    teacher_cos = np.sum(fraud_img_n * real_img_n, axis=1).astype(np.float32)
    teacher_logit = (args.teacher_scale * teacher_cos).astype(np.float32)

    # ---- output dataframe
    out = pd.DataFrame({"orig_row_id": np.arange(N, dtype=np.int64)})

    # keep metadata
    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"keep-col '{c}' not found in input")
        out[c] = df[c].to_numpy()

    if args.label_col:
        if args.label_col not in df.columns:
            raise ValueError(f"label-col '{args.label_col}' not found")
        out[args.label_col] = df[args.label_col].to_numpy()

    out["teacher_cos"] = teacher_cos
    out["teacher_logit"] = teacher_logit

    # ---- store text embeddings
    out[[f"fraud_txt_{i}" for i in range(D)]] = fraud_txt
    out[[f"real_txt_{i}" for i in range(D)]]  = real_txt

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_parquet(args.output, index=False)

    print("DONE")
    print("rows:", N)
    print("text_dim:", D, "img_dim:", K)
    print("teacher_cos mean/std:", teacher_cos.mean(), teacher_cos.std())
    print("teacher_logit scale:", args.teacher_scale)


if __name__ == "__main__":
    main()
    
    """
    python seton_notebooks/build_text_to_img_logit_distill_dataset.py \
  --input text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/Golden_and_Text/test_logit_distill.parquet \
  --label-col label \
  --keep-cols real_name,fraudulent_name
"""
