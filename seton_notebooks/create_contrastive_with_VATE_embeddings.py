#!/usr/bin/env python3
"""
build_t2i_4pairs_dataset.py

For each original row, produce 4 training pairs:

Always-positive (label=1):
  1) fraud_txt  -> fraud_img
  2) real_txt   -> real_img

Cross pairs (label = original spoof label):
  3) fraud_txt  -> real_img   (label = spoof_attempt)
  4) real_txt   -> fraud_img  (label = spoof_attempt)

Output is a single parquet with 4*N rows:
- left_txt_emb_* : text vector (fraud or real)
- right_img_emb_*: image/aligned vector (fraud or real)
- label: 0/1
- pair_kind, orig_row_id

Run:
  python3 build_t2i_4pairs_dataset.py \
    --input  Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
    --output Golden_and_Text/t2i_4pairs.parquet \
    --label-col spoof_attempt \
    --fraud-txt-prefix fraud_txt_emb_ \
    --real-txt-prefix real_txt_emb_ \
    --fraud-img-prefix fraud_aligned_ \
    --real-img-prefix real_aligned_
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _infer_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix={prefix!r}")
    # Sort by trailing integer if present, else lexicographic
    def key(c: str):
        s = c[len(prefix):]
        return int(s) if s.isdigit() else s
    return sorted(cols, key=key)


def _as_float32_2d(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    arr = df[cols].to_numpy()
    # ensure numeric
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array for embeddings")
    return arr


def build_4pairs_dataset(
    input_parquet: str,
    output_parquet: str,
    *,
    label_col: str = "spoof_attempt",
    fraud_txt_prefix: str = "fraud_txt_emb_",
    real_txt_prefix: str = "real_txt_emb_",
    fraud_img_prefix: str = "fraud_aligned_",
    real_img_prefix: str = "real_aligned_",
    keep_metadata_cols: Tuple[str, ...] = ("real_name", "fraud_name"),
) -> None:
    df = pd.read_parquet(input_parquet)
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col={label_col!r}. Available: {list(df.columns[:50])} ...")

    # Get embedding columns
    ftxt_cols = _infer_cols(df, fraud_txt_prefix)
    rtxt_cols = _infer_cols(df, real_txt_prefix)
    fimg_cols = _infer_cols(df, fraud_img_prefix)
    rimg_cols = _infer_cols(df, real_img_prefix)

    # Convert to arrays (float32)
    ftxt = _as_float32_2d(df, ftxt_cols)
    rtxt = _as_float32_2d(df, rtxt_cols)
    fimg = _as_float32_2d(df, fimg_cols)
    rimg = _as_float32_2d(df, rimg_cols)

    N = len(df)
    if N == 0:
        raise ValueError("Input parquet has 0 rows")

    # Labels must be 0/1
    y = df[label_col].to_numpy()
    # robust cast: accepts bool, int, floats 0/1
    y = (y.astype(np.int64) != 0).astype(np.int8)

    text_dim = ftxt.shape[1]
    img_dim = fimg.shape[1]
    if rtxt.shape[1] != text_dim:
        raise ValueError(f"text dims mismatch: fraud_txt={text_dim}, real_txt={rtxt.shape[1]}")
    if rimg.shape[1] != img_dim:
        raise ValueError(f"img dims mismatch: fraud_img={img_dim}, real_img={rimg.shape[1]}")

    # Allocate output arrays: 4N rows
    out_left = np.empty((4 * N, text_dim), dtype=np.float32)
    out_right = np.empty((4 * N, img_dim), dtype=np.float32)
    out_label = np.empty((4 * N,), dtype=np.int8)

    # Block 1: fraud_txt -> fraud_img (label=1)
    out_left[0:N] = ftxt
    out_right[0:N] = fimg
    out_label[0:N] = 1

    # Block 2: real_txt -> real_img (label=1)
    out_left[N:2 * N] = rtxt
    out_right[N:2 * N] = rimg
    out_label[N:2 * N] = 1

    # Block 3: fraud_txt -> real_img (label = y)
    out_left[2 * N:3 * N] = ftxt
    out_right[2 * N:3 * N] = rimg
    out_label[2 * N:3 * N] = y

    # Block 4: real_txt -> fraud_img (label = y)
    out_left[3 * N:4 * N] = rtxt
    out_right[3 * N:4 * N] = fimg
    out_label[3 * N:4 * N] = y

    # Build output DataFrame (wide embeddings)
    out = pd.DataFrame({
        "orig_row_id": np.tile(np.arange(N, dtype=np.int64), 4),
        "pair_kind": (
            ["fraud_txt__fraud_img"] * N
            + ["real_txt__real_img"] * N
            + ["fraud_txt__real_img"] * N
            + ["real_txt__fraud_img"] * N
        ),
        "label": out_label.astype(np.int64),
    })

    # Optional: carry through metadata if present
    for c in keep_metadata_cols:
        if c in df.columns:
            out[c] = np.tile(df[c].to_numpy(), 4)

    # Add embedding columns
    out_left_cols = [f"left_txt_emb_{i}" for i in range(text_dim)]
    out_right_cols = [f"right_img_emb_{i}" for i in range(img_dim)]
    out[out_left_cols] = out_left
    out[out_right_cols] = out_right

    # Write
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    out.to_parquet(output_parquet, index=False)

    # Summary
    print("DONE")
    print(f"Input rows: {N}")
    print(f"Output rows: {len(out)} (should be 4x)")
    print("label counts:", out["label"].value_counts().to_dict())
    print("pair_kind counts:", out["pair_kind"].value_counts().to_dict())
    print(f"text_dim={text_dim}, img_dim={img_dim}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label-col", default="spoof_attempt")
    ap.add_argument("--fraud-txt-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-txt-prefix", default="real_txt_emb_")
    ap.add_argument("--fraud-img-prefix", default="fraud_aligned_")
    ap.add_argument("--real-img-prefix", default="real_aligned_")
    args = ap.parse_args()

    build_4pairs_dataset(
        input_parquet=args.input,
        output_parquet=args.output,
        label_col=args.label_col,
        fraud_txt_prefix=args.fraud_txt_prefix,
        real_txt_prefix=args.real_txt_prefix,
        fraud_img_prefix=args.fraud_img_prefix,
        real_img_prefix=args.real_img_prefix,
    )


if __name__ == "__main__":
    main()

"""
Example:

python ../seton_notebooks/create_contrastive_with_VATE_embeddings.py \
  --input Golden_and_Text/validate_pairs_with_img_and_vate_txt_embs.parquet \
  --output Golden_and_Text/validate_4pairs.parquet \
  --label-col label \
  --fraud-txt-prefix fraud_txt_emb_ \
  --real-txt-prefix real_txt_emb_ \
  --fraud-img-prefix fraud_aligned_ \
  --real-img-prefix real_aligned_
"""

"""
Example:

python ../seton_notebooks/create_contrastive_with_VATE_embeddings.py \
  --input  Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --output Golden_and_Text/test_contrastive_pairs.parquet \
  --negatives-per-row 1 \
  --seed 123 \
  --require-different-real-name \
  --parquet-compression snappy \
  --chunk-size 200000
"""