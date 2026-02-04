#!/usr/bin/env python3
"""
build_t2i_positive_only_dataset.py

For each original row, produce ONLY positive training pairs:

  1) fraud_txt -> fraud_img  (label = 1)
  2) real_txt  -> real_img   (label = 1)

No cross-pairs.
No negatives.
Output has 2*N rows.

Output columns:
- left_txt_emb_*   : text embedding
- right_img_emb_*  : image/aligned embedding
- label            : always 1
- pair_kind
- orig_row_id
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# helpers
# -------------------------

def _infer_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix={prefix!r}")

    def key(c: str):
        s = c[len(prefix):]
        return int(s) if s.isdigit() else s

    return sorted(cols, key=key)


def _as_float32_2d(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    arr = df[cols].to_numpy()
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim != 2:
        raise ValueError("Expected 2D embedding array")

    return arr


# -------------------------
# main builder
# -------------------------

def build_positive_only_dataset(
    input_parquet: str,
    output_parquet: str,
    *,
    fraud_txt_prefix: str = "fraud_txt_emb_",
    real_txt_prefix: str = "real_txt_emb_",
    fraud_img_prefix: str = "fraud_aligned_",
    real_img_prefix: str = "real_aligned_",
    keep_metadata_cols: Tuple[str, ...] = ("real_name", "fraud_name"),
) -> None:

    df = pd.read_parquet(input_parquet)
    N = len(df)
    if N == 0:
        raise ValueError("Input parquet has 0 rows")

    # Infer embedding columns
    ftxt_cols = _infer_cols(df, fraud_txt_prefix)
    rtxt_cols = _infer_cols(df, real_txt_prefix)
    fimg_cols = _infer_cols(df, fraud_img_prefix)
    rimg_cols = _infer_cols(df, real_img_prefix)

    # Convert to arrays
    ftxt = _as_float32_2d(df, ftxt_cols)
    rtxt = _as_float32_2d(df, rtxt_cols)
    fimg = _as_float32_2d(df, fimg_cols)
    rimg = _as_float32_2d(df, rimg_cols)

    text_dim = ftxt.shape[1]
    img_dim = fimg.shape[1]

    if rtxt.shape[1] != text_dim:
        raise ValueError("fraud_txt and real_txt dims mismatch")
    if rimg.shape[1] != img_dim:
        raise ValueError("fraud_img and real_img dims mismatch")

    # Allocate outputs (2N rows)
    out_left = np.empty((2 * N, text_dim), dtype=np.float32)
    out_right = np.empty((2 * N, img_dim), dtype=np.float32)
    out_label = np.ones((2 * N,), dtype=np.int8)

    # Block 1: fraud_txt -> fraud_img
    out_left[0:N] = ftxt
    out_right[0:N] = fimg

    # Block 2: real_txt -> real_img
    out_left[N:2 * N] = rtxt
    out_right[N:2 * N] = rimg

    # Build DataFrame
    out = pd.DataFrame({
        "orig_row_id": np.repeat(np.arange(N, dtype=np.int64), 2),
        "pair_kind": (
            ["fraud_txt__fraud_img"] * N
            + ["real_txt__real_img"] * N
        ),
        "label": out_label.astype(np.int64),
    })

    # Carry metadata if present
    for c in keep_metadata_cols:
        if c in df.columns:
            out[c] = np.repeat(df[c].to_numpy(), 2)

    # Add embedding columns
    out_left_cols = [f"left_txt_emb_{i}" for i in range(text_dim)]
    out_right_cols = [f"right_img_emb_{i}" for i in range(img_dim)]

    out[out_left_cols] = out_left
    out[out_right_cols] = out_right

    # Write parquet
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    out.to_parquet(output_parquet, index=False)

    # Summary
    print("DONE")
    print(f"Input rows:  {N}")
    print(f"Output rows: {len(out)} (2x)")
    print("label counts:", out["label"].value_counts().to_dict())
    print("pair_kind counts:", out["pair_kind"].value_counts().to_dict())
    print(f"text_dim={text_dim}, img_dim={img_dim}")


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fraud-txt-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-txt-prefix", default="real_txt_emb_")
    ap.add_argument("--fraud-img-prefix", default="fraud_aligned_")
    ap.add_argument("--real-img-prefix", default="real_aligned_")
    args = ap.parse_args()

    build_positive_only_dataset(
        input_parquet=args.input,
        output_parquet=args.output,
        fraud_txt_prefix=args.fraud_txt_prefix,
        real_txt_prefix=args.real_txt_prefix,
        fraud_img_prefix=args.fraud_img_prefix,
        real_img_prefix=args.real_img_prefix,
    )


if __name__ == "__main__":
    main()

"""
python seton_notebooks/build_t2i_positive_only_dataset.py \
  --input  text_to_image/Golden_and_Text/validate_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/Golden_and_Text/validate.parquet
"""