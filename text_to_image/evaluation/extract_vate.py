#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Drop aligned image embeddings and rename text embedding prefixes"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet file",
    )
    args = parser.parse_args()

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_parquet(args.input)

    # -------------------------
    # Drop aligned image columns
    # -------------------------
    drop_cols = [
        c for c in df.columns
        if c.startswith("fraud_aligned_") or c.startswith("real_aligned_")
    ]

    if drop_cols:
        print(f"[INFO] Dropping {len(drop_cols)} aligned columns")
        df = df.drop(columns=drop_cols)
    else:
        print("[INFO] No aligned columns found to drop")

    # -------------------------
    # Rename text embedding columns
    # -------------------------
    rename_map = {}

    for c in df.columns:
        if c.startswith("fraud_txt_emb_"):
            rename_map[c] = c.replace("fraud_txt_emb_", "fraud_emb_")
        elif c.startswith("real_txt_emb_"):
            rename_map[c] = c.replace("real_txt_emb_", "real_emb_")

    if rename_map:
        print(f"[INFO] Renaming {len(rename_map)} text embedding columns")
        df = df.rename(columns=rename_map)
    else:
        print("[INFO] No text embedding columns found to rename")

    # -------------------------
    # Save
    # -------------------------
    df.to_parquet(args.output, index=False)
    print(f"[DONE] Wrote output to {args.output}")


if __name__ == "__main__":
    main()

"""
python text_to_image/evaluation/extract_vate.py \
  --input text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/evaluation/vate_test.parquet
"""