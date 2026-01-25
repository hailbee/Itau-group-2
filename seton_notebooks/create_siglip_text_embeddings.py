#!/usr/bin/env python3
"""
Append SigLIP TEXT embeddings to an existing file that already contains IMAGE embeddings.

Input file: already contains at least these columns:
  - fraudulent_name
  - real_name
  - label   (optional but you said first 3 columns are these)

This script:
  - reads those name columns,
  - computes SigLIP text embeddings for each unique string,
  - appends columns:
      fraud_txt_emb_0..D-1
      real_txt_emb_0..D-1
  - keeps ALL existing columns intact (including image embeddings)

Example:
  python create_siglip_text_embeddings.py \
    --input 'Downloads/validate_pairs_with_siglip_embeddings' \
    --output validate_pairs_with_img_and_txt_embs.parquet \
    --batch-size 256 \
    --strip-com
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoProcessor, SiglipTextModel


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
# Normalization
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
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


# ---------------------------
# Embedding
# ---------------------------
@torch.no_grad()
def embed_unique_texts(
    uniq_texts: List[str],
    model: SiglipTextModel,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
    do_l2_normalize: bool,
    max_length: Optional[int],
) -> np.ndarray:
    """
    Returns float32 embeddings of shape (N, D) on CPU.
    Uses model.pooler_output.
    """
    n = len(uniq_texts)
    if n == 0:
        raise ValueError("No texts to embed.")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    # first batch to infer dim
    first = uniq_texts[: min(batch_size, n)]
    b0 = processor(
        text=first,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    b0 = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in b0.items()}

    with autocast_ctx:
        out0 = model(**b0)
        e0 = out0.pooler_output

    if do_l2_normalize:
        e0 = F.normalize(e0, dim=-1, eps=1e-8)

    e0 = e0.float().cpu().numpy()
    dim = int(e0.shape[1])

    emb = np.empty((n, dim), dtype=np.float32)
    emb[: len(first)] = e0

    for start in tqdm(range(len(first), n, batch_size), desc="Embedding text batches"):
        chunk = uniq_texts[start : start + batch_size]
        b = processor(
            text=chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        b = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in b.items()}

        with autocast_ctx:
            out = model(**b)
            e = out.pooler_output

        if do_l2_normalize:
            e = F.normalize(e, dim=-1, eps=1e-8)

        emb[start : start + len(chunk)] = e.float().cpu().numpy()

    return emb


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Existing CSV/Parquet that already has image embeddings")
    ap.add_argument("--output", required=True, help="Output CSV/Parquet with image + text embeddings")
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help='Override: "cuda", "cpu", or "mps"')
    ap.add_argument("--strip-com", action="store_true")
    ap.add_argument("--no-normalize", action="store_true", help="Disable L2-normalization of text embeddings")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting output file if it exists")
    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output} (use --overwrite)")

    df = load_table(args.input)

    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    else:
        df = df.copy()

    # You said the first 3 columns are: fraudulent_name, real_name, label
    # We'll use column names if present; otherwise fall back to first columns.
    if "fraudulent_name" in df.columns and "real_name" in df.columns:
        fraud_col = "fraudulent_name"
        real_col = "real_name"
    else:
        fraud_col = df.columns[0]
        real_col = df.columns[1]

    # Normalize names for embedding consistency (optional .com stripping)
    df[fraud_col] = df[fraud_col].map(lambda x: normalize_name(x, args.strip_com))
    df[real_col] = df[real_col].map(lambda x: normalize_name(x, args.strip_com))

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = SiglipTextModel.from_pretrained(args.model, torch_dtype=torch_dtype).to(device).eval()

    # Deduplicate strings across both columns
    all_texts = pd.concat([df[fraud_col], df[real_col]], ignore_index=True).astype(str)
    uniq = pd.unique(all_texts).tolist()
    print(f"[INFO] unique strings={len(uniq):,}")

    emb_mat = embed_unique_texts(
        uniq_texts=uniq,
        model=model,
        processor=processor,
        device=device,
        batch_size=int(args.batch_size),
        do_l2_normalize=(not args.no_normalize),
        max_length=args.max_length,
    )
    dim = int(emb_mat.shape[1])
    print(f"[INFO] text embedding dim={dim}")

    # Map string -> embedding row
    name_to_idx = pd.Series(np.arange(len(uniq), dtype=np.int64), index=pd.Index(uniq, dtype="object"))
    fraud_idx = df[fraud_col].astype(str).map(name_to_idx).to_numpy(np.int64)
    real_idx = df[real_col].astype(str).map(name_to_idx).to_numpy(np.int64)

    fraud_embs = emb_mat[fraud_idx]
    real_embs = emb_mat[real_idx]

    # Append after existing columns (including your image embeddings)
    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols = [f"real_txt_emb_{i}" for i in range(dim)]

    # If re-running, avoid duplicate columns
    for c in fraud_cols + real_cols:
        if c in df.columns:
            raise ValueError(f"Column already exists in input: {c}. Did you already append text embeddings?")

    out_df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(fraud_embs, columns=fraud_cols),
            pd.DataFrame(real_embs, columns=real_cols),
        ],
        axis=1,
    )

    save_table(out_df, args.output)
    print(f"[INFO] wrote: {args.output}")


if __name__ == "__main__":
    main()
