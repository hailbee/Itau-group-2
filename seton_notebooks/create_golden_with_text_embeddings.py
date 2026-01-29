#!/usr/bin/env python3
"""
Append SigLIP TEXT embeddings to an existing file that already contains IMAGE embeddings.

SAFE VERSION:
- Preserves all existing columns exactly
- Appends fraud_txt_emb_* and real_txt_emb_* as float32
- Hard-fails on ANY column collision
- No index reset, no reordering, no dtype pollution

Example:
python seton_notebooks/create_golden_with_text_embeddings.py \
  --input text_to_image/Golden/golden_embeddings_validate.parquet \
  --output text_to_image/Golden_and_Text/validate_pairs_with_img_and_txt_embs.parquet \
  --batch-size 256
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
# Text normalization
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
    Returns float32 embeddings of shape (N, D).
    """
    n = len(uniq_texts)
    if n == 0:
        raise ValueError("No texts to embed.")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    embeddings = []

    for start in tqdm(range(0, n, batch_size), desc="Embedding text"):
        chunk = uniq_texts[start : start + batch_size]
        batch = processor(
            text=chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in batch.items()}

        with autocast_ctx:
            out = model(**batch)
            e = out.pooler_output

        if do_l2_normalize:
            e = F.normalize(e, dim=-1, eps=1e-8)

        embeddings.append(e.float().cpu())

    emb = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
    return emb


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet with image embeddings (golden or raw)")
    ap.add_argument("--output", required=True, help="Output parquet with image + text embeddings")
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--strip-com", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output):
        raise FileExistsError(f"Output already exists: {args.output}")

    df = load_table(args.input)

    if args.max_rows is not None:
        df = df.head(int(args.max_rows))

    # Required columns
    if "fraudulent_name" not in df.columns or "real_name" not in df.columns:
        raise RuntimeError("Input must contain 'fraudulent_name' and 'real_name' columns.")

    df = df.copy()
    df["fraudulent_name"] = df["fraudulent_name"].map(lambda x: normalize_name(x, args.strip_com))
    df["real_name"] = df["real_name"].map(lambda x: normalize_name(x, args.strip_com))

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = SiglipTextModel.from_pretrained(args.model, torch_dtype=torch_dtype).to(device).eval()

    # Deduplicate text
    all_texts = pd.concat(
        [df["fraudulent_name"], df["real_name"]],
        ignore_index=True
    ).astype(str)

    uniq_texts = pd.unique(all_texts).tolist()
    print(f"[INFO] unique text strings: {len(uniq_texts):,}")

    emb_mat = embed_unique_texts(
        uniq_texts=uniq_texts,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
        do_l2_normalize=(not args.no_normalize),
        max_length=args.max_length,
    )

    dim = emb_mat.shape[1]
    print(f"[INFO] text embedding dim = {dim}")

    # Map text → embedding
    text_to_idx = {t: i for i, t in enumerate(uniq_texts)}
    fraud_idx = df["fraudulent_name"].map(text_to_idx).to_numpy()
    real_idx  = df["real_name"].map(text_to_idx).to_numpy()

    fraud_embs = emb_mat[fraud_idx]
    real_embs  = emb_mat[real_idx]

    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols  = [f"real_txt_emb_{i}" for i in range(dim)]

    # HARD FAIL on collision
    for c in fraud_cols + real_cols:
        if c in df.columns:
            raise RuntimeError(f"Column collision detected: {c}")

    text_df = pd.DataFrame(
        np.hstack([fraud_embs, real_embs]),
        columns=fraud_cols + real_cols,
        dtype=np.float32,
    )

    out_df = pd.concat([df, text_df], axis=1)

    save_table(out_df, args.output)
    print(f"[INFO] wrote clean merged file → {args.output}")


if __name__ == "__main__":
    main()
