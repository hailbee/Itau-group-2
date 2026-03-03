#!/usr/bin/env python3
"""
Write plain backbone text embeddings (SigLIP/CLIP/CoCa/FLAVA) into a parquet/CSV.

- No trained Siamese head, no projection, no checkpoint loading.
- Uses BaselineTester(model_type=...) and its model_wrapper only.
- Writes fraud_txt_emb_* and real_txt_emb_* columns.
- Optional "text-only output" file (overwrites by default).

Example:
python3 create_siglip_text_only_embeddings.py \
  --input ../Downloads/train_pairs_with_siglip_embeddings.parquet \
  --text-only-output ../Downloads/text_train.parquet \
  --include-keys fraudulent_name real_name label \
  --backbone siglip \
  --batch-size 256 \
  --overwrite-cols
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.baseline.baseline_tester import BaselineTester


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
# Text embedding (backbone-only)
# ---------------------------
def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if x.device != device else x


def _call_text_embedder(model_wrapper: object, texts: Sequence[str]) -> torch.Tensor:
    """
    Tries a few common entrypoints used by backbone wrappers.

    Expected output: torch.Tensor [B, D]
    """
    for name in ("embed_text", "encode_text", "get_text_features", "text_features"):
        if hasattr(model_wrapper, name):
            fn = getattr(model_wrapper, name)
            if callable(fn):
                out = fn(list(texts))
                if isinstance(out, torch.Tensor):
                    return out

    if hasattr(model_wrapper, "model"):
        m = getattr(model_wrapper, "model")
        for name in ("encode_text", "get_text_features"):
            if hasattr(m, name):
                fn = getattr(m, name)
                if callable(fn):
                    out = fn(list(texts))
                    if isinstance(out, torch.Tensor):
                        return out

    if callable(model_wrapper):
        out = model_wrapper(list(texts))
        if isinstance(out, torch.Tensor):
            return out

    raise RuntimeError(
        "Could not find a text-embedding entrypoint on BaselineTester.model_wrapper. "
        "Tried: embed_text/encode_text/get_text_features/text_features, wrapper.model.*, and calling the wrapper."
    )


@torch.no_grad()
def batched_text_embedding(
    model_wrapper: object,
    texts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    all_out: List[torch.Tensor] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i : i + batch_size]
        out = _call_text_embedder(model_wrapper, chunk)

        if not isinstance(out, torch.Tensor):
            raise RuntimeError("Text embedder did not return a torch.Tensor.")
        out = _to_device(out, device)
        if out.ndim != 2:
            raise RuntimeError(f"Expected [B, D] tensor, got shape {tuple(out.shape)}")

        all_out.append(out.detach())

    return torch.cat(all_out, dim=0)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)

    ap.add_argument("--backbone", default="siglip", choices=["clip", "coca", "flava", "siglip"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)

    ap.add_argument(
        "--overwrite-cols",
        action="store_true",
        help="If fraud_txt_emb_* / real_txt_emb_* already exist, overwrite them in-place instead of failing.",
    )

    ap.add_argument(
        "--text-only-output",
        default=None,
        help="If set, also write a second file containing ONLY the backbone text embedding columns (plus optional keys).",
    )
    ap.add_argument(
        "--include-keys",
        nargs="*",
        default=[],
        help="Column names to keep in the text-only output (e.g., fraudulent_name real_name label).",
    )

    args = ap.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"[INFO] device={device}")

    df = load_table(args.input).copy()

    for col in ("fraudulent_name", "real_name"):
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    tester = BaselineTester(model_type=args.backbone, batch_size=args.batch_size, device=str(device))
    backbone = tester.model_wrapper

    fraud_names = df["fraudulent_name"].astype(str).tolist()
    real_names = df["real_name"].astype(str).tolist()

    fraud_embs = batched_text_embedding(backbone, fraud_names, args.batch_size, device=device)
    real_embs = batched_text_embedding(backbone, real_names, args.batch_size, device=device)

    if fraud_embs.shape != real_embs.shape:
        raise RuntimeError(f"Embedding shape mismatch: fraud={tuple(fraud_embs.shape)}, real={tuple(real_embs.shape)}")

    dim = int(fraud_embs.shape[1])
    print(f"[INFO] text emb dim={dim}")

    fraud_np = fraud_embs.detach().to("cpu", dtype=torch.float32).numpy()
    real_np = real_embs.detach().to("cpu", dtype=torch.float32).numpy()

    fraud_cols = [f"fraud_txt_emb_{i}" for i in range(dim)]
    real_cols = [f"real_txt_emb_{i}" for i in range(dim)]
    embed_cols = fraud_cols + real_cols

    collisions = [c for c in embed_cols if c in df.columns]
    if collisions and not args.overwrite_cols:
        raise RuntimeError(
            "Column collision(s) detected.\n"
            f"Collisions: {collisions[:10]}{' ...' if len(collisions) > 10 else ''}\n"
            "If you intend to overwrite these columns, pass --overwrite-cols."
        )

    if collisions and args.overwrite_cols:
        df = df.drop(columns=collisions)

    text_df = pd.DataFrame(
        np.hstack([fraud_np, real_np]),
        columns=embed_cols,
        index=df.index,
        dtype=np.float32,
    )
    out_df = pd.concat([df, text_df], axis=1)

    if args.text_only_output is not None:
        keep_keys: List[str] = []
        for k in args.include_keys:
            if k not in out_df.columns:
                raise RuntimeError(f"Requested key column for text-only output not found: {k}")
            keep_keys.append(k)

        only_cols = keep_keys + embed_cols
        only_df = out_df.loc[:, only_cols].copy()
        for c in embed_cols:
            only_df[c] = only_df[c].astype(np.float32, copy=False)

        save_table(only_df, args.text_only_output)
        print(f"[INFO] wrote text-only file → {args.text_only_output}")

    if "label" in out_df.columns:
        with torch.no_grad():
            sims = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()

        y = out_df["label"].astype(float).to_numpy()

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y, sims)
        print(f"[INFO] sanity ROC AUC (cosine sim): {auc(fpr, tpr):.4f}")
    else:
        print("[INFO] No 'label' column found; skipping sanity ROC-AUC.")


if __name__ == "__main__":
    main()