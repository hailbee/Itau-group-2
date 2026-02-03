#!/usr/bin/env python3
"""
apply_distill_to_pairs.py

Loads a trained distillation checkpoint (teacher-only) and applies it to a
text-text pairs file like:
  - fraud_emb_0..fraud_emb_{D-1}
  - real_emb_0..real_emb_{D-1}
  - plus any metadata columns (names, label, etc.)

It OVERWRITES the fraud_emb_* and real_emb_* columns with the model outputs
(i.e., embeddings in the new aligned space), preserving all other columns.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from text_to_image.siamese import SiameseEmbeddingModel


def pick_device(device_override: Optional[str] = None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def cols_with_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'")
    # keep stable order by numeric suffix if present
    def key(c):
        try:
            return int(c.split(prefix)[1])
        except Exception:
            return c
    return sorted(cols, key=key)


def safe_torch_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ckpt", required=True, help="saved_models/distill_teacher_only.pt")
    ap.add_argument("--input_filepath", required=True, help=".parquet or .csv with fraud_emb_* and real_emb_*")
    ap.add_argument("--output_filepath", required=True, help=".parquet or .csv output")
    ap.add_argument("--fraud_prefix", default="fraud_emb_")
    ap.add_argument("--real_prefix", default="real_emb_")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", default=None)

    # Optional overrides (normally pulled from checkpoint)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--out_dim", type=int, default=None)
    ap.add_argument("--normalize", choices=["True", "False"], default="True")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    do_norm = (args.normalize == "True")
    print(f"[INFO] device={device} normalize={do_norm}")

    # Load checkpoint
    ckpt = safe_torch_load(args.model_ckpt, map_location=device)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        raise ValueError("Expected checkpoint dict with key 'model_state' (from main_distill_teacher_only.py).")

    # Load data
    df = load_table(args.input_filepath)

    fraud_cols = cols_with_prefix(df, args.fraud_prefix)
    real_cols = cols_with_prefix(df, args.real_prefix)

    D_in_fraud = len(fraud_cols)
    D_in_real = len(real_cols)
    if D_in_fraud != D_in_real:
        raise ValueError(f"fraud dim {D_in_fraud} != real dim {D_in_real}")

    # Determine model dims (prefer checkpoint)
    text_dim = int(ckpt.get("text_dim", D_in_fraud))
    teacher_dim = int(ckpt.get("teacher_dim", ckpt.get("out_dim", D_in_fraud)))
    share_text_heads = bool(ckpt.get("share_text_heads", False))

    if text_dim != D_in_fraud:
        raise ValueError(
            f"Checkpoint text_dim={text_dim} but file has {D_in_fraud} dims. "
            f"Are you applying the right model to the right embeddings?"
        )

    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        # checkpoint stores cfg dict from training
        cfg = ckpt.get("cfg", {})
        hidden_dim = int(cfg.get("hidden_dim", 512))

    out_dim = args.out_dim if args.out_dim is not None else teacher_dim

    # Build model
    model = SiameseEmbeddingModel(
        embedding_dim=int(text_dim),
        hidden_dim=int(hidden_dim),
        out_dim=int(out_dim),
        teacher_dim=None,
        share_text_heads=bool(share_text_heads),
        share_teacher_heads=True,
        dropout=0.0,
        activation="relu",
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded model: text_dim={text_dim} hidden_dim={hidden_dim} out_dim={out_dim} share_text_heads={share_text_heads}")

    # Build tensors
    fraud_x = torch.from_numpy(df[fraud_cols].to_numpy(dtype=np.float32))
    real_x = torch.from_numpy(df[real_cols].to_numpy(dtype=np.float32))

    loader = DataLoader(
        TensorDataset(fraud_x, real_x),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    fraud_out_chunks = []
    real_out_chunks = []

    with torch.no_grad():
        for xb_f, xb_r in loader:
            xb_f = xb_f.to(device)
            xb_r = xb_r.to(device)

            # IMPORTANT: your model.forward is text-text and returns (z_fraud, z_real)
            zf, zr = model(xb_f, xb_r)

            if do_norm:
                zf = F.normalize(zf, dim=1)
                zr = F.normalize(zr, dim=1)

            fraud_out_chunks.append(zf.detach().cpu())
            real_out_chunks.append(zr.detach().cpu())

    fraud_out = torch.cat(fraud_out_chunks, dim=0).numpy().astype(np.float32)
    real_out = torch.cat(real_out_chunks, dim=0).numpy().astype(np.float32)

    if fraud_out.shape[1] != len(fraud_cols):
        # If out_dim differs from original, you can't "keep same format" without changing columns.
        raise ValueError(
            f"Model output dim={fraud_out.shape[1]} but input columns dim={len(fraud_cols)}. "
            f"To overwrite columns, out_dim must equal the number of fraud_emb_* columns."
        )

    # Overwrite columns in-place (preserve file format)
    df.loc[:, fraud_cols] = fraud_out
    df.loc[:, real_cols] = real_out

    save_table(df, args.output_filepath)
    print(f"[INFO] Saved translated file -> {args.output_filepath}")
    print(f"[INFO] Rows={len(df)} cols={df.shape[1]} overwritten: {len(fraud_cols)} + {len(real_cols)}")


if __name__ == "__main__":
    main()

"""
python seton_notebooks/apply_distill_to_pairs.py \
  --model_ckpt saved_models/text_to_img_contrastive.pt \
  --input_filepath ../Downloads/test_pairs_with_siglip_embeddings.parquet \
  --output_filepath ../Downloads/aligned_test_pairs_with_siglip_embeddings.parquet\
  --batch_size 4096
"""