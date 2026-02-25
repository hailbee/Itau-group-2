#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_multiple_students_to_single_parquet.py

This is the version you are asking for:

- You have ONE parquet (e.g., vate_onehot.parquet) that contains ONLY VATE embeddings:
      fraud_txt_emb_0.., real_txt_emb_0..
  plus lots of other columns (labels, names, one-hot mechanisms, etc.)

- You have MULTIPLE student checkpoints (e.g., deja/source/pacifico) and you want to
  apply EACH checkpoint to the SAME rows (same VATE embeddings), producing multiple
  aligned embedding sets, then save a NEW parquet WITHOUT changing anything else.

Output
------
For each checkpoint tag T (e.g., deja), this appends:
    fraud_txt_emb_aligned_{T}_0.., real_txt_emb_aligned_{T}_0..
leaving ALL existing columns untouched.

Example
-------
python for_paper/apply_multiple_students_to_single_parquet.py \
  --data for_paper/vate_onehot.parquet \
  --out  for_paper/vate_onehot_with_3aligned.parquet \
  --ckpt-map "deja=saved_models/deja_best_model.pt,source=saved_models/source_best_model.pt,pacifico=saved_models/pacifico_best_model.pt" \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix  real_txt_emb_ \
  --out-fraud-base fraud_txt_emb_aligned_ \
  --out-real-base  real_txt_emb_aligned_ \
  --out-dim 768 \
  --batch 4096 \
  --device cuda

Safety
------
- Refuses to overwrite existing output columns unless --overwrite-output-cols is set.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from text_to_image.siamese import SiameseEmbeddingModel


# -------------------------
# Device
# -------------------------

def pick_device(override: Optional[str]) -> torch.device:
    if override:
        d = torch.device(override)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Column helpers
# -------------------------

def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str) -> int:
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def mat_from_prefix(df: pd.DataFrame, prefix: str) -> Tuple[np.ndarray, List[str]]:
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=np.float32, copy=True)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix for prefix {prefix!r}, got {mat.shape}.")
    return mat, cols


def build_out_cols(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(int(dim))]


# -------------------------
# Checkpoint loading (matches your other scripts)
# -------------------------

def evaluator2_style_state(ckpt):
    return ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt


def as_state_dict(state_obj) -> Dict[str, torch.Tensor]:
    if isinstance(state_obj, nn.Module):
        return state_obj.state_dict()
    if isinstance(state_obj, dict) and state_obj and all(isinstance(v, torch.Tensor) for v in state_obj.values()):
        return state_obj
    raise RuntimeError(f"Checkpoint state is not a state_dict or nn.Module. Type={type(state_obj)}")


def infer_hidden_dim_from_state(sd: Dict[str, torch.Tensor], text_dim: int) -> int:
    candidates: List[int] = []
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[1]) == int(text_dim):
            candidates.append(int(v.shape[0]))
    if not candidates:
        raise RuntimeError("Could not infer hidden_dim from checkpoint. Provide --hidden-dim explicitly.")
    counts: Dict[int, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    return int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0])


def load_student_model(
    ckpt_path: str,
    text_dim: int,
    out_dim: int,
    hidden_dim_override: Optional[int],
    device: torch.device,
) -> SiameseEmbeddingModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = as_state_dict(evaluator2_style_state(ckpt))
    inferred_hidden = infer_hidden_dim_from_state(sd, text_dim=text_dim)
    hidden_dim = inferred_hidden if hidden_dim_override is None else int(hidden_dim_override)
    if hidden_dim_override is not None and hidden_dim != int(inferred_hidden):
        raise ValueError(f"hidden_dim mismatch: ckpt implies {inferred_hidden}, but --hidden-dim={hidden_dim}")
    model = SiameseEmbeddingModel(text_dim=text_dim, hidden_dim=hidden_dim, image_dim=int(out_dim)).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# -------------------------
# Encoding
# -------------------------

@torch.inference_mode()
def encode_mat(model: SiameseEmbeddingModel, mat: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    x = torch.from_numpy(mat).to(device=device, dtype=torch.float32)
    bs = int(max(1, batch))
    outs: List[np.ndarray] = []
    for i0 in range(0, x.shape[0], bs):
        i1 = min(i0 + bs, x.shape[0])
        z = model.encode_text(x[i0:i1])
        outs.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, int(model.encode_text(x[:1]).shape[1])), dtype=np.float32)


# -------------------------
# Map parsing
# -------------------------

def parse_ckpt_map(s: str) -> Dict[str, str]:
    """
    Parse "deja=path,source=path,pacifico=path" into dict.
    """
    out: Dict[str, str] = {}
    s = str(s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --ckpt-map entry {p!r}. Expected 'tag=path'.")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Bad --ckpt-map entry {p!r}. Expected 'tag=path'.")
        out[k] = v
    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True, help="Input parquet (e.g., vate_onehot.parquet).")
    ap.add_argument("--out", required=True, help="Output parquet path.")

    ap.add_argument(
        "--ckpt-map",
        required=True,
        help="Comma-separated mapping: 'deja=path,source=path,pacifico=path'. Each will be applied to ALL rows.",
    )

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_", help="Input fraud embedding prefix.")
    ap.add_argument("--real-prefix", default="real_txt_emb_", help="Input real embedding prefix.")

    ap.add_argument(
        "--out-fraud-base",
        default="fraud_txt_emb_aligned_",
        help="Base output prefix. Final prefix becomes '{base}{tag}_'.",
    )
    ap.add_argument(
        "--out-real-base",
        default="real_txt_emb_aligned_",
        help="Base output prefix. Final prefix becomes '{base}{tag}_'.",
    )

    ap.add_argument("--out-dim", type=int, default=768, help="Aligned embedding dim.")
    ap.add_argument("--hidden-dim", type=int, default=None, help="Override hidden_dim if inference fails.")
    ap.add_argument("--batch", type=int, default=4096, help="Batch size for encode_text.")
    ap.add_argument("--device", default=None, help="cuda|cpu|mps (optional override).")

    ap.add_argument(
        "--overwrite-output-cols",
        action="store_true",
        help="Allow overwriting output columns if they already exist.",
    )

    args = ap.parse_args()

    ckpt_map = parse_ckpt_map(args.ckpt_map)
    if not ckpt_map:
        raise ValueError("--ckpt-map parsed empty.")

    df = pd.read_parquet(args.data)

    fraud_mat, _ = mat_from_prefix(df, args.fraud_prefix)
    real_mat, _ = mat_from_prefix(df, args.real_prefix)
    text_dim = int(fraud_mat.shape[1])
    if int(real_mat.shape[1]) != text_dim:
        raise ValueError("fraud and real embedding dims do not match")

    device = pick_device(args.device)
    print(f"[INFO] device={device}")
    print(f"[INFO] N={len(df):,} | text_dim={text_dim} -> out_dim={int(args.out_dim)}")
    print(f"[INFO] checkpoints: {list(ckpt_map.keys())}")

    # Collision check (all tags)
    all_new_cols: List[str] = []
    for tag in ckpt_map.keys():
        fraud_pref = f"{args.out_fraud_base}{tag}_"
        real_pref = f"{args.out_real_base}{tag}_"
        all_new_cols.extend(build_out_cols(fraud_pref, int(args.out_dim)))
        all_new_cols.extend(build_out_cols(real_pref, int(args.out_dim)))

    collisions = [c for c in all_new_cols if c in df.columns]
    if collisions and not args.overwrite_output_cols:
        raise ValueError(
            "Output columns already exist (first few shown): "
            + ", ".join(collisions[:10])
            + ". Re-run with --overwrite-output-cols to overwrite them."
        )

    # Load/cache models by path
    model_cache: Dict[str, SiameseEmbeddingModel] = {}

    # Start from original df; append (or overwrite) per tag
    df_out = df.copy() if (args.overwrite_output_cols and collisions) else df

    for tag, ckpt_path in ckpt_map.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\n=== APPLY tag={tag} | ckpt={ckpt_path} ===")

        if ckpt_path in model_cache:
            model = model_cache[ckpt_path]
        else:
            model = load_student_model(
                ckpt_path=ckpt_path,
                text_dim=text_dim,
                out_dim=int(args.out_dim),
                hidden_dim_override=args.hidden_dim,
                device=device,
            )
            model_cache[ckpt_path] = model

        aligned_fraud = encode_mat(model, fraud_mat, device=device, batch=int(args.batch))
        aligned_real = encode_mat(model, real_mat, device=device, batch=int(args.batch))

        if aligned_fraud.shape != (len(df), int(args.out_dim)):
            raise RuntimeError(f"Unexpected aligned_fraud shape: {aligned_fraud.shape}")
        if aligned_real.shape != (len(df), int(args.out_dim)):
            raise RuntimeError(f"Unexpected aligned_real shape: {aligned_real.shape}")

        fraud_pref = f"{args.out_fraud_base}{tag}_"
        real_pref = f"{args.out_real_base}{tag}_"
        fraud_cols = build_out_cols(fraud_pref, int(args.out_dim))
        real_cols = build_out_cols(real_pref, int(args.out_dim))

        fraud_df = pd.DataFrame(aligned_fraud, columns=fraud_cols, index=df.index)
        real_df = pd.DataFrame(aligned_real, columns=real_cols, index=df.index)

        if args.overwrite_output_cols and any(c in df_out.columns for c in fraud_cols + real_cols):
            for c in fraud_cols:
                df_out[c] = fraud_df[c].to_numpy(dtype=np.float32, copy=False)
            for c in real_cols:
                df_out[c] = real_df[c].to_numpy(dtype=np.float32, copy=False)
        else:
            df_out = pd.concat([df_out, fraud_df, real_df], axis=1)

        print(f"[OK] appended: {fraud_cols[0]}..{fraud_cols[-1]} and {real_cols[0]}..{real_cols[-1]}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_parquet(args.out, index=False)
    print(f"\n[OK] wrote: {args.out}")
    print("[OK] non-embedding columns preserved unchanged.")


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLE

python for_paper/apply_multiple_students_to_single_parquet.py \
  --data for_paper/vate_onehot.parquet \
  --out  for_paper/vate_onehot_with_3aligned.parquet \
  --ckpt-map "deja=saved_models/deja_best_model.pt,source=saved_models/source_best_model.pt,pacifico=saved_models/pacifico_best_model.pt" \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix  real_txt_emb_ \
  --out-fraud-base fraud_txt_emb_aligned_ \
  --out-real-base  real_txt_emb_aligned_ \
  --out-dim 768 \
  --batch 4096 \
  --device cuda
"""
