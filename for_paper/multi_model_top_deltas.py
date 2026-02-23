#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from text_to_image.siamese import SiameseEmbeddingModel


# ============================================================
# Helpers
# ============================================================

def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def _mat_np(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    return df[cols].to_numpy(dtype=np.float32, copy=False)


def pick_device(device_override=None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=1)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def auto_orient_score(y: np.ndarray, raw_cos: np.ndarray):
    auc_pos = float(roc_auc_score(y, raw_cos))
    auc_neg = float(roc_auc_score(y, -raw_cos))
    flipped = auc_neg > auc_pos
    score = (-raw_cos) if flipped else raw_cos
    return score.astype(np.float32), auc_pos


def load_model(path: str, text_dim: int, hidden_dim: int, out_dim: int, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        image_dim=out_dim,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def compute_student_cos(model, fraud_np, real_np, device, batch_size):
    n = fraud_np.shape[0]
    sims = np.empty(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)

        f = torch.from_numpy(fraud_np[i:j]).to(device)
        r = torch.from_numpy(real_np[i:j]).to(device)

        zf, zr = model(f, r)
        zf = F.normalize(zf, dim=1)
        zr = F.normalize(zr, dim=1)

        sims[i:j] = F.cosine_similarity(zf, zr, dim=1).cpu().numpy()

    return sims


# ============================================================
# Build Tables
# ============================================================

def build_tables(df_pairs, y, base_margin, stu_margin, top_k):
    delta = stu_margin - base_margin

    pos = y == 1
    neg = y == 0

    def make(mask, largest):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return pd.DataFrame()

        order = np.argsort(-delta[idx]) if largest else np.argsort(delta[idx])
        sel = idx[order[:top_k]]

        out = df_pairs.iloc[sel].copy()
        out["y"] = y[sel]
        out["baseline_margin"] = base_margin[sel]
        out["student_margin"] = stu_margin[sel]
        out["delta_margin"] = delta[sel]
        return out.reset_index(drop=True)

    return {
        "pos_better": make(pos, True),
        "pos_worse": make(pos, False),
        "neg_better": make(neg, False),
        "neg_worse": make(neg, True),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--fraud-txt-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-txt-prefix", default="real_txt_emb_")
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--out-dim", type=int, required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="multi_model_report.txt")

    args = ap.parse_args()
    device = pick_device(args.device)

    df = pd.read_parquet(args.dataset) if args.dataset.endswith(".parquet") else pd.read_csv(args.dataset)
    y = df[args.label_col].astype(int).to_numpy()

    df_pairs = df[[args.fraud_col, args.real_col]].rename(
        columns={args.fraud_col: "fraudulent_name", args.real_col: "real_name"}
    )

    fraud_np = _mat_np(df, args.fraud_txt_prefix)
    real_np = _mat_np(df, args.real_txt_prefix)

    base_raw = cosine_np(fraud_np, real_np)
    base_score, base_auc = auto_orient_score(y, base_raw)
    base_thr = youden_threshold(y, base_score)
    base_margin = base_score - base_thr

    with open(args.output, "w") as f:

        f.write(f"BASELINE AUC={base_auc:.6f} | thr={base_thr:.6f}\n\n")

        for spec in args.ckpt:
            name, path = spec.split("=", 1)

            model = load_model(path, fraud_np.shape[1], args.hidden_dim, args.out_dim, device)
            stu_raw = compute_student_cos(model, fraud_np, real_np, device, args.batch_size)

            stu_score, stu_auc = auto_orient_score(y, stu_raw)
            stu_thr = youden_threshold(y, stu_score)
            stu_margin = stu_score - stu_thr

            f.write("=" * 80 + "\n")
            f.write(f"MODEL {name} | AUC={stu_auc:.6f} | thr={stu_thr:.6f}\n")
            f.write("=" * 80 + "\n\n")

            tables = build_tables(df_pairs, y, base_margin, stu_margin, args.top_k)

            for i, (key, title) in enumerate([
                ("pos_better", "POSITIVES — MADE BETTER"),
                ("pos_worse",  "POSITIVES — MADE WORSE"),
                ("neg_better", "NEGATIVES — MADE BETTER"),
                ("neg_worse",  "NEGATIVES — MADE WORSE"),
            ], 1):

                f.write(f"[{i}/4] {title}\n")

                df_section = tables[key]
                if df_section.empty:
                    f.write("  (none)\n\n")
                    continue

                for _, row in df_section.iterrows():
                    f.write(
                        f"  {row['fraudulent_name']} vs {row['real_name']} "
                        f"| y={row['y']} "
                        f"| base_m={row['baseline_margin']:.4f} "
                        f"| stu_m={row['student_margin']:.4f} "
                        f"| Δ={row['delta_margin']:.4f}\n"
                    )
                f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
"""
python for_paper/multi_model_top_deltas.py \
  --dataset ../Downloads/vate_test.parquet \
  --label-col label \
  --hidden-dim 1024 \
  --out-dim 768 \
  --ckpt deja=saved_models/deja_best_model.pt \
  --ckpt source=saved_models/source_best_model.pt \
  --ckpt pacifico=saved_models/pacifico_best_model.pt \
  --top-k 20 \
  --output for_paper/results.txt
"""