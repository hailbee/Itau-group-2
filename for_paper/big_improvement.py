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


# -------------------------
# Helpers
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def _mat(df: pd.DataFrame, prefix: str) -> torch.Tensor:
    cols = _sorted_prefixed_cols(df, prefix)
    return torch.tensor(df[cols].to_numpy(dtype=np.float32, copy=False))


def pick_device(device_override=None):
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thresholds[i])


def cosine_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=1)


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--out-dim", type=int, required=True)

    ap.add_argument("--fraud-txt-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-txt-prefix", default="real_txt_emb_")
    ap.add_argument("--label-col", default="label")

    ap.add_argument("--top-k", type=int, default=25)

    ap.add_argument(
        "--mode",
        choices=["improve", "worsen", "flip_correct", "worse_confidence"],
        default="improve",
    )

    ap.add_argument("--output", default="top_improvements.parquet")
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    df = pd.read_parquet(args.test)
    y = df[args.label_col].astype(int).to_numpy()

    fraud_txt = _mat(df, args.fraud_txt_prefix)
    real_txt = _mat(df, args.real_txt_prefix)
    text_dim = fraud_txt.shape[1]

    # -------------------------
    # Baseline
    # -------------------------
    base_cos = cosine_np(fraud_txt.numpy(), real_txt.numpy())
    base_score = -base_cos if roc_auc_score(y, -base_cos) > roc_auc_score(y, base_cos) else base_cos
    base_thr = _youden_threshold(y, base_score)
    base_pred = (base_score >= base_thr).astype(int)

    # -------------------------
    # Model
    # -------------------------
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        image_dim=args.out_dim,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    z_f, z_r = model(fraud_txt.to(device), real_txt.to(device))
    z_f = F.normalize(z_f, dim=1)
    z_r = F.normalize(z_r, dim=1)

    student_cos = F.cosine_similarity(z_f, z_r, dim=1).cpu().numpy()
    student_score = -student_cos if roc_auc_score(y, -student_cos) > roc_auc_score(y, student_cos) else student_cos
    student_thr = _youden_threshold(y, student_score)
    student_pred = (student_score >= student_thr).astype(int)

    # -------------------------
    # Margins
    # -------------------------
    base_margin = np.abs(base_score - base_thr)
    student_margin = np.abs(student_score - student_thr)
    improvement = student_margin - base_margin

    result = df.copy()
    result["baseline_margin"] = base_margin
    result["student_margin"] = student_margin
    result["improvement_margin"] = improvement
    result["baseline_pred"] = base_pred
    result["student_pred"] = student_pred

    # -------------------------
    # Modes
    # -------------------------
    if args.mode == "worse_confidence":
        view = result[(student_pred != y) & (student_margin > base_margin)]
        view = view.sort_values("student_margin", ascending=False)

    elif args.mode == "worsen":
        view = result.sort_values("improvement_margin", ascending=True)

    elif args.mode == "flip_correct":
        view = result[(base_pred != y) & (student_pred == y)]
        view = view.sort_values("improvement_margin", ascending=False)

    else:  # improve
        view = result.sort_values("improvement_margin", ascending=False)

    view = view.head(args.top_k)

    print("\nTOP RESULTS:\n")
    cols = [
        c for c in [
            "fraudulent_name",
            "real_name",
            "baseline_pred",
            "student_pred",
            "baseline_margin",
            "student_margin",
            "improvement_margin",
        ] if c in view.columns
    ]
    print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
    
"""
python for_paper/big_improvement.py \
  --test ../Downloads/vate_test.parquet \
  --model-path saved_models/deja_best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --mode worse_confidence \
  --top-k 20
"""