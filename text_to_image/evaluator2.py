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

from siamese import SiameseEmbeddingModel


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


def _save_table(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output file must end with .parquet or .csv")


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))

    best_threshold = float(thresholds[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])
    best_specificity = float(1.0 - best_fpr)
    best_j = float(youden_j[best_idx])

    return best_threshold, best_j, best_tpr, best_fpr, best_specificity


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate trained embeddings using cosine similarity (ROC AUC) and export student embeddings."
    )

    # files
    ap.add_argument("--test", required=True, help="Input parquet or csv")
    ap.add_argument(
        "--model-path",
        default="saved_models/best_model.pt",
        help="Path to trained model checkpoint",
    )

    # architecture (MUST MATCH TRAINING)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--out-dim", type=int, required=True)

    # prefixes (input text embeddings)
    ap.add_argument("--fraud-txt-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-txt-prefix", default="real_txt_emb_")
    ap.add_argument("--label-col", default="label")

    # export
    ap.add_argument("--output", required=True, help="Output parquet/csv with student embeddings")
    ap.add_argument("--out-fraud-prefix", default="fraud_student_")
    ap.add_argument("--out-real-prefix", default="real_student_")
    ap.add_argument(
        "--keep-original-embeddings",
        action="store_true",
        help="If set, keep the original input embedding columns too (default is to drop them).",
    )
    ap.add_argument(
        "--save-unnormalized",
        action="store_true",
        help="If set, also save unnormalized student vectors with suffix '_raw'.",
    )

    # runtime
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_parquet(args.test) if args.test.lower().endswith(".parquet") else pd.read_csv(args.test)
    y = df[args.label_col].astype(int).to_numpy()

    fraud_txt = _mat(df, args.fraud_txt_prefix)
    real_txt = _mat(df, args.real_txt_prefix)

    text_dim = fraud_txt.shape[1]

    print(f"[INFO] text_dim={text_dim} | hidden_dim={args.hidden_dim} | out_dim={args.out_dim}")

    # -------------------------
    # Load model
    # -------------------------
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        image_dim=args.out_dim,
    ).to(device)

    load_info = model.load_state_dict(state, strict=False)
    model.eval()

    if hasattr(load_info, "missing_keys") and hasattr(load_info, "unexpected_keys"):
        print(
            f"[INFO] load_state_dict: missing_keys={len(load_info.missing_keys)} "
            f"unexpected_keys={len(load_info.unexpected_keys)}"
        )
        if load_info.missing_keys:
            print("  missing_keys (first 10):", load_info.missing_keys[:10])
        if load_info.unexpected_keys:
            print("  unexpected_keys (first 10):", load_info.unexpected_keys[:10])

    print("[INFO] Model loaded successfully")

    # -------------------------
    # Forward pass: student embeddings + sims
    # -------------------------
    bs = int(args.batch_size)
    n = len(df)

    zf_all = np.empty((n, args.out_dim), dtype=np.float32)
    zr_all = np.empty((n, args.out_dim), dtype=np.float32)
    sims_list = []

    if args.save_unnormalized:
        zf_raw_all = np.empty((n, args.out_dim), dtype=np.float32)
        zr_raw_all = np.empty((n, args.out_dim), dtype=np.float32)

    for start in range(0, n, bs):
        end = min(start + bs, n)

        f = fraud_txt[start:end].to(device)
        r = real_txt[start:end].to(device)

        z_f, z_r = model(f, r)

        if args.save_unnormalized:
            zf_raw_all[start:end] = z_f.detach().float().cpu().numpy()
            zr_raw_all[start:end] = z_r.detach().float().cpu().numpy()

        z_f = F.normalize(z_f, dim=1)
        z_r = F.normalize(z_r, dim=1)

        zf_all[start:end] = z_f.detach().float().cpu().numpy()
        zr_all[start:end] = z_r.detach().float().cpu().numpy()

        sims_list.append(F.cosine_similarity(z_f, z_r, dim=1).detach().cpu())

    sims = torch.cat(sims_list).numpy()

    # -------------------------
    # Metrics
    # -------------------------
    auc_pos = float(roc_auc_score(y, sims))
    auc_neg = float(roc_auc_score(y, -sims))
    auc_best = max(auc_pos, auc_neg)
    direction = "score_higher_for_label1" if auc_pos >= auc_neg else "score_lower_for_label1 (use -score)"

    # choose correct scoring direction for threshold
    if auc_pos >= auc_neg:
        scores_for_threshold = sims
        threshold_note = "threshold applies to cosine similarity score directly"
    else:
        scores_for_threshold = -sims
        threshold_note = "threshold applies to flipped score (-cosine similarity)"

    best_thresh, best_j, best_tpr, best_fpr, best_spec = _youden_threshold(y, scores_for_threshold)

    print("\n==============================")
    print(" STUDENT EMBEDDING EVALUATION")
    print("==============================")
    print(f"ROC AUC (cosine): {auc_pos:.6f}")
    print(f"ROC AUC (cosine, flipped): {auc_neg:.6f}")
    print(f"ROC AUC (best): {auc_best:.6f}  |  direction: {direction}")
    print("------------------------------")
    print(" Youden Threshold (ROC Optimal)")
    print("------------------------------")
    print(f"Best threshold: {best_thresh:.6f}")
    print(f"Youden J:       {best_j:.6f}")
    print(f"TPR (recall):   {best_tpr:.6f}")
    print(f"FPR:            {best_fpr:.6f}")
    print(f"Specificity:    {best_spec:.6f}")
    print(f"[INFO] {threshold_note}")
    print("==============================\n")

    # -------------------------
    # SAVE OUTPUT DATAFRAME (FIX)
    # -------------------------
    out_df = df.copy()

    # optionally drop original embedding columns
    if not args.keep_original_embeddings:
        fraud_cols = _sorted_prefixed_cols(out_df, args.fraud_txt_prefix)
        real_cols = _sorted_prefixed_cols(out_df, args.real_txt_prefix)
        out_df = out_df.drop(columns=fraud_cols + real_cols)

    # add normalized student embeddings
    for j in range(args.out_dim):
        out_df[f"{args.out_fraud_prefix}{j}"] = zf_all[:, j]
        out_df[f"{args.out_real_prefix}{j}"] = zr_all[:, j]

    # add unnormalized student embeddings if requested
    if args.save_unnormalized:
        for j in range(args.out_dim):
            out_df[f"{args.out_fraud_prefix}{j}_raw"] = zf_raw_all[:, j]
            out_df[f"{args.out_real_prefix}{j}_raw"] = zr_raw_all[:, j]

    _save_table(out_df, args.output)
    print(f"[INFO] Saved output table to: {args.output}")


if __name__ == "__main__":
    main()

"""
Example:
python text_to_image/evaluator2.py \
  --test text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet  \
  --model-path saved_models/best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --output text_to_image/evaluation/vate_test_student_only.parquet
"""
