#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

from siamese import SiameseEmbeddingModel


# -------------------------
# Helpers
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix) :]
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


def _drop_prefixed_cols(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    drop_cols = []
    for p in prefixes:
        drop_cols.extend([c for c in df.columns if isinstance(c, str) and c.startswith(p)])
    if drop_cols:
        return df.drop(columns=drop_cols)
    return df


def _make_default_misclassified_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    if ext.lower() not in [".parquet", ".csv"]:
        return output_path + "_misclassified.parquet"
    return base + "_misclassified" + ext


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (best_threshold, best_tpr, best_fpr) maximizing Youden's J = TPR - FPR.
    Assumes: higher score => more likely label 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thresholds[i]), float(tpr[i]), float(fpr[i])


def _max_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, int, int, int, int]:
    """
    Finds threshold that maximizes accuracy for rule: pred = (score >= thr).
    Returns (best_thr, best_acc, tn, fp, fn, tp) at that threshold.

    Uses ROC curve points: for each threshold, TPR and FPR define TP and FP rates,
    allowing accuracy computation without scanning all thresholds over all samples.
    """
    y_true = y_true.astype(int)
    n = int(y_true.shape[0])
    p = int(np.sum(y_true == 1))
    nneg = n - p
    if p == 0 or nneg == 0:
        raise ValueError("Cannot compute max-accuracy threshold: need both classes present in y_true.")

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Convert rates to counts at each threshold
    tp = tpr * p
    fp = fpr * nneg
    fn = p - tp
    tn = nneg - fp

    acc = (tp + tn) / (p + nneg)
    i = int(np.argmax(acc))

    best_thr = float(thresholds[i])
    best_acc = float(acc[i])

    # Round counts to nearest int; at ROC points these should be integer-valued up to float error
    tn_i = int(np.rint(tn[i]))
    fp_i = int(np.rint(fp[i]))
    fn_i = int(np.rint(fn[i]))
    tp_i = int(np.rint(tp[i]))

    return best_thr, best_acc, tn_i, fp_i, fn_i, tp_i


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate trained embeddings using cosine similarity (ROC AUC), compute Youden-threshold accuracy, "
        "optionally export student embeddings, and optionally save misclassified samples."
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

    # export (NOW OPTIONAL)
    ap.add_argument(
        "--output",
        default=None,
        help="Optional: output parquet/csv with student embeddings + preds. If omitted, nothing is written.",
    )
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

    # accuracy / misclassified
    ap.add_argument(
        "--threshold-method",
        choices=["youden", "zero"],
        default="youden",
        help="How to choose the classification threshold on the similarity score. "
        "'youden' maximizes (TPR-FPR) on the ROC curve; 'zero' uses threshold=0.",
    )
    ap.add_argument(
        "--misclassified-out",
        default=None,
        help="Optional: path to save misclassified rows (parquet/csv). "
        "If omitted, defaults to <output>_misclassified.<ext> ONLY if --output is provided. "
        "If neither is provided, misclassified rows are not written.",
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
    if args.label_col not in df.columns:
        raise KeyError(f"Label column '{args.label_col}' not found in input. Available: {list(df.columns)[:50]} ...")

    y = df[args.label_col].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Expected binary labels in '{args.label_col}' with values in {{0,1}}; got {np.unique(y)}")

    fraud_txt = _mat(df, args.fraud_txt_prefix)
    real_txt = _mat(df, args.real_txt_prefix)

    text_dim = fraud_txt.shape[1]
    if real_txt.shape[1] != text_dim:
        raise ValueError(f"Fraud/real text dims differ: {fraud_txt.shape[1]} vs {real_txt.shape[1]}")

    print(f"[INFO] text_dim={text_dim} | hidden_dim={args.hidden_dim} | out_dim={args.out_dim} | n={len(df)}")

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

    need_embeddings = args.output is not None
    if need_embeddings:
        zf_all = np.empty((n, args.out_dim), dtype=np.float32)
        zr_all = np.empty((n, args.out_dim), dtype=np.float32)
        if args.save_unnormalized:
            zf_raw_all = np.empty((n, args.out_dim), dtype=np.float32)
            zr_raw_all = np.empty((n, args.out_dim), dtype=np.float32)

    sims_list = []

    for start in range(0, n, bs):
        end = min(start + bs, n)

        f = fraud_txt[start:end].to(device)
        r = real_txt[start:end].to(device)

        z_f, z_r = model(f, r)

        if need_embeddings and args.save_unnormalized:
            zf_raw_all[start:end] = z_f.detach().float().cpu().numpy()
            zr_raw_all[start:end] = z_r.detach().float().cpu().numpy()

        z_f = F.normalize(z_f, dim=1)
        z_r = F.normalize(z_r, dim=1)

        if need_embeddings:
            zf_all[start:end] = z_f.detach().float().cpu().numpy()
            zr_all[start:end] = z_r.detach().float().cpu().numpy()

        sims_list.append(F.cosine_similarity(z_f, z_r, dim=1).detach().cpu())

    sims = torch.cat(sims_list).numpy().astype(np.float32, copy=False)

    # -------------------------
    # Metrics: ROC AUC + choose score direction
    # -------------------------
    auc_pos = float(roc_auc_score(y, sims))
    auc_neg = float(roc_auc_score(y, -sims))
    use_flipped = auc_neg > auc_pos

    score = (-sims) if use_flipped else sims
    auc_best = float(max(auc_pos, auc_neg))
    direction = "higher_score_more_label1 (using -cosine)" if use_flipped else "higher_score_more_label1 (using cosine)"

    # -------------------------
    # Youden / zero threshold accuracy
    # -------------------------
    if args.threshold_method == "youden":
        thr, thr_tpr, thr_fpr = _youden_threshold(y, score)
        thr_desc = f"youden (J=max), thr={thr:.6f}, TPR={thr_tpr:.4f}, FPR={thr_fpr:.4f}"
    else:
        thr = 0.0
        thr_desc = "zero (thr=0.0)"

    pred = (score >= thr).astype(np.int32)
    acc = float(accuracy_score(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # -------------------------
    # Max-accuracy threshold (over all thresholds)
    # -------------------------
    best_thr_acc, best_acc, best_tn, best_fp, best_fn, best_tp = _max_accuracy_threshold(y, score)

    # -------------------------
    # Misclassified rows (based on chosen threshold_method)
    # -------------------------
    mis_mask = pred != y
    mis_df = df.loc[mis_mask].copy()
    mis_df["score"] = score[mis_mask]
    mis_df["pred"] = pred[mis_mask]
    mis_df["y_true"] = y[mis_mask]
    mis_df["threshold"] = thr
    mis_df["threshold_method"] = args.threshold_method
    mis_df["score_direction"] = direction

    if args.misclassified_out is not None:
        mis_out = args.misclassified_out
    elif args.output is not None:
        mis_out = _make_default_misclassified_path(args.output)
    else:
        mis_out = None

    # -------------------------
    # Optionally save embeddings output
    # -------------------------
    if args.output is not None:
        out_df = df.copy()

        if not args.keep_original_embeddings:
            out_df = _drop_prefixed_cols(out_df, [args.fraud_txt_prefix, args.real_txt_prefix])

        for i in range(args.out_dim):
            out_df[f"{args.out_fraud_prefix}{i}"] = zf_all[:, i]
            out_df[f"{args.out_real_prefix}{i}"] = zr_all[:, i]

        if args.save_unnormalized:
            for i in range(args.out_dim):
                out_df[f"{args.out_fraud_prefix}{i}_raw"] = zf_raw_all[:, i]
                out_df[f"{args.out_real_prefix}{i}_raw"] = zr_raw_all[:, i]

        out_df["cosine_sim"] = sims
        out_df["score_used_for_label1"] = score
        out_df["pred"] = pred
        out_df["threshold"] = thr
        out_df["threshold_method"] = args.threshold_method
        out_df["score_direction"] = direction

        _save_table(out_df, args.output)
        print(f"[SAVED] embeddings+preds:   {args.output}")

    if mis_out is not None:
        _save_table(mis_df, mis_out)
        print(f"[SAVED] misclassified:     {mis_out}")

    # -------------------------
    # Print summary
    # -------------------------
    print("\n==============================")
    print(" STUDENT EMBEDDING EVALUATION")
    print("==============================")
    print(f"ROC AUC (cosine):          {auc_pos:.6f}")
    print(f"ROC AUC (cosine flipped):  {auc_neg:.6f}")
    print(f"ROC AUC (best):            {auc_best:.6f}")
    print(f"Direction:                 {direction}")
    print("------------------------------")
    print(f"Threshold ({args.threshold_method}):  {thr_desc}")
    print(f"Accuracy @ {args.threshold_method}:  {acc:.6f}")
    print(f"Confusion @ {args.threshold_method} (tn, fp, fn, tp): ({tn}, {fp}, {fn}, {tp})")
    print("------------------------------")
    print(f"Max Accuracy:              {best_acc:.6f}")
    print(f"Max-Acc Threshold:         thr={best_thr_acc:.6f}")
    print(f"Confusion @ max-acc (tn, fp, fn, tp): ({best_tn}, {best_fp}, {best_fn}, {best_tp})")
    print("------------------------------")
    print(f"Misclassified count (@{args.threshold_method}): {int(mis_mask.sum())} / {n}")
    print("==============================\n")


if __name__ == "__main__":
    main()
"""
Examples:

1) Metrics only (writes nothing):
python text_to_image/evaluator2.py \
  --test ../Downloads/vate_test.parquet \
  --model-path saved_models/pacifico_best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768

2) Save only misclassified (still writes nothing else):
python text_to_image/evaluator2.py \
  --test ../Downloads/vate_test.parquet \
  --model-path saved_models/pacifico_best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --misclassified-out text_to_image/Golden_and_Text/pacifico_misclassified.parquet

3) Save embeddings+preds AND (by default) misclassified:
python text_to_image/evaluator2.py \
  --test ../Downloads/vate_test.parquet \
  --model-path saved_models/pacifico_best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --output text_to_image/Golden_and_Text/test_with_student_embs.parquet
"""