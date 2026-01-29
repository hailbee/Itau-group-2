#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

"""
# Golden / spoof-aware image embeddings
python golden_eval.py \
  --input Golden_and_Text/test_pairs_with_img_and_txt_embs.parquet \
  --space raw_text
  """

def compute_metrics(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    youden_j = tpr - fpr
    idx = int(np.argmax(youden_j))
    thr = float(thresholds[idx])

    y_pred = (y_scores >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "youden_threshold": thr,
        "accuracy_youden": acc,
        "confusion_matrix_youden": cm.tolist(),
    }


def get_sorted_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) == 0:
        raise ValueError(f"No columns found with prefix '{prefix}'")

    # sort by numeric suffix
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    return cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--space",
        choices=["aligned_img", "raw_text"],
        default="aligned_img",
        help="Which space to evaluate",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)

    if args.space == "raw_img":
        fraud_cols = get_sorted_cols(df, "fraud_raw_")
        real_cols  = get_sorted_cols(df, "real_raw_")
    elif args.space == "aligned_img":
        fraud_cols = get_sorted_cols(df, "fraud_aligned_")
        real_cols  = get_sorted_cols(df, "real_aligned_")
    elif args.space == "raw_text":
        fraud_cols = get_sorted_cols(df, "fraud_txt_emb_")
        real_cols  = get_sorted_cols(df, "real_txt_emb_")
    else:
        raise ValueError(args.space)

    assert len(fraud_cols) == len(real_cols), "Fraud/real dim mismatch"

    fraud = torch.tensor(df[fraud_cols].to_numpy(np.float32))
    real  = torch.tensor(df[real_cols].to_numpy(np.float32))
    labels = df["label"].to_numpy(int)

    fraud = F.normalize(fraud, dim=1)
    real  = F.normalize(real, dim=1)
    sims = F.cosine_similarity(fraud, real, dim=1).cpu().numpy()

    metrics = compute_metrics(labels, sims)

    print(f"\n[{args.space.upper()} BASELINE]")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
