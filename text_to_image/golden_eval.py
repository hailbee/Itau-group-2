import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

"""

python3 golden_eval.py \
  --input Golden/golden_embeddings_test.parquet \
  --space aligned

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--space", choices=["raw", "aligned"], default="raw")
    args = p.parse_args()

    df = pd.read_parquet(args.input)

    if args.space == "raw":
        fraud_cols = [c for c in df.columns if c.startswith("fraud_raw_")]
        real_cols  = [c for c in df.columns if c.startswith("real_raw_")]
    else:
        fraud_cols = [c for c in df.columns if c.startswith("fraud_aligned_")]
        real_cols  = [c for c in df.columns if c.startswith("real_aligned_")]

    fraud = torch.tensor(df[fraud_cols].to_numpy(np.float32))
    real  = torch.tensor(df[real_cols].to_numpy(np.float32))
    labels = df["label"].to_numpy(int)

    fraud = F.normalize(fraud, dim=1)
    real  = F.normalize(real, dim=1)
    sims = F.cosine_similarity(fraud, real, dim=1).cpu().numpy()

    metrics = compute_metrics(labels, sims)

    print(f"[{args.space.upper()} BASELINE]")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
