import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score
from sklearn.metrics import auc
from utils.evals import (
    find_best_threshold_youden,
    plot_roc_curve,
    plot_confusion_matrix,
    find_best_threshold_accuracy
)

class Evaluator:
    """
    Evaluation for pairwise Siamese model where model(x1, x2) -> (z1, z2).
    Computes similarity scores from the MODEL OUTPUTS (not raw embeddings).
    """
    def __init__(self, model, batch_size=32, model_type=None):
        self.model = model
        self.batch_size = batch_size
        self.model_type = model_type

    def compute_metrics(self, results_df, plot=False):
        y_true = results_df["label"].astype(int)
        y_scores = results_df["similarity"].astype(float)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")

        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)
        best_acc, best_acc_threshold = find_best_threshold_accuracy(y_true, y_scores, thresholds)

        y_pred = (y_scores > youden_thresh).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "threshold": float(youden_thresh),
            "roc_curve": (fpr, tpr, thresholds),
            "roc_auc": float(roc_auc),
            "best_accuracy": float(best_acc),
            "best_accuracy_threshold": float(best_acc_threshold),
        }

        if plot:
            plot_roc_curve(results_df)
            print(f"Plotting confusion matrix at Youden's threshold: {youden_thresh:.3f}")
            plot_confusion_matrix(y_true, y_scores, youden_thresh)
            print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
            plot_confusion_matrix(y_true, y_scores, best_acc_threshold)

        return metrics

    def evaluate(self, test_filepath, plot=False, max_rows=None):
        return self.test_pairs(test_filepath, plot=plot, max_rows=max_rows)

    def test_pairs(self, test_filepath, plot=False, max_rows=None):
        # Load data
        if test_filepath.endswith(".csv"):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)

        if max_rows is not None:
            df = df.head(int(max_rows))

        # Names/labels
        fraud_names = df["fraudulent_name"].astype(str).tolist()
        real_names = df["real_name"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        # Pull embeddings (adjust slices if your parquet layout changes)
        fraud_np = df.iloc[:, 3:771].to_numpy(dtype=np.float32, copy=False)
        real_np  = df.iloc[:, 771:1539].to_numpy(dtype=np.float32, copy=False)

        fraud_embs = torch.from_numpy(fraud_np)  # CPU tensor
        real_embs  = torch.from_numpy(real_np)   # CPU tensor

        # Decide device from model
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval()

        sims_all = []
        bs = int(self.batch_size)

        with torch.no_grad():
            for start in range(0, len(df), bs):
                end = start + bs
                x1 = fraud_embs[start:end].to(device, non_blocking=True)
                x2 = real_embs[start:end].to(device, non_blocking=True)

                # Run model (learned projection space)
                z1, z2 = self.model(x1, x2)

                # Match training behavior: L2 normalize before comparing
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)

                # Higher similarity => more likely positive (label 1)
                sims = F.cosine_similarity(z1, z2, dim=1)

                sims_all.append(sims.detach().cpu())

        similarities = torch.cat(sims_all, dim=0).numpy()

        results_df = pd.DataFrame({
            "fraudulent_name": fraud_names,
            "real_name": real_names,
            "label": labels,
            "similarity": similarities
        })

        metrics = self.compute_metrics(results_df, plot=plot)
        return results_df, metrics
