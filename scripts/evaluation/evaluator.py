import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score


class Evaluator:
    """
    Threshold-free evaluation for pairwise Siamese model where model(x1, x2) -> (z1, z2).
    Computes similarity scores in the learned projection space and reports ROC AUC only.
    Optionally saves ROC curve plot as a PNG.
    """
    def __init__(self, model, batch_size=32, model_type=None):
        self.model = model
        self.batch_size = batch_size
        self.model_type = model_type

    def _save_roc_plot(self, fpr, tpr, roc_auc, save_path, title="ROC Curve"):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved ROC curve to {save_path}")

    def compute_metrics(self, results_df, plot=False, roc_png_path=None):
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df["similarity"].astype(float).to_numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))

        # -------------------------------
        # Youden threshold
        # -------------------------------
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        youden_threshold = thresholds[best_idx]

        # Predictions using Youden threshold
        y_pred = (y_scores >= youden_threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)

        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Youden threshold: {youden_threshold:.4f}")
        print(f"Accuracy (Youden): {accuracy:.4f}")

        if plot:
            # Default save path if none provided
            if roc_png_path is None:
                os.makedirs("images", exist_ok=True)
                roc_png_path = "images/roc_curve.png"

            self._save_roc_plot(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                save_path=roc_png_path,
                title="ROC Curve"
            )
        
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_youden_j = youden_j[best_idx]

        return {
            "roc_auc": roc_auc,
            "youden_j": float(best_youden_j),
            "youden_threshold": float(youden_threshold),
            "accuracy": float(accuracy),
    }

    def evaluate(self, test_filepath, plot=False, max_rows=None, roc_png_path=None):
        return self.test_pairs(
            test_filepath,
            plot=plot,
            max_rows=max_rows,
            roc_png_path=roc_png_path
        )

    def test_pairs(self, test_filepath, plot=False, max_rows=None, roc_png_path=None):
        # Load data
        if test_filepath.endswith(".csv"):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)

        if max_rows is not None:
            df = df.head(int(max_rows))

        fraud_names = df["fraudulent_name"].astype(str).tolist()
        real_names = df["real_name"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        # Pull embeddings (adjust slices if your parquet layout changes)
        fraud_np = df.iloc[:, 3:771].to_numpy(dtype=np.float32, copy=False)
        real_np  = df.iloc[:, 771:1539].to_numpy(dtype=np.float32, copy=False)

        fraud_embs = torch.from_numpy(fraud_np)
        real_embs  = torch.from_numpy(real_np)

        # Decide device from model
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

        self.model.eval()
        sims_all = []
        bs = int(self.batch_size)

        with torch.no_grad():
            for start in range(0, len(df), bs):
                end = start + bs
                x1 = fraud_embs[start:end].to(device)
                x2 = real_embs[start:end].to(device)

                z1, z2 = self.model(x1, x2)

                # Match training behavior: L2 normalize before comparing
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)

                sims = F.cosine_similarity(z1, z2, dim=1)
                sims_all.append(sims.detach().cpu())

        similarities = torch.cat(sims_all, dim=0).numpy()

        results_df = pd.DataFrame({
            "fraudulent_name": fraud_names,
            "real_name": real_names,
            "label": labels,
            "similarity": similarities
        })

        metrics = self.compute_metrics(results_df, plot=plot, roc_png_path=roc_png_path)
        return results_df, metrics
