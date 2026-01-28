# scripts/evaluation/evaluator.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


class Evaluator:
    """
    Evaluation for pairwise Siamese model where model(x1, x2) -> (z1, z2).

    Computes cosine similarity scores in the learned projection space and reports:
      - ROC AUC
      - Youden-optimal threshold (maximizes TPR - FPR) + accuracy at that threshold
      - Best-accuracy threshold (any threshold maximizing accuracy) + that best accuracy
      - Confusion matrix at Youden threshold

    Optionally saves:
      - ROC curve PNG
      - Confusion matrix PNG (Youden threshold)
    """

    def __init__(self, model, batch_size=32, model_type=None):
        self.model = model
        self.batch_size = int(batch_size)
        self.model_type = model_type

    # -------------------------------
    # Plot helpers
    # -------------------------------
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

    def _save_confusion_matrix_plot(
        self,
        cm,
        save_path,
        title="Confusion Matrix (Youden)",
        class_names=("0", "1"),
    ):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, [f"Pred {c}" for c in class_names])
        plt.yticks(tick_marks, [f"True {c}" for c in class_names])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Saved confusion matrix to {save_path}")

    # -------------------------------
    # Metrics
    # -------------------------------
    def compute_metrics(
        self,
        results_df,
        plot=False,
        roc_png_path=None,
        cm_png_path=None,
        title_prefix="Test",
    ):
        """
        results_df must contain:
          - label (0/1)
          - similarity (float cosine similarity)
        """
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df["similarity"].astype(float).to_numpy()

        # ROC + thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))

        # -------------------------------
        # Youden threshold (maximize TPR - FPR)
        # -------------------------------
        youden_j = tpr - fpr
        youden_best_idx = int(np.argmax(youden_j))
        youden_threshold = float(thresholds[youden_best_idx])
        best_youden_j = float(youden_j[youden_best_idx])

        y_pred_youden = (y_scores >= youden_threshold).astype(int)
        accuracy_youden = float(accuracy_score(y_true, y_pred_youden))
        cm_youden = confusion_matrix(y_true, y_pred_youden)

        # -------------------------------
        # Best-accuracy threshold (maximize accuracy)
        # -------------------------------
        finite_mask = np.isfinite(thresholds)
        finite_thresholds = thresholds[finite_mask]

        accs = []
        for thr in finite_thresholds:
            y_pred = (y_scores >= thr).astype(int)
            accs.append(accuracy_score(y_true, y_pred))
        accs = np.asarray(accs, dtype=float)

        best_acc_idx = int(np.argmax(accs))
        best_accuracy_threshold = float(finite_thresholds[best_acc_idx])
        best_accuracy = float(accs[best_acc_idx])

        # -------------------------------
        # Logging
        # -------------------------------
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Youden threshold: {youden_threshold:.4f}")
        print(f"Accuracy (Youden): {accuracy_youden:.4f}")
        print(f"Best-Acc threshold: {best_accuracy_threshold:.4f}")
        print(f"Best Accuracy: {best_accuracy:.4f}")

        # -------------------------------
        # Plots
        # -------------------------------
        if plot:
            # ROC
            if roc_png_path is None:
                os.makedirs("images", exist_ok=True)
                roc_png_path = "images/roc_curve.png"
            self._save_roc_plot(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                save_path=roc_png_path,
                title=f"{title_prefix} ROC Curve",
            )

            # Confusion matrix (Youden)
            if cm_png_path is None:
                os.makedirs("images", exist_ok=True)
                cm_png_path = "images/confusion_matrix_youden.png"
            self._save_confusion_matrix_plot(
                cm=cm_youden,
                save_path=cm_png_path,
                title=f"{title_prefix} Confusion Matrix (Youden)",
                class_names=("0", "1"),
            )

        return {
            "roc_auc": roc_auc,

            "youden_j": best_youden_j,
            "youden_threshold": youden_threshold,
            "accuracy_youden": accuracy_youden,
            "confusion_matrix_youden": cm_youden.tolist(),

            "best_accuracy": best_accuracy,
            "best_accuracy_threshold": best_accuracy_threshold,
        }

    # -------------------------------
    # Public entrypoints
    # -------------------------------
    def evaluate(
        self,
        test_filepath,
        plot=False,
        max_rows=None,
        roc_png_path=None,
        cm_png_path=None,
    ):
        return self.test_pairs(
            test_filepath,
            plot=plot,
            max_rows=max_rows,
            roc_png_path=roc_png_path,
            cm_png_path=cm_png_path,
        )

    def test_pairs(
        self,
        test_filepath,
        plot=False,
        max_rows=None,
        roc_png_path=None,
        cm_png_path=None,
    ):
        # Load data
        if str(test_filepath).endswith(".csv"):
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
        real_np = df.iloc[:, 771:1539].to_numpy(dtype=np.float32, copy=False)

        fraud_embs = torch.from_numpy(fraud_np)
        real_embs = torch.from_numpy(real_np)

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
            "similarity": similarities,
        })

        metrics = self.compute_metrics(
            results_df,
            plot=plot,
            roc_png_path=roc_png_path,
            cm_png_path=cm_png_path,
            title_prefix="Test",
        )
        return results_df, metrics