import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

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

        def compute_metrics(
            self,
            results_df,
            plot=False,
            roc_png_path=None,
            acc_curve_png_path=None,
            cm_png_path=None,
            title_prefix="Test",
        ):
            y_true = results_df["label"].astype(int).to_numpy()
            y_scores = results_df["similarity"].astype(float).to_numpy()

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = float(auc(fpr, tpr))

            # -------------------------------
            # Youden threshold
            # -------------------------------
            youden_j = tpr - fpr
            youden_best_idx = int(np.argmax(youden_j))
            youden_threshold = float(thresholds[youden_best_idx])

            y_pred_youden = (y_scores >= youden_threshold).astype(int)
            accuracy_youden = float(accuracy_score(y_true, y_pred_youden))
            cm_youden = confusion_matrix(y_true, y_pred_youden)

            # -------------------------------
            # Best-accuracy threshold (NOT necessarily Youden)
            # -------------------------------
            # roc_curve can include inf as a threshold; ignore non-finite thresholds
            finite_mask = np.isfinite(thresholds)
            finite_thresholds = thresholds[finite_mask]

            # Compute accuracy at each threshold
            accs = []
            for thr in finite_thresholds:
                y_pred = (y_scores >= thr).astype(int)
                accs.append(accuracy_score(y_true, y_pred))
            accs = np.asarray(accs, dtype=float)

            best_acc_idx = int(np.argmax(accs))
            best_acc_threshold = float(finite_thresholds[best_acc_idx])
            best_accuracy = float(accs[best_acc_idx])

            # -------------------------------
            # Prints
            # -------------------------------
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Youden threshold: {youden_threshold:.4f}")
            print(f"Accuracy (Youden): {accuracy_youden:.4f}")
            print(f"Best-Acc threshold: {best_acc_threshold:.4f}")
            print(f"Best Accuracy: {best_accuracy:.4f}")

            # -------------------------------
            # Plots
            # -------------------------------
            if plot:
                # ROC plot (existing behavior)
                if roc_png_path is None:
                    os.makedirs("images", exist_ok=True)
                    roc_png_path = "images/roc_curve.png"
                self._save_roc_plot(
                    fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                    save_path=roc_png_path,
                    title=f"{title_prefix} ROC Curve"
                )

                # Accuracy vs threshold plot (shows best and Youden)
                if acc_curve_png_path is None:
                    os.makedirs("images", exist_ok=True)
                    acc_curve_png_path = "images/accuracy_vs_threshold.png"

                os.makedirs(os.path.dirname(acc_curve_png_path) or ".", exist_ok=True)
                plt.figure()
                plt.plot(finite_thresholds, accs, label="Accuracy")
                plt.axvline(youden_threshold, linestyle="--", label=f"Youden thr={youden_threshold:.4f}")
                plt.axvline(best_acc_threshold, linestyle="--", label=f"Best-Acc thr={best_acc_threshold:.4f}")
                plt.xlabel("Threshold (cosine similarity)")
                plt.ylabel("Accuracy")
                plt.title(f"{title_prefix} Accuracy vs Threshold")
                plt.legend()
                plt.tight_layout()
                plt.savefig(acc_curve_png_path)
                plt.close()
                print(f"[DEBUG] Saved accuracy-vs-threshold curve to {acc_curve_png_path}")

                # Confusion matrix plot (at Youden threshold)
                if cm_png_path is None:
                    os.makedirs("images", exist_ok=True)
                    cm_png_path = "images/confusion_matrix_youden.png"

                os.makedirs(os.path.dirname(cm_png_path) or ".", exist_ok=True)
                plt.figure()
                plt.imshow(cm_youden, interpolation="nearest")
                plt.title(f"{title_prefix} Confusion Matrix (Youden)")
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
                plt.yticks(tick_marks, ["True 0", "True 1"])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(cm_youden[i, j]), ha="center", va="center")

                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.tight_layout()
                plt.savefig(cm_png_path)
                plt.close()
                print(f"[DEBUG] Saved confusion matrix to {cm_png_path}")

            # Youden J best value
            best_youden_j = float(youden_j[youden_best_idx])

            return {
                "roc_auc": roc_auc,

                "youden_j": best_youden_j,
                "youden_threshold": youden_threshold,
                "accuracy_youden": accuracy_youden,
                "confusion_matrix_youden": cm_youden.tolist(),

                "best_accuracy": best_accuracy,
                "best_accuracy_threshold": best_acc_threshold,
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
