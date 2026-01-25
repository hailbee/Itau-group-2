import os
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


@dataclass
class EvalConfig:
    batch_size: int = 256
    fraud_txt_slice: tuple[int, int] = (1539, 2307)
    real_txt_slice: tuple[int, int] = (2307, 3075)


class Evaluator2:
    """
    Evaluator for text2img student:
      model(fraud_txt, real_txt) -> (z_fraud, z_real)
      similarity = cosine(z_fraud, z_real)
    """

    def __init__(self, model, cfg: EvalConfig | None = None):
        self.model = model
        self.cfg = cfg or EvalConfig()

    def _save_roc_plot(self, fpr, tpr, roc_auc, save_path, title="ROC Curve"):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _save_cm_plot(self, cm, save_path, title="Confusion Matrix"):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure()
        plt.imshow(cm)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def compute_metrics(
        self,
        results_df,
        plot=False,
        roc_png_path=None,
        cm_png_path=None,
        title_prefix="Test",
    ):
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df["similarity"].astype(float).to_numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))

        youden_j = tpr - fpr
        youden_best_idx = int(np.argmax(youden_j))
        youden_threshold = float(thresholds[youden_best_idx])
        best_youden_j = float(youden_j[youden_best_idx])

        y_pred_youden = (y_scores >= youden_threshold).astype(int)
        accuracy_youden = float(accuracy_score(y_true, y_pred_youden))
        cm_youden = confusion_matrix(y_true, y_pred_youden)

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

        if plot:
            if roc_png_path is None:
                os.makedirs("images", exist_ok=True)
                roc_png_path = "images/roc_curve_text2img.png"
            if cm_png_path is None:
                os.makedirs("images", exist_ok=True)
                cm_png_path = "images/confusion_matrix_text2img.png"

            self._save_roc_plot(fpr, tpr, roc_auc, roc_png_path, title=f"{title_prefix} ROC Curve")
            self._save_cm_plot(cm_youden, cm_png_path, title=f"{title_prefix} Confusion Matrix (Youden)")

        return {
            "roc_auc": roc_auc,
            "youden_j": best_youden_j,
            "youden_threshold": youden_threshold,
            "accuracy": accuracy_youden,
            "best_accuracy": best_accuracy,
            "best_accuracy_threshold": best_accuracy_threshold,
            "confusion_matrix_youden": cm_youden.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        }

    @torch.no_grad()
    def evaluate(
        self,
        test_filepath,
        plot=False,
        max_rows=None,
        roc_png_path=None,
        cm_png_path=None,
    ):
        if str(test_filepath).endswith(".csv"):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)

        if max_rows is not None:
            df = df.head(int(max_rows))

        fraud_names = df["fraudulent_name"].astype(str).tolist() if "fraudulent_name" in df.columns else [""] * len(df)
        real_names = df["real_name"].astype(str).tolist() if "real_name" in df.columns else [""] * len(df)
        labels = df["label"].astype(int).tolist() if "label" in df.columns else [0] * len(df)

        fs, fe = self.cfg.fraud_txt_slice
        rs, re_ = self.cfg.real_txt_slice

        fraud_txt_np = df.iloc[:, fs:fe].to_numpy(dtype=np.float32, copy=False)
        real_txt_np = df.iloc[:, rs:re_].to_numpy(dtype=np.float32, copy=False)

        fraud_txt = torch.from_numpy(fraud_txt_np)
        real_txt = torch.from_numpy(real_txt_np)
        y = torch.tensor(labels, dtype=torch.int64)

        device = next(self.model.parameters()).device
        self.model.eval()

        sims, ys = [], []
        bs = int(self.cfg.batch_size)
        for start in range(0, len(df), bs):
            end = min(len(df), start + bs)
            x1 = fraud_txt[start:end].to(device)
            x2 = real_txt[start:end].to(device)

            z1, z2 = self.model(x1, x2)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            sim = F.cosine_similarity(z1, z2, dim=1)

            sims.append(sim.detach().cpu())
            ys.append(y[start:end].cpu())

        similarities = torch.cat(sims, dim=0).numpy()
        labels_np = torch.cat(ys, dim=0).numpy().astype(int)

        results_df = pd.DataFrame({
            "fraudulent_name": fraud_names,
            "real_name": real_names,
            "label": labels_np,
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
