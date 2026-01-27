# scripts/evaluation/evaluator2.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


@dataclass
class EvalConfig:
    """
    Mirrors your old Evaluator style, but for Text->Img distill setup.
    We evaluate:
      - Alignment: cos(pred_*_txt, *_img_teacher)
      - Downstream spoof score: cos(pred_fraud, pred_real)
    """
    batch_size: int = 1024

    # embedding column slices (by integer index)
    fraud_img_slice: tuple[int, int] = (3, 771)
    real_img_slice: tuple[int, int] = (771, 1539)
    fraud_txt_slice: tuple[int, int] = (1539, 2307)
    real_txt_slice: tuple[int, int] = (2307, 3075)

    # column names
    fraud_name_col: str = "fraudulent_name"
    real_name_col: str = "real_name"
    label_col: str = "label"

    # fallback label index if label_col not present
    fallback_label_idx: int = 2


class Evaluator2:
    """
    Evaluator for text2img-distill models where:
      model(fraud_txt, real_txt) -> (pred_fraud, pred_real)

    Produces a results_df with:
      - sim_aligned: cosine(pred_fraud, pred_real)   [THIS is your ROC-AUC score]
      - cos_align_fraud_to_img: cosine(pred_fraud, fraud_img_teacher)
      - cos_align_real_to_img:  cosine(pred_real,  real_img_teacher)

    Optionally includes:
      - sim_raw_text:  cosine(raw_fraud_txt, raw_real_txt)  (baseline you said you can compute separately)
      - sim_teacher_img: cosine(fraud_img_teacher, real_img_teacher) (reference / upper bound-ish)

    And reports metrics in the same format as scripts/evaluation/evaluator.py:
      - ROC AUC
      - Youden threshold + accuracy + confusion matrix
      - Best-accuracy threshold + best accuracy
      - Optional ROC + CM plots
    """

    def __init__(self, model, cfg: EvalConfig | None = None):
        self.model = model
        self.cfg = cfg or EvalConfig()

    # -------------------------------
    # Plot helpers (same style as old)
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
    # Metrics (same as old, but score_col configurable)
    # -------------------------------
    def compute_metrics(
        self,
        results_df: pd.DataFrame,
        score_col: str = "similarity",
        plot: bool = False,
        roc_png_path: str | None = None,
        cm_png_path: str | None = None,
        title_prefix: str = "Test",
    ):
        """
        results_df must contain:
          - label (0/1)
          - score_col (float score; higher => more spoof per your convention)
        """
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df[score_col].astype(float).to_numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))

        # Youden threshold (maximize TPR - FPR)
        youden_j = tpr - fpr
        youden_best_idx = int(np.argmax(youden_j))
        youden_threshold = float(thresholds[youden_best_idx])
        best_youden_j = float(youden_j[youden_best_idx])

        y_pred_youden = (y_scores >= youden_threshold).astype(int)
        accuracy_youden = float(accuracy_score(y_true, y_pred_youden))
        cm_youden = confusion_matrix(y_true, y_pred_youden)

        # Best-accuracy threshold (maximize accuracy)
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

        # Logging
        print(f"[{title_prefix}] ROC AUC ({score_col}): {roc_auc:.4f}")
        print(f"[{title_prefix}] Youden threshold: {youden_threshold:.4f}")
        print(f"[{title_prefix}] Accuracy (Youden): {accuracy_youden:.4f}")
        print(f"[{title_prefix}] Best-Acc threshold: {best_accuracy_threshold:.4f}")
        print(f"[{title_prefix}] Best Accuracy: {best_accuracy:.4f}")

        # Plots
        if plot:
            if roc_png_path is None:
                os.makedirs("images", exist_ok=True)
                roc_png_path = f"images/roc_curve_{score_col}.png"
            self._save_roc_plot(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                save_path=roc_png_path,
                title=f"{title_prefix} ROC Curve ({score_col})",
            )

            if cm_png_path is None:
                os.makedirs("images", exist_ok=True)
                cm_png_path = f"images/confusion_matrix_youden_{score_col}.png"
            self._save_confusion_matrix_plot(
                cm=cm_youden,
                save_path=cm_png_path,
                title=f"{title_prefix} Confusion Matrix (Youden) ({score_col})",
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
    # Public entrypoints (same as old)
    # -------------------------------
    def evaluate(
        self,
        test_filepath,
        plot: bool = False,
        max_rows: int | None = None,
        roc_png_path: str | None = None,
        cm_png_path: str | None = None,
    ):
        return self.test_pairs(
            test_filepath,
            plot=plot,
            max_rows=max_rows,
            roc_png_path=roc_png_path,
            cm_png_path=cm_png_path,
        )

    @torch.no_grad()
    def test_pairs(
        self,
        test_filepath,
        plot: bool = False,
        max_rows: int | None = None,
        roc_png_path: str | None = None,
        cm_png_path: str | None = None,
    ):
        # Load data
        if str(test_filepath).endswith(".csv"):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)

        if max_rows is not None:
            df = df.head(int(max_rows))

        # Names + labels
        fraud_names = (
            df[self.cfg.fraud_name_col].astype(str).tolist()
            if self.cfg.fraud_name_col in df.columns
            else [""] * len(df)
        )
        real_names = (
            df[self.cfg.real_name_col].astype(str).tolist()
            if self.cfg.real_name_col in df.columns
            else [""] * len(df)
        )

        if self.cfg.label_col in df.columns:
            labels = df[self.cfg.label_col].astype(int).to_numpy()
        else:
            labels = df.iloc[:, int(self.cfg.fallback_label_idx)].astype(int).to_numpy()

        # Pull embeddings (CPU tensors)
        fi, fe = self.cfg.fraud_img_slice
        ri, re = self.cfg.real_img_slice
        ft, fte = self.cfg.fraud_txt_slice
        rt, rte = self.cfg.real_txt_slice

        fraud_img = torch.from_numpy(df.iloc[:, fi:fe].to_numpy(dtype=np.float32, copy=False))
        real_img  = torch.from_numpy(df.iloc[:, ri:re].to_numpy(dtype=np.float32, copy=False))
        fraud_txt = torch.from_numpy(df.iloc[:, ft:fte].to_numpy(dtype=np.float32, copy=False))
        real_txt  = torch.from_numpy(df.iloc[:, rt:rte].to_numpy(dtype=np.float32, copy=False))

        # Decide device from model
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )

        self.model.eval()

        # Baseline/reference scores computed on CPU (fast and avoids GPU mem)
        fraud_txt_n = F.normalize(fraud_txt, dim=1)
        real_txt_n  = F.normalize(real_txt,  dim=1)
        sim_raw_text = F.cosine_similarity(fraud_txt_n, real_txt_n, dim=1).numpy()

        fraud_img_n = F.normalize(fraud_img, dim=1)
        real_img_n  = F.normalize(real_img,  dim=1)
        sim_teacher_img = F.cosine_similarity(fraud_img_n, real_img_n, dim=1).numpy()

        # Model outputs in batches (move only slices to device)
        bs = int(self.cfg.batch_size)
        sim_aligned_all = []
        align_fraud_all = []
        align_real_all = []

        for start in range(0, len(df), bs):
            end = min(start + bs, len(df))

            x_fraud_txt = fraud_txt[start:end].to(device, non_blocking=True)
            x_real_txt  = real_txt[start:end].to(device, non_blocking=True)

            t_fraud_img = fraud_img[start:end].to(device, non_blocking=True)
            t_real_img  = real_img[start:end].to(device, non_blocking=True)

            pred_fraud, pred_real = self.model(x_fraud_txt, x_real_txt)

            pred_fraud = F.normalize(pred_fraud, dim=1)
            pred_real  = F.normalize(pred_real,  dim=1)
            t_fraud_img = F.normalize(t_fraud_img, dim=1)
            t_real_img  = F.normalize(t_real_img,  dim=1)

            sim_aligned = F.cosine_similarity(pred_fraud, pred_real, dim=1)
            cos_align_fraud = F.cosine_similarity(pred_fraud, t_fraud_img, dim=1)
            cos_align_real  = F.cosine_similarity(pred_real,  t_real_img,  dim=1)

            sim_aligned_all.append(sim_aligned.detach().cpu())
            align_fraud_all.append(cos_align_fraud.detach().cpu())
            align_real_all.append(cos_align_real.detach().cpu())

        sim_aligned = torch.cat(sim_aligned_all, dim=0).numpy()
        cos_align_fraud_to_img = torch.cat(align_fraud_all, dim=0).numpy()
        cos_align_real_to_img  = torch.cat(align_real_all,  dim=0).numpy()

        # Results DF (keep everything you might want to inspect)
        results_df = pd.DataFrame({
            "fraudulent_name": fraud_names,
            "real_name": real_names,
            "label": labels.astype(int),

            # Scores you care about:
            "sim_aligned": sim_aligned,           # main KPI space
            "sim_raw_text": sim_raw_text,         # baseline (optional)
            "sim_teacher_img": sim_teacher_img,   # reference (optional)

            # Alignment debug:
            "cos_align_fraud_to_img": cos_align_fraud_to_img,
            "cos_align_real_to_img": cos_align_real_to_img,
        })

        # Main metrics: ALIGNED (plots optional)
        aligned_metrics = self.compute_metrics(
            results_df.rename(columns={"sim_aligned": "similarity"})[["label", "similarity"]],
            score_col="similarity",
            plot=plot,
            roc_png_path=roc_png_path,
            cm_png_path=cm_png_path,
            title_prefix="ALIGNED",
        )

        # Optional extra blocks for quick comparison (no plots by default)
        raw_metrics = self.compute_metrics(
            results_df.rename(columns={"sim_raw_text": "similarity"})[["label", "similarity"]],
            score_col="similarity",
            plot=False,
            title_prefix="RAW_TEXT",
        )
        teacher_metrics = self.compute_metrics(
            results_df.rename(columns={"sim_teacher_img": "similarity"})[["label", "similarity"]],
            score_col="similarity",
            plot=False,
            title_prefix="TEACHER_IMG",
        )

        alignment_debug = {
            "fraud_mean_cosine_to_image": float(np.mean(cos_align_fraud_to_img)),
            "real_mean_cosine_to_image": float(np.mean(cos_align_real_to_img)),
            "fraud_median_cosine_to_image": float(np.median(cos_align_fraud_to_img)),
            "real_median_cosine_to_image": float(np.median(cos_align_real_to_img)),
            "fraud_std_cosine_to_image": float(np.std(cos_align_fraud_to_img)),
            "real_std_cosine_to_image": float(np.std(cos_align_real_to_img)),
        }

        metrics = {
            "aligned_text_space": aligned_metrics,
            "raw_text_space": raw_metrics,
            "teacher_image_space": teacher_metrics,
            "alignment_debug": alignment_debug,
            "deltas": {
                "auc_aligned_minus_raw": float(aligned_metrics["roc_auc"] - raw_metrics["roc_auc"]),
                "auc_teacher_minus_aligned": float(teacher_metrics["roc_auc"] - aligned_metrics["roc_auc"]),
            },
        }

        return results_df, metrics
