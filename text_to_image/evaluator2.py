# evaluator2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix


def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    cols.sort(key=lambda c: (key_fn(c), c))
    return cols


def _mat(df: pd.DataFrame, prefix: str) -> torch.Tensor:
    cols = _sorted_prefixed_cols(df, prefix)
    return torch.tensor(df[cols].to_numpy(dtype=np.float32, copy=False))


@dataclass
class EvalConfig:
    batch_size: int = 2048

    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"

    fraud_teacher_prefix: str = "fraud_aligned_"
    real_teacher_prefix: str = "real_aligned_"

    label_col: str = "label"
    fraud_name_col: str = "fraudulent_name"
    real_name_col: str = "real_name"


class Evaluator2:
    """
    Evaluation aligned with your reported teacher baseline:

      - RAW_TEXT:  cos(fraud_txt, real_txt)
      - TEACHER:   cos(fraud_teacher, real_teacher)
      - STUDENT:   cos(pred_fraud, pred_real)

    Diagnostics:
      - cos(pred_fraud, fraud_teacher)
      - cos(pred_real,  real_teacher)
    """

    def __init__(self, model, cfg: Optional[EvalConfig] = None):
        self.model = model
        self.cfg = cfg or EvalConfig()

    def _compute_metrics(self, y_true, y_scores, tag: str) -> Dict[str, Any]:
        """
        IMPORTANT: ROC AUC computed with roc_auc_score() to MATCH Optuna exactly.
        Threshold/accuracy still uses roc_curve for Youden's J.
        """
        # Match Optuna label handling as closely as possible
        y_true = np.asarray(y_true).astype(np.int32, copy=False)
        y_scores = np.asarray(y_scores)

        # Optuna-style AUC
        roc_auc = float(roc_auc_score(y_true, y_scores))

        # Youden threshold (needs roc_curve)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        youden_j = tpr - fpr
        idx = int(np.argmax(youden_j))
        thr = float(thresholds[idx])

        y_pred = (y_scores >= thr).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)

        print(f"[{tag}] ROC AUC: {roc_auc:.4f}")
        print(f"[{tag}] Youden threshold: {thr:.4f}")
        print(f"[{tag}] Accuracy (Youden): {acc:.4f}")

        return {
            "roc_auc": roc_auc,
            "youden_threshold": thr,
            "accuracy_youden": acc,
            "confusion_matrix_youden": cm.tolist(),
        }

    @torch.inference_mode()
    def evaluate(
        self,
        filepath: str,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:

        df = pd.read_parquet(filepath) if filepath.endswith(".parquet") else pd.read_csv(filepath)
        if max_rows is not None:
            df = df.head(int(max_rows)).copy()

        y = df[self.cfg.label_col].astype(int).to_numpy()

        fraud_txt = _mat(df, self.cfg.fraud_txt_prefix)
        real_txt  = _mat(df, self.cfg.real_txt_prefix)

        fraud_teacher = _mat(df, self.cfg.fraud_teacher_prefix)
        real_teacher  = _mat(df, self.cfg.real_teacher_prefix)

        # RAW TEXT score
        sim_raw = F.cosine_similarity(
            F.normalize(fraud_txt, dim=1),
            F.normalize(real_txt, dim=1),
            dim=1
        ).cpu().numpy()

        # TEACHER score
        sim_teacher = F.cosine_similarity(
            F.normalize(fraud_teacher, dim=1),
            F.normalize(real_teacher, dim=1),
            dim=1
        ).cpu().numpy()

        # STUDENT score
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.model.eval()
        bs = int(self.cfg.batch_size)

        sims_student = []
        align_fraud = []
        align_real = []

        for start in range(0, len(df), bs):
            end = min(start + bs, len(df))

            f_txt = fraud_txt[start:end].to(device)
            r_txt = real_txt[start:end].to(device)

            f_t = fraud_teacher[start:end].to(device)
            r_t = real_teacher[start:end].to(device)

            p_f, p_r = self.model(f_txt, r_txt)

            p_f = F.normalize(p_f, dim=1)
            p_r = F.normalize(p_r, dim=1)
            f_t = F.normalize(f_t, dim=1)
            r_t = F.normalize(r_t, dim=1)

            sims_student.append(F.cosine_similarity(p_f, p_r, dim=1).cpu())

            align_fraud.append(F.cosine_similarity(p_f, f_t, dim=1).cpu())
            align_real.append(F.cosine_similarity(p_r, r_t, dim=1).cpu())

        sim_student = torch.cat(sims_student).numpy()
        cos_fraud_align = torch.cat(align_fraud).numpy()
        cos_real_align  = torch.cat(align_real).numpy()

        student_metrics = self._compute_metrics(y, sim_student, "STUDENT")
        raw_metrics     = self._compute_metrics(y, sim_raw, "RAW_TEXT")
        teacher_metrics = self._compute_metrics(y, sim_teacher, "TEACHER")

        results_df = pd.DataFrame({
            "label": y,
            "sim_student": sim_student,
            "sim_raw_text": sim_raw,
            "sim_teacher": sim_teacher,
            "cos_align_fraud": cos_fraud_align,
            "cos_align_real": cos_real_align,
        })

        metrics = {
            "student": student_metrics,
            "raw_text": raw_metrics,
            "teacher": teacher_metrics,
            "alignment_debug": {
                "mean_cos_fraud_to_teacher": float(np.mean(cos_fraud_align)),
                "mean_cos_real_to_teacher": float(np.mean(cos_real_align)),
                "std_cos_fraud_to_teacher": float(np.std(cos_fraud_align)),
                "std_cos_real_to_teacher": float(np.std(cos_real_align)),
            },
            "deltas": {
                "auc_student_minus_raw": float(student_metrics["roc_auc"] - raw_metrics["roc_auc"]),
                "auc_teacher_minus_student": float(teacher_metrics["roc_auc"] - student_metrics["roc_auc"]),
            },
        }

        return results_df, metrics
