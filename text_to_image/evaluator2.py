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


def _np_mat(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Return float32 numpy matrix for columns matching prefix."""
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=np.float32, copy=False)
    if mat.ndim != 2:
        raise ValueError(f"Prefix '{prefix}' did not produce a 2D matrix")
    return mat


def _pick_label_col(df: pd.DataFrame, preferred: str) -> str:
    """Choose a label column with a graceful fallback."""
    if preferred in df.columns:
        return preferred
    for alt in ("spoof_attempt", "label", "y"):
        if alt in df.columns:
            print(f"[WARN] label_col '{preferred}' not found; using '{alt}' instead.")
            return alt
    raise KeyError(f"Could not find label column. Tried '{preferred}', 'spoof_attempt', 'label', 'y'.")


@dataclass
class EvalConfig:
    batch_size: int = 2048

    # text-text eval input prefixes
    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"

    # optional teacher/debug prefixes (if present)
    fraud_teacher_prefix: str = "fraud_aligned_"
    real_teacher_prefix: str = "real_aligned_"

    # labels + optional names
    label_col: str = "label"
    fraud_name_col: str = "fraudulent_name"
    real_name_col: str = "real_name"

    # toggles
    compute_raw_text: bool = True
    compute_teacher: bool = True


class Evaluator2:
    """
    NEW TASK EVALUATOR:

    Primary metric (what you described):
      STUDENT: cosine( normalize(Pf(fraud_txt)), normalize(Pr(real_txt)) )
    ROC AUC computed on that cosine score.

    Optional baselines (if enabled and columns exist):
      RAW_TEXT: cosine(fraud_txt, real_txt)
      TEACHER:  cosine(fraud_teacher, real_teacher)

    Returns:
      (results_df, metrics_dict)
    """

    def __init__(self, model, cfg: Optional[EvalConfig] = None):
        self.model = model
        self.cfg = cfg or EvalConfig()

    def _compute_metrics(self, y_true, y_scores, tag: str) -> Dict[str, Any]:
        y_true = np.asarray(y_true).astype(np.int32, copy=False)
        y_scores = np.asarray(y_scores, dtype=np.float64)

        # If all labels are the same, roc_auc_score will error; guard it
        if np.unique(y_true).size < 2:
            print(f"[{tag}] Only one class present in y_true; ROC AUC undefined.")
            return {
                "roc_auc": float("nan"),
                "youden_threshold": float("nan"),
                "accuracy_youden": float("nan"),
                "confusion_matrix_youden": None,
            }

        roc_auc = float(roc_auc_score(y_true, y_scores))

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

        # ---- load ----
        df = pd.read_parquet(filepath) if filepath.endswith(".parquet") else pd.read_csv(filepath)
        if max_rows is not None:
            df = df.head(int(max_rows)).copy()

        # ---- labels ----
        label_col = _pick_label_col(df, self.cfg.label_col)
        y = df[label_col].astype(int).to_numpy()

        # ---- pull arrays (numpy) ----
        fraud_txt_np = _np_mat(df, self.cfg.fraud_txt_prefix)
        real_txt_np = _np_mat(df, self.cfg.real_txt_prefix)

        # Optional teacher arrays if present
        have_teacher = (
            self.cfg.compute_teacher
            and any(isinstance(c, str) and c.startswith(self.cfg.fraud_teacher_prefix) for c in df.columns)
            and any(isinstance(c, str) and c.startswith(self.cfg.real_teacher_prefix) for c in df.columns)
        )
        fraud_teacher_np = real_teacher_np = None
        if have_teacher:
            try:
                fraud_teacher_np = _np_mat(df, self.cfg.fraud_teacher_prefix)
                real_teacher_np = _np_mat(df, self.cfg.real_teacher_prefix)
            except Exception as e:
                print(f"[WARN] teacher columns requested but failed to load: {e}")
                have_teacher = False

        # ---- device ----
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.model.eval()
        bs = int(self.cfg.batch_size)

        # ---- STUDENT cosine (main metric) ----
        sims_student: List[torch.Tensor] = []
        for start in range(0, len(df), bs):
            end = min(start + bs, len(df))

            f_txt = torch.from_numpy(fraud_txt_np[start:end]).to(device)
            r_txt = torch.from_numpy(real_txt_np[start:end]).to(device)

            # Your SiameseEmbeddingModel.forward returns (z_fraud, z_real)
            z_f, z_r = self.model(f_txt, r_txt)

            z_f = F.normalize(z_f, dim=1)
            z_r = F.normalize(z_r, dim=1)

            sims_student.append(F.cosine_similarity(z_f, z_r, dim=1).detach().cpu())

        sim_student = torch.cat(sims_student).numpy()

        # ---- RAW TEXT cosine (baseline) ----
        sim_raw = None
        if self.cfg.compute_raw_text:
            ft = torch.from_numpy(fraud_txt_np)
            rt = torch.from_numpy(real_txt_np)
            sim_raw = F.cosine_similarity(F.normalize(ft, dim=1), F.normalize(rt, dim=1), dim=1).numpy()

        # ---- TEACHER cosine (debug baseline) ----
        sim_teacher = None
        if have_teacher and fraud_teacher_np is not None and real_teacher_np is not None:
            ft = torch.from_numpy(fraud_teacher_np)
            rt = torch.from_numpy(real_teacher_np)
            sim_teacher = F.cosine_similarity(F.normalize(ft, dim=1), F.normalize(rt, dim=1), dim=1).numpy()

        # ---- metrics ----
        student_metrics = self._compute_metrics(y, sim_student, "STUDENT")
        metrics: Dict[str, Any] = {"student": student_metrics}

        if sim_raw is not None:
            raw_metrics = self._compute_metrics(y, sim_raw, "RAW_TEXT")
            metrics["raw_text"] = raw_metrics
            metrics.setdefault("deltas", {})
            metrics["deltas"]["auc_student_minus_raw"] = float(student_metrics["roc_auc"] - raw_metrics["roc_auc"])

        if sim_teacher is not None:
            teacher_metrics = self._compute_metrics(y, sim_teacher, "TEACHER")
            metrics["teacher"] = teacher_metrics
            metrics.setdefault("deltas", {})
            metrics["deltas"]["auc_teacher_minus_student"] = float(teacher_metrics["roc_auc"] - student_metrics["roc_auc"])

        # ---- results df ----
        out = {
            "label": y,
            "sim_student": sim_student,
        }
        if sim_raw is not None:
            out["sim_raw_text"] = sim_raw
        if sim_teacher is not None:
            out["sim_teacher"] = sim_teacher

        # Optional names if present
        if self.cfg.fraud_name_col in df.columns:
            out[self.cfg.fraud_name_col] = df[self.cfg.fraud_name_col].astype(str).to_numpy()
        if self.cfg.real_name_col in df.columns:
            out[self.cfg.real_name_col] = df[self.cfg.real_name_col].astype(str).to_numpy()

        results_df = pd.DataFrame(out)
        return results_df, metrics
