from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _sorted_prefix_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'.")

    # Sort by numeric suffix if possible
    def key_fn(c: str):
        tail = c[len(prefix):]
        try:
            return int(tail)
        except Exception:
            return tail

    return sorted(cols, key=key_fn)


@dataclass
class EvalConfig:
    batch_size: int = 256
    device: Optional[str] = None

    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"

    # teacher (spoof-aware image) targets
    fraud_teacher_prefix: str = "fraud_img_emb_"
    real_teacher_prefix: str = "real_img_emb_"

    label_col: str = "label"


class Evaluator2:
    """
    Evaluator for the text->(spoof-aware image space) student.

    Expected model behavior:
      model(x_fraud, x_real) -> (z_fraud, z_real)

    Metrics:
      - Alignment: cosine(student, teacher) for fraud/real + mean
      - Pair AUC: cosine(student_fraud, student_real) vs label (if label exists)
    """

    def __init__(self, model: torch.nn.Module, cfg: EvalConfig):
        self.model = model
        self.cfg = cfg

        if cfg.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_file(
        self,
        data_path: str,
        save_metrics_path: Optional[str] = None,
        return_per_row: bool = False,
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        df = _load_table(data_path)
        metrics, per_row = self.evaluate_df(df, return_per_row=return_per_row)

        if save_metrics_path is not None:
            os.makedirs(os.path.dirname(save_metrics_path) or ".", exist_ok=True)
            with open(save_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        return metrics, per_row

    @torch.no_grad()
    def evaluate_df(
        self,
        df: pd.DataFrame,
        return_per_row: bool = False,
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        cfg = self.cfg

        fraud_txt_cols = _sorted_prefix_cols(df, cfg.fraud_txt_prefix)
        real_txt_cols = _sorted_prefix_cols(df, cfg.real_txt_prefix)
        fraud_t_cols = _sorted_prefix_cols(df, cfg.fraud_teacher_prefix)
        real_t_cols = _sorted_prefix_cols(df, cfg.real_teacher_prefix)

        # Optional labels
        has_label = cfg.label_col in df.columns
        labels = df[cfg.label_col].to_numpy(np.float32, copy=False) if has_label else None

        fraud_txt = df[fraud_txt_cols].to_numpy(np.float32, copy=False)
        real_txt = df[real_txt_cols].to_numpy(np.float32, copy=False)
        fraud_teacher = df[fraud_t_cols].to_numpy(np.float32, copy=False)
        real_teacher = df[real_t_cols].to_numpy(np.float32, copy=False)

        n = fraud_txt.shape[0]
        bs = int(cfg.batch_size)

        align_fraud_all = []
        align_real_all = []
        pair_score_all = []
        labels_all = []  # only if exists

        # Optional per-row output
        per_row_out = [] if return_per_row else None

        for start in range(0, n, bs):
            end = min(start + bs, n)

            ft = torch.from_numpy(fraud_txt[start:end]).to(self.device)
            rt = torch.from_numpy(real_txt[start:end]).to(self.device)
            fT = torch.from_numpy(fraud_teacher[start:end]).to(self.device)
            rT = torch.from_numpy(real_teacher[start:end]).to(self.device)

            zf, zr = self.model(ft, rt)

            zf_n = F.normalize(zf, dim=-1)
            zr_n = F.normalize(zr, dim=-1)
            fT_n = F.normalize(fT, dim=-1)
            rT_n = F.normalize(rT, dim=-1)

            # alignment
            align_f = (zf_n * fT_n).sum(dim=-1)  # cosine
            align_r = (zr_n * rT_n).sum(dim=-1)

            # pair score (student fraud vs student real) for ROC/AUC
            pair_score = (zf_n * zr_n).sum(dim=-1)

            align_fraud_all.append(align_f.detach().cpu().numpy())
            align_real_all.append(align_r.detach().cpu().numpy())
            pair_score_all.append(pair_score.detach().cpu().numpy())

            if labels is not None:
                labels_all.append(labels[start:end])

            if return_per_row:
                # store per-row computed values
                for i in range(end - start):
                    per_row_out.append(
                        {
                            "row_index": start + i,
                            "align_fraud": float(align_f[i].item()),
                            "align_real": float(align_r[i].item()),
                            "pair_score": float(pair_score[i].item()),
                            "label": float(labels[start + i]) if labels is not None else None,
                        }
                    )

        align_fraud = np.concatenate(align_fraud_all, axis=0)
        align_real = np.concatenate(align_real_all, axis=0)
        pair_scores = np.concatenate(pair_score_all, axis=0)

        metrics: Dict[str, float] = {}
        metrics["mean_align_fraud"] = float(np.mean(align_fraud))
        metrics["mean_align_real"] = float(np.mean(align_real))
        metrics["mean_alignment_cosine"] = float(np.mean(0.5 * (align_fraud + align_real)))

        # label-based metrics (optional)
        if labels is not None and len(labels_all) > 0 and np.any(np.isfinite(labels)):
            y = np.concatenate(labels_all, axis=0)

            # Only attempt AUC if labels look like binary
            try:
                auc = roc_auc_score(y, pair_scores)
                metrics["pair_roc_auc"] = float(auc)

                fpr, tpr, thr = roc_curve(y, pair_scores)
                j = tpr - fpr
                best_i = int(np.argmax(j))
                best_thr = float(thr[best_i])
                preds = (pair_scores >= best_thr).astype(np.int32)
                acc = accuracy_score(y, preds)

                metrics["youden_threshold"] = best_thr
                metrics["youden_accuracy"] = float(acc)
            except Exception as e:
                # Keep evaluation resilient
                metrics["pair_roc_auc_error"] = float("nan")
                metrics["pair_roc_auc_error_msg"] = str(e)

        per_row_df = pd.DataFrame(per_row_out) if return_per_row else None
        return metrics, per_row_df
