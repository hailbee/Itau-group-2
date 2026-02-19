#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as npx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from siamese import SiameseEmbeddingModel


# -------------------------
# IO / device
# -------------------------
def pick_device(override: Optional[str]) -> torch.device:
    if override:
        d = torch.device(override)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is not available.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


# -------------------------
# Column helpers (stable numeric ordering)
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str) -> int:
        suf = c[len(prefix) :]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def mat_from_prefix(df: pd.DataFrame, prefix: str) -> npx.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=npx.float32, copy=True)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix for prefix {prefix!r}, got shape {mat.shape}.")
    return mat


# -------------------------
# Checkpoint loading
# -------------------------
def evaluator2_style_state(ckpt):
    return ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt


def as_state_dict(state_obj) -> Dict[str, torch.Tensor]:
    if isinstance(state_obj, nn.Module):
        return state_obj.state_dict()
    if isinstance(state_obj, dict) and state_obj and all(isinstance(v, torch.Tensor) for v in state_obj.values()):
        return state_obj
    raise RuntimeError(f"Checkpoint state is not a state_dict or nn.Module. Type={type(state_obj)}")


def infer_hidden_dim_from_state(sd: Dict[str, torch.Tensor], text_dim: int) -> int:
    candidates: List[int] = []
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[1]) == int(text_dim):
            candidates.append(int(v.shape[0]))
    if not candidates:
        raise RuntimeError("Could not infer hidden_dim from checkpoint. Provide --hidden-dim explicitly.")
    # Most common; tie-break by larger
    counts: Dict[int, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    return int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0])


def has_both_classes(y: npx.ndarray) -> bool:
    if y.size == 0:
        return False
    u = npx.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


# -------------------------
# Threshold selection
# -------------------------
def youden_threshold(y_true: npx.ndarray, scores: npx.ndarray) -> Tuple[float, float]:
    """Maximize Youden's J = TPR - FPR for pred=(scores>=thr). Returns (thr, best_J)."""
    if not has_both_classes(y_true):
        raise ValueError("Cannot pick Youden threshold: evaluation set is one-class.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(npx.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


def best_accuracy_threshold(y_true: npx.ndarray, scores: npx.ndarray) -> Tuple[float, float]:
    """Maximize accuracy for pred=(scores>=thr). Returns (thr, best_accuracy)."""
    if not has_both_classes(y_true):
        raise ValueError("Cannot pick accuracy-max threshold: evaluation set is one-class.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    P = float(y_true.sum())
    N_ = float(len(y_true) - y_true.sum())
    accs = (tpr * P + (1.0 - fpr) * N_) / (P + N_ + 1e-12)
    best_idx = int(npx.argmax(accs))
    return float(thresholds[best_idx]), float(accs[best_idx])


# -------------------------
# Metrics at a threshold
# -------------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else float("nan")


def confusion_counts(y_true: npx.ndarray, y_pred: npx.ndarray) -> Tuple[int, int, int, int]:
    tp = int(npx.sum((y_true == 1) & (y_pred == 1)))
    fp = int(npx.sum((y_true == 0) & (y_pred == 1)))
    tn = int(npx.sum((y_true == 0) & (y_pred == 0)))
    fn = int(npx.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def print_metrics(title: str, thr: float, y_true: npx.ndarray, scores: npx.ndarray) -> float:
    """
    Prints accuracy/precision/recall/F1 plus a few extras.
    Returns accuracy (for easy comparison).
    """
    y_pred = (scores >= thr).astype(npx.int32)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # TPR
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if npx.isfinite(precision) and npx.isfinite(recall) else float("nan")
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)

    fpr = _safe_div(fp, fp + tn)
    specificity = _safe_div(tn, tn + fp)
    balanced_acc = _safe_div(recall + specificity, 2.0) if npx.isfinite(recall) and npx.isfinite(specificity) else float("nan")
    alert_rate = float(npx.mean(y_pred)) if y_pred.size > 0 else float("nan")

    print("==============================")
    print(f"METRICS @ THRESHOLD: {title}")
    print("==============================")
    print(f"[THR  ] {thr:.6f}")
    print(f"[CM   ] TP={tp:,} | FP={fp:,} | TN={tn:,} | FN={fn:,}")
    print(f"[BASIC] Accuracy={accuracy:.6f} | Precision={precision:.6f} | Recall={recall:.6f} | F1={f1:.6f}")
    print(f"[RATE ] FPR={fpr:.6f} | Specificity={specificity:.6f} | BalancedAcc={balanced_acc:.6f}")
    print(f"[OPS  ] AlertRate(pred==1)={alert_rate:.6f}")
    print("==============================\n")

    return float(accuracy)


# -------------------------
# Names
# -------------------------
def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().casefold()
    return s


# -------------------------
# Core: max cosine excluding same-name candidates (memory-safe)
# -------------------------
@torch.inference_mode()
def max_cos_excluding_name(
    zq: torch.Tensor,  # (N, D) normalized
    zk: torch.Tensor,  # (M, D) normalized
    key_name_ids: torch.Tensor,  # (M,) int64 group id per key
    query_name_ids: torch.Tensor,  # (N,) int64 group id per query (-1 if not in bank)
    query_batch_size: int,
    key_chunk_size: int,
) -> npx.ndarray:
    device = zq.device
    N = int(zq.shape[0])
    M = int(zk.shape[0])

    qbs = max(1, int(query_batch_size))
    kcs = max(1, int(key_chunk_size))

    max_vals_all = torch.full((N,), -1e9, device=device, dtype=torch.float32)

    for q0 in range(0, N, qbs):
        q1 = min(q0 + qbs, N)
        q = zq[q0:q1]
        q_ids = query_name_ids[q0:q1]

        best = torch.full((q1 - q0,), -1e9, device=device)

        for k0 in range(0, M, kcs):
            k1 = min(k0 + kcs, M)
            k = zk[k0:k1]
            k_ids = key_name_ids[k0:k1].view(1, -1)

            scores = q @ k.T
            same = (q_ids.view(-1, 1) == k_ids)
            scores = scores.masked_fill(same, -1e9)

            vals, _ = torch.max(scores, dim=1)
            best = torch.maximum(best, vals)

        max_vals_all[q0:q1] = best

    return max_vals_all.detach().cpu().numpy().astype(npx.float32, copy=False)


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Inference + evaluation (prints only metrics):\n"
            "  - ROC AUC\n"
            "  - Youden threshold vs Max-Accuracy threshold\n"
            "  - Accuracy, Precision, Recall, F1 (and a few basics)\n"
            "Eval mask is Option B: exclude in-bank positives only.\n"
        )
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--student-model-path", required=True)

    ap.add_argument("--label-col", default="label")  # 1=spoof
    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--out-dim", type=int, default=768)
    ap.add_argument("--device", default=None)

    ap.add_argument("--dedup-real-names", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--query-batch-size", type=int, default=2048)
    ap.add_argument("--real-chunk-size", type=int, default=20000)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    df = read_table(args.data)
    for col in [args.label_col, args.fraud_name_col, args.real_name_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")

    y_raw = df[args.label_col].to_numpy()
    y = (y_raw.astype(npx.float32) >= 0.5).astype(npx.int32)
    n = int(len(df))
    print(f"[INFO] n={n:,} | spoof_rate(label=1)={float(y.mean()):.6f}")

    fraud_mat = mat_from_prefix(df, args.fraud_prefix)
    real_mat = mat_from_prefix(df, args.real_prefix)
    text_dim = int(fraud_mat.shape[1])
    if int(real_mat.shape[1]) != text_dim:
        raise ValueError("fraud and real embedding dims do not match")

    # ---- Build bank (optionally dedup) ----
    real_names_full = df[args.real_name_col].astype(str).to_numpy()
    if args.dedup_real_names:
        seen = set()
        keep_idx: List[int] = []
        keep_names: List[str] = []
        for i, nm in enumerate(real_names_full):
            k = normalize_name(nm)
            if k not in seen:
                seen.add(k)
                keep_idx.append(i)
                keep_names.append(nm)
        keep_idx_np = npx.array(keep_idx, dtype=npx.int64)
        real_bank = real_mat[keep_idx_np]
        real_names_bank = npx.array(keep_names, dtype=object)
        print(f"[INFO] dedup_real_names: {len(real_names_full):,} -> {len(real_bank):,} candidates")
    else:
        real_bank = real_mat
        real_names_bank = real_names_full

    # ---- Map bank name -> id ----
    name_to_id: Dict[str, int] = {}
    key_ids_list: List[int] = []
    for nm in real_names_bank.tolist():
        k = normalize_name(nm)
        if k not in name_to_id:
            name_to_id[k] = len(name_to_id)
        key_ids_list.append(name_to_id[k])
    key_name_ids_np = npx.array(key_ids_list, dtype=npx.int64)

    fraud_names = df[args.fraud_name_col].astype(str).to_numpy()
    query_name_ids_np = npx.array([name_to_id.get(normalize_name(nm), -1) for nm in fraud_names.tolist()], dtype=npx.int64)
    exact_in_bank = (query_name_ids_np != -1)

    # ---- Load model ----
    ckpt = torch.load(args.student_model_path, map_location=device)
    sd = as_state_dict(evaluator2_style_state(ckpt))

    inferred_hidden = infer_hidden_dim_from_state(sd, text_dim=text_dim)
    hidden_dim = inferred_hidden if args.hidden_dim is None else int(args.hidden_dim)
    if args.hidden_dim is not None and hidden_dim != int(inferred_hidden):
        raise ValueError(f"hidden_dim mismatch: ckpt implies {inferred_hidden}, but --hidden-dim={hidden_dim}")

    model = SiameseEmbeddingModel(text_dim=text_dim, hidden_dim=hidden_dim, image_dim=int(args.out_dim)).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ---- Encode bank + queries ----
    bs = max(1, int(args.batch_size))

    real_bank_t = torch.from_numpy(real_bank).to(device)
    zr = torch.empty((real_bank_t.shape[0], int(args.out_dim)), device=device, dtype=torch.float32)
    for i0 in range(0, real_bank_t.shape[0], bs):
        i1 = min(i0 + bs, real_bank_t.shape[0])
        z = model.encode_text(real_bank_t[i0:i1])
        zr[i0:i1] = F.normalize(z, dim=1)

    fraud_t = torch.from_numpy(fraud_mat).to(device)
    zf = torch.empty((n, int(args.out_dim)), device=device, dtype=torch.float32)
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        z = model.encode_text(fraud_t[i0:i1])
        zf[i0:i1] = F.normalize(z, dim=1)

    # ---- Score = max cosine excluding same-name ----
    key_name_ids = torch.from_numpy(key_name_ids_np).to(device)
    query_name_ids = torch.from_numpy(query_name_ids_np).to(device)

    score = max_cos_excluding_name(
        zq=zf,
        zk=zr,
        key_name_ids=key_name_ids,
        query_name_ids=query_name_ids,
        query_batch_size=int(args.query_batch_size),
        key_chunk_size=int(args.real_chunk_size),
    )
    score = npx.clip(score, -1.0, 1.0)

    # ---- Option B eval mask: exclude in-bank positives only ----
    eval_mask = ~(exact_in_bank & (y == 1))
    y_eval = y[eval_mask]
    s_eval = score[eval_mask]

    print("\n==============================")
    print("EVALUATION SET (OPTION B)")
    print("==============================")
    print(f"[EVAL] n={int(eval_mask.sum()):,} | pos={int(y_eval.sum()):,} | neg={int(len(y_eval) - y_eval.sum()):,}")
    print("==============================\n")

    # ---- ROC AUC + AUPRC ----
    print("==============================")
    print("THRESHOLD-FREE METRICS")
    print("==============================")
    if not has_both_classes(y_eval):
        print("[AUC] AUROC/AUPRC undefined (one-class eval set).")
        return
    auroc = float(roc_auc_score(y_eval, s_eval))
    auprc = float(average_precision_score(y_eval, s_eval))
    print(f"[AUC] ROC AUC (AUROC)={auroc:.6f}")
    print(f"[AUC] PR  AUC (AUPRC/AP)={auprc:.6f}")
    print("==============================\n")

    # ---- thresholds ----
    thr_y, best_j = youden_threshold(y_eval, s_eval)
    thr_a, best_acc = best_accuracy_threshold(y_eval, s_eval)

    print("==============================")
    print("THRESHOLDS")
    print("==============================")
    print(f"[THR] Youden:       thr={thr_y:.6f} | best_J={best_j:.6f}")
    print(f"[THR] Max-Accuracy: thr={thr_a:.6f} | best_acc={best_acc:.6f}")
    print("==============================\n")

    # ---- metrics at both thresholds ----
    acc_y = print_metrics("Youden threshold", thr_y, y_eval, s_eval)
    acc_a = print_metrics("Max-Accuracy threshold", thr_a, y_eval, s_eval)

    print("==============================")
    print("ACCURACY COMPARISON")
    print("==============================")
    print(f"[ACC] Youden accuracy      = {acc_y:.6f}")
    print(f"[ACC] Max-accuracy accuracy = {acc_a:.6f}")
    print("==============================\n")


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLE

python inference/inference.py \
  --data ../Test/deja_test_pairs_with_img_and_vate_txt_embs.parquet \
  --student-model-path saved_models/deja_best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix real_txt_emb_ \
  --dedup-real-names
"""
