#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

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
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


# -------------------------
# Column helpers (evaluator2-like sorting)
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix) :]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def mat_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=np.float32, copy=True)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix for prefix {prefix!r}, got shape {mat.shape}.")
    return mat


# -------------------------
# Checkpoint loading (match evaluator2)
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
    candidates = []
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[1]) == int(text_dim):
            candidates.append(int(v.shape[0]))
    if not candidates:
        raise RuntimeError("Could not infer hidden_dim from checkpoint. Provide --hidden-dim explicitly.")
    return max(set(candidates), key=candidates.count)


def has_both_classes(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


# -------------------------
# Threshold rules (picked on the same eval set, like your old evaluator.py)
# -------------------------
def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Choose thr maximizing Youden's J = TPR - FPR for pred=(scores>=thr).
    Returns (thr, best_J).
    """
    if not has_both_classes(y_true):
        raise ValueError("Cannot pick Youden threshold: evaluation set is one-class.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


def best_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Choose thr maximizing accuracy for pred=(scores>=thr).
    Returns (thr, best_accuracy).
    """
    if not has_both_classes(y_true):
        raise ValueError("Cannot pick accuracy-max threshold: evaluation set is one-class.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    P = float(y_true.sum())
    N = float(len(y_true) - y_true.sum())
    accs = (tpr * P + (1.0 - fpr) * N) / (P + N + 1e-12)
    best_idx = int(np.argmax(accs))
    return float(thresholds[best_idx]), float(accs[best_idx])


# -------------------------
# Metrics helpers
# -------------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return _safe_div(num, float(np.sqrt(den))) if den > 0 else float("nan")


def report_threshold_metrics(title: str, thr: float, y_true: np.ndarray, scores: np.ndarray) -> None:
    y_pred = (scores >= thr).astype(np.int32)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    tpr = _safe_div(tp, tp + fn)  # recall
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)  # specificity
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2.0 * prec * tpr, prec + tpr) if np.isfinite(prec) and np.isfinite(tpr) else float("nan")
    acc = _safe_div(tp + tn, tp + fp + tn + fn)
    bacc = _safe_div(tpr + tnr, 2.0) if np.isfinite(tpr) and np.isfinite(tnr) else float("nan")
    _mcc = mcc(tp, fp, tn, fn)
    alert = float(np.mean(y_pred)) if y_pred.size > 0 else float("nan")

    print("==============================")
    print(f"THRESHOLD METRICS (EVAL-SET): {title}")
    print("==============================")
    print(f"[THR] threshold={thr:.6f}")
    print(f"[CM ] TP={tp:,} | FP={fp:,} | TN={tn:,} | FN={fn:,}")
    print(f"[RATE] TPR/Recall={tpr:.6f} | FPR={fpr:.6f} | TNR/Spec={tnr:.6f}")
    print(f"[RATE] Precision={prec:.6f} | F1={f1:.6f}")
    print(f"[SUM] Accuracy={acc:.6f} | BalancedAcc={bacc:.6f} | MCC={_mcc:.6f}")
    print(f"[OPS] AlertRate(pred==1)={alert:.6f}")
    print("==============================\n")


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
) -> Tuple[np.ndarray, np.ndarray]:
    device = zq.device
    N = int(zq.shape[0])
    M = int(zk.shape[0])

    qbs = max(1, int(query_batch_size))
    kcs = max(1, int(key_chunk_size))

    max_vals_all = torch.full((N,), -1e9, device=device, dtype=torch.float32)
    max_idxs_all = torch.full((N,), -1, device=device, dtype=torch.long)

    for q0 in range(0, N, qbs):
        q1 = min(q0 + qbs, N)
        q = zq[q0:q1]
        q_ids = query_name_ids[q0:q1]

        best = torch.full((q1 - q0,), -1e9, device=device)
        best_idx = torch.full((q1 - q0,), -1, device=device, dtype=torch.long)

        for k0 in range(0, M, kcs):
            k1 = min(k0 + kcs, M)
            k = zk[k0:k1]
            k_ids = key_name_ids[k0:k1].view(1, -1)

            scores = q @ k.T
            same = (q_ids.view(-1, 1) == k_ids)
            scores = scores.masked_fill(same, -1e9)

            vals, idxs = torch.max(scores, dim=1)
            better = vals > best
            best = torch.where(better, vals, best)
            best_idx = torch.where(better, idxs + k0, best_idx)

        max_vals_all[q0:q1] = best
        max_idxs_all[q0:q1] = best_idx

    return (
        max_vals_all.detach().cpu().numpy().astype(np.float32, copy=False),
        max_idxs_all.detach().cpu().numpy(),
    )


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Bank spoof inference with Option B evaluation (NO calibration split):\n"
            "  - Simulate open-set negatives by evaluating on open_mask = ~(exact_in_bank & y==1).\n"
            "  - Compute AUROC/AUPRC on that eval set.\n"
            "  - Pick Youden and Max-Accuracy thresholds on that SAME eval set (like evaluator.py).\n"
            "  - Report confusion-matrix-derived metrics at each threshold.\n"
        )
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default=None, help="Unused; kept for compatibility.")

    ap.add_argument("--label-col", default="label")  # 1=spoof
    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

    ap.add_argument("--student-model-path", required=True)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--out-dim", type=int, default=768)
    ap.add_argument("--device", default=None)

    ap.add_argument("--dedup-real-names", action="store_true")

    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--query-batch-size", type=int, default=2048)
    ap.add_argument("--real-chunk-size", type=int, default=20000)

    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If provided, also report metrics at this threshold (on the same eval set).",
    )

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    df = read_table(args.data)
    for col in [args.label_col, args.fraud_name_col, args.real_name_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")

    y_raw = df[args.label_col].to_numpy()
    y = (y_raw.astype(np.float32) >= 0.5).astype(np.int32)
    n = len(df)
    print(f"[INFO] n={n:,} | spoof_rate(label=1)={float(y.mean()):.6f}")

    fraud_np = mat_from_prefix(df, args.fraud_prefix)
    real_np = mat_from_prefix(df, args.real_prefix)
    text_dim = int(fraud_np.shape[1])
    if int(real_np.shape[1]) != text_dim:
        raise ValueError("fraud and real embedding dims do not match")

    # -------------------------
    # Build bank (optionally dedup)
    # -------------------------
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
        keep_idx_np = np.array(keep_idx, dtype=np.int64)
        real_np_bank = real_np[keep_idx_np]
        real_names_bank = np.array(keep_names, dtype=object)
        print(f"[INFO] dedup_real_names: {len(real_names_full):,} -> {len(real_np_bank):,} candidates")
    else:
        real_np_bank = real_np
        real_names_bank = real_names_full

    # Map normalized bank name -> id
    name_to_id: Dict[str, int] = {}
    key_ids_list: List[int] = []
    for nm in real_names_bank.tolist():
        k = normalize_name(nm)
        if k not in name_to_id:
            name_to_id[k] = len(name_to_id)
        key_ids_list.append(name_to_id[k])
    key_name_ids_np = np.array(key_ids_list, dtype=np.int64)

    fraud_names = df[args.fraud_name_col].astype(str).to_numpy()
    query_name_ids_np = np.array(
        [name_to_id.get(normalize_name(nm), -1) for nm in fraud_names.tolist()],
        dtype=np.int64,
    )
    exact_in_bank = (query_name_ids_np != -1)
    print(f"[INFO] exact_in_bank_rate={float(exact_in_bank.mean()):.6f}")

    layup = (y == 0) & exact_in_bank
    print(f"[INFO] layup_rate(label0 & exact)= {float(layup.mean()):.6f}")

    # -------------------------
    # Load model
    # -------------------------
    ckpt = torch.load(args.student_model_path, map_location=device)
    sd = as_state_dict(evaluator2_style_state(ckpt))
    inferred_hidden = infer_hidden_dim_from_state(sd, text_dim=text_dim)
    hidden_dim = inferred_hidden if args.hidden_dim is None else int(args.hidden_dim)
    if args.hidden_dim is not None and hidden_dim != int(inferred_hidden):
        raise ValueError(f"hidden_dim mismatch: ckpt implies {inferred_hidden}, but --hidden-dim={hidden_dim}")

    model = SiameseEmbeddingModel(text_dim=text_dim, hidden_dim=hidden_dim, image_dim=int(args.out_dim)).to(device)
    load_info = model.load_state_dict(sd, strict=False)
    model.eval()
    print(
        f"[INFO] load_state_dict: missing_keys={len(getattr(load_info,'missing_keys',[]))} "
        f"unexpected_keys={len(getattr(load_info,'unexpected_keys',[]))}"
    )

    # -------------------------
    # Encode bank + queries through student
    # -------------------------
    bs = max(1, int(args.batch_size))

    real_bank_t = torch.from_numpy(real_np_bank).to(device)
    zr = torch.empty((real_bank_t.shape[0], int(args.out_dim)), device=device, dtype=torch.float32)
    for i0 in range(0, real_bank_t.shape[0], bs):
        i1 = min(i0 + bs, real_bank_t.shape[0])
        z = model.encode_text(real_bank_t[i0:i1])
        zr[i0:i1] = F.normalize(z, dim=1)

    fraud_t = torch.from_numpy(fraud_np).to(device)
    zf = torch.empty((n, int(args.out_dim)), device=device, dtype=torch.float32)
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        z = model.encode_text(fraud_t[i0:i1])
        zf[i0:i1] = F.normalize(z, dim=1)

    key_name_ids = torch.from_numpy(key_name_ids_np).to(device)
    query_name_ids = torch.from_numpy(query_name_ids_np).to(device)

    score, _ = max_cos_excluding_name(
        zq=zf,
        zk=zr,
        key_name_ids=key_name_ids,
        query_name_ids=query_name_ids,
        query_batch_size=int(args.query_batch_size),
        key_chunk_size=int(args.real_chunk_size),
    )
    score = np.clip(score, -1.0, 1.0)

    # -------------------------
    # Option B eval mask (no split):
    #   Evaluate only on open_mask = ~(exact_in_bank & y==1)
    #   i.e., exclude in-bank positives; keep all negatives + hard positives.
    # -------------------------
    eval_exact_in_bank = exact_in_bank & (y == 1)
    open_mask = ~eval_exact_in_bank

    y_eval = y[open_mask]
    s_eval = score[open_mask]

    open_n = int(open_mask.sum())
    open_pos = int(y_eval.sum())
    open_neg = int(open_n - open_pos)
    excl_inbank_pos = int(np.sum((y == 1) & exact_in_bank))

    print("\n==============================")
    print("OPTION B EVALUATION SET (NO CALIB SPLIT)")
    print("==============================")
    print("eval_exact_in_bank = exact_in_bank & (y==1)")
    print("open_mask = ~eval_exact_in_bank")
    print(f"[EVAL] n={open_n:,} | pos(y=1)={open_pos:,} | neg(y=0)={open_neg:,}")
    print(f"[EVAL] excluded_in-bank_positives(y=1 & exact_in_bank)={excl_inbank_pos:,}")
    print("==============================\n")

    # -------------------------
    # Threshold-free metrics (on eval set)
    # -------------------------
    print("==============================")
    print("THRESHOLD-FREE METRICS (EVAL SET)")
    print("==============================")
    if has_both_classes(y_eval):
        auroc = float(roc_auc_score(y_eval, s_eval))
        auprc = float(average_precision_score(y_eval, s_eval))
        print(f"[AUC] AUROC={auroc:.6f}")
        print(f"[AUC] AUPRC(AP)={auprc:.6f}")
    else:
        print("[AUC] AUROC/AUPRC undefined (evaluation set is one-class).")
    print("==============================\n")

    # -------------------------
    # Thresholds on the same eval set (like evaluator.py)
    # -------------------------
    thresholds: List[Tuple[str, float]] = []
    if args.threshold is not None:
        thresholds.append(("user_provided", float(args.threshold)))

    if has_both_classes(y_eval):
        thr_y, best_j = youden_threshold(y_eval, s_eval)
        thr_a, best_acc = best_accuracy_threshold(y_eval, s_eval)
        thresholds.append(("youden_J_on_eval_set", thr_y))
        thresholds.append(("max_accuracy_on_eval_set", thr_a))

        print("==============================")
        print("THRESHOLDS (PICKED ON EVAL SET)")
        print("==============================")
        print(f"[THR] Youden:       thr={thr_y:.6f} | best_J={best_j:.6f}")
        print(f"[THR] Max-Accuracy: thr={thr_a:.6f} | best_acc={best_acc:.6f}")
        if args.threshold is not None:
            print(f"[THR] User-provided thr={float(args.threshold):.6f}")
        print("==============================\n")
    else:
        print("[WARN] Cannot compute Youden/max-accuracy thresholds: eval set is one-class.\n")

    if len(thresholds) == 0:
        print("[WARN] No thresholds available to evaluate (provide --threshold or ensure eval set has both classes).")
        return

    for name, thr in thresholds:
        report_threshold_metrics(name, thr, y_eval, s_eval)

    # -------------------------
    # Hard spoof slice (original definition)
    # -------------------------
    hard_spoof_mask = (y == 1) & (~exact_in_bank)
    hard_idx = np.where(hard_spoof_mask)[0]
    hard_n = int(len(hard_idx))

    print("==============================")
    print("HARD SPOOF SLICE (ORIGINAL DEFINITION)")
    print("==============================")
    print("Hard spoof set = (y==1) & (exact_in_bank==0)")
    print(f"[SLICE] hard_n={hard_n:,}")
    if hard_n > 0:
        print(f"[SLICE] mean(score)={float(score[hard_idx].mean()):.6f}")
        print(f"[SLICE] median(score)={float(np.median(score[hard_idx])):.6f}")
        for name, thr in thresholds:
            hard_recall = float(np.mean((score[hard_idx] >= thr).astype(np.float32)))
            print(f"[SLICE] hard_spoof_recall at {name} thr = {hard_recall:.6f}")
    else:
        print("[SLICE] No hard spoof samples present.")
    print("==============================\n")


if __name__ == "__main__":
    main()


"""
Example (no calibration split; thresholds picked on the eval set):

python inference/inference.py \
  --data text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --student-model-path saved_models/best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix real_txt_emb_ \
  --dedup-real-names

Optionally also evaluate a fixed threshold:

python inference/inference.py \
  --data ... \
  --student-model-path ... \
  --hidden-dim 1024 \
  --out-dim 768 \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix real_txt_emb_ \
  --dedup-real-names \
  --threshold 0.976462
"""
