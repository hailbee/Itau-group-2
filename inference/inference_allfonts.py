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
    return _safe_div(num, float(np.sqrt(float(den)))) if den > 0 else float("nan")


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
 
 
 
from sklearn.metrics import precision_recall_curve

# ============================================================
# NEW THRESHOLD METHODS
# ============================================================

def compute_two_thresholds(y, scores, method):
    fpr, tpr, roc_thr = roc_curve(y, scores)
    prec, rec, pr_thr = precision_recall_curve(y, scores)

    if method == "tpr_fpr":
        pos_idx = np.where(tpr >= 0.95)[0][0]
        T_pos = roc_thr[pos_idx]

        valid = np.where(fpr <= 0.05)[0]
        T_neg = roc_thr[valid[-1]]

        return T_pos, T_neg

    elif method == "fixed_fpr":
        neg_scores = scores[y == 0]
        T = np.percentile(neg_scores, 95)
        return T, T

    elif method == "fixed_precision":
        target_precision = 0.99
        valid = np.where(prec[:-1] >= target_precision)[0]
        thr = pr_thr[valid[0]] if len(valid) > 0 else pr_thr[0]
        return thr, thr

    elif method == "symmetric_percentile":
        pos_scores = scores[y == 1]
        neg_scores = scores[y == 0]
        T_pos = np.percentile(pos_scores, 5)
        T_neg = np.percentile(neg_scores, 95)
        return T_pos, T_neg

    else:
        raise ValueError("Unknown two-threshold method")


def compute_single_threshold(y, scores, method):
    fpr, tpr, roc_thr = roc_curve(y, scores)
    prec, rec, pr_thr = precision_recall_curve(y, scores)

    if method == "youden":
        j = tpr - fpr
        return roc_thr[np.argmax(j)]

    elif method == "max_accuracy":
        P = y.sum()
        N = len(y) - P
        acc = (tpr * P + (1 - fpr) * N) / (P + N)
        return roc_thr[np.argmax(acc)]

    elif method == "max_f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return pr_thr[np.argmax(f1[:-1])]

    elif method == "max_mcc":
        best_mcc = -1
        best_thr = roc_thr[0]
        for thr in roc_thr:
            y_pred = (scores >= thr)
            tp, fp, tn, fn = confusion_counts(y, y_pred)
            val = mcc(tp, fp, tn, fn)
            if val > best_mcc:
                best_mcc = val
                best_thr = thr
        return best_thr

    else:
        raise ValueError("Unknown single-threshold method")   

# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    # Font datasets
    ap.add_argument("--dejavusans-val-data", required=True)
    ap.add_argument("--dejavusans-test-data", required=True)
    ap.add_argument("--sourcecodepro-val-data", required=True)
    ap.add_argument("--sourcecodepro-test-data", required=True)
    ap.add_argument("--pacifico-val-data", required=True)
    ap.add_argument("--pacifico-test-data", required=True)

    # Model
    ap.add_argument("--student-model-path", required=True)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--out-dim", type=int, default=768)

    # Columns
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")
    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

    # Other options
    ap.add_argument("--dedup-real-names", action="store_true")
    ap.add_argument("--query-batch-size", type=int, default=2048)
    ap.add_argument("--real-chunk-size", type=int, default=20000)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")
    
    
    
    # --------------------------------------------------------
    # Infer text_dim from first available validation file
    # --------------------------------------------------------
    
    sample_path = None
    for p in [
        args.dejavusans_val_data,
        args.sourcecodepro_val_data,
        args.pacifico_val_data,
    ]:
        if p is not None:
            sample_path = p
            break
    
    if sample_path is None:
        raise ValueError("At least one validation dataset must be provided to infer text_dim.")
    
    sample_df = read_table(sample_path)
    
    fraud_np = mat_from_prefix(sample_df, args.fraud_prefix)
    real_np  = mat_from_prefix(sample_df, args.real_prefix)
    
    text_dim = int(fraud_np.shape[1])
    if int(real_np.shape[1]) != text_dim:
        raise ValueError("fraud and real embedding dims do not match")
    
    
    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Process each font
    # --------------------------------------------------------
    fonts = [
        ("dejavusans", args.dejavusans_val_data, args.dejavusans_test_data),
        ("sourcecodepro", args.sourcecodepro_val_data, args.sourcecodepro_test_data),
        ("pacifico", args.pacifico_val_data, args.pacifico_test_data),
    ]

    all_val = []
    all_test = []

    for name, val_path, test_path in fonts:
        print(f"\n====== FONT: {name} ======")
    
        val_df = read_table(val_path)
        test_df = read_table(test_path)
    
        # --------------------------------------------------------
        # Build REAL BANK for THIS FONT
        # --------------------------------------------------------
    
        real_np_bank = mat_from_prefix(test_df, args.real_prefix)
        real_names = test_df[args.real_name_col].values
    
        if args.dedup_real_names:
            seen = set()
            keep = []
            for i, n in enumerate(real_names):
                k = normalize_name(n)
                if k not in seen:
                    seen.add(k)
                    keep.append(i)
            real_np_bank = real_np_bank[keep]
            real_names = real_names[keep]
            print(f"[INFO] dedup_real_names enabled → {len(real_names)} unique names")
    
        # Build name_to_id mapping for THIS FONT
        name_to_id = {}
        ids = []
        for n in real_names:
            k = normalize_name(n)
            if k not in name_to_id:
                name_to_id[k] = len(name_to_id)
            ids.append(name_to_id[k])
    
        key_name_ids = torch.tensor(ids, device=device)
    
        # --------------------------------------------------------
        # Encode bank through student (batched)
        # --------------------------------------------------------
    
        bs = max(1, int(args.query_batch_size))
        real_bank_t = torch.from_numpy(real_np_bank).to(device)
    
        zr = torch.empty(
            (real_bank_t.shape[0], int(args.out_dim)),
            device=device,
            dtype=torch.float32
        )
    
        for i0 in range(0, real_bank_t.shape[0], bs):
            i1 = min(i0 + bs, real_bank_t.shape[0])
            z = model.encode_text(real_bank_t[i0:i1])
            zr[i0:i1] = F.normalize(z, dim=1)
    
        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------
    
        y_val = (val_df[args.label_col].values >= 0.5).astype(int)
        y_test = (test_df[args.label_col].values >= 0.5).astype(int)
    
        # --------------------------------------------------------
        # Encode FRAUD queries
        # --------------------------------------------------------
    
        fraud_val = mat_from_prefix(val_df, args.fraud_prefix)
        fraud_test = mat_from_prefix(test_df, args.fraud_prefix)
    
        zf_val = F.normalize(
            model.encode_text(torch.from_numpy(fraud_val).to(device)),
            dim=1
        )
    
        zf_test = F.normalize(
            model.encode_text(torch.from_numpy(fraud_test).to(device)),
            dim=1
        )
    
        # --------------------------------------------------------
        # Query name IDs (relative to THIS FONT bank)
        # --------------------------------------------------------
    
        query_val_ids = torch.tensor(
            [name_to_id.get(normalize_name(x), -1)
             for x in val_df[args.fraud_name_col]],
            device=device
        )
    
        query_test_ids = torch.tensor(
            [name_to_id.get(normalize_name(x), -1)
             for x in test_df[args.fraud_name_col]],
            device=device
        )
    
        # --------------------------------------------------------
        # Cosine (exclude same name)
        # --------------------------------------------------------
    
        cos_val, _ = max_cos_excluding_name(
            zf_val, zr, key_name_ids, query_val_ids,
            args.query_batch_size, args.real_chunk_size
        )
    
        cos_test, _ = max_cos_excluding_name(
            zf_test, zr, key_name_ids, query_test_ids,
            args.query_batch_size, args.real_chunk_size
        )
    
        # --------------------------------------------------------
        # PER-FONT THRESHOLDING (CHOOSE METHOD HERE)
        # --------------------------------------------------------
        
        method = "symmetric_percentile"   # change this to test methods
        
        if method in ["tpr_fpr", "fixed_fpr", "fixed_precision", "symmetric_percentile"]:
            T_pos, T_neg = compute_two_thresholds(y_val, cos_val, method)
        
            delta_val = np.where(cos_val >= 0,
                                 cos_val - T_pos,
                                 cos_val - T_neg)
        
            delta_test = np.where(cos_test >= 0,
                                  cos_test - T_pos,
                                  cos_test - T_neg)
        
        else:
            T = compute_single_threshold(y_val, cos_val, method)
        
            delta_val = cos_val - T
            delta_test = cos_test - T
        
        print(f"[INFO] Font={name} | method={method}")
    
        all_val.append(delta_val)
        all_test.append(delta_test)
    
    all_val = np.stack(all_val, axis=1)
    all_test = np.stack(all_test, axis=1)
    
    best_val = np.argmax(np.abs(all_val), axis=1)
    final_val = all_val[np.arange(len(all_val)), best_val]
    
    best_test = np.argmax(np.abs(all_test), axis=1)
    final_test = all_test[np.arange(len(all_test)), best_test]

    # --------------------------------------------------------
    # OPTION B EVALUATION (ALL FONTS)
    # --------------------------------------------------------
    
    # Recompute exact_in_bank + y_test per font and stack
    # Use ONE test_df (they are aligned across fonts)
    test_df = read_table(fonts[0][2])  # first font test path
    
    y_all = (test_df[args.label_col].values >= 0.5).astype(int)
    
    # Build name set per font and OR them
    exact_masks = []
    
    for _, _, test_path in fonts:
        df = read_table(test_path)
        real_names = df[args.real_name_col].values
        name_set = set(normalize_name(n) for n in real_names)
    
        mask = np.array([
            normalize_name(n) in name_set
            for n in df[args.fraud_name_col]
        ])
    
        exact_masks.append(mask)
    
    # A sample is "exact in bank" if ANY font contains it
    exact_in_bank_all = np.logical_or.reduce(exact_masks)
    
    # IMPORTANT:
    # final_test already corresponds to stacked multi-font fusion.
    # So we must also concatenate per-font final_test BEFORE fusion.
    # If your fusion assumed identical ordering across fonts,
    # then final_test already matches y_all ordering.
    # Otherwise you must also concatenate per-font test scores before fusion.
    
    # For your current structure (aligned datasets assumed):
    s_all = final_test
    
    eval_exact_in_bank = exact_in_bank_all & (y_all == 1)
    open_mask = ~eval_exact_in_bank

    y_eval = y_all[open_mask]
    s_eval = s_all[open_mask]

    open_n = int(open_mask.sum())
    open_pos = int(y_eval.sum())
    open_neg = int(open_n - open_pos)
    excl_inbank_pos = int(np.sum((y_all == 1) & exact_in_bank_all))

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
    
    # --------------------------------------------------------
    # FINAL THRESHOLD ON FUSED VALIDATION SCORES
    # --------------------------------------------------------
    
    # Use fused validation scores to compute one global threshold
    final_method = "max_mcc"  # change for experiments
    
    final_thr = compute_single_threshold(y_val, final_val, final_method)
    
    print(f"[INFO] Final fused threshold method={final_method} | thr={final_thr:.6f}")
    
    report_threshold_metrics(
        f"FINAL_FUSED_{final_method}",
        final_thr,
        y_all,
        final_test
    )

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
    hard_spoof_mask = (y_all == 1) & (~exact_in_bank_all)
    hard_idx = np.where(hard_spoof_mask)[0]
    hard_n = int(len(hard_idx))

    print("==============================")
    print("HARD SPOOF SLICE (ORIGINAL DEFINITION)")
    print("==============================")
    print("Hard spoof set = (y==1) & (exact_in_bank==0)")
    print(f"[SLICE] hard_n={hard_n:,}")
    if hard_n > 0:
        print(f"[SLICE] mean(s_all)={float(s_all[hard_idx].mean()):.6f}")
        print(f"[SLICE] median(s_all)={float(np.median(s_all[hard_idx])):.6f}")
        for name, thr in thresholds:
            hard_recall = float(np.mean((s_all[hard_idx] >= thr).astype(np.float32)))
            print(f"[SLICE] hard_spoof_recall at {name} thr = {hard_recall:.6f}")
    else:
        print("[SLICE] No hard spoof samples present.")
    print("==============================\n")


if __name__ == "__main__":
    main()