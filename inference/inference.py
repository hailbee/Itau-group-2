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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

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


def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output type: {path}")


# -------------------------
# Column helpers (evaluator2-like sorting)
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str):
        suf = c[len(prefix):]
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
# Threshold helpers (robust to one-class)
# -------------------------
def _has_both_classes(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def best_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    P = float(y_true.sum())
    N = float(len(y_true) - y_true.sum())
    accs = (tpr * P + (1.0 - fpr) * N) / (P + N + 1e-12)
    best_idx = int(np.argmax(accs))
    return float(thresholds[best_idx]), float(accs[best_idx])


def metrics_report(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = float((pred == y).mean()) if y.size else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return acc, tpr, tnr, cm


# -------------------------
# Core: max cosine excluding same-name candidates (memory-safe)
# -------------------------
@torch.inference_mode()
def max_cos_excluding_name(
    zq: torch.Tensor,                      # (N, D) normalized
    zk: torch.Tensor,                      # (M, D) normalized
    key_name_ids: torch.Tensor,            # (M,) int64 group id per key
    query_name_ids: torch.Tensor,          # (N,) int64 group id per query (-1 if not in bank)
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
# Counts table printer
# -------------------------
def print_counts(name: str, y: np.ndarray, exact: np.ndarray) -> None:
    # counts[label][exact]
    c00 = int(np.sum((y == 0) & (exact == 0)))
    c01 = int(np.sum((y == 0) & (exact == 1)))
    c10 = int(np.sum((y == 1) & (exact == 0)))
    c11 = int(np.sum((y == 1) & (exact == 1)))
    print(f"[COUNTS] {name}:")
    print(f"         label0_exact0={c00:,} | label0_exact1={c01:,}")
    print(f"         label1_exact0={c10:,} | label1_exact1={c11:,}")


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Bank spoof inference (Fix A):\n"
            "  score = max cosine to bank excluding same-name candidates\n"
            "  policy (MANDATORY): if fraudulent_name is exactly in bank => pred=0\n"
            "Reports BOTH:\n"
            "  (1) SYSTEM metrics (includes policy)\n"
            "  (2) NON-LAYUP metrics (excludes layup rows: label=0 AND exact_in_bank=1)\n"
            "Also prints both Youden and Accuracy-opt thresholds.\n"
        )
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)

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

    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

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

    # Build candidate bank
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

    # Name IDs for exclusion masking
    name_to_id: Dict[str, int] = {}
    key_ids_list: List[int] = []
    for nm in real_names_bank.tolist():
        k = normalize_name(nm)
        if k not in name_to_id:
            name_to_id[k] = len(name_to_id)
        key_ids_list.append(name_to_id[k])
    key_name_ids_np = np.array(key_ids_list, dtype=np.int64)

    fraud_names = df[args.fraud_name_col].astype(str).to_numpy()
    query_name_ids_np = np.array([name_to_id.get(normalize_name(nm), -1) for nm in fraud_names.tolist()], dtype=np.int64)
    exact_in_bank = (query_name_ids_np != -1)
    print(f"[INFO] exact_in_bank_rate={float(exact_in_bank.mean()):.6f}")

    # Load model
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

    # Encode both sides through student model
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

    # Compute Fix-A score
    score, argmax_idx = max_cos_excluding_name(
        zq=zf,
        zk=zr,
        key_name_ids=key_name_ids,
        query_name_ids=query_name_ids,
        query_batch_size=int(args.query_batch_size),
        key_chunk_size=int(args.real_chunk_size),
    )
    score = np.clip(score, -1.0, 1.0)

    # AUC on ALL (always defined if both classes exist in full data)
    auc_all = float(roc_auc_score(y, score)) if _has_both_classes(y) else float("nan")
    print(f"[METRIC] ROC AUC(top1_excl_self) ALL={auc_all:.6f} (higher => spoof)")

    # Split
    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    calib_n = int(max(1, round(float(args.calib_frac) * n)))
    calib_idx = idx[:calib_n]
    test_idx = idx[calib_n:]
    print(f"[INFO] calib_n={len(calib_idx):,} | test_n={len(test_idx):,}")

    # Define "layups" as EXACT matches that are truly NON-SPOOF: (label=0 AND exact=1)
    layup = (y == 0) & exact_in_bank

    # For reporting "non-layup" metrics: exclude layups
    test_nonlayup = test_idx[~layup[test_idx]]

    # For threshold calibration:
    # Prefer calibrating on non-layups too (to avoid bias), BUT only if both classes exist.
    calib_nonlayup = calib_idx[~layup[calib_idx]]

    # Print counts to make this obvious
    print_counts("FULL", y, exact_in_bank.astype(np.int32))
    print_counts("CALIB", y[calib_idx], exact_in_bank[calib_idx].astype(np.int32))
    print_counts("CALIB_NONLAYUP", y[calib_nonlayup], exact_in_bank[calib_nonlayup].astype(np.int32))
    print_counts("TEST", y[test_idx], exact_in_bank[test_idx].astype(np.int32))
    print_counts("TEST_NONLAYUP", y[test_nonlayup], exact_in_bank[test_nonlayup].astype(np.int32))

    # Choose calibration set for thresholds robustly
    if _has_both_classes(y[calib_nonlayup]):
        thr_calib_idx = calib_nonlayup
        thr_note = "calibrated on CALIB_NONLAYUP (excludes layups)"
    else:
        thr_calib_idx = calib_idx
        thr_note = "calibrated on CALIB (fallback; nonlayup was one-class)"

    # Compute thresholds
    if not _has_both_classes(y[thr_calib_idx]):
        # This should not happen if full data has both classes, but guard anyway.
        thr_youden = float("inf")
        thr_acc = float("inf")
        calib_acc_est = float("nan")
        print("[WARN] Calibration subset has one class; thresholds undefined (set to inf).")
    else:
        thr_youden = youden_threshold(y[thr_calib_idx], score[thr_calib_idx])
        thr_acc, calib_acc_est = best_accuracy_threshold(y[thr_calib_idx], score[thr_calib_idx])

    # Prediction with MANDATORY policy
    def predict_system(scores_subset: np.ndarray, exact_subset: np.ndarray, thr: float) -> np.ndarray:
        pred = (scores_subset >= thr).astype(np.int32)
        pred[exact_subset] = 0
        return pred

    # SYSTEM predictions (on TEST)
    pred_y_test = predict_system(score[test_idx], exact_in_bank[test_idx], thr_youden)
    pred_a_test = predict_system(score[test_idx], exact_in_bank[test_idx], thr_acc)

    # NON-LAYUP predictions (same rule, just evaluated on subset)
    pred_y_nonlay = predict_system(score[test_nonlayup], exact_in_bank[test_nonlayup], thr_youden)
    pred_a_nonlay = predict_system(score[test_nonlayup], exact_in_bank[test_nonlayup], thr_acc)

    # Metrics
    acc_y_sys, tpr_y_sys, tnr_y_sys, cm_y_sys = metrics_report(y[test_idx], pred_y_test)
    acc_a_sys, tpr_a_sys, tnr_a_sys, cm_a_sys = metrics_report(y[test_idx], pred_a_test)

    acc_y_nl, tpr_y_nl, tnr_y_nl, cm_y_nl = metrics_report(y[test_nonlayup], pred_y_nonlay)
    acc_a_nl, tpr_a_nl, tnr_a_nl, cm_a_nl = metrics_report(y[test_nonlayup], pred_a_nonlay)

    # AUC on non-layup subset if both classes exist
    auc_nonlay = float(roc_auc_score(y[test_nonlayup], score[test_nonlayup])) if _has_both_classes(y[test_nonlayup]) else float("nan")

    print("\n==============================")
    print("THRESHOLDS")
    print("==============================")
    print(f"[THR] youden_thr={thr_youden:.6f}")
    print(f"[THR] acc_thr={thr_acc:.6f} | calib_acc_estimate={calib_acc_est:.6f}")
    print(f"[THR] note: {thr_note}")
    print("Policy: exact_in_bank => pred=0 (MANDATORY)")
    print("==============================\n")

    print("==============================")
    print("SYSTEM TEST METRICS (includes layups; production headline)")
    print("==============================")
    print(f"[YOUDEN] acc={acc_y_sys:.6f} | TPR(spoof)={tpr_y_sys:.6f} | TNR(non-spoof)={tnr_y_sys:.6f}")
    print(f"[YOUDEN] cm [true0/true1 x pred0/pred1]:\n{cm_y_sys}")
    print("------------------------------")
    print(f"[ACC]    acc={acc_a_sys:.6f} | TPR(spoof)={tpr_a_sys:.6f} | TNR(non-spoof)={tnr_a_sys:.6f}")
    print(f"[ACC]    cm [true0/true1 x pred0/pred1]:\n{cm_a_sys}")
    print("==============================\n")

    print("==============================")
    print("NON-LAYUP TEST METRICS (excludes layups: label0 & exact_in_bank)")
    print("==============================")
    print(f"[NONLAYUP] ROC AUC={auc_nonlay:.6f} (nan means subset is one-class)")
    print(f"[YOUDEN] acc={acc_y_nl:.6f} | TPR(spoof)={tpr_y_nl:.6f} | TNR(non-spoof)={tnr_y_nl:.6f}")
    print(f"[YOUDEN] cm [true0/true1 x pred0/pred1]:\n{cm_y_nl}")
    print("------------------------------")
    print(f"[ACC]    acc={acc_a_nl:.6f} | TPR(spoof)={tpr_a_nl:.6f} | TNR(non-spoof)={tnr_a_nl:.6f}")
    print(f"[ACC]    cm [true0/true1 x pred0/pred1]:\n{cm_a_nl}")
    print("==============================\n")

    # Save outputs
    out = df.copy()
    out["top1_excl_self"] = score.astype(np.float32, copy=False)
    out["exact_in_bank"] = exact_in_bank.astype(np.int32)
    out["layup"] = layup.astype(np.int32)

    out["argmax_real_index_excl_self"] = argmax_idx
    out["argmax_real_name_excl_self"] = np.array(
        [real_names_bank[int(j)] if int(j) >= 0 else "" for j in argmax_idx],
        dtype=object,
    )

    out["youden_thr"] = float(thr_youden)
    out["acc_thr"] = float(thr_acc)
    out["thr_note"] = thr_note

    out["pred_youden_system"] = predict_system(score, exact_in_bank, thr_youden)
    out["pred_acc_system"] = predict_system(score, exact_in_bank, thr_acc)

    write_table(out, args.output)
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()


"""
Run:

python inference/inference.py \
  --data text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --student-model-path saved_models/best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix real_txt_emb_ \
  --dedup-real-names \
  --calib-frac 0.2 \
  --seed 0 \
  --output text_to_image/evaluation/bank_spoof_preds_report_both.parquet
"""
