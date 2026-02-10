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
# Basic IO / device
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
# Column helpers
# -------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    # Try to sort by numeric suffix if present (matches evaluator2 idea)
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
# Checkpoint loading (evaluator2-compatible)
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
# Name normalization
# -------------------------
def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().casefold()
    return s


# -------------------------
# Youden threshold (higher score => label 1)
# -------------------------
def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def has_both_classes(y: np.ndarray) -> bool:
    y = y.astype(np.int32)
    return (y.min() != y.max())


# -------------------------
# Core: max cosine excluding "self-name" candidates
# -------------------------
@torch.inference_mode()
def max_cos_excluding_name(
    zq: torch.Tensor,                      # (N, D) normalized
    zk: torch.Tensor,                      # (M, D) normalized
    key_name_ids: torch.Tensor,            # (M,) int64, group id for each key (normalized name)
    query_name_ids: torch.Tensor,          # (N,) int64, group id for each query
    query_batch_size: int,
    key_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each query i:
      score_i = max_j cos(zq[i], zk[j]) over j where key_name_ids[j] != query_name_ids[i].

    Returns:
      max_vals (N,), argmax_idxs (N,) (index into keys, -1 if no eligible keys)
    """
    device = zq.device
    N = int(zq.shape[0])
    M = int(zk.shape[0])

    qbs = max(1, int(query_batch_size))
    kcs = max(1, int(key_chunk_size))

    max_vals_all = torch.full((N,), -1e9, device=device, dtype=torch.float32)
    max_idxs_all = torch.full((N,), -1, device=device, dtype=torch.long)

    for q0 in range(0, N, qbs):
        q1 = min(q0 + qbs, N)
        q = zq[q0:q1]                    # (Q, D)
        q_ids = query_name_ids[q0:q1]    # (Q,)

        best = torch.full((q1 - q0,), -1e9, device=device)
        best_idx = torch.full((q1 - q0,), -1, device=device, dtype=torch.long)

        for k0 in range(0, M, kcs):
            k1 = min(k0 + kcs, M)
            k = zk[k0:k1]                                # (K, D)
            k_ids = key_name_ids[k0:k1].view(1, -1)      # (1, K)

            scores = q @ k.T                              # (Q, K)

            # Mask out keys whose name_id == query's name_id
            # broadcast compare: (Q,1) vs (1,K) -> (Q,K)
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
            "Fix A: bank spoof score = max cosine similarity EXCLUDING candidates whose normalized real_name == "
            "normalized fraudulent_name.\n"
            "This removes trivial self-matches and should restore 'spoof => higher similarity'.\n"
            "Prediction: pred=1 if score >= threshold (Youden by default).\n"
        )
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--label-col", default="label")  # 1=spoof, 0=non-spoof
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

    ap.add_argument("--threshold", type=float, default=None, help="If provided, use this threshold on score.")
    ap.add_argument("--threshold-mode", choices=["youden"], default="youden")

    ap.add_argument("--use-margin", action="store_true", help="Optional: require top1 - top2 >= margin (NOT used here).")
    ap.add_argument("--margin", type=float, default=0.0)

    # Important policy toggle:
    # For Fix A experiment, default is NOT to force exact names to non-spoof;
    # we are removing their self-match from scoring instead.
    ap.add_argument(
        "--force-exact-nonspoof",
        action="store_true",
        help="If set: if fraudulent_name is an exact real_name, force pred=0 (non-spoof). Default off.",
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
    print(f"[INFO] n={len(df):,} | spoof_rate(label=1)={float(y.mean()):.6f}")

    fraud_np = mat_from_prefix(df, args.fraud_prefix)
    real_np = mat_from_prefix(df, args.real_prefix)

    n, text_dim = fraud_np.shape
    n_real_full, text_dim2 = real_np.shape
    if text_dim2 != text_dim:
        raise ValueError(f"fraud_dim={text_dim} but real_dim={text_dim2}; expected same dim.")
    print(f"[INFO] fraud_mat={fraud_np.shape} | real_mat(full)={real_np.shape}")

    # -------------------------
    # Build candidate bank: (real vectors, real names)
    # -------------------------
    real_names_full = df[args.real_name_col].astype(str).to_numpy()

    if args.dedup_real_names:
        seen = set()
        keep_idx: List[int] = []
        keep_names: List[str] = []
        for i, nm in enumerate(real_names_full):
            key = normalize_name(nm)
            if key not in seen:
                seen.add(key)
                keep_idx.append(i)
                keep_names.append(nm)
        keep_idx_np = np.array(keep_idx, dtype=np.int64)
        real_np_bank = real_np[keep_idx_np]
        real_names_bank = np.array(keep_names, dtype=object)
        print(f"[INFO] dedup_real_names: {n_real_full:,} -> {len(real_np_bank):,} candidates")
    else:
        real_np_bank = real_np
        real_names_bank = real_names_full

    # Name IDs (group ids) for exclusion masking
    # Build a mapping from normalized name -> integer id
    name_to_id: Dict[str, int] = {}
    key_ids_list: List[int] = []
    for nm in real_names_bank.tolist():
        k = normalize_name(nm)
        if k not in name_to_id:
            name_to_id[k] = len(name_to_id)
        key_ids_list.append(name_to_id[k])
    key_name_ids_np = np.array(key_ids_list, dtype=np.int64)

    fraud_names = df[args.fraud_name_col].astype(str).to_numpy()
    query_ids_list: List[int] = []
    for nm in fraud_names.tolist():
        k = normalize_name(nm)
        # If a query name isn't in bank at all, give it a special id that won't match any key
        if k not in name_to_id:
            query_ids_list.append(-1)
        else:
            query_ids_list.append(name_to_id[k])
    query_name_ids_np = np.array(query_ids_list, dtype=np.int64)

    exact_in_bank = (query_name_ids_np != -1)
    print(f"[INFO] exact_in_bank_rate={float(exact_in_bank.mean()):.6f}")

    # -------------------------
    # Load student model (evaluator2 style)
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
    missing = getattr(load_info, "missing_keys", [])
    unexpected = getattr(load_info, "unexpected_keys", [])
    print(f"[INFO] load_state_dict: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    # -------------------------
    # Encode BOTH sides through student model (apples-to-apples)
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

    # -------------------------
    # Score: max cosine excluding same-name candidates
    # -------------------------
    key_name_ids = torch.from_numpy(key_name_ids_np).to(device)
    query_name_ids = torch.from_numpy(query_name_ids_np).to(device)

    score_excl, argmax_idx = max_cos_excluding_name(
        zq=zf,
        zk=zr,
        key_name_ids=key_name_ids,
        query_name_ids=query_name_ids,
        query_batch_size=int(args.query_batch_size),
        key_chunk_size=int(args.real_chunk_size),
    )

    # If a query is in-bank and the bank only contains that one name (rare), score may stay -1e9.
    # Clamp to [-1, 1] range for sanity; -1e9 means "no eligible candidates".
    score_excl_clamped = np.clip(score_excl, -1.0, 1.0)

    # -------------------------
    # Threshold (Youden) on score_excl where higher => spoof
    # -------------------------
    if args.threshold is None:
        # Prefer rows where the score is meaningful (argmax exists)
        ok = (argmax_idx >= 0)
        if ok.any() and has_both_classes(y[ok]):
            thr = youden_threshold(y[ok], score_excl_clamped[ok])
            thr_note = "Youden on score_excl (rows with eligible candidates)"
        elif has_both_classes(y):
            thr = youden_threshold(y, score_excl_clamped)
            thr_note = "Youden on score_excl (ALL rows; ok-subset was one-class)"
        else:
            thr = float("inf")
            thr_note = "degenerate: only one class exists"
    else:
        thr = float(args.threshold)
        thr_note = "user provided"

    pred = (score_excl_clamped >= thr).astype(np.int32)

    # Optional policy: force exact names to non-spoof
    if args.force_exact_nonspoof:
        pred[exact_in_bank] = 0

    # -------------------------
    # Metrics
    # -------------------------
    try:
        auc = float(roc_auc_score(y, score_excl_clamped))
        auc_flip = float(roc_auc_score(y, -score_excl_clamped))
        auc_best = max(auc, auc_flip)
        direction = "score_higher_for_spoof" if auc >= auc_flip else "score_lower_for_spoof (use -score)"
    except Exception:
        auc = float("nan")
        auc_flip = float("nan")
        auc_best = float("nan")
        direction = "n/a"

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = float((pred == y).mean())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"[METRIC] ROC AUC(score_excl)={auc:.6f} | flipped={auc_flip:.6f} | best={auc_best:.6f} | dir: {direction}")
    print(f"[METRIC] threshold={thr:.6f} ({thr_note})")
    if args.force_exact_nonspoof:
        print("[METRIC] policy: force exact-in-bank queries to pred=0")
    print(f"[METRIC] accuracy={acc:.6f} | TPR(spoof)= {tpr:.6f} | TNR(non-spoof)= {tnr:.6f}")
    print(f"[METRIC] confusion_matrix [true0/true1 x pred0/pred1]:\n{cm}")

    # Sanity: class means
    if np.any(y == 0):
        print(f"[METRIC] mean(score_excl) label0={float(score_excl_clamped[y==0].mean()):.6f}")
    if np.any(y == 1):
        print(f"[METRIC] mean(score_excl) label1={float(score_excl_clamped[y==1].mean()):.6f}")

    # -------------------------
    # Save output
    # -------------------------
    out = df.copy()
    out["score_excl_self"] = score_excl_clamped.astype(np.float32, copy=False)
    out["argmax_real_index_excl_self"] = argmax_idx
    out["argmax_real_name_excl_self"] = np.array(
        [real_names_bank[int(j)] if int(j) >= 0 else "" for j in argmax_idx],
        dtype=object,
    )
    out["exact_in_bank"] = exact_in_bank.astype(np.int32)
    out["pred_label"] = pred
    out["used_threshold"] = float(thr)

    write_table(out, args.output)
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()


"""
Example:

python inference/inference.py \
  --data text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --student-model-path saved_models/best_model.pt \
  --hidden-dim 1024 \
  --out-dim 768 \
  --fraud-prefix fraud_txt_emb_ \
  --real-prefix real_txt_emb_ \
  --dedup-real-names \
  --output text_to_image/evaluation/bank_spoof_preds_excl_self.parquet

If you want to also enforce your real-world policy (exact real names are never spoof):
  --force-exact-nonspoof
"""