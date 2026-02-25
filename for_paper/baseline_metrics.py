#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# -----------------------------
# Optional fast libs
# -----------------------------
HAVE_RAPIDFUZZ = False
HAVE_FUZZYWUZZY = False
HAVE_PY_LEV = False

try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz.distance import Levenshtein as rf_lev
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

if not HAVE_RAPIDFUZZ:
    try:
        from fuzzywuzzy import fuzz as fw_fuzz
        HAVE_FUZZYWUZZY = True
    except Exception:
        HAVE_FUZZYWUZZY = False

try:
    import Levenshtein as py_lev  # python-Levenshtein
    HAVE_PY_LEV = True
except Exception:
    HAVE_PY_LEV = False


# -----------------------------
# I/O
# -----------------------------
def load_table(path: str) -> pd.DataFrame:
    pl = path.lower()
    if pl.endswith(".csv"):
        return pd.read_csv(path)
    if pl.endswith(".parquet") or pl.endswith(".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path} (expected .csv or .parquet)")


def safe_str_list(s: pd.Series) -> List[str]:
    return s.fillna("").astype(str).tolist()


# -----------------------------
# Levenshtein (distance + similarity)
# -----------------------------
def levenshtein_distance(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(rf_lev.distance(a, b))
    if HAVE_PY_LEV:
        return int(py_lev.distance(a, b))

    # Pure Python DP fallback (correct but slower)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return int(prev[-1])


def levenshtein_similarity(a: str, b: str) -> float:
    # Normalized to [0,1], where 1 means identical
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    d = levenshtein_distance(a, b)
    sim = 1.0 - (d / m)
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return float(sim)


# -----------------------------
# Fuzzy matching (Token Set Ratio, Partial Ratio)
# -----------------------------
def token_set_ratio(a: str, b: str) -> float:
    # Correct implementation: token_set_ratio; normalize to [0,1]
    if HAVE_RAPIDFUZZ:
        return float(rf_fuzz.token_set_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        return float(fw_fuzz.token_set_ratio(a, b)) / 100.0
    raise RuntimeError(
        "Missing fuzzy library. Install rapidfuzz (recommended) or fuzzywuzzy.\n"
        "Example: pip install rapidfuzz"
    )


def partial_ratio(a: str, b: str) -> float:
    # Correct implementation: partial_ratio; normalize to [0,1]
    if HAVE_RAPIDFUZZ:
        return float(rf_fuzz.partial_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        return float(fw_fuzz.partial_ratio(a, b)) / 100.0
    raise RuntimeError(
        "Missing fuzzy library. Install rapidfuzz (recommended) or fuzzywuzzy.\n"
        "Example: pip install rapidfuzz"
    )


# -----------------------------
# Evaluation
# -----------------------------
@dataclass
class MetricResult:
    metric: str
    roc_auc: float
    inverted: bool
    youden_threshold: float
    accuracy: float
    precision: float
    recall: float


def youden_j_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def align_score_direction(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    Ensures "higher score => more positive".
    If AUC < 0.5, invert scores with (1 - score). This is appropriate for [0,1] similarity scores.
    For non-[0,1] scores (e.g., -distance), we instead multiply by -1 as the inversion.
    """
    auc0 = float(roc_auc_score(y_true, scores))
    if auc0 >= 0.5:
        return scores, False, auc0

    # Heuristic inversion:
    # If scores look like similarities in [0,1], use 1-score. Otherwise, use -score.
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smin >= 0.0 and smax <= 1.0:
        inv = 1.0 - scores
    else:
        inv = -scores

    auc1 = float(roc_auc_score(y_true, inv))
    return inv, True, auc1


def evaluate(y_true: np.ndarray, scores: np.ndarray, name: str) -> MetricResult:
    scores = scores.astype(np.float64, copy=False)

    scores_aligned, inverted, auc_val = align_score_direction(y_true, scores)
    thr = youden_j_threshold(y_true, scores_aligned)

    y_pred = (scores_aligned >= thr).astype(np.int32)

    return MetricResult(
        metric=name,
        roc_auc=float(auc_val),
        inverted=bool(inverted),
        youden_threshold=float(thr),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
    )


def compute_all_scores(fraud: List[str], real: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(fraud)

    # Levenshtein raw distance (integer)
    lev_dist = np.fromiter(
        (levenshtein_distance(a, b) for a, b in zip(fraud, real)),
        dtype=np.int32,
        count=n,
    )

    # Use score = -distance so "higher => more similar"
    lev_dist_score = -lev_dist.astype(np.float64)

    # Levenshtein normalized similarity in [0,1]
    lev_sim = np.fromiter(
        (levenshtein_similarity(a, b) for a, b in zip(fraud, real)),
        dtype=np.float64,
        count=n,
    )

    # Token set ratio in [0,1]
    tsr = np.fromiter(
        (token_set_ratio(a, b) for a, b in zip(fraud, real)),
        dtype=np.float64,
        count=n,
    )

    # Partial ratio in [0,1]
    pr = np.fromiter(
        (partial_ratio(a, b) for a, b in zip(fraud, real)),
        dtype=np.float64,
        count=n,
    )

    return lev_dist_score, lev_sim, tsr, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to parquet/csv with fraudulent_name, real_name, label")
    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1, help="Value treated as positive class (default: 1)")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests")
    args = ap.parse_args()

    df = load_table(args.input).copy()
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    for c in (args.fraud_col, args.real_col, args.label_col):
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    fraud = safe_str_list(df[args.fraud_col])
    real = safe_str_list(df[args.real_col])

    y_raw = df[args.label_col].to_numpy()
    y = (y_raw == args.positive_label).astype(np.int32)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Label column has <2 classes; ROC-AUC/Youden require both classes present.")

    print(f"N={len(df)}  positive_rate={float(y.mean()):.6f}")
    if HAVE_RAPIDFUZZ:
        print("[INFO] Using rapidfuzz for fuzzy + Levenshtein distance.")
    else:
        print("[INFO] rapidfuzz not available; using fallbacks (may be slower).")

    lev_dist_score, lev_sim, tsr, pr = compute_all_scores(fraud, real)

    results: List[MetricResult] = [
        evaluate(y, lev_dist_score, "levenshtein_distance (score=-distance)"),
        evaluate(y, lev_sim, "levenshtein_similarity (1 - d/maxlen)"),
        evaluate(y, tsr, "token_set_ratio"),
        evaluate(y, pr, "partial_ratio"),
    ]

    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("roc_auc", ascending=False)

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()


"""
Usage example:

python3 for_paper/baseline_metrics.py \
  --input ../Downloads/vate_test.parquet \
  --fraud-col fraudulent_name \
  --real-col real_name \
  --label-col label \
  --positive-label 1
"""