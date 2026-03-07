#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier


# =========================
# Optional string libs
# =========================
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
        from fuzzywuzzy import fuzz as fw_fuzz  # type: ignore

        HAVE_FUZZYWUZZY = True
    except Exception:
        HAVE_FUZZYWUZZY = False

try:
    import Levenshtein as py_lev  # type: ignore

    HAVE_PY_LEV = True
except Exception:
    HAVE_PY_LEV = False


# =========================
# IO
# =========================
def load_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path} (expected .parquet/.pq or .csv)")


def safe_str_list(s: pd.Series) -> List[str]:
    return s.fillna("").astype(str).tolist()


# =========================
# String features
# =========================
def levenshtein_distance(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(rf_lev.distance(a, b))
    if HAVE_PY_LEV:
        return int(py_lev.distance(a, b))

    # Pure Python DP fallback
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


def token_set_ratio(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return float(rf_fuzz.token_set_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        return float(fw_fuzz.token_set_ratio(a, b)) / 100.0
    raise RuntimeError("Install rapidfuzz (recommended) or fuzzywuzzy to use token_set_ratio.")


def partial_ratio(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return float(rf_fuzz.partial_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        return float(fw_fuzz.partial_ratio(a, b)) / 100.0
    raise RuntimeError("Install rapidfuzz (recommended) or fuzzywuzzy to use partial_ratio.")


# =========================
# Thresholding / metrics
# =========================
def _has_both_classes(y: np.ndarray) -> bool:
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    finite = np.isfinite(thr)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thr = thr[finite]
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[float, float, float]:
    yhat = (scores >= thr).astype(np.int32)
    acc = float(accuracy_score(y_true, yhat))
    prec = float(precision_score(y_true, yhat, zero_division=0))
    rec = float(recall_score(y_true, yhat, zero_division=0))
    return acc, prec, rec


# =========================
# Features (3 text metrics only)
# =========================
def build_features(
    df: pd.DataFrame,
    fraud_col: str,
    real_col: str,
    label_col: str,
    positive_label: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    for c in (fraud_col, real_col, label_col):
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    fraud_names = safe_str_list(df[fraud_col])
    real_names = safe_str_list(df[real_col])

    y_raw = df[label_col].to_numpy()
    y = (y_raw == positive_label).astype(np.int32)
    if not _has_both_classes(y):
        raise RuntimeError("Need both classes (0 and 1) in this split to compute AUROC/Youden.")

    n = len(df)

    lev_dist = np.empty((n,), dtype=np.int32)
    tsr = np.empty((n,), dtype=np.float32)
    pr = np.empty((n,), dtype=np.float32)

    for i, (a, b) in enumerate(zip(fraud_names, real_names)):
        lev_dist[i] = levenshtein_distance(a, b)
        tsr[i] = float(token_set_ratio(a, b))
        pr[i] = float(partial_ratio(a, b))

    lev_dist_score = (-lev_dist).astype(np.float32)

    feat_names = [
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
    ]
    X = np.stack([tsr, lev_dist_score, pr], axis=1).astype(np.float32)
    return X, y, feat_names


# =========================
# Ensemble model
# =========================
def make_model(kind: str, seed: int) -> object:
    kind = kind.lower().strip()
    if kind == "adaboost":
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        try:
            return AdaBoostClassifier(estimator=base, n_estimators=300, learning_rate=0.05, random_state=seed)
        except TypeError:
            return AdaBoostClassifier(base_estimator=base, n_estimators=300, learning_rate=0.05, random_state=seed)
    if kind == "gboost":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError("Unknown model. Use 'adaboost', 'gboost', or 'none'.")


def model_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = 1.0 / (1.0 + np.exp(-s))
        return s.astype(np.float32)
    raise RuntimeError("Model does not support predict_proba or decision_function.")


# =========================
# Tables
# =========================
@dataclass
class SingleFeatureRow:
    feature: str
    test_auroc: float
    youden_thr_test: float
    test_accuracy: float
    test_precision: float
    test_recall: float


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    args = ap.parse_args()

    if not (HAVE_RAPIDFUZZ or HAVE_FUZZYWUZZY):
        raise RuntimeError("You need rapidfuzz (recommended) or fuzzywuzzy installed for token_set_ratio/partial_ratio.")

    df_tr = load_table(args.train)
    df_te = load_table(args.test)

    X_tr, y_tr, feat_names = build_features(
        df_tr,
        args.fraud_col,
        args.real_col,
        args.label_col,
        args.positive_label,
    )
    X_te, y_te, _ = build_features(
        df_te,
        args.fraud_col,
        args.real_col,
        args.label_col,
        args.positive_label,
    )

    # -----------------------------
    # Single-feature (test-Youden, paper style)
    # -----------------------------
    rows: List[SingleFeatureRow] = []
    for j, name in enumerate(feat_names):
        s_te = X_te[:, j].astype(np.float64)

        test_auroc = float(roc_auc_score(y_te, s_te))

        thr_test = youden_threshold(y_te, s_te)  # keep Youden on TEST
        acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

        rows.append(
            SingleFeatureRow(
                feature=name,
                test_auroc=test_auroc,
                youden_thr_test=float(thr_test),
                test_accuracy=acc,
                test_precision=prec,
                test_recall=rec,
            )
        )

    single_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("test_auroc", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print(single_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # -----------------------------
    # Ensemble (always prints)
    #
    # Train on train, score on test with predict_proba
    # Threshold for accuracy is test-Youden (paper style)
    # -----------------------------
    if args.model == "none":
        print("Model needed!")
        return

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    s_te = model_scores(clf, X_te).astype(np.float64)

    test_auroc = float(roc_auc_score(y_te, s_te))

    thr_test = youden_threshold(y_te, s_te)  # keep Youden on TEST
    acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

    print()
    print("Ensemble")
    print(
        f"model={args.model} test_auroc={test_auroc:.6f} youden_thr_test={thr_test:.6f} "
        f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}"
    )

    false_negative_mask = (y_te == 1) & (s_te < thr_test)
    df_fn = df_te.loc[false_negative_mask].copy().reset_index(drop=True)
    df_fn["score"] = s_te[false_negative_mask]
    df_fn["pred_label"] = 0
    df_fn["is_false_negative"] = 1

    output_dir = "for_paper/error_outputs"
    os.makedirs(output_dir, exist_ok=True)

    fn_parquet = os.path.join(output_dir, "text_metrics_false_negatives.parquet")
    fn_csv = os.path.join(output_dir, "text_metrics_false_negatives.csv")

    df_fn.to_parquet(fn_parquet, index=False)
    df_fn.to_csv(fn_csv, index=False)

    print(f"[OK] wrote false-negative parquet: {fn_parquet}")
    print(f"[OK] wrote false-negative csv: {fn_csv}")


if __name__ == "__main__":
    main()

"""
python3 for_paper/text_metrics.py \
  --train ../Downloads/text_train.parquet \
  --test  ../Downloads/text_test.parquet \
  --model adaboost
"""

"""
python3 for_paper/text_metrics.py \
  --train ../Downloads/text_train.parquet \
  --test  ../Ref/typosquat_ref.parquet \
  --model adaboost
"""