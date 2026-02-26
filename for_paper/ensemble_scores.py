#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

from text_to_image.siamese import SiameseEmbeddingModel


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
# Column helpers
# =========================
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str) -> int:
        suf = c[len(prefix) :]
        if re.fullmatch(r"-?\d+", suf):
            return int(suf)
        return 10**18

    return sorted(cols, key=lambda c: (key_fn(c), c))


def mat_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=np.float32, copy=False)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix for prefix {prefix!r}, got shape {mat.shape}.")
    return mat


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
# PT loading + transform (optional)
# =========================
def load_checkpoint_safely(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        # raw state dict
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        # common wrappers
        for k in ("model_state", "state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict) and ckpt[k]:
                sd = ckpt[k]
                if all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd
    raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt)}")


def infer_hidden_dim_from_state(sd: Dict[str, torch.Tensor], text_dim: int) -> int:
    candidates: List[int] = []
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and int(v.shape[1]) == int(text_dim):
            candidates.append(int(v.shape[0]))
    if not candidates:
        raise RuntimeError("Could not infer hidden_dim from checkpoint.")
    counts: Dict[int, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    return int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0])


@torch.inference_mode()
def cosine_from_optional_student(
    fraud_mat: np.ndarray,
    real_mat: np.ndarray,
    student_model_path: Optional[str],
    device: torch.device,
    pt_batch_size: int,
    pt_out_dim: int,
) -> np.ndarray:
    """
    If student_model_path is None: cosine on raw mats.
    Else: transform both via encode_text, L2-normalize, cosine on transformed.
    """
    if student_model_path is None:
        a = fraud_mat.astype(np.float32, copy=False)
        b = real_mat.astype(np.float32, copy=False)
        dot = np.sum(a * b, axis=1)
        na = np.sqrt(np.sum(a * a, axis=1)) + 1e-12
        nb = np.sqrt(np.sum(b * b, axis=1)) + 1e-12
        return (dot / (na * nb)).astype(np.float32)

    ckpt = load_checkpoint_safely(student_model_path, map_location=device)
    sd = extract_state_dict(ckpt)

    text_dim = int(fraud_mat.shape[1])
    hidden_dim = infer_hidden_dim_from_state(sd, text_dim=text_dim)

    model = SiameseEmbeddingModel(text_dim=text_dim, hidden_dim=hidden_dim, image_dim=int(pt_out_dim)).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    n = int(fraud_mat.shape[0])
    bs = max(1, int(pt_batch_size))

    fraud_t = torch.from_numpy(fraud_mat).to(device=device, dtype=torch.float32)
    real_t = torch.from_numpy(real_mat).to(device=device, dtype=torch.float32)

    sims = torch.empty((n,), device=device, dtype=torch.float32)

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        zf = model.encode_text(fraud_t[i0:i1])
        zr = model.encode_text(real_t[i0:i1])

        zf = F.normalize(zf, dim=1)
        zr = F.normalize(zr, dim=1)

        sims[i0:i1] = F.cosine_similarity(zf, zr, dim=1)

    return sims.detach().cpu().numpy().astype(np.float32, copy=False)


# =========================
# Features (your 4)
# =========================
def build_features(
    df: pd.DataFrame,
    fraud_col: str,
    real_col: str,
    label_col: str,
    positive_label: int,
    fraud_prefix: str,
    real_prefix: str,
    student_model_path: Optional[str],
    device: torch.device,
    pt_batch_size: int,
    pt_out_dim: int,
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

    fraud_mat = mat_from_prefix(df, fraud_prefix)
    real_mat = mat_from_prefix(df, real_prefix)
    if fraud_mat.shape != real_mat.shape:
        raise RuntimeError(f"Embedding shape mismatch: fraud={fraud_mat.shape} real={real_mat.shape}")

    cos = cosine_from_optional_student(
        fraud_mat=fraud_mat,
        real_mat=real_mat,
        student_model_path=student_model_path,
        device=device,
        pt_batch_size=pt_batch_size,
        pt_out_dim=pt_out_dim,
    )

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
        "cosine_sim",
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
    ]
    X = np.stack([cos, tsr, lev_dist_score, pr], axis=1).astype(np.float32)
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
    val_auroc: float
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
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)

    ap.add_argument("--student-model-path", default=None, help="Optional .pt to transform embeddings before cosine_sim.")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--pt-batch-size", type=int, default=4096)
    ap.add_argument("--pt-out-dim", type=int, default=768)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

    args = ap.parse_args()

    if not (HAVE_RAPIDFUZZ or HAVE_FUZZYWUZZY):
        raise RuntimeError("You need rapidfuzz (recommended) or fuzzywuzzy installed for token_set_ratio/partial_ratio.")

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    df_tr = load_table(args.train)
    df_va = load_table(args.val)
    df_te = load_table(args.test)

    X_tr, y_tr, feat_names = build_features(
        df_tr,
        args.fraud_col, args.real_col, args.label_col, args.positive_label,
        args.fraud_prefix, args.real_prefix,
        student_model_path=args.student_model_path,
        device=device,
        pt_batch_size=args.pt_batch_size,
        pt_out_dim=args.pt_out_dim,
    )
    X_va, y_va, _ = build_features(
        df_va,
        args.fraud_col, args.real_col, args.label_col, args.positive_label,
        args.fraud_prefix, args.real_prefix,
        student_model_path=args.student_model_path,
        device=device,
        pt_batch_size=args.pt_batch_size,
        pt_out_dim=args.pt_out_dim,
    )
    X_te, y_te, _ = build_features(
        df_te,
        args.fraud_col, args.real_col, args.label_col, args.positive_label,
        args.fraud_prefix, args.real_prefix,
        student_model_path=args.student_model_path,
        device=device,
        pt_batch_size=args.pt_batch_size,
        pt_out_dim=args.pt_out_dim,
    )

    # -----------------------------
    # Single-feature (test-Youden, paper style)
    # -----------------------------
    rows: List[SingleFeatureRow] = []
    for j, name in enumerate(feat_names):
        s_va = X_va[:, j].astype(np.float64)
        s_te = X_te[:, j].astype(np.float64)

        val_auroc = float(roc_auc_score(y_va, s_va))
        test_auroc = float(roc_auc_score(y_te, s_te))

        thr_test = youden_threshold(y_te, s_te)
        acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

        rows.append(
            SingleFeatureRow(
                feature=name,
                val_auroc=val_auroc,
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
    # If --model none: define "ensemble score" = cosine_sim (still gives an accuracy line)
    # If classifier: train on train, score on val/test with predict_proba
    # Threshold for accuracy is test-Youden (paper style)
    # -----------------------------
    if args.model == "none":
        print("Model needed!")
        return

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    s_va = model_scores(clf, X_va).astype(np.float64)
    s_te = model_scores(clf, X_te).astype(np.float64)

    val_auroc = float(roc_auc_score(y_va, s_va))
    test_auroc = float(roc_auc_score(y_te, s_te))

    thr_test = youden_threshold(y_te, s_te)
    acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

    print()
    print("Ensemble")
    print(f"model={args.model} val_auroc={val_auroc:.6f} test_auroc={test_auroc:.6f} youden_thr_test={thr_test:.6f} "
          f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}")


if __name__ == "__main__":
    main()
    
"""
python3 for_paper/ensemble_scores.py \
  --train ../Downloads/vate_train.parquet \
  --val   ../Downloads/vate_validate.parquet \
  --test  ../Downloads/vate_test.parquet \
  --student-model-path saved_models/deja_best_model.pt \
  --model adaboost
"""

"""
python3 for_paper/ensemble_scores.py \
  --train ../Downloads/vate_train.parquet \
  --val   ../Downloads/vate_validate.parquet \
  --test  ../Downloads/vate_test.parquet \
  --model adaboost
"""