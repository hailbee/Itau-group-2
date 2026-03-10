#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier


# =========================
# Optional string libs
# =========================

try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz.distance import Levenshtein as rf_lev

    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

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
# Column helpers (embeddings)
# =========================
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key_fn(c: str) -> int:
        suf = c[len(prefix):]
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
# PT loading + projection model
# =========================
def load_checkpoint_safely(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        for k in ("model_state", "state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict) and ckpt[k]:
                sd = ckpt[k]
                if all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return sd
    raise RuntimeError(f"Unrecognized checkpoint format: {type(ckpt)}")


def strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def infer_dims_from_head(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    required = ["head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"]
    for k in required:
        if k not in sd:
            raise KeyError(f"Expected key '{k}' not found.")

    w0 = sd["head.0.weight"]
    w2 = sd["head.2.weight"]

    hidden_dim = int(w0.shape[0])
    in_dim = int(w0.shape[1])
    out_dim = int(w2.shape[0])

    return in_dim, hidden_dim, out_dim


class SiameseEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def load_golden_projector(pt_path: str, device: torch.device) -> Tuple[SiameseEmbeddingModel, int]:
    ckpt = load_checkpoint_safely(pt_path, map_location=torch.device("cpu"))
    sd = strip_known_prefixes(extract_state_dict(ckpt))

    in_dim, hidden_dim, out_dim = infer_dims_from_head(sd)

    model = SiameseEmbeddingModel(in_dim, hidden_dim, out_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    return model, in_dim


@torch.inference_mode()
def projected_cosine(
    model: SiameseEmbeddingModel,
    in_dim: int,
    fraud_mat: np.ndarray,
    real_mat: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:

    a = fraud_mat.astype(np.float32, copy=False)
    b = real_mat.astype(np.float32, copy=False)

    n = int(a.shape[0])
    bs = max(1, int(batch_size))

    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)

    sims = torch.empty((n,), device="cpu", dtype=torch.float32)

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        x1 = a_t[i0:i1].to(device=device)
        x2 = b_t[i0:i1].to(device=device)

        z1 = model.encode(x1)
        z2 = model.encode(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        s = F.cosine_similarity(z1, z2, dim=1).detach().cpu()
        sims[i0:i1] = s

    return sims.numpy().astype(np.float32, copy=False)


def build_features(
    df: pd.DataFrame,
    fraud_col: str,
    real_col: str,
    label_col: str,
    positive_label: int,
    fraud_prefix: str,
    real_prefix: str,
    projector: SiameseEmbeddingModel,
    projector_in_dim: int,
    device: torch.device,
    pt_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:

    fraud_names = safe_str_list(df[fraud_col])
    real_names = safe_str_list(df[real_col])

    y_raw = df[label_col].to_numpy()
    y = (y_raw == positive_label).astype(np.int32)

    fraud_mat = mat_from_prefix(df, fraud_prefix)
    real_mat = mat_from_prefix(df, real_prefix)

    cos = projected_cosine(
        model=projector,
        in_dim=projector_in_dim,
        fraud_mat=fraud_mat,
        real_mat=real_mat,
        device=device,
        batch_size=pt_batch_size,
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

    X = np.stack([cos, tsr, lev_dist_score, pr], axis=1).astype(np.float32)
    return X, y


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
    raise ValueError("Unknown model.")


def model_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-s))).astype(np.float32)
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

    ap.add_argument("--deja-train", required=True)
    ap.add_argument("--deja-test", required=True)
    ap.add_argument("--deja-pt", required=True)

    ap.add_argument("--device", default=None)
    ap.add_argument("--pt-batch-size", type=int, default=8192)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    ap.add_argument("--deja-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--deja-real-prefix", default="real_emb_")
    ap.add_argument("--model-output-path", default="saved_models/image_model.joblib")

    args = ap.parse_args()

    if not (HAVE_RAPIDFUZZ or HAVE_FUZZYWUZZY):
        raise RuntimeError("You need rapidfuzz (recommended) or fuzzywuzzy installed for token_set_ratio/partial_ratio.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_tr_dj = load_table(args.deja_train)
    df_te_dj = load_table(args.deja_test)

    y_tr = (df_tr_dj[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
    y_te = (df_te_dj[args.label_col].to_numpy() == args.positive_label).astype(np.int32)

    proj_dj, in_dim_dj = load_golden_projector(args.deja_pt, device)

    X_tr, _ = build_features(
        df_tr_dj,
        args.fraud_col,
        args.real_col,
        args.label_col,
        args.positive_label,
        args.deja_fraud_prefix,
        args.deja_real_prefix,
        proj_dj,
        in_dim_dj,
        device,
        args.pt_batch_size,
    )
    X_te, _ = build_features(
        df_te_dj,
        args.fraud_col,
        args.real_col,
        args.label_col,
        args.positive_label,
        args.deja_fraud_prefix,
        args.deja_real_prefix,
        proj_dj,
        in_dim_dj,
        device,
        args.pt_batch_size,
    )

    if args.model == "none":
        print("Model needed!")
        return

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    model_output_dir = os.path.dirname(args.model_output_path)
    if model_output_dir:
        os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(
        {
            "model": clf,
            "model_type": args.model,
            "seed": args.seed,
            "feature_names": [
                "cosine_sim",
                "token_set_ratio",
                "levenshtein_distance_score",
                "partial_ratio",
            ],
            "positive_label": args.positive_label,
            "fraud_col": args.fraud_col,
            "real_col": args.real_col,
            "label_col": args.label_col,
            "fraud_prefix": args.deja_fraud_prefix,
            "real_prefix": args.deja_real_prefix,
        },
        args.model_output_path,
    )

    print(f"[OK] wrote saved model: {args.model_output_path}")

    s_te = model_scores(clf, X_te).astype(np.float64)
    test_auroc = float(roc_auc_score(y_te, s_te))

    yhat_te = clf.predict(X_te)
    acc = float(accuracy_score(y_te, yhat_te))
    prec = float(precision_score(y_te, yhat_te, zero_division=0))
    rec = float(recall_score(y_te, yhat_te, zero_division=0))

    print()
    print("Ensemble")
    print(
        f"model={args.model} test_auroc={test_auroc:.6f} "
        f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}"
    )


if __name__ == "__main__":
    main()

"""
python3 for_paper/image.py \
  --deja-train ../Deja/train_pairs_with_siglip_embeddings.parquet \
  --deja-test  ../Deja/test_pairs_with_siglip_embeddings.parquet \
  --deja-pt    ../Deja/single_run_model.pt \
  --model adaboost \
  --model-output-path saved_models/image_model.joblib
"""