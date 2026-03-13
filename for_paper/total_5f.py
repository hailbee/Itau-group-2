#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
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

try:
    import Levenshtein as py_lev

    HAVE_PY_LEV = True
except Exception:
    HAVE_PY_LEV = False

try:
    from fuzzywuzzy import fuzz as fw_fuzz

    HAVE_FUZZYWUZZY = True
except Exception:
    HAVE_FUZZYWUZZY = False

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
                    kk = kk[len(p) :]
                    changed = True
        out[kk] = v
    return out


def infer_dims_from_head(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    required = ["head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"]
    for k in required:
        if k not in sd:
            raise KeyError(f"Expected key '{k}' not found. Keys present (sample): {list(sd.keys())[:20]}")

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
    sd = extract_state_dict(ckpt)
    sd = strip_known_prefixes(sd)

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

    if int(a.shape[1]) != int(in_dim) or int(b.shape[1]) != int(in_dim):
        raise RuntimeError(f"Embedding dim mismatch: data_dim={a.shape[1]} vs model_in_dim={in_dim}.")

    n = int(a.shape[0])
    bs = max(1, int(batch_size))

    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)

    sims = torch.empty((n,), device="cpu", dtype=torch.float32)

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        x1 = a_t[i0:i1].to(device=device, dtype=torch.float32, non_blocking=True)
        x2 = b_t[i0:i1].to(device=device, dtype=torch.float32, non_blocking=True)

        z1 = model.encode(x1)
        z2 = model.encode(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        s = F.cosine_similarity(z1, z2, dim=1).detach().to("cpu")
        sims[i0:i1] = s

    return sims.numpy().astype(np.float32, copy=False)


# =========================
# Feature builders
# =========================
def build_text_features(
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
    for c in (fraud_col, real_col, label_col):
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    fraud_names = safe_str_list(df[fraud_col])
    real_names = safe_str_list(df[real_col])

    y_raw = df[label_col].to_numpy()
    y = (y_raw == positive_label).astype(np.int32)
    if not _has_both_classes(y):
        raise RuntimeError("Need both classes (0 and 1) in this split to compute AUROC.")

    fraud_mat = mat_from_prefix(df, fraud_prefix)
    real_mat = mat_from_prefix(df, real_prefix)
    if fraud_mat.shape != real_mat.shape:
        raise RuntimeError(f"Embedding shape mismatch: fraud={fraud_mat.shape} real={real_mat.shape}")

    text_cos = projected_cosine(
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

    X = np.stack([text_cos, tsr, lev_dist_score, pr], axis=1).astype(np.float32, copy=False)
    return X, y


def build_single_font_cosine(
    df: pd.DataFrame,
    label_col: str,
    positive_label: int,
    fraud_prefix: str,
    real_prefix: str,
    projector: SiameseEmbeddingModel,
    projector_in_dim: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if label_col not in df.columns:
        raise RuntimeError(f"Missing required column: {label_col}")

    y_raw = df[label_col].to_numpy()
    y = (y_raw == positive_label).astype(np.int32)
    if not _has_both_classes(y):
        raise RuntimeError("Need both classes (0 and 1) in this split to compute AUROC.")

    fraud_mat = mat_from_prefix(df, fraud_prefix)
    real_mat = mat_from_prefix(df, real_prefix)
    if fraud_mat.shape != real_mat.shape:
        raise RuntimeError(f"Embedding shape mismatch: fraud={fraud_mat.shape} real={real_mat.shape}")

    cos = projected_cosine(
        model=projector,
        in_dim=projector_in_dim,
        fraud_mat=fraud_mat,
        real_mat=real_mat,
        device=device,
        batch_size=batch_size,
    )
    return cos, y


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
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()

    # Downloads text data / text embeddings
    ap.add_argument("--downloads-train", required=True)
    ap.add_argument("--downloads-test", required=True)
    ap.add_argument("--downloads-pt", required=True)

    # Exact image fonts you specified
    ap.add_argument("--deja-train", required=True)
    ap.add_argument("--deja-test", required=True)
    ap.add_argument("--deja-pt", required=True)

    ap.add_argument("--unifont-train", required=True)
    ap.add_argument("--unifont-test", required=True)
    ap.add_argument("--unifont-pt", required=True)

    ap.add_argument("--libre-train", required=True)
    ap.add_argument("--libre-test", required=True)
    ap.add_argument("--libre-pt", required=True)

    ap.add_argument("--doulos-train", required=True)
    ap.add_argument("--doulos-test", required=True)
    ap.add_argument("--doulos-pt", required=True)

    ap.add_argument("--cousine-train", required=True)
    ap.add_argument("--cousine-test", required=True)
    ap.add_argument("--cousine-pt", required=True)

    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--pt-batch-size", type=int, default=8192)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    ap.add_argument("--downloads-fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--downloads-real-prefix", default="real_txt_emb_")

    ap.add_argument("--deja-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--deja-real-prefix", default="real_emb_")
    ap.add_argument("--unifont-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--unifont-real-prefix", default="real_emb_")
    ap.add_argument("--libre-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--libre-real-prefix", default="real_emb_")
    ap.add_argument("--doulos-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--doulos-real-prefix", default="real_emb_")
    ap.add_argument("--cousine-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--cousine-real-prefix", default="real_emb_")

    ap.add_argument("--model-output-path", default="saved_models/best_ensemble_model.joblib")

    args = ap.parse_args()

    if not (HAVE_RAPIDFUZZ or HAVE_FUZZYWUZZY):
        raise RuntimeError("You need rapidfuzz (recommended) or fuzzywuzzy installed for token_set_ratio/partial_ratio.")

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Load Downloads text data
    df_tr_txt = load_table(args.downloads_train)
    df_te_txt = load_table(args.downloads_test)

    # Load exact font datasets
    df_tr_dj = load_table(args.deja_train)
    df_te_dj = load_table(args.deja_test)

    df_tr_uf = load_table(args.unifont_train)
    df_te_uf = load_table(args.unifont_test)

    df_tr_li = load_table(args.libre_train)
    df_te_li = load_table(args.libre_test)

    df_tr_do = load_table(args.doulos_train)
    df_te_do = load_table(args.doulos_test)

    df_tr_co = load_table(args.cousine_train)
    df_te_co = load_table(args.cousine_test)

    for df in (
        df_tr_txt, df_te_txt,
        df_tr_dj, df_te_dj,
        df_tr_uf, df_te_uf,
        df_tr_li, df_te_li,
        df_tr_do, df_te_do,
        df_tr_co, df_te_co,
    ):
        if args.label_col not in df.columns:
            raise RuntimeError(f"Missing label_col={args.label_col!r} in one of the tables.")
        if args.fraud_col not in df.columns:
            raise RuntimeError(f"Missing fraud_col={args.fraud_col!r} in one of the tables.")
        if args.real_col not in df.columns:
            raise RuntimeError(f"Missing real_col={args.real_col!r} in one of the tables.")

    # Row-count checks
    n_tr = len(df_tr_txt)
    n_te = len(df_te_txt)

    if (
        len(df_tr_dj) != n_tr
        or len(df_tr_uf) != n_tr
        or len(df_tr_li) != n_tr
        or len(df_tr_do) != n_tr
        or len(df_tr_co) != n_tr
    ):
        raise RuntimeError(
            "Train row-count mismatch: "
            f"Downloads={n_tr} Deja={len(df_tr_dj)} Unifont={len(df_tr_uf)} "
            f"Libre={len(df_tr_li)} Doulos={len(df_tr_do)} Cousine={len(df_tr_co)}"
        )

    if (
        len(df_te_dj) != n_te
        or len(df_te_uf) != n_te
        or len(df_te_li) != n_te
        or len(df_te_do) != n_te
        or len(df_te_co) != n_te
    ):
        raise RuntimeError(
            "Test row-count mismatch: "
            f"Downloads={n_te} Deja={len(df_te_dj)} Unifont={len(df_te_uf)} "
            f"Libre={len(df_te_li)} Doulos={len(df_te_do)} Cousine={len(df_te_co)}"
        )

    # Label checks
    y_tr = (df_tr_txt[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
    y_te = (df_te_txt[args.label_col].to_numpy() == args.positive_label).astype(np.int32)

    for name, df in [
        ("Deja train", df_tr_dj),
        ("Unifont train", df_tr_uf),
        ("Libre train", df_tr_li),
        ("Doulos train", df_tr_do),
        ("Cousine train", df_tr_co),
    ]:
        y = (df[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
        if not np.array_equal(y_tr, y):
            raise RuntimeError(f"Train labels mismatch: Downloads vs {name}")

    for name, df in [
        ("Deja test", df_te_dj),
        ("Unifont test", df_te_uf),
        ("Libre test", df_te_li),
        ("Doulos test", df_te_do),
        ("Cousine test", df_te_co),
    ]:
        y = (df[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
        if not np.array_equal(y_te, y):
            raise RuntimeError(f"Test labels mismatch: Downloads vs {name}")

    # Load projectors
    proj_txt, in_dim_txt = load_golden_projector(args.downloads_pt, device=device)
    proj_dj, in_dim_dj = load_golden_projector(args.deja_pt, device=device)
    proj_uf, in_dim_uf = load_golden_projector(args.unifont_pt, device=device)
    proj_li, in_dim_li = load_golden_projector(args.libre_pt, device=device)
    proj_do, in_dim_do = load_golden_projector(args.doulos_pt, device=device)
    proj_co, in_dim_co = load_golden_projector(args.cousine_pt, device=device)

    # Build text-side features: [text cosine, token_set_ratio, levenshtein_distance_score, partial_ratio]
    X_tr_txt, y_tr_txt = build_text_features(
        df=df_tr_txt,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.downloads_fraud_prefix,
        real_prefix=args.downloads_real_prefix,
        projector=proj_txt,
        projector_in_dim=in_dim_txt,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )
    X_te_txt, y_te_txt = build_text_features(
        df=df_te_txt,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.downloads_fraud_prefix,
        real_prefix=args.downloads_real_prefix,
        projector=proj_txt,
        projector_in_dim=in_dim_txt,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )

    # Build image-side features: 5 cosine features using the exact image-ensemble fonts
    cos_tr_dj, y_tr_dj = build_single_font_cosine(
        df=df_tr_dj,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.deja_fraud_prefix,
        real_prefix=args.deja_real_prefix,
        projector=proj_dj,
        projector_in_dim=in_dim_dj,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_dj, y_te_dj = build_single_font_cosine(
        df=df_te_dj,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.deja_fraud_prefix,
        real_prefix=args.deja_real_prefix,
        projector=proj_dj,
        projector_in_dim=in_dim_dj,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_uf, y_tr_uf = build_single_font_cosine(
        df=df_tr_uf,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.unifont_fraud_prefix,
        real_prefix=args.unifont_real_prefix,
        projector=proj_uf,
        projector_in_dim=in_dim_uf,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_uf, y_te_uf = build_single_font_cosine(
        df=df_te_uf,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.unifont_fraud_prefix,
        real_prefix=args.unifont_real_prefix,
        projector=proj_uf,
        projector_in_dim=in_dim_uf,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_li, y_tr_li = build_single_font_cosine(
        df=df_tr_li,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.libre_fraud_prefix,
        real_prefix=args.libre_real_prefix,
        projector=proj_li,
        projector_in_dim=in_dim_li,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_li, y_te_li = build_single_font_cosine(
        df=df_te_li,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.libre_fraud_prefix,
        real_prefix=args.libre_real_prefix,
        projector=proj_li,
        projector_in_dim=in_dim_li,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_do, y_tr_do = build_single_font_cosine(
        df=df_tr_do,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.doulos_fraud_prefix,
        real_prefix=args.doulos_real_prefix,
        projector=proj_do,
        projector_in_dim=in_dim_do,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_do, y_te_do = build_single_font_cosine(
        df=df_te_do,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.doulos_fraud_prefix,
        real_prefix=args.doulos_real_prefix,
        projector=proj_do,
        projector_in_dim=in_dim_do,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_co, y_tr_co = build_single_font_cosine(
        df=df_tr_co,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.cousine_fraud_prefix,
        real_prefix=args.cousine_real_prefix,
        projector=proj_co,
        projector_in_dim=in_dim_co,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_co, y_te_co = build_single_font_cosine(
        df=df_te_co,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.cousine_fraud_prefix,
        real_prefix=args.cousine_real_prefix,
        projector=proj_co,
        projector_in_dim=in_dim_co,
        device=device,
        batch_size=args.pt_batch_size,
    )

    # Sanity label checks after feature extraction
    for name, y_other in [
        ("text train", y_tr_txt),
        ("deja train", y_tr_dj),
        ("unifont train", y_tr_uf),
        ("libre train", y_tr_li),
        ("doulos train", y_tr_do),
        ("cousine train", y_tr_co),
    ]:
        if not np.array_equal(y_tr, y_other):
            raise RuntimeError(f"Train labels mismatch after feature extraction: {name}")

    for name, y_other in [
        ("text test", y_te_txt),
        ("deja test", y_te_dj),
        ("unifont test", y_te_uf),
        ("libre test", y_te_li),
        ("doulos test", y_te_do),
        ("cousine test", y_te_co),
    ]:
        if not np.array_equal(y_te, y_other):
            raise RuntimeError(f"Test labels mismatch after feature extraction: {name}")

    # Final ensemble feature matrix:
    # 4 text-side features + 5 image-side cosine features
    X_tr = np.column_stack(
        [
            X_tr_txt,
            cos_tr_dj,
            cos_tr_uf,
            cos_tr_li,
            cos_tr_do,
            cos_tr_co,
        ]
    ).astype(np.float32, copy=False)

    X_te = np.column_stack(
        [
            X_te_txt,
            cos_te_dj,
            cos_te_uf,
            cos_te_li,
            cos_te_do,
            cos_te_co,
        ]
    ).astype(np.float32, copy=False)

    if args.model == "none":
        print("Model needed!")
        return

    if not _has_both_classes(y_tr) or not _has_both_classes(y_te):
        raise RuntimeError("Need both classes (0 and 1) in both train and test splits.")

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    model_output_dir = os.path.dirname(args.model_output_path)
    if model_output_dir:
        os.makedirs(model_output_dir, exist_ok=True)

    feature_names = [
        "text_cosine",
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
        "cosine_deja",
        "cosine_unifont",
        "cosine_libre",
        "cosine_doulos",
        "cosine_cousine",
    ]

    joblib.dump(
        {
            "model": clf,
            "model_type": args.model,
            "seed": args.seed,
            "feature_names": feature_names,
            "positive_label": args.positive_label,
            "fraud_col": args.fraud_col,
            "real_col": args.real_col,
            "label_col": args.label_col,
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

    print("Ensemble")
    print(
        f"model={args.model} test_auroc={test_auroc:.6f} "
        f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}"
    )


if __name__ == "__main__":
    main()

# =========================
# Example run
# =========================

"""
python3 for_paper/total_5f.py \
  --downloads-train ../Downloads/text_train.parquet \
  --downloads-test  ../Downloads/text_test.parquet \
  --downloads-pt    ../Downloads/single_run_model.pt \
  --deja-train      ../Deja/train_pairs_with_siglip_embeddings.parquet \
  --deja-test       ../Deja/test_pairs_with_siglip_embeddings.parquet \
  --deja-pt         ../Deja/single_run_model.pt \
  --unifont-train   ../Unifont/train_pairs_with_siglip_embeddings.parquet \
  --unifont-test    ../Unifont/test_pairs_with_siglip_embeddings.parquet \
  --unifont-pt      ../Unifont/single_run_model.pt \
  --libre-train     ../Libre/train_pairs_with_siglip_embeddings.parquet \
  --libre-test      ../Libre/test_pairs_with_siglip_embeddings.parquet \
  --libre-pt        ../Libre/single_run_model.pt \
  --doulos-train    ../Doulos/train_pairs_with_siglip_embeddings.parquet \
  --doulos-test     ../Doulos/test_pairs_with_siglip_embeddings.parquet \
  --doulos-pt       ../Doulos/single_run_model.pt \
  --cousine-train   ../Cousine/train_pairs_with_siglip_embeddings.parquet \
  --cousine-test    ../Cousine/test_pairs_with_siglip_embeddings.parquet \
  --cousine-pt      ../Cousine/single_run_model.pt \
  --model adaboost \
  --model-output-path saved_models/total_5f_model.joblib
"""
