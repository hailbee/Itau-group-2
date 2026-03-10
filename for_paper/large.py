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
# IO
# =========================
def load_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path} (expected .parquet/.pq or .csv)")


# =========================
# Column helpers (embeddings)
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


def cosine_feature_from_dataset(
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
        raise RuntimeError("Need both classes (0 and 1) in this split to compute AUROC/Youden.")

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

    # Downloads (always included)
    ap.add_argument("--downloads-train", required=True)
    ap.add_argument("--downloads-test", required=True)
    ap.add_argument("--downloads-pt", required=True)

    # Fonts (6)
    ap.add_argument("--deja-train", required=True)
    ap.add_argument("--deja-test", required=True)
    ap.add_argument("--deja-pt", required=True)

    ap.add_argument("--unifont-train", required=True)
    ap.add_argument("--unifont-test", required=True)
    ap.add_argument("--unifont-pt", required=True)

    ap.add_argument("--libre-train", required=True)
    ap.add_argument("--libre-test", required=True)
    ap.add_argument("--libre-pt", required=True)

    ap.add_argument("--exo2-train", required=True)
    ap.add_argument("--exo2-test", required=True)
    ap.add_argument("--exo2-pt", required=True)

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

    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    # Prefixes (Downloads differs)
    ap.add_argument("--downloads-fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--downloads-real-prefix", default="real_txt_emb_")

    # Font folders (default)
    ap.add_argument("--deja-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--deja-real-prefix", default="real_emb_")
    ap.add_argument("--unifont-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--unifont-real-prefix", default="real_emb_")
    ap.add_argument("--libre-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--libre-real-prefix", default="real_emb_")
    ap.add_argument("--exo2-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--exo2-real-prefix", default="real_emb_")
    ap.add_argument("--doulos-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--doulos-real-prefix", default="real_emb_")
    ap.add_argument("--cousine-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--cousine-real-prefix", default="real_emb_")
    ap.add_argument("--model-output-path", default="saved_models/large_model.joblib")

    args = ap.parse_args()

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

    # Load data
    df_tr_dl = load_table(args.downloads_train)
    df_te_dl = load_table(args.downloads_test)

    df_tr_dj = load_table(args.deja_train)
    df_te_dj = load_table(args.deja_test)

    df_tr_uf = load_table(args.unifont_train)
    df_te_uf = load_table(args.unifont_test)

    df_tr_li = load_table(args.libre_train)
    df_te_li = load_table(args.libre_test)

    df_tr_ex = load_table(args.exo2_train)
    df_te_ex = load_table(args.exo2_test)

    df_tr_do = load_table(args.doulos_train)
    df_te_do = load_table(args.doulos_test)

    df_tr_co = load_table(args.cousine_train)
    df_te_co = load_table(args.cousine_test)

    # Alignment checks (row-wise feature stacking)
    n_tr = len(df_tr_dl)
    n_te = len(df_te_dl)

    if (
        len(df_tr_dj) != n_tr
        or len(df_tr_uf) != n_tr
        or len(df_tr_li) != n_tr
        or len(df_tr_ex) != n_tr
        or len(df_tr_do) != n_tr
        or len(df_tr_co) != n_tr
    ):
        raise RuntimeError(
            "Train row-count mismatch: "
            f"Downloads={n_tr} Deja={len(df_tr_dj)} Unifont={len(df_tr_uf)} Libre={len(df_tr_li)} "
            f"Exo2={len(df_tr_ex)} Doulos={len(df_tr_do)} Cousine={len(df_tr_co)}"
        )

    if (
        len(df_te_dj) != n_te
        or len(df_te_uf) != n_te
        or len(df_te_li) != n_te
        or len(df_te_ex) != n_te
        or len(df_te_do) != n_te
        or len(df_te_co) != n_te
    ):
        raise RuntimeError(
            "Test row-count mismatch: "
            f"Downloads={n_te} Deja={len(df_te_dj)} Unifont={len(df_te_uf)} Libre={len(df_te_li)} "
            f"Exo2={len(df_te_ex)} Doulos={len(df_te_do)} Cousine={len(df_te_co)}"
        )

    for df in (
        df_tr_dl,
        df_te_dl,
        df_tr_dj,
        df_te_dj,
        df_tr_uf,
        df_te_uf,
        df_tr_li,
        df_te_li,
        df_tr_ex,
        df_te_ex,
        df_tr_do,
        df_te_do,
        df_tr_co,
        df_te_co,
    ):
        if args.label_col not in df.columns:
            raise RuntimeError(f"Missing label_col={args.label_col!r} in one of the tables.")

    # Ensure labels match exactly across all tables (same row order)
    y_tr = (df_tr_dl[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
    y_te = (df_te_dl[args.label_col].to_numpy() == args.positive_label).astype(np.int32)

    for name, df in [
        ("Deja train", df_tr_dj),
        ("Unifont train", df_tr_uf),
        ("Libre train", df_tr_li),
        ("Exo2 train", df_tr_ex),
        ("Doulos train", df_tr_do),
        ("Cousine train", df_tr_co),
    ]:
        y = (df[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
        if not np.array_equal(y_tr, y):
            raise RuntimeError(f"Train labels mismatch: Downloads vs {name} (row order / dataset mismatch).")

    for name, df in [
        ("Deja test", df_te_dj),
        ("Unifont test", df_te_uf),
        ("Libre test", df_te_li),
        ("Exo2 test", df_te_ex),
        ("Doulos test", df_te_do),
        ("Cousine test", df_te_co),
    ]:
        y = (df[args.label_col].to_numpy() == args.positive_label).astype(np.int32)
        if not np.array_equal(y_te, y):
            raise RuntimeError(f"Test labels mismatch: Downloads vs {name} (row order / dataset mismatch).")

    # Load projectors (no switching / no overlap)
    proj_dl, in_dim_dl = load_golden_projector(args.downloads_pt, device=device)
    proj_dj, in_dim_dj = load_golden_projector(args.deja_pt, device=device)
    proj_uf, in_dim_uf = load_golden_projector(args.unifont_pt, device=device)
    proj_li, in_dim_li = load_golden_projector(args.libre_pt, device=device)
    proj_ex, in_dim_ex = load_golden_projector(args.exo2_pt, device=device)
    proj_do, in_dim_do = load_golden_projector(args.doulos_pt, device=device)
    proj_co, in_dim_co = load_golden_projector(args.cousine_pt, device=device)

    # Compute features (one cosine per source)
    cos_tr_dl, _ = cosine_feature_from_dataset(
        df=df_tr_dl,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.downloads_fraud_prefix,
        real_prefix=args.downloads_real_prefix,
        projector=proj_dl,
        projector_in_dim=in_dim_dl,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_dl, _ = cosine_feature_from_dataset(
        df=df_te_dl,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.downloads_fraud_prefix,
        real_prefix=args.downloads_real_prefix,
        projector=proj_dl,
        projector_in_dim=in_dim_dl,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_dj, _ = cosine_feature_from_dataset(
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
    cos_te_dj, _ = cosine_feature_from_dataset(
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

    cos_tr_uf, _ = cosine_feature_from_dataset(
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
    cos_te_uf, _ = cosine_feature_from_dataset(
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

    cos_tr_li, _ = cosine_feature_from_dataset(
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
    cos_te_li, _ = cosine_feature_from_dataset(
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

    cos_tr_ex, _ = cosine_feature_from_dataset(
        df=df_tr_ex,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.exo2_fraud_prefix,
        real_prefix=args.exo2_real_prefix,
        projector=proj_ex,
        projector_in_dim=in_dim_ex,
        device=device,
        batch_size=args.pt_batch_size,
    )
    cos_te_ex, _ = cosine_feature_from_dataset(
        df=df_te_ex,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.exo2_fraud_prefix,
        real_prefix=args.exo2_real_prefix,
        projector=proj_ex,
        projector_in_dim=in_dim_ex,
        device=device,
        batch_size=args.pt_batch_size,
    )

    cos_tr_do, _ = cosine_feature_from_dataset(
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
    cos_te_do, _ = cosine_feature_from_dataset(
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

    cos_tr_co, _ = cosine_feature_from_dataset(
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
    cos_te_co, _ = cosine_feature_from_dataset(
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

    # Build feature matrices (7 cosine similarities: Downloads + 6 fonts)
    X_tr = np.stack([cos_tr_dl, cos_tr_dj, cos_tr_uf, cos_tr_li, cos_tr_ex, cos_tr_do, cos_tr_co], axis=1).astype(
        np.float32, copy=False
    )
    X_te = np.stack([cos_te_dl, cos_te_dj, cos_te_uf, cos_te_li, cos_te_ex, cos_te_do, cos_te_co], axis=1).astype(
        np.float32, copy=False
    )
    feat_names = [
        "cosine_downloads",
        "cosine_deja",
        "cosine_unifont",
        "cosine_libre",
        "cosine_exo2",
        "cosine_doulos",
        "cosine_cousine",
    ]

    """
    # -----------------------------
    # Single-feature (test-Youden, paper style)
    # -----------------------------
    rows: List[SingleFeatureRow] = []
    for j, name in enumerate(feat_names):
        s_te = X_te[:, j].astype(np.float64)
        test_auroc = float(roc_auc_score(y_te, s_te))
        thr_test = youden_threshold(y_te, s_te)  # Youden is on TEST (do not change)
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
    """

    # -----------------------------
    # Ensemble (7 cosine features)
    # Default classification strategy for accuracy/precision/recall
    # -----------------------------
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

    joblib.dump(
        {
            "model": clf,
            "model_type": args.model,
            "seed": args.seed,
            "feature_names": feat_names,
            "positive_label": args.positive_label,
            "label_col": args.label_col,
            "downloads_fraud_prefix": args.downloads_fraud_prefix,
            "downloads_real_prefix": args.downloads_real_prefix,
            "deja_fraud_prefix": args.deja_fraud_prefix,
            "deja_real_prefix": args.deja_real_prefix,
            "unifont_fraud_prefix": args.unifont_fraud_prefix,
            "unifont_real_prefix": args.unifont_real_prefix,
            "libre_fraud_prefix": args.libre_fraud_prefix,
            "libre_real_prefix": args.libre_real_prefix,
            "exo2_fraud_prefix": args.exo2_fraud_prefix,
            "exo2_real_prefix": args.exo2_real_prefix,
            "doulos_fraud_prefix": args.doulos_fraud_prefix,
            "doulos_real_prefix": args.doulos_real_prefix,
            "cousine_fraud_prefix": args.cousine_fraud_prefix,
            "cousine_real_prefix": args.cousine_real_prefix,
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
python3 for_paper/large.py \
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
  --exo2-train      ../Exo2/train_pairs_with_siglip_embeddings.parquet \
  --exo2-test       ../Exo2/test_pairs_with_siglip_embeddings.parquet \
  --exo2-pt         ../Exo2/single_run_model.pt \
  --doulos-train    ../Doulos/train_pairs_with_siglip_embeddings.parquet \
  --doulos-test     ../Doulos/test_pairs_with_siglip_embeddings.parquet \
  --doulos-pt       ../Doulos/single_run_model.pt \
  --cousine-train   ../Cousine/train_pairs_with_siglip_embeddings.parquet \
  --cousine-test    ../Cousine/test_pairs_with_siglip_embeddings.parquet \
  --cousine-pt      ../Cousine/single_run_model.pt \
  --model adaboost \
  --model-output-path saved_models/large_model.joblib
"""