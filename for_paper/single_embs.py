#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

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
# PT loading + "golden model" projection
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
    # Common wrappers: "module." (DDP), "model."
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
    # Expect keys like head.0.weight and head.2.weight (as in your notebook cell)
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


@torch.inference_mode()
def cosine_from_optional_golden_model(
    fraud_mat: np.ndarray,
    real_mat: np.ndarray,
    golden_model_path: Optional[str],
    device: torch.device,
    pt_batch_size: int,
) -> np.ndarray:
    """
    If golden_model_path is None: cosine on raw mats.
    Else: transform both via golden model head(), L2-normalize, cosine on transformed.
    """
    a_np = fraud_mat.astype(np.float32, copy=False)
    b_np = real_mat.astype(np.float32, copy=False)

    if golden_model_path is None:
        dot = np.sum(a_np * b_np, axis=1)
        na = np.sqrt(np.sum(a_np * a_np, axis=1)) + 1e-12
        nb = np.sqrt(np.sum(b_np * b_np, axis=1)) + 1e-12
        return (dot / (na * nb)).astype(np.float32, copy=False)

    ckpt = load_checkpoint_safely(golden_model_path, map_location=torch.device("cpu"))
    sd = extract_state_dict(ckpt)
    sd = strip_known_prefixes(sd)

    in_dim, hidden_dim, out_dim = infer_dims_from_head(sd)

    if int(a_np.shape[1]) != int(in_dim) or int(b_np.shape[1]) != int(in_dim):
        raise RuntimeError(
            f"Embedding dim mismatch vs golden model: data_dim={a_np.shape[1]} but model_in_dim={in_dim}."
        )

    model = SiameseEmbeddingModel(in_dim, hidden_dim, out_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    n = int(a_np.shape[0])
    bs = max(1, int(pt_batch_size))

    fraud_t = torch.from_numpy(a_np)
    real_t = torch.from_numpy(b_np)

    sims = torch.empty((n,), device="cpu", dtype=torch.float32)

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        x1 = fraud_t[i0:i1].to(device=device, dtype=torch.float32, non_blocking=True)
        x2 = real_t[i0:i1].to(device=device, dtype=torch.float32, non_blocking=True)

        z1 = model.encode(x1)
        z2 = model.encode(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        s = F.cosine_similarity(z1, z2, dim=1).detach().to("cpu")
        sims[i0:i1] = s

    return sims.numpy().astype(np.float32, copy=False)


# =========================
# Features (cosine sim only)
# =========================
def build_features(
    df: pd.DataFrame,
    label_col: str,
    positive_label: int,
    fraud_prefix: str,
    real_prefix: str,
    golden_model_path: Optional[str],
    device: torch.device,
    pt_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

    cos = cosine_from_optional_golden_model(
        fraud_mat=fraud_mat,
        real_mat=real_mat,
        golden_model_path=golden_model_path,
        device=device,
        pt_batch_size=pt_batch_size,
    )

    feat_names = ["cosine_sim"]
    X = cos.reshape(-1, 1).astype(np.float32, copy=False)
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

    ap.add_argument("--golden_model", default=None, help="Optional .pt to transform embeddings before cosine_sim.")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--pt-batch-size", type=int, default=8192)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

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

    df_tr = load_table(args.train)
    df_te = load_table(args.test)

    X_tr, y_tr, feat_names = build_features(
        df_tr,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.fraud_prefix,
        real_prefix=args.real_prefix,
        golden_model_path=args.golden_model,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )
    X_te, y_te, _ = build_features(
        df_te,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.fraud_prefix,
        real_prefix=args.real_prefix,
        golden_model_path=args.golden_model,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )

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

    # -----------------------------
    # Ensemble
    #
    # Train on train, score on test
    # Threshold for accuracy is test-Youden (paper style)
    # -----------------------------
    if args.model == "none":
        print("Model needed!")
        return

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    s_te = model_scores(clf, X_te).astype(np.float64)
    test_auroc = float(roc_auc_score(y_te, s_te))

    thr_test = youden_threshold(y_te, s_te)  # Youden is on TEST (do not change)
    acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

    print()
    print("Ensemble")
    print(
        f"model={args.model} test_auroc={test_auroc:.6f} youden_thr_test={thr_test:.6f} "
        f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}"
    )


if __name__ == "__main__":
    main()

"""
python3 for_paper/single_embs.py \
  --train ../Downloads/text_train.parquet \
  --test  ../Downloads/text_test.parquet \
  --golden_model ../Downloads/single_run_model.pt \
  --model adaboost
"""

"""
python3 for_paper/single_embs.py \
  --train ../Unifont/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Unifont/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Unifont/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Cousine/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Cousine/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Cousine/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Libre/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Libre/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Libre/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Gentium/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Gentium/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Gentium/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Exo2/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Exo2/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Exo2/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Doulos/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Doulos/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Doulos/single_run_model.pt \
  --model adaboost

python3 for_paper/single_embs.py \
  --train ../Deja/train_pairs_with_siglip_embeddings.parquet \
  --test  ../Deja/test_pairs_with_siglip_embeddings.parquet \
  --golden_model ../Deja/single_run_model.pt \
  --model adaboost
"""