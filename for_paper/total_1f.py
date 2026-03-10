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

import difflib

HAVE_RAPIDFUZZ = False

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
# Column helpers
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
# String metrics
# =========================
def levenshtein_distance(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(rf_lev.distance(a, b))

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
    return float(rf_fuzz.token_set_ratio(a, b)) / 100.0


def partial_ratio(a: str, b: str) -> float:
    return float(rf_fuzz.partial_ratio(a, b)) / 100.0


# =========================
# Thresholding
# =========================
def _has_both_classes(y: np.ndarray) -> bool:
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
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
                return ckpt[k]
    raise RuntimeError("Unrecognized checkpoint format")


def strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out


def infer_dims_from_head(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    w0 = sd["head.0.weight"]
    w2 = sd["head.2.weight"]
    return int(w0.shape[1]), int(w0.shape[0]), int(w2.shape[0])


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


def load_golden_projector(pt_path: str, device: torch.device):
    ckpt = load_checkpoint_safely(pt_path, map_location=torch.device("cpu"))
    sd = strip_known_prefixes(extract_state_dict(ckpt))

    in_dim, hidden_dim, out_dim = infer_dims_from_head(sd)
    model = SiameseEmbeddingModel(in_dim, hidden_dim, out_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, in_dim


@torch.inference_mode()
def projected_cosine(model, in_dim, fraud_mat, real_mat, device, batch_size):

    a = fraud_mat.astype(np.float32)
    b = real_mat.astype(np.float32)

    n = a.shape[0]
    sims = np.zeros(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        x1 = torch.from_numpy(a[i:i+batch_size]).to(device)
        x2 = torch.from_numpy(b[i:i+batch_size]).to(device)

        z1 = F.normalize(model.encode(x1), dim=1)
        z2 = F.normalize(model.encode(x2), dim=1)

        sims[i:i+batch_size] = F.cosine_similarity(z1, z2).cpu().numpy()

    return sims


# =========================
# Text feature builder
# =========================
def build_text_features(df, fraud_col, real_col, text_cos):

    fraud = safe_str_list(df[fraud_col])
    real = safe_str_list(df[real_col])

    n = len(df)

    lev = np.zeros(n)
    tsr = np.zeros(n)
    pr = np.zeros(n)

    for i,(a,b) in enumerate(zip(fraud,real)):
        lev[i] = levenshtein_distance(a,b)
        tsr[i] = token_set_ratio(a,b)
        pr[i] = partial_ratio(a,b)

    lev_score = -lev

    return np.stack([text_cos, tsr, lev_score, pr], axis=1)


# =========================
# Ensemble
# =========================
def make_model(kind: str, seed: int):
    if kind=="adaboost":
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        return AdaBoostClassifier(estimator=base,n_estimators=300,learning_rate=0.05,random_state=seed)
    if kind=="gboost":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError


def model_scores(model, X):
    return model.predict_proba(X)[:,1]


# =========================
# Main
# =========================
def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--downloads-train",required=True)
    ap.add_argument("--downloads-test",required=True)
    ap.add_argument("--downloads-pt",required=True)

    ap.add_argument("--deja-train",required=True)
    ap.add_argument("--deja-test",required=True)
    ap.add_argument("--deja-pt",required=True)

    ap.add_argument("--fraud-col",default="fraudulent_name")
    ap.add_argument("--real-col",default="real_name")

    ap.add_argument("--label-col",default="label")
    ap.add_argument("--positive-label",type=int,default=1)

    ap.add_argument("--model",default="adaboost")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--model-output-path", default="saved_models/total_1f_model.joblib")

    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_tr_dl=load_table(args.downloads_train)
    df_te_dl=load_table(args.downloads_test)

    df_tr_dj=load_table(args.deja_train)
    df_te_dj=load_table(args.deja_test)

    proj_dl,in_dim_dl=load_golden_projector(args.downloads_pt,device)
    proj_dj,in_dim_dj=load_golden_projector(args.deja_pt,device)

    cos_tr_dl=projected_cosine(
        proj_dl,in_dim_dl,
        mat_from_prefix(df_tr_dl,"fraud_txt_emb_"),
        mat_from_prefix(df_tr_dl,"real_txt_emb_"),
        device,8192)

    cos_te_dl=projected_cosine(
        proj_dl,in_dim_dl,
        mat_from_prefix(df_te_dl,"fraud_txt_emb_"),
        mat_from_prefix(df_te_dl,"real_txt_emb_"),
        device,8192)

    text_tr=build_text_features(df_tr_dl,args.fraud_col,args.real_col,cos_tr_dl)
    text_te=build_text_features(df_te_dl,args.fraud_col,args.real_col,cos_te_dl)

    cos_tr_dj=projected_cosine(
        proj_dj,in_dim_dj,
        mat_from_prefix(df_tr_dj,"fraud_emb_"),
        mat_from_prefix(df_tr_dj,"real_emb_"),
        device,8192)

    cos_te_dj=projected_cosine(
        proj_dj,in_dim_dj,
        mat_from_prefix(df_te_dj,"fraud_emb_"),
        mat_from_prefix(df_te_dj,"real_emb_"),
        device,8192)

    X_tr=np.column_stack([text_tr,cos_tr_dj])
    X_te=np.column_stack([text_te,cos_te_dj])

    y_tr=(df_tr_dl[args.label_col].values==args.positive_label).astype(int)
    y_te=(df_te_dl[args.label_col].values==args.positive_label).astype(int)

    clf=make_model(args.model,args.seed)
    clf.fit(X_tr,y_tr)

    model_output_dir = os.path.dirname(args.model_output_path)
    if model_output_dir:
        os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(
        {
            "model": clf,
            "model_type": args.model,
            "seed": args.seed,
            "feature_names": [
                "text_cosine",
                "token_set_ratio",
                "levenshtein_distance_score",
                "partial_ratio",
                "cosine_deja",
            ],
            "positive_label": args.positive_label,
            "fraud_col": args.fraud_col,
            "real_col": args.real_col,
            "label_col": args.label_col,
        },
        args.model_output_path,
    )

    print(f"[OK] wrote saved model: {args.model_output_path}")

    s=model_scores(clf,X_te)

    yhat=clf.predict(X_te)

    acc=float(accuracy_score(y_te,yhat))
    prec=float(precision_score(y_te,yhat, zero_division=0))
    rec=float(recall_score(y_te,yhat, zero_division=0))

    print("Ensemble")
    print(
        f"test_auroc={roc_auc_score(y_te,s):.6f} "
        f"test_accuracy={acc:.6f} "
        f"test_precision={prec:.6f} "
        f"test_recall={rec:.6f}"
    )


if __name__=="__main__":
    main()
    
"""
python3 for_paper/total_1f.py \
  --downloads-train ../Downloads/text_train.parquet \
  --downloads-test  ../Downloads/text_test.parquet \
  --downloads-pt    ../Downloads/single_run_model.pt \
  --deja-train      ../Deja/train_pairs_with_siglip_embeddings.parquet \
  --deja-test       ../Deja/test_pairs_with_siglip_embeddings.parquet \
  --deja-pt         ../Deja/single_run_model.pt \
  --model adaboost \
  --model-output-path saved_models/total_1f_model.joblib
"""