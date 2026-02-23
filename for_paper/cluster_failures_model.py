#!/usr/bin/env python3
"""
cluster_from_model.py

Correct pipeline:

  1) Load VATE dataset
  2) Run student model
  3) Compute max cosine (excluding same-name)
  4) Compute Youden threshold (Option B eval mask)
  5) Define misclassified via Youden threshold
  6) Optionally filter FN / FP
  7) Cluster Δ = z_fraud - z_real
  8) Save clustered mistakes

NO joins.
NO external misclassified file.
"""

from __future__ import annotations
import argparse
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from text_to_image.siamese import SiameseEmbeddingModel


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
def pick_device(override: Optional[str]) -> torch.device:
    if override:
        d = torch.device(override)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------
def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns with prefix {prefix}")
    def key_fn(c):
        suf = c[len(prefix):]
        return int(suf) if re.fullmatch(r"-?\d+", suf) else 10**18
    return sorted(cols, key=key_fn)


def mat_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    return df[cols].to_numpy(dtype=np.float32, copy=True)


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------
def as_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj.get("model_state", obj)
    raise RuntimeError("Invalid checkpoint format.")


def infer_hidden_dim(sd: Dict[str, torch.Tensor], text_dim: int) -> int:
    candidates = []
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == text_dim:
            candidates.append(v.shape[0])
    if not candidates:
        raise RuntimeError("Cannot infer hidden_dim.")
    return max(set(candidates), key=candidates.count)


# ------------------------------------------------------------
# Youden threshold
# ------------------------------------------------------------
def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


# ------------------------------------------------------------
# Max cosine excluding same-name
# ------------------------------------------------------------
@torch.inference_mode()
def max_cos_excluding_name(
    zq: torch.Tensor,
    zk: torch.Tensor,
    key_ids: torch.Tensor,
    query_ids: torch.Tensor,
    batch_q: int,
    batch_k: int,
) -> np.ndarray:

    N = zq.shape[0]
    result = torch.full((N,), -1e9, device=zq.device)

    for i in range(0, N, batch_q):
        q = zq[i:i+batch_q]
        q_ids = query_ids[i:i+batch_q]
        best = torch.full((q.shape[0],), -1e9, device=zq.device)

        for j in range(0, zk.shape[0], batch_k):
            k = zk[j:j+batch_k]
            k_ids = key_ids[j:j+batch_k].view(1, -1)
            scores = q @ k.T
            mask = (q_ids.view(-1,1) == k_ids)
            scores = scores.masked_fill(mask, -1e9)
            vals, _ = torch.max(scores, dim=1)
            best = torch.maximum(best, vals)

        result[i:i+batch_q] = best

    return result.cpu().numpy()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@torch.inference_mode()
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--student-model-path", required=True)
    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--only", choices=["all","fn","fp"], default="all")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--out-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    df = pd.read_parquet(args.data)
    y = (df[args.label_col].to_numpy().astype(np.float32) >= 0.5).astype(np.int32)

    fraud_mat = mat_from_prefix(df, args.fraud_prefix)
    real_mat = mat_from_prefix(df, args.real_prefix)
    text_dim = fraud_mat.shape[1]

    # ----- Load model
    ckpt = torch.load(args.student_model_path, map_location=device)
    sd = as_state_dict(ckpt)
    hidden_dim = infer_hidden_dim(sd, text_dim)

    model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        image_dim=args.out_dim,
    ).to(device)

    model.load_state_dict(sd, strict=False)
    model.eval()

    # ----- Encode
    ft = torch.from_numpy(fraud_mat).to(device)
    rt = torch.from_numpy(real_mat).to(device)

    def encode(x):
        z = torch.empty((x.shape[0], args.out_dim), device=device)
        for i in range(0, x.shape[0], args.batch_size):
            z[i:i+args.batch_size] = F.normalize(
                model.encode_text(x[i:i+args.batch_size]),
                dim=1
            )
        return z

    zf = encode(ft)
    zr = encode(rt)

    # ----- Build name IDs
    names = df["real_name"].astype(str).to_numpy()
    norm = lambda s: unicodedata.normalize("NFKC", s).strip()
    name_to_id = {}
    ids = []
    for n in names:
        k = norm(n)
        if k not in name_to_id:
            name_to_id[k] = len(name_to_id)
        ids.append(name_to_id[k])
    key_ids = torch.tensor(ids, device=device)

    fraud_names = df["fraudulent_name"].astype(str).to_numpy()
    query_ids = torch.tensor(
        [name_to_id.get(norm(n), -1) for n in fraud_names],
        device=device
    )

    # ----- Score
    score = max_cos_excluding_name(
        zf, zr, key_ids, query_ids,
        batch_q=2048, batch_k=20000
    )
    score = np.clip(score, -1.0, 1.0)

    # ----- Option B eval mask
    exact_in_bank = (query_ids.cpu().numpy() != -1)
    eval_mask = ~(exact_in_bank & (y == 1))

    thr = youden_threshold(y[eval_mask], score[eval_mask])
    print(f"[INFO] Youden threshold = {thr:.9f}")

    pred = (score >= thr).astype(np.int32)
    mis_mask = (pred != y)

    # ----- Split FN/FP if requested
    if args.only == "fn":
        mis_mask &= (y == 1)
    elif args.only == "fp":
        mis_mask &= (y == 0)

    idx = np.where(mis_mask)[0]
    print(f"[INFO] misclassified selected = {len(idx)}")

    if len(idx) == 0:
        print("No mistakes found.")
        return

    # ----- Δ features
    delta = (zf - zr)[idx]
    norms = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
    direction = delta / norms
    feats = torch.cat([direction, norms], dim=1).cpu().numpy()

    # ----- PCA + UMAP + HDBSCAN (with noise filtering)
    import umap
    import hdbscan
    
    pca = PCA(n_components=min(args.pca_dim, feats.shape[1]))
    Zp = pca.fit_transform(feats)
    
    reducer = umap.UMAP(
        n_components=15,
        metric="cosine",
        random_state=42,
    )
    Zu = reducer.fit_transform(Zp)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=100,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(Zu)
    # ---- Ignore noise
    valid = labels != -1
    labels = labels[valid]
    idx = idx[valid]
    norms = norms[valid]

    # ----- Save
    out = df.iloc[idx].copy()
    out["score"] = score[idx]
    out["pred"] = pred[idx]
    out["threshold"] = thr
    out["margin"] = score[idx] - thr
    out["cluster_id"] = labels
    out["delta_norm"] = norms.cpu().numpy().reshape(-1)

    out.to_parquet(args.output, index=False)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
    
"""
python for_paper/cluster_failures_model.py \
  --data ../Downloads/vate_test.parquet \
  --student-model-path saved_models/deja_best_model.pt \
  --only fn \
  --k 8 \
  --output for_paper/deja_fn_clustered.parquet
"""