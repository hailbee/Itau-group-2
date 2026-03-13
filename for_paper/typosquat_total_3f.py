#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# =========================
# Paths / config
# =========================
DOWNLOADS_TEST = "../Downloads/typosquat_text_test.parquet"
DOWNLOADS_PT = "../Downloads/single_run_model.pt"

DEJA_TEST = "../Deja/typosquat_test_pairs_with_siglip_embeddings.parquet"
DEJA_PT = "../Deja/single_run_model.pt"

GENTIUM_TEST = "../Gentium/typosquat_test_pairs_with_siglip_embeddings.parquet"
GENTIUM_PT = "../Gentium/single_run_model.pt"

UNIFONT_TEST = "../Unifont/typosquat_test_pairs_with_siglip_embeddings.parquet"
UNIFONT_PT = "../Unifont/single_run_model.pt"

SAVED_MODEL_PATH = "saved_models/total_3f_model.joblib"
ERROR_OUTPUT_DIR = "for_paper/error_outputs"

DEVICE = None
PT_BATCH_SIZE = 8192

FRAUD_COL = "fraudulent_name"
REAL_COL = "real_name"
LABEL_COL = "label"
POSITIVE_LABEL = 1

DOWNLOADS_FRAUD_PREFIX = "fraud_txt_emb_"
DOWNLOADS_REAL_PREFIX = "real_txt_emb_"

DEJA_FRAUD_PREFIX = "fraud_emb_"
DEJA_REAL_PREFIX = "real_emb_"
GENTIUM_FRAUD_PREFIX = "fraud_emb_"
GENTIUM_REAL_PREFIX = "real_emb_"
UNIFONT_FRAUD_PREFIX = "fraud_emb_"
UNIFONT_REAL_PREFIX = "real_emb_"


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
        from fuzzywuzzy import fuzz as fw_fuzz

        HAVE_FUZZYWUZZY = True
    except Exception:
        HAVE_FUZZYWUZZY = False

try:
    import Levenshtein as py_lev

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
# Basic checks
# =========================
def _has_both_classes(y: np.ndarray) -> bool:
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


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
        raise RuntimeError("Need both classes (0 and 1) in this split.")

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
        raise RuntimeError("Need both classes (0 and 1) in this split.")

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
# Saved model loading
# =========================
def load_saved_model_bundle(path: str) -> Tuple[Any, List[str] | None]:
    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feature_names = obj.get("feature_names", None)
    else:
        model = obj
        feature_names = None

    if feature_names is not None:
        if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
            raise RuntimeError("Saved model bundle has invalid feature_names.")

    return model, feature_names


def model_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = 1.0 / (1.0 + np.exp(-s))
        return s.astype(np.float32)
    raise RuntimeError("Model does not support predict_proba or decision_function.")


# =========================
# Threshold search
# =========================
def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(thr),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold_max_precision_under_recall_constraint(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_recall: float,
) -> Dict[str, float]:
    thresholds = np.unique(scores.astype(np.float64))
    candidates: List[Dict[str, float]] = []

    for thr in thresholds:
        metrics = evaluate_at_threshold(y_true, scores, float(thr))
        if metrics["recall"] >= min_recall:
            candidates.append(metrics)

    if not candidates:
        raise RuntimeError(f"No threshold achieved recall >= {min_recall:.3f}")

    candidates.sort(
        key=lambda d: (
            d["precision"],
            d["recall"],
            d["accuracy"],
            d["threshold"],
        ),
        reverse=True,
    )
    return candidates[0]


# =========================
# Error output
# =========================
def save_error_lists(
    df_text: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    fn_filename: str,
    fp_filename: str,
    fraud_col: str,
    real_col: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    fn_path = os.path.join(out_dir, fn_filename)
    fp_path = os.path.join(out_dir, fp_filename)

    false_negatives = df_text[(y_true == 1) & (y_pred == 0)][[fraud_col, real_col]]
    false_positives = df_text[(y_true == 0) & (y_pred == 1)][[fraud_col, real_col]]

    with open(fn_path, "w", encoding="utf-8") as f:
        for _, row in false_negatives.iterrows():
            f.write(f"{row[fraud_col]},{row[real_col]}\n")

    with open(fp_path, "w", encoding="utf-8") as f:
        for _, row in false_positives.iterrows():
            f.write(f"{row[fraud_col]},{row[real_col]}\n")

    return fn_path, fp_path


def write_summary_file(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# =========================
# Main
# =========================
def main() -> None:
    if not (HAVE_RAPIDFUZZ or HAVE_FUZZYWUZZY):
        raise RuntimeError("You need rapidfuzz (recommended) or fuzzywuzzy installed for token_set_ratio/partial_ratio.")

    if DEVICE is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(DEVICE)

    df_te_txt = load_table(DOWNLOADS_TEST)
    df_te_dj = load_table(DEJA_TEST)
    df_te_ge = load_table(GENTIUM_TEST)
    df_te_uf = load_table(UNIFONT_TEST)

    for df in (df_te_txt, df_te_dj, df_te_ge, df_te_uf):
        if LABEL_COL not in df.columns:
            raise RuntimeError(f"Missing label_col={LABEL_COL!r} in one of the tables.")
        if FRAUD_COL not in df.columns:
            raise RuntimeError(f"Missing fraud_col={FRAUD_COL!r} in one of the tables.")
        if REAL_COL not in df.columns:
            raise RuntimeError(f"Missing real_col={REAL_COL!r} in one of the tables.")

    n_te = len(df_te_txt)
    if (
        len(df_te_dj) != n_te
        or len(df_te_ge) != n_te
        or len(df_te_uf) != n_te
    ):
        raise RuntimeError(
            "Test row-count mismatch: "
            f"Downloads={n_te} Deja={len(df_te_dj)} Gentium={len(df_te_ge)} Unifont={len(df_te_uf)}"
        )

    y_te = (df_te_txt[LABEL_COL].to_numpy() == POSITIVE_LABEL).astype(np.int32)

    for name, df in [
        ("Deja test", df_te_dj),
        ("Gentium test", df_te_ge),
        ("Unifont test", df_te_uf),
    ]:
        y = (df[LABEL_COL].to_numpy() == POSITIVE_LABEL).astype(np.int32)
        if not np.array_equal(y_te, y):
            raise RuntimeError(f"Test labels mismatch: Downloads vs {name}")

    if not _has_both_classes(y_te):
        raise RuntimeError("Need both classes (0 and 1) in test split.")

    clf, saved_feature_names = load_saved_model_bundle(SAVED_MODEL_PATH)

    proj_txt, in_dim_txt = load_golden_projector(DOWNLOADS_PT, device=device)
    proj_dj, in_dim_dj = load_golden_projector(DEJA_PT, device=device)
    proj_ge, in_dim_ge = load_golden_projector(GENTIUM_PT, device=device)
    proj_uf, in_dim_uf = load_golden_projector(UNIFONT_PT, device=device)

    X_te_txt, y_te_txt = build_text_features(
        df=df_te_txt,
        fraud_col=FRAUD_COL,
        real_col=REAL_COL,
        label_col=LABEL_COL,
        positive_label=POSITIVE_LABEL,
        fraud_prefix=DOWNLOADS_FRAUD_PREFIX,
        real_prefix=DOWNLOADS_REAL_PREFIX,
        projector=proj_txt,
        projector_in_dim=in_dim_txt,
        device=device,
        pt_batch_size=PT_BATCH_SIZE,
    )

    cos_te_dj, y_te_dj = build_single_font_cosine(
        df=df_te_dj,
        label_col=LABEL_COL,
        positive_label=POSITIVE_LABEL,
        fraud_prefix=DEJA_FRAUD_PREFIX,
        real_prefix=DEJA_REAL_PREFIX,
        projector=proj_dj,
        projector_in_dim=in_dim_dj,
        device=device,
        batch_size=PT_BATCH_SIZE,
    )
    cos_te_ge, y_te_ge = build_single_font_cosine(
        df=df_te_ge,
        label_col=LABEL_COL,
        positive_label=POSITIVE_LABEL,
        fraud_prefix=GENTIUM_FRAUD_PREFIX,
        real_prefix=GENTIUM_REAL_PREFIX,
        projector=proj_ge,
        projector_in_dim=in_dim_ge,
        device=device,
        batch_size=PT_BATCH_SIZE,
    )
    cos_te_uf, y_te_uf = build_single_font_cosine(
        df=df_te_uf,
        label_col=LABEL_COL,
        positive_label=POSITIVE_LABEL,
        fraud_prefix=UNIFONT_FRAUD_PREFIX,
        real_prefix=UNIFONT_REAL_PREFIX,
        projector=proj_uf,
        projector_in_dim=in_dim_uf,
        device=device,
        batch_size=PT_BATCH_SIZE,
    )

    for name, y_other in [
        ("text test", y_te_txt),
        ("deja test", y_te_dj),
        ("gentium test", y_te_ge),
        ("unifont test", y_te_uf),
    ]:
        if not np.array_equal(y_te, y_other):
            raise RuntimeError(f"Test labels mismatch after feature extraction: {name}")

    available_features: Dict[str, np.ndarray] = {
        "text_cosine": X_te_txt[:, 0],
        "token_set_ratio": X_te_txt[:, 1],
        "levenshtein_distance_score": X_te_txt[:, 2],
        "partial_ratio": X_te_txt[:, 3],
        "cosine_deja": cos_te_dj,
        "cosine_gentium": cos_te_ge,
        "cosine_unifont": cos_te_uf,
    }

    default_feature_names = [
        "text_cosine",
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
        "cosine_deja",
        "cosine_gentium",
        "cosine_unifont",
    ]

    if saved_feature_names is None:
        if hasattr(clf, "n_features_in_"):
            expected = int(clf.n_features_in_)
            if expected != len(default_feature_names):
                raise RuntimeError(
                    "Saved model does not include feature_names, and its expected feature count "
                    f"({expected}) does not match the default feature count ({len(default_feature_names)})."
                )
        feature_names = default_feature_names
    else:
        feature_names = list(saved_feature_names)
        if hasattr(clf, "n_features_in_") and int(clf.n_features_in_) != len(feature_names):
            raise RuntimeError(
                f"Saved model expects {int(clf.n_features_in_)} features but bundle feature_names has {len(feature_names)}."
            )

    missing = [f for f in feature_names if f not in available_features]
    if missing:
        raise RuntimeError(f"Saved model requires unknown features: {missing}")

    X_te = np.column_stack([available_features[f] for f in feature_names]).astype(np.float32, copy=False)

    scores = model_scores(clf, X_te).astype(np.float64)
    test_auroc = float(roc_auc_score(y_te, scores))

    targets = [0.975, 0.95, 0.925, 0.9]
    lines: List[str] = []
    lines.append("ensemble_best:")
    lines.append("")

    for target in targets:
        best = choose_threshold_max_precision_under_recall_constraint(y_te, scores, min_recall=target)
        lines.append(f"recall ≥ {target:.3f}, maximize precision:")
        lines.append(
            f"test_auroc={test_auroc:.6f} "
            f"threshold={best['threshold']:.6f} "
            f"accuracy={best['accuracy']:.6f} "
            f"precision={best['precision']:.6f} "
            f"recall={best['recall']:.6f} "
            f"(FPR = {best['fpr']:.6f})"
        )
        lines.append("")

        if abs(target - 0.975) < 1e-12:
            y_pred_0975 = (scores >= best["threshold"]).astype(np.int32)
            fn_path, fp_path = save_error_lists(
                df_text=df_te_txt,
                y_true=y_te,
                y_pred=y_pred_0975,
                out_dir=ERROR_OUTPUT_DIR,
                fn_filename="ensemble_best_false_negatives_recall_ge_0.975.txt",
                fp_filename="ensemble_best_false_positives_recall_ge_0.975.txt",
                fraud_col=FRAUD_COL,
                real_col=REAL_COL,
            )
            lines.append(f"saved false negatives: {fn_path}")
            lines.append(f"saved false positives: {fp_path}")
            lines.append("")

    os.makedirs(ERROR_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(ERROR_OUTPUT_DIR, "typosquat_total_3f_metrics.txt")
    write_summary_file(summary_path, lines)
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()

"""
python3 for_paper/typosquat_total_3f.py
"""