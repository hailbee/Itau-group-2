#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
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
# PT loading + "golden model" projection for cosine
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
# Multi-hot mechanism flags
# EXACT criteria copied from mechanism_multihot_audit.py
# =========================

COMMON_SUFFIXES: set[str] = {
    "com", "net", "org", "io", "co", "gov", "edu", "us", "uk", "de", "fr", "jp", "br", "ru", "cn", "in",
    "info", "biz", "app", "dev", "ai", "me", "tv",
    "exe", "dll", "sys", "scr", "bat", "cmd", "ps1", "vbs", "js", "jar", "msi",
}

SEPARATORS: set[str] = set(" \t\r\n-_./\\:·•—–‐-‒―")

INVISIBLE_CODEPOINTS: set[str] = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u00ad",
    "\u2060",
    "\u180e",
}

CONFUSABLE_CHAR_MAP: Dict[str, str] = {
    "ł": "l", "Ł": "l",
    "ø": "o", "Ø": "o",
    "đ": "d", "Đ": "d",
    "ı": "i",
    "а": "a", "А": "a",
    "е": "e", "Е": "e",
    "о": "o", "О": "o",
    "р": "p", "Р": "p",
    "с": "c", "С": "c",
    "х": "x", "Х": "x",
    "у": "y", "У": "y",
    "к": "k", "К": "k",
    "м": "m", "М": "m",
    "т": "t", "Т": "t",
    "н": "h", "Н": "h",
    "і": "i", "І": "i",
    "α": "a", "Α": "a",
    "ο": "o", "Ο": "o",
    "ρ": "p", "Ρ": "p",
    "χ": "x", "Χ": "x",
    "ν": "v", "Ν": "v",
}

MULTICHAR_CONFUSABLES: List[Tuple[str, str]] = [
    ("rn", "m"),
    ("cl", "d"),
    ("vv", "w"),
    ("l1", "h"),
    ("1l", "h"),
]

LEET_MAP: Dict[str, str] = {
    "0": "o",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "1": "l",
}
LEET_CHARS: set[str] = set(LEET_MAP.keys())

FLAG_KEYS: List[str] = [
    "punycode",
    "non_ascii",
    "unicode_homoglyph",
    "hyphen_change",
    "digit_change",
    "transposition",
    "insertion",
    "deletion",
    "substitution",
    "digit_substitution",
    "mixed_script",
    "separator_change",
    "repeat_char",
    "affix",
    "numeric_affix",
    "visual_confusable",
    "multichar_confusable",
    "leet_pair",
]


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).strip()


def _strip_marks(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) >= 128 for ch in s)


def _contains_invisible_or_format(s: str) -> bool:
    for ch in s:
        if ch in INVISIBLE_CODEPOINTS:
            return True
        if unicodedata.category(ch) == "Cf":
            return True
    return False


def _script_of_char(ch: str) -> str:
    if not ch.isalpha():
        return "NA"
    name = unicodedata.name(ch, "")
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "GREEK" in name:
        return "GREEK"
    if "LATIN" in name:
        return "LATIN"
    if "ARABIC" in name:
        return "ARABIC"
    if "HEBREW" in name:
        return "HEBREW"
    if "DEVANAGARI" in name:
        return "DEVANAGARI"
    if "HANGUL" in name:
        return "HANGUL"
    if "HIRAGANA" in name or "KATAKANA" in name or "CJK UNIFIED IDEOGRAPH" in name:
        return "CJK"
    return "OTHER"


def _has_mixed_script(s: str) -> bool:
    scripts = set()
    for ch in s:
        sc = _script_of_char(ch)
        if sc not in ("NA", "OTHER"):
            scripts.add(sc)
        if len(scripts) >= 2:
            return True
    return False


def _is_single_adjacent_swap(a: str, b: str) -> bool:
    if len(a) != len(b) or a == b:
        return False
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) != 2:
        return False
    i, j = diffs
    if j != i + 1:
        return False
    aa = list(a)
    aa[i], aa[j] = aa[j], aa[i]
    return "".join(aa) == b


def _diff_ops(real_s: str, fraud_s: str) -> Tuple[int, int, int]:
    sm = difflib.SequenceMatcher(a=real_s, b=fraud_s)
    n_ins = 0
    n_del = 0
    n_rep = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            n_ins += (j2 - j1)
        elif tag == "delete":
            n_del += (i2 - i1)
        elif tag == "replace":
            n_rep += max(i2 - i1, j2 - j1)
    return n_ins, n_del, n_rep


def _remove_separators(s: str) -> str:
    return "".join(ch for ch in s if (ch not in SEPARATORS and not ch.isspace()))


def _collapse_runs(s: str) -> str:
    if not s:
        return s
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _affix_extra(a: str, b: str) -> Tuple[bool, str]:
    if not a or not b or a == b:
        return False, ""
    if len(a) < len(b):
        a, b = b, a

    MAX_EXTRA = 10
    if len(a) - len(b) > MAX_EXTRA:
        return False, ""

    if a.startswith(b):
        return True, a[len(b):]
    if a.endswith(b):
        return True, a[: len(a) - len(b)]
    return False, ""


def _multichar_skeleton(s: str) -> str:
    t = s.casefold()
    for pat, rep in MULTICHAR_CONFUSABLES:
        t = t.replace(pat, rep)
    return t


def _confusable_skeleton(s: str) -> str:
    t = _multichar_skeleton(s)
    out_chars: List[str] = []
    for ch in t:
        out_chars.append(CONFUSABLE_CHAR_MAP.get(ch, ch))
    return "".join(out_chars)


def _is_strict_leet_pair(f: str, r: str) -> bool:
    f_cf = f.casefold()
    r_cf = r.casefold()
    if not f_cf or not r_cf or len(f_cf) != len(r_cf):
        return False

    used = False
    for fc, rc in zip(f_cf, r_cf):
        if fc == rc:
            continue
        if fc in LEET_MAP and LEET_MAP[fc] == rc:
            used = True
            continue
        return False
    return used


def mechanism_flags(fraudulent: str, real: str) -> Dict[str, int]:
    out: Dict[str, int] = {k: 0 for k in FLAG_KEYS}

    f_raw = str(fraudulent or "")
    r_raw = str(real or "")

    f = _nfkc(f_raw)
    r = _nfkc(r_raw)

    if not f and not r:
        return out

    out["punycode"] = int(("xn--" in f) or ("xn--" in r))
    out["non_ascii"] = int(_has_non_ascii(f) or _has_non_ascii(r))
    out["hyphen_change"] = int(("-" in f) ^ ("-" in r))
    out["digit_change"] = int((any(ch.isdigit() for ch in f)) ^ (any(ch.isdigit() for ch in r)))

    f_stripped = _strip_marks(f)
    r_stripped = _strip_marks(r)
    if (f_stripped.casefold() == r_stripped.casefold()) and (f.casefold() != r.casefold()):
        if out["non_ascii"]:
            out["unicode_homoglyph"] = 1

    out["transposition"] = int(_is_single_adjacent_swap(r, f) or _is_single_adjacent_swap(f, r))

    n_ins, n_del, n_rep = _diff_ops(r.casefold(), f.casefold())
    out["insertion"] = int(n_ins > 0)
    out["deletion"] = int(n_del > 0)
    out["substitution"] = int(n_rep > 0)

    if len(f) == len(r) and f != r:
        diffs = [(a, b) for a, b in zip(f, r) if a != b]
        if diffs:
            digitish = 0
            for a, b in diffs:
                if (a.isdigit() and b.isdigit()) or (a.isdigit() and not b.isdigit()) or (b.isdigit() and not a.isdigit()):
                    digitish += 1
            if digitish >= 1 and digitish == len(diffs) and len(diffs) <= 8:
                out["digit_substitution"] = 1

    out["mixed_script"] = int(_has_mixed_script(f) or _has_mixed_script(r))

    f_nosep = _remove_separators(f).casefold()
    r_nosep = _remove_separators(r).casefold()
    if f != r and f_nosep and (f_nosep == r_nosep) and (
        any(ch in SEPARATORS or ch.isspace() for ch in f) or any(ch in SEPARATORS or ch.isspace() for ch in r)
    ):
        out["separator_change"] = 1

    f_coll = _collapse_runs(f).casefold()
    r_coll = _collapse_runs(r).casefold()
    if f != r and f_coll and (f_coll == r_coll) and (len(f) != len(r)):
        out["repeat_char"] = 1

    is_aff, extra = _affix_extra(f.casefold(), r.casefold())
    out["affix"] = int(is_aff)
    if is_aff:
        extra_compact = _remove_separators(extra)
        out["numeric_affix"] = int(extra_compact.isdigit() and len(extra_compact) > 0)

    if f != r and (f.casefold() != r.casefold()):
        sk_f = _confusable_skeleton(f)
        sk_r = _confusable_skeleton(r)
        if sk_f == sk_r and sk_f != "":
            out["visual_confusable"] = 1
        if _multichar_skeleton(f) == _multichar_skeleton(r) and _multichar_skeleton(f) != "":
            out["multichar_confusable"] = 1

    if f != r and (f.casefold() != r.casefold()):
        f_has_leet = any(ch in LEET_CHARS for ch in f.casefold())
        r_has_leet = any(ch in LEET_CHARS for ch in r.casefold())
        if f_has_leet and (not r_has_leet) and _is_strict_leet_pair(f, r):
            out["leet_pair"] = 1

    return out


def mechanism_flags_df(fraud_series: Sequence[str], real_series: Sequence[str]) -> pd.DataFrame:
    rows = [mechanism_flags(f, r) for f, r in zip(fraud_series, real_series)]
    return pd.DataFrame(rows, columns=FLAG_KEYS).fillna(0).astype(int)


def combo_id(flags_row: Dict[str, int] | pd.Series) -> int:
    total = 0
    for i, k in enumerate(FLAG_KEYS):
        v = int(flags_row[k]) if k in flags_row else 0
        if v:
            total |= (1 << i)
    return total


def combo_label(flags_row: Dict[str, int] | pd.Series) -> str:
    keys = [k for k in FLAG_KEYS if int(flags_row[k]) == 1]
    return "+".join(keys) if keys else "NONE"


# =========================
# Features (cosine + 3 text metrics)
# =========================
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

    feat_names = [
        "cosine_sim",
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
    ]
    X = np.stack([cos, tsr, lev_dist_score, pr], axis=1).astype(np.float32, copy=False)
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
# Error analysis outputs
# =========================
def build_positive_mechanism_tables(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_col: str,
    real_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_mask = y_true == 1
    df_pos = df_test.loc[pos_mask].copy().reset_index(drop=True)

    flags = mechanism_flags_df(
        df_pos[fraud_col].astype(str).tolist(),
        df_pos[real_col].astype(str).tolist(),
    ).reset_index(drop=True)

    df_pos = pd.concat([df_pos.reset_index(drop=True), flags], axis=1)
    df_pos["mech_combo_id"] = [combo_id(r) for _, r in flags.iterrows()]
    df_pos["mech_combo"] = [combo_label(r) for _, r in flags.iterrows()]
    df_pos["mech_all_zero"] = (flags.sum(axis=1) == 0).astype(int)

    y_pred_pos = y_pred[pos_mask]
    df_pos["pred_label"] = y_pred_pos.astype(np.int32)
    df_pos["is_false_negative"] = ((df_pos["pred_label"] == 0) & (df_pos["label"] == 1)).astype(int)

    df_fn = df_pos.loc[df_pos["is_false_negative"] == 1].copy().reset_index(drop=True)
    return df_pos, df_fn


def build_mechanism_error_summary(df_pos: pd.DataFrame, df_fn: pd.DataFrame) -> pd.DataFrame:
    total_positive = int(len(df_pos))
    rows: List[Dict[str, float | int | str]] = []

    for k in FLAG_KEYS:
        n_pos_with_type = int(df_pos[k].sum())
        n_fn_with_type = int(df_fn[k].sum())

        prevalence_among_positive = (n_pos_with_type / total_positive) if total_positive > 0 else 0.0
        error_rate_within_type = (n_fn_with_type / n_pos_with_type) if n_pos_with_type > 0 else 0.0

        rows.append(
            {
                "mechanism": k,
                "n_positive_with_type": n_pos_with_type,
                "n_false_negative_with_type": n_fn_with_type,
                "prevalence_among_positive": prevalence_among_positive,
                "prevalence_among_positive_pct": 100.0 * prevalence_among_positive,
                "false_negative_rate_within_type": error_rate_within_type,
                "false_negative_rate_within_type_pct": 100.0 * error_rate_within_type,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["n_false_negative_with_type", "n_positive_with_type", "mechanism"],
        ascending=[False, False, True],
    )
    return summary


def save_mechanism_summary_txt(
    summary_df: pd.DataFrame,
    total_positive: int,
    total_false_negative: int,
    out_txt: str,
) -> None:
    lines: List[str] = []
    lines.append(f"total_positive_test_examples={total_positive}")
    lines.append(f"total_false_negative_test_examples={total_false_negative}")
    lines.append("")
    lines.append(
        "mechanism | n_positive_with_type | n_false_negative_with_type | "
        "prevalence_among_positive_pct | false_negative_rate_within_type_pct"
    )

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['mechanism']} | "
            f"{int(row['n_positive_with_type'])} | "
            f"{int(row['n_false_negative_with_type'])} | "
            f"{float(row['prevalence_among_positive_pct']):.6f} | "
            f"{float(row['false_negative_rate_within_type_pct']):.6f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_scatter_plot(summary_df: pd.DataFrame, out_png: str) -> None:
    x = summary_df["prevalence_among_positive_pct"].to_numpy(dtype=float)
    y = summary_df["false_negative_rate_within_type_pct"].to_numpy(dtype=float)
    labels = summary_df["mechanism"].astype(str).tolist()

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)

    plt.xlabel("Proportion of label-1 examples with mechanism (%)")
    plt.ylabel("Misclassification rate within mechanism (%)")
    plt.title("Text ensemble false-negative rate by spoof classification")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def save_grouped_bar_chart(summary_df: pd.DataFrame, out_png: str) -> None:
    labels = summary_df["mechanism"].astype(str).tolist()
    prevalence = summary_df["prevalence_among_positive_pct"].to_numpy(dtype=float)
    error_rate = summary_df["false_negative_rate_within_type_pct"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(16, 8))
    plt.bar(x - width / 2, prevalence, width=width, label="Prevalence among label-1 (%)")
    plt.bar(x + width / 2, error_rate, width=width, label="False-negative rate within type (%)")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Percent")
    plt.title("Text ensemble spoof classification prevalence vs false-negative rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)

    ap.add_argument("--golden_model", required=True, help="Path to .pt (applied before cosine_sim).")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--pt-batch-size", type=int, default=8192)

    ap.add_argument("--model", default="adaboost", choices=["adaboost", "gboost", "none"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fraud-col", default="fraudulent_name")
    ap.add_argument("--real-col", default="real_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1)

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")

    ap.add_argument("--error-output-dir", default="for_paper/error_outputs")

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

    df_tr = load_table(args.train)
    df_te = load_table(args.test)

    projector, projector_in_dim = load_golden_projector(args.golden_model, device=device)

    X_tr, y_tr, feat_names = build_features(
        df_tr,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.fraud_prefix,
        real_prefix=args.real_prefix,
        projector=projector,
        projector_in_dim=projector_in_dim,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )
    X_te, y_te, _ = build_features(
        df_te,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
        fraud_prefix=args.fraud_prefix,
        real_prefix=args.real_prefix,
        projector=projector,
        projector_in_dim=projector_in_dim,
        device=device,
        pt_batch_size=args.pt_batch_size,
    )

    rows: List[SingleFeatureRow] = []
    for j, name in enumerate(feat_names):
        s_te = X_te[:, j].astype(np.float64)

        test_auroc = float(roc_auc_score(y_te, s_te))
        thr_test = youden_threshold(y_te, s_te)
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

    if args.model == "none":
        print("Model needed!")
        return

    clf = make_model(args.model, args.seed)
    clf.fit(X_tr, y_tr)

    s_te = model_scores(clf, X_te).astype(np.float64)
    test_auroc = float(roc_auc_score(y_te, s_te))

    thr_test = youden_threshold(y_te, s_te)
    acc, prec, rec = metrics_at_threshold(y_te, s_te, thr_test)

    print()
    print("Ensemble")
    print(
        f"model={args.model} test_auroc={test_auroc:.6f} youden_thr_test={thr_test:.6f} "
        f"test_accuracy={acc:.6f} test_precision={prec:.6f} test_recall={rec:.6f}"
    )

    yhat_te = (s_te >= thr_test).astype(np.int32)

    os.makedirs(args.error_output_dir, exist_ok=True)

    df_pos, df_fn = build_positive_mechanism_tables(
        df_test=df_te,
        y_true=y_te,
        y_pred=yhat_te,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
    )

    summary_df = build_mechanism_error_summary(df_pos=df_pos, df_fn=df_fn)

    positives_parquet = os.path.join(args.error_output_dir, "text_ensemble_test_positives_multihot.parquet")
    fn_parquet = os.path.join(args.error_output_dir, "text_ensemble_false_negatives_multihot.parquet")
    summary_csv = os.path.join(args.error_output_dir, "text_ensemble_mechanism_error_summary.csv")
    summary_txt = os.path.join(args.error_output_dir, "text_ensemble_mechanism_error_summary.txt")
    scatter_png = os.path.join(args.error_output_dir, "text_ensemble_mechanism_error_scatter.png")
    bar_png = os.path.join(args.error_output_dir, "text_ensemble_mechanism_error_bar.png")

    df_pos.to_parquet(positives_parquet, index=False)
    df_fn.to_parquet(fn_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_mechanism_summary_txt(
        summary_df=summary_df,
        total_positive=int(len(df_pos)),
        total_false_negative=int(len(df_fn)),
        out_txt=summary_txt,
    )
    save_scatter_plot(summary_df=summary_df, out_png=scatter_png)
    save_grouped_bar_chart(summary_df=summary_df, out_png=bar_png)

    print()
    print(f"[OK] wrote positives multihot parquet: {positives_parquet}")
    print(f"[OK] wrote false-negative multihot parquet: {fn_parquet}")
    print(f"[OK] wrote mechanism summary csv: {summary_csv}")
    print(f"[OK] wrote mechanism summary txt: {summary_txt}")
    print(f"[OK] wrote scatter plot: {scatter_png}")
    print(f"[OK] wrote bar chart: {bar_png}")


if __name__ == "__main__":
    main()

"""
python3 for_paper/text_ensemble_error.py \
  --train ../Downloads/text_train.parquet \
  --test  ../Downloads/text_test.parquet \
  --golden_model ../Downloads/single_run_model.pt \
  --model adaboost
"""