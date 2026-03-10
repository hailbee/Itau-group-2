#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Small output settings
# =========================
REPRESENTATIVE_MIN_NAME_LEN = 5
AVG_STRING_LENGTH_BUCKET_WIDTH = 2.0
ABS_LENGTH_DIFFERENCE_BUCKET_WIDTH = 2


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


def clean_text_cell(x: Any) -> str:
    return str(x).replace("\n", " ").replace("\r", " ").replace("|", "/").strip()


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
# =========================
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

    max_extra = 10
    if len(a) - len(b) > max_extra:
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

    if not hasattr(model, "predict"):
        raise RuntimeError(f"Loaded object from {path!r} does not have a predict() method.")

    if feature_names is not None:
        if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
            raise RuntimeError("Saved model bundle has invalid feature_names.")

    return model, feature_names


# =========================
# Error analysis helpers
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

    y_pred_pos = y_pred[pos_mask]
    df_pos["pred_label"] = y_pred_pos.astype(np.int32)
    df_pos["is_false_negative"] = ((y_pred_pos == 0) & (y_true[pos_mask] == 1)).astype(int)

    df_fn = df_pos.loc[df_pos["is_false_negative"] == 1].copy().reset_index(drop=True)
    return df_pos, df_fn


def build_mechanism_error_summary(df_pos: pd.DataFrame, df_fn: pd.DataFrame) -> pd.DataFrame:
    total_positive = int(len(df_pos))
    rows: List[Dict[str, float | str]] = []

    for k in FLAG_KEYS:
        n_pos_with_type = int(df_pos[k].sum())
        n_fn_with_type = int(df_fn[k].sum())

        frequency_pct = (100.0 * n_pos_with_type / total_positive) if total_positive > 0 else 0.0
        model_missed_pct = (100.0 * n_fn_with_type / n_pos_with_type) if n_pos_with_type > 0 else 0.0

        rows.append(
            {
                "classification_type": k,
                "frequency_pct": frequency_pct,
                "model_missed_pct": model_missed_pct,
                "raw_errors": n_fn_with_type,
            }
        )

    return pd.DataFrame(rows)


def save_mechanism_summary_txt(summary_df: pd.DataFrame, out_txt: str) -> None:
    lines: List[str] = []
    lines.append("classification_type | frequency_pct | model_missed_pct | raw_errors")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{clean_text_cell(row['classification_type'])} | "
            f"{float(row['frequency_pct']):.6f} | "
            f"{float(row['model_missed_pct']):.6f} | "
            f"{int(row['raw_errors'])}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_representative_errors_txt(
    df_fn: pd.DataFrame,
    fraud_col: str,
    real_col: str,
    label_col: str,
    out_txt: str,
) -> None:
    lines: List[str] = []

    for k in FLAG_KEYS:
        lines.append(f"classification_type={k}")
        lines.append(f"{real_col} | {fraud_col} | {label_col} | classification_type")

        subset = df_fn.loc[df_fn[k] == 1, [real_col, fraud_col, label_col]].copy()

        real_len = subset[real_col].fillna("").astype(str).str.len()
        fraud_len = subset[fraud_col].fillna("").astype(str).str.len()
        subset = subset.loc[(real_len >= REPRESENTATIVE_MIN_NAME_LEN) & (fraud_len >= REPRESENTATIVE_MIN_NAME_LEN)]

        subset = subset.drop_duplicates().head(3).reset_index(drop=True)

        if subset.empty:
            lines.append(f"NO_EXAMPLES | NO_EXAMPLES | NO_EXAMPLES | {k}")
        else:
            for _, row in subset.iterrows():
                lines.append(
                    f"{clean_text_cell(row[real_col])} | "
                    f"{clean_text_cell(row[fraud_col])} | "
                    f"{clean_text_cell(row[label_col])} | "
                    f"{k}"
                )

        lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def build_bucketed_accuracy_summary(
    values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_width: float | None,
    bucket_col: str,
    use_quantiles: bool = False,
) -> pd.DataFrame:
    values = values.astype(np.float64, copy=False)
    correct = (y_true == y_pred).astype(np.int32)

    df = pd.DataFrame(
        {
            "values": values,
            "correct": correct,
        }
    )

    if use_quantiles:
        bucket_codes, bins = pd.qcut(
            df["values"],
            q=5,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        df["bucket"] = bucket_codes.astype("Int64")
        df = df.loc[df["bucket"].notna()].copy()

        df["bucket_start"] = df["bucket"].map(lambda i: float(bins[int(i)]))
        df["bucket_end"] = df["bucket"].map(lambda i: float(bins[int(i) + 1]))

        grouped = (
            df.groupby(["bucket_start", "bucket_end"], as_index=False)
            .agg(
                count=("correct", "size"),
                accuracy=("correct", "mean"),
                bucket_mean=("values", "mean"),
            )
            .sort_values(["bucket_start", "bucket_end"])
            .reset_index(drop=True)
        )

        total_n = int(len(df))
        grouped["accuracy_pct"] = 100.0 * grouped["accuracy"]
        grouped["frequency_pct"] = 100.0 * grouped["count"] / total_n

        grouped = grouped.rename(
            columns={
                "bucket_start": f"{bucket_col}_start",
                "bucket_end": f"{bucket_col}_end",
                "bucket_mean": f"{bucket_col}_mean",
            }
        )

        return grouped[
            [
                f"{bucket_col}_start",
                f"{bucket_col}_end",
                f"{bucket_col}_mean",
                "accuracy_pct",
                "frequency_pct",
            ]
        ]

    if bucket_width is None or bucket_width <= 0:
        raise ValueError("bucket_width must be > 0 when not using quantiles")

    bucket_start = np.floor(values / bucket_width) * bucket_width
    bucket_end = bucket_start + bucket_width

    df["bucket_start"] = bucket_start
    df["bucket_end"] = bucket_end

    grouped = (
        df.groupby(["bucket_start", "bucket_end"], as_index=False)
        .agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
            bucket_mean=("values", "mean"),
        )
        .sort_values(["bucket_start", "bucket_end"])
        .reset_index(drop=True)
    )

    total_n = int(len(df))
    grouped["accuracy_pct"] = 100.0 * grouped["accuracy"]
    grouped["frequency_pct"] = 100.0 * grouped["count"] / total_n

    grouped = grouped.rename(
        columns={
            "bucket_start": f"{bucket_col}_start",
            "bucket_end": f"{bucket_col}_end",
            "bucket_mean": f"{bucket_col}_mean",
        }
    )

    return grouped[
        [
            f"{bucket_col}_start",
            f"{bucket_col}_end",
            f"{bucket_col}_mean",
            "accuracy_pct",
            "frequency_pct",
        ]
    ]


def build_raw_value_accuracy_summary(
    values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    value_col: str,
) -> pd.DataFrame:
    values = values.astype(np.float64, copy=False)
    correct = (y_true == y_pred).astype(np.int32)

    df = pd.DataFrame(
        {
            "values": values,
            "correct": correct,
        }
    )

    grouped = (
        df.groupby("values", as_index=False)
        .agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        )
        .sort_values("values")
        .reset_index(drop=True)
    )

    total_n = int(len(df))
    grouped["accuracy_pct"] = 100.0 * grouped["accuracy"]
    grouped["frequency_pct"] = 100.0 * grouped["count"] / total_n

    grouped = grouped.rename(columns={"values": value_col})

    return grouped[[value_col, "accuracy_pct", "frequency_pct"]]


def save_bucketed_accuracy_txt(
    summary_df: pd.DataFrame,
    bucket_col: str,
    out_txt: str,
    float_format: str,
) -> None:
    start_col = f"{bucket_col}_start"
    end_col = f"{bucket_col}_end"
    mean_col = f"{bucket_col}_mean"

    lines: List[str] = []
    lines.append(f"{start_col} | {end_col} | {mean_col} | accuracy_pct | frequency_pct")

    for _, row in summary_df.iterrows():
        start_val = float(row[start_col])
        end_val = float(row[end_col])
        mean_val = float(row[mean_col])

        if float_format == "int":
            start_str = f"{int(round(start_val))}"
            end_str = f"{int(round(end_val))}"
            mean_str = f"{int(round(mean_val))}"
        else:
            start_str = f"{start_val:.1f}"
            end_str = f"{end_val:.1f}"
            mean_str = f"{mean_val:.6f}"

        lines.append(
            f"{start_str} | {end_str} | {mean_str} | "
            f"{float(row['accuracy_pct']):.6f} | "
            f"{float(row['frequency_pct']):.6f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_raw_value_accuracy_txt(
    summary_df: pd.DataFrame,
    value_col: str,
    out_txt: str,
    float_format: str,
) -> None:
    lines: List[str] = []
    lines.append(f"{value_col} | accuracy_pct | frequency_pct")

    for _, row in summary_df.iterrows():
        value = float(row[value_col])

        if float_format == "int":
            value_str = f"{int(round(value))}"
        else:
            value_str = f"{value:.6f}"

        lines.append(
            f"{value_str} | "
            f"{float(row['accuracy_pct']):.6f} | "
            f"{float(row['frequency_pct']):.6f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()

    # Kept accepted for compatibility with old command, but not used anymore.
    ap.add_argument("--downloads-train", default=None)
    ap.add_argument("--deja-train", default=None)
    ap.add_argument("--unifont-train", default=None)
    ap.add_argument("--libre-train", default=None)
    ap.add_argument("--exo2-train", default=None)
    ap.add_argument("--doulos-train", default=None)
    ap.add_argument("--cousine-train", default=None)
    ap.add_argument("--model", default="adaboost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-output-path", default=None)

    ap.add_argument("--downloads-test", required=True)
    ap.add_argument("--downloads-pt", required=True)

    ap.add_argument("--deja-test", required=True)
    ap.add_argument("--deja-pt", required=True)

    ap.add_argument("--unifont-test", required=True)
    ap.add_argument("--unifont-pt", required=True)

    ap.add_argument("--libre-test", required=True)
    ap.add_argument("--libre-pt", required=True)

    ap.add_argument("--exo2-test", required=True)
    ap.add_argument("--exo2-pt", required=True)

    ap.add_argument("--doulos-test", required=True)
    ap.add_argument("--doulos-pt", required=True)

    ap.add_argument("--cousine-test", required=True)
    ap.add_argument("--cousine-pt", required=True)

    ap.add_argument("--saved-model-path", default="saved_models/total_5f_model.joblib")
    ap.add_argument("--error-output-dir", default="for_paper/error_outputs")

    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--pt-batch-size", type=int, default=8192)

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
    ap.add_argument("--exo2-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--exo2-real-prefix", default="real_emb_")
    ap.add_argument("--doulos-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--doulos-real-prefix", default="real_emb_")
    ap.add_argument("--cousine-fraud-prefix", default="fraud_emb_")
    ap.add_argument("--cousine-real-prefix", default="real_emb_")

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

    # Load test data only
    df_te_txt = load_table(args.downloads_test)
    df_te_dj = load_table(args.deja_test)
    df_te_uf = load_table(args.unifont_test)
    df_te_li = load_table(args.libre_test)
    df_te_ex = load_table(args.exo2_test)
    df_te_do = load_table(args.doulos_test)
    df_te_co = load_table(args.cousine_test)

    for df in (df_te_txt, df_te_dj, df_te_uf, df_te_li, df_te_ex, df_te_do, df_te_co):
        if args.label_col not in df.columns:
            raise RuntimeError(f"Missing label_col={args.label_col!r} in one of the tables.")
        if args.fraud_col not in df.columns:
            raise RuntimeError(f"Missing fraud_col={args.fraud_col!r} in one of the tables.")
        if args.real_col not in df.columns:
            raise RuntimeError(f"Missing real_col={args.real_col!r} in one of the tables.")

    n_te = len(df_te_txt)
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
            f"Downloads={n_te} Deja={len(df_te_dj)} Unifont={len(df_te_uf)} "
            f"Libre={len(df_te_li)} Exo2={len(df_te_ex)} Doulos={len(df_te_do)} Cousine={len(df_te_co)}"
        )

    y_te = (df_te_txt[args.label_col].to_numpy() == args.positive_label).astype(np.int32)

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
            raise RuntimeError(f"Test labels mismatch: Downloads vs {name}")

    if not _has_both_classes(y_te):
        raise RuntimeError("Need both classes (0 and 1) in test split.")

    clf, saved_feature_names = load_saved_model_bundle(args.saved_model_path)

    proj_txt, in_dim_txt = load_golden_projector(args.downloads_pt, device=device)
    proj_dj, in_dim_dj = load_golden_projector(args.deja_pt, device=device)
    proj_uf, in_dim_uf = load_golden_projector(args.unifont_pt, device=device)
    proj_li, in_dim_li = load_golden_projector(args.libre_pt, device=device)
    proj_ex, in_dim_ex = load_golden_projector(args.exo2_pt, device=device)
    proj_do, in_dim_do = load_golden_projector(args.doulos_pt, device=device)
    proj_co, in_dim_co = load_golden_projector(args.cousine_pt, device=device)

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
    cos_te_ex, y_te_ex = build_single_font_cosine(
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

    for name, y_other in [
        ("text test", y_te_txt),
        ("deja test", y_te_dj),
        ("unifont test", y_te_uf),
        ("libre test", y_te_li),
        ("exo2 test", y_te_ex),
        ("doulos test", y_te_do),
        ("cousine test", y_te_co),
    ]:
        if not np.array_equal(y_te, y_other):
            raise RuntimeError(f"Test labels mismatch after feature extraction: {name}")

    available_features: Dict[str, np.ndarray] = {
        "text_cosine": X_te_txt[:, 0],
        "token_set_ratio": X_te_txt[:, 1],
        "levenshtein_distance_score": X_te_txt[:, 2],
        "partial_ratio": X_te_txt[:, 3],
        "cosine_deja": cos_te_dj,
        "cosine_unifont": cos_te_uf,
        "cosine_libre": cos_te_li,
        "cosine_exo2": cos_te_ex,
        "cosine_doulos": cos_te_do,
        "cosine_cousine": cos_te_co,
    }

    default_feature_names = [
        "text_cosine",
        "token_set_ratio",
        "levenshtein_distance_score",
        "partial_ratio",
        "cosine_deja",
        "cosine_unifont",
        "cosine_libre",
        "cosine_exo2",
        "cosine_doulos",
        "cosine_cousine",
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
    yhat_te = clf.predict(X_te).astype(np.int32)

    os.makedirs(args.error_output_dir, exist_ok=True)

    df_pos, df_fn = build_positive_mechanism_tables(
        df_test=df_te_txt,
        y_true=y_te,
        y_pred=yhat_te,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
    )

    mechanism_summary = build_mechanism_error_summary(df_pos=df_pos, df_fn=df_fn)

    fraud_names = df_te_txt[args.fraud_col].fillna("").astype(str)
    real_names = df_te_txt[args.real_col].fillna("").astype(str)

    fraud_len = fraud_names.str.len().to_numpy(dtype=np.int32)
    real_len = real_names.str.len().to_numpy(dtype=np.int32)

    avg_string_length = (fraud_len + real_len) / 2.0
    abs_length_difference = np.abs(fraud_len - real_len)

    accuracy_by_avg_len = build_bucketed_accuracy_summary(
        values=avg_string_length,
        y_true=y_te,
        y_pred=yhat_te,
        bucket_width=None,
        bucket_col="avg_string_length_bucket",
        use_quantiles=True,
    )

    pos_mask = y_te == 1

    accuracy_by_abs_diff = build_raw_value_accuracy_summary(
        values=abs_length_difference[pos_mask].astype(np.float64),
        y_true=y_te[pos_mask],
        y_pred=yhat_te[pos_mask],
        value_col="abs_length_difference",
    )

    summary_txt = os.path.join(args.error_output_dir, "total_5f_error_type_summary.txt")
    examples_txt = os.path.join(args.error_output_dir, "total_5f_representative_example_errors.txt")
    avg_len_txt = os.path.join(args.error_output_dir, "total_5f_accuracy_by_avg_string_length.txt")
    abs_diff_txt = os.path.join(args.error_output_dir, "total_5f_accuracy_by_abs_length_difference.txt")

    save_mechanism_summary_txt(mechanism_summary, summary_txt)
    save_representative_errors_txt(
        df_fn=df_fn,
        fraud_col=args.fraud_col,
        real_col=args.real_col,
        label_col=args.label_col,
        out_txt=examples_txt,
    )
    save_bucketed_accuracy_txt(
        summary_df=accuracy_by_avg_len,
        bucket_col="avg_string_length_bucket",
        out_txt=avg_len_txt,
        float_format="float",
    )
    save_raw_value_accuracy_txt(
        summary_df=accuracy_by_abs_diff,
        value_col="abs_length_difference",
        out_txt=abs_diff_txt,
        float_format="int",
    )

    print(f"[OK] wrote {summary_txt}")
    print(f"[OK] wrote {examples_txt}")
    print(f"[OK] wrote {avg_len_txt}")
    print(f"[OK] wrote {abs_diff_txt}")


if __name__ == "__main__":
    main()

"""
python3 for_paper/error_from_saved.py \
  --downloads-test  ../Downloads/text_test.parquet \
  --downloads-pt    ../Downloads/single_run_model.pt \
  --deja-test       ../Deja/test_pairs_with_siglip_embeddings.parquet \
  --deja-pt         ../Deja/single_run_model.pt \
  --unifont-test    ../Unifont/test_pairs_with_siglip_embeddings.parquet \
  --unifont-pt      ../Unifont/single_run_model.pt \
  --libre-test      ../Libre/test_pairs_with_siglip_embeddings.parquet \
  --libre-pt        ../Libre/single_run_model.pt \
  --exo2-test       ../Exo2/test_pairs_with_siglip_embeddings.parquet \
  --exo2-pt         ../Exo2/single_run_model.pt \
  --doulos-test     ../Doulos/test_pairs_with_siglip_embeddings.parquet \
  --doulos-pt       ../Doulos/single_run_model.pt \
  --cousine-test    ../Cousine/test_pairs_with_siglip_embeddings.parquet \
  --cousine-pt      ../Cousine/single_run_model.pt \
  --saved-model-path saved_models/total_5f_model.joblib
"""