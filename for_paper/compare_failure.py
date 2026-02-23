#!/usr/bin/env python3
"""
compare_failure.py

Compare misclassified samples across THREE result files (same schema as deja_misclassified.parquet).

Assumed columns (present in each file):
  - fraudulent_name (str)
  - real_name       (str)
  - y_true          (int 0/1)
  - pred            (int 0/1)
  - score           (float)
  - threshold (float; constant within each file)

What it prints (no files saved):
  1) Overall overlap breakdown (only A / only B / only C / pairwise overlaps / all-3)
  2) Same breakdown for FN vs FP separately
  3) Jaccard similarities (overall + FN + FP)
  4) Score similarity across shared errors (Pearson + Spearman) for pairwise overlaps
  5) Flag-rate comparisons (non-ascii, punycode, hyphen-change, digit-change) for:
        - all-3 shared errors
        - unique-to-each-model errors
  6) Representative examples:
        - "Hard shared" (all-3, farthest from threshold on average)
        - "Borderline shared" (all-3, closest to threshold on average)
        - "Unique hard" per model (only that model, farthest from threshold)

  7) NEW: Margin diagnostics (using each font/model's OWN threshold):
        - abs(margin) summaries for each overlap group (all3, only_*, ab_only, ac_only, bc_only)
        - within each group, split into FN vs FP
        - "borderline fractions": % with |margin| <= {1e-3, 5e-3, 1e-2}
        - on the all-3 shared set: which model is MOST confident wrong (largest |margin|) counts overall + split FN/FP

Key definition (simple):
  key = (fraudulent_name, real_name, y_true)

So swaps (fraud<->real) are treated as different keys by design.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -------------------------
# I/O
# -------------------------
def _read_table(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path, columns=usecols) if usecols else pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path, usecols=usecols) if usecols else pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", usecols=usecols) if usecols else pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported extension: {ext} (use .parquet/.pq/.csv/.tsv)")


# -------------------------
# Core feature engineering
# -------------------------
@dataclass(frozen=True)
class Cols:
    fraud: str = "fraudulent_name"
    real: str = "real_name"
    y: str = "y_true"
    pred: str = "pred"
    score: str = "score"
    thr: str = "threshold"


RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")
RE_DIGIT = re.compile(r"\d")


def _prep_df(df: pd.DataFrame, cols: Cols, tag: str) -> pd.DataFrame:
    """
    Produces:
      - key: (fraud, real, y_true)
      - error_type: 'FN' / 'FP' / 'OTHER'
      - margin: score - threshold
      - flags: has_non_ascii, has_punycode, hyphen_change, digit_change
      - lengths

    Note: We drop duplicate keys to keep all set-based computations consistent.
    """
    df = df.copy()
    df[cols.fraud] = df[cols.fraud].astype(str)
    df[cols.real] = df[cols.real].astype(str)
    df[cols.y] = df[cols.y].astype(int)
    df[cols.pred] = df[cols.pred].astype(int)
    df[cols.score] = df[cols.score].astype(float)

    thr_vals = pd.unique(df[cols.thr].dropna())
    if len(thr_vals) != 1:
        raise ValueError(f"[{tag}] threshold must be a single constant, got {len(thr_vals)} unique values.")
    thr = float(thr_vals[0])

    df["threshold"] = thr
    df["margin"] = df[cols.score] - thr

    y = df[cols.y].to_numpy()
    p = df[cols.pred].to_numpy()
    err = np.full(len(df), "OTHER", dtype=object)
    err[(y == 1) & (p == 0)] = "FN"
    err[(y == 0) & (p == 1)] = "FP"
    df["error_type"] = err

    df["fraud_len"] = df[cols.fraud].str.len()
    df["real_len"] = df[cols.real].str.len()
    df["len_diff"] = (df["fraud_len"] - df["real_len"]).abs()

    # Flags (row-level)
    f = df[cols.fraud].astype(str)
    r = df[cols.real].astype(str)
    df["has_non_ascii"] = f.str.contains(RE_NON_ASCII) | r.str.contains(RE_NON_ASCII)
    df["has_punycode"] = f.str.contains("xn--", regex=False) | r.str.contains("xn--", regex=False)
    df["hyphen_change"] = f.str.contains("-", regex=False) ^ r.str.contains("-", regex=False)
    df["digit_change"] = f.str.contains(RE_DIGIT) ^ r.str.contains(RE_DIGIT)

    df["key"] = list(zip(df[cols.fraud].tolist(), df[cols.real].tolist(), df[cols.y].tolist()))

    # Ensure one row per key for clean set logic and margin summaries
    df = df.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)
    return df


# -------------------------
# Set overlap utilities
# -------------------------
def _breakdown(sa: Set[Tuple], sb: Set[Tuple], sc: Set[Tuple]) -> Dict[str, Set[Tuple]]:
    all3 = sa & sb & sc
    ab = (sa & sb) - all3
    ac = (sa & sc) - all3
    bc = (sb & sc) - all3
    only_a = sa - sb - sc
    only_b = sb - sa - sc
    only_c = sc - sa - sb
    return {
        "all3": all3,
        "ab_only": ab,
        "ac_only": ac,
        "bc_only": bc,
        "only_a": only_a,
        "only_b": only_b,
        "only_c": only_c,
    }


def _jaccard(sa: Set[Tuple], sb: Set[Tuple]) -> float:
    u = sa | sb
    if not u:
        return 0.0
    return len(sa & sb) / len(u)


# -------------------------
# Correlations on shared keys
# -------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# -------------------------
# Flag summaries
# -------------------------
FLAG_COLS = ["has_non_ascii", "has_punycode", "hyphen_change", "digit_change"]


def _flag_rates(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {c: float("nan") for c in FLAG_COLS}
    return {c: float(df[c].mean()) for c in FLAG_COLS}


def _slice_by_keys(df: pd.DataFrame, keys: Set[Tuple]) -> pd.DataFrame:
    if not keys:
        return df.head(0).copy()
    return df[df["key"].isin(keys)].copy()


# -------------------------
# Margin diagnostics (NEW)
# -------------------------
def _margin_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Summaries on abs(margin) and signed margin.
    """
    if len(df) == 0:
        return {
            "n": 0.0,
            "abs_mean": float("nan"),
            "abs_median": float("nan"),
            "abs_q10": float("nan"),
            "abs_q90": float("nan"),
            "abs_q99": float("nan"),
            "signed_mean": float("nan"),
            "signed_median": float("nan"),
            "frac_abs_le_1e-3": float("nan"),
            "frac_abs_le_5e-3": float("nan"),
            "frac_abs_le_1e-2": float("nan"),
        }

    m = df["margin"].astype(float)
    am = m.abs()

    return {
        "n": float(len(df)),
        "abs_mean": float(am.mean()),
        "abs_median": float(am.median()),
        "abs_q10": float(am.quantile(0.10)),
        "abs_q90": float(am.quantile(0.90)),
        "abs_q99": float(am.quantile(0.99)),
        "signed_mean": float(m.mean()),
        "signed_median": float(m.median()),
        "frac_abs_le_1e-3": float((am <= 1e-3).mean()),
        "frac_abs_le_5e-3": float((am <= 5e-3).mean()),
        "frac_abs_le_1e-2": float((am <= 1e-2).mean()),
    }


def _print_margin_line(prefix: str, s: Dict[str, float]) -> None:
    print(
        f"{prefix} n={int(s['n']):5d} | "
        f"|m| mean={s['abs_mean']:.6f} med={s['abs_median']:.6f} "
        f"q10={s['abs_q10']:.6f} q90={s['abs_q90']:.6f} q99={s['abs_q99']:.6f} | "
        f"border(|m|<=1e-3)={s['frac_abs_le_1e-3']:.3f} "
        f"<=5e-3={s['frac_abs_le_5e-3']:.3f} "
        f"<=1e-2={s['frac_abs_le_1e-2']:.3f}"
    )


def _print_margin_block(title: str, df: pd.DataFrame, model_name: str) -> None:
    print(f"\n{title} [{model_name}]")
    print("-" * (len(title) + len(model_name) + 3))
    overall = _margin_stats(df)
    _print_margin_line("ALL ", overall)

    fn = df[df["error_type"] == "FN"]
    fp = df[df["error_type"] == "FP"]
    _print_margin_line(" FN ", _margin_stats(fn))
    _print_margin_line(" FP ", _margin_stats(fp))


def _merge_all3(da: pd.DataFrame, db: pd.DataFrame, dc: pd.DataFrame, keys: Set[Tuple]) -> pd.DataFrame:
    """
    For all-3 shared keys, build a single table containing margins/scores for all three.
    Uses A for string fields + error_type (should match across files for a given y_true).
    """
    a = _slice_by_keys(da, keys)[["key", "fraudulent_name", "real_name", "y_true", "error_type", "score", "margin",
                                 "has_non_ascii", "has_punycode", "hyphen_change", "digit_change"]].copy()
    b = _slice_by_keys(db, keys)[["key", "score", "margin"]].copy()
    c = _slice_by_keys(dc, keys)[["key", "score", "margin"]].copy()

    a = a.rename(columns={"score": "score_a", "margin": "margin_a"})
    b = b.rename(columns={"score": "score_b", "margin": "margin_b"})
    c = c.rename(columns={"score": "score_c", "margin": "margin_c"})

    m = a.merge(b, on="key", how="inner").merge(c, on="key", how="inner")
    m["abs_a"] = m["margin_a"].abs()
    m["abs_b"] = m["margin_b"].abs()
    m["abs_c"] = m["margin_c"].abs()
    m["mean_abs_margin"] = (m["abs_a"] + m["abs_b"] + m["abs_c"]) / 3.0
    return m


def _winner_counts(m: pd.DataFrame, name_a: str, name_b: str, name_c: str) -> Dict[str, int]:
    """
    On all-3 shared keys: which model has the largest |margin| (most confident wrong).
    """
    if len(m) == 0:
        return {name_a: 0, name_b: 0, name_c: 0}

    abs_mat = np.vstack([m["abs_a"].to_numpy(), m["abs_b"].to_numpy(), m["abs_c"].to_numpy()]).T
    idx = np.argmax(abs_mat, axis=1)
    names = np.array([name_a, name_b, name_c], dtype=object)
    winners = names[idx]
    out = {name_a: 0, name_b: 0, name_c: 0}
    for w in winners:
        out[str(w)] += 1
    return out


# -------------------------
# Representative examples
# -------------------------
def _example_lines(df: pd.DataFrame, k: int) -> List[str]:
    out = []
    if len(df) == 0:
        return out
    for _, r in df.head(k).iterrows():
        out.append(
            f"fraud={r['fraudulent_name']!r} | real={r['real_name']!r} | y={int(r['y_true'])} | "
            f"type={r['error_type']} | score={float(r['score']):.6f} | margin={float(r['margin']):+.6f} | "
            f"flags={{non_ascii:{bool(r['has_non_ascii'])}, puny:{bool(r['has_punycode'])}, hyphen:{bool(r['hyphen_change'])}, digit:{bool(r['digit_change'])}}}"
        )
    return out


def _rank_shared_examples(da: pd.DataFrame, db: pd.DataFrame, dc: pd.DataFrame, shared_keys: Set[Tuple], topk: int):
    """
    For all-3 shared keys, compute mean_abs_margin across A/B/C.
    Return:
      - hard_shared: largest mean_abs_margin
      - borderline_shared: smallest mean_abs_margin
    """
    m = _merge_all3(da, db, dc, shared_keys)
    if len(m) == 0:
        empty = da.head(0).copy()
        return empty, empty

    hard = m.sort_values("mean_abs_margin", ascending=False).head(topk)
    border = m.sort_values("mean_abs_margin", ascending=True).head(topk)
    return hard, border


# -------------------------
# Printing
# -------------------------
def _print_breakdown(name: str, sets: Dict[str, Set[Tuple]]) -> None:
    print(f"\n--- Overlap breakdown ({name}) ---")
    for k in ["only_a", "only_b", "only_c", "ab_only", "ac_only", "bc_only", "all3"]:
        print(f"{k:>8}: {len(sets[k])}")


def _print_flag_block(title: str, d: Dict[str, float]) -> None:
    print(title)
    for k in FLAG_COLS:
        v = d[k]
        print(f"  {k:>12}: {v:.4f}")


def _print_examples_block(title: str, lines: List[str]) -> None:
    print("\n" + title)
    print("-" * len(title))
    if not lines:
        print("(none)")
        return
    for i, s in enumerate(lines, 1):
        print(f"[{i}] {s}")


def _print_winner_dict(title: str, d: Dict[str, int]) -> None:
    total = sum(d.values())
    print(title)
    for k, v in d.items():
        frac = (v / total) if total else 0.0
        print(f"  {k:>10}: {v:6d}  ({frac:.3f})")


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to results file A (parquet/csv/tsv).")
    ap.add_argument("--b", required=True, help="Path to results file B (parquet/csv/tsv).")
    ap.add_argument("--c", required=True, help="Path to results file C (parquet/csv/tsv).")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--name-c", default="C")

    ap.add_argument("--topk", type=int, default=10, help="How many patterns/examples to print per block.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = Cols()
    needed = [cols.fraud, cols.real, cols.y, cols.pred, cols.score, cols.thr]

    df_a = _prep_df(_read_table(args.a, usecols=needed), cols, tag=args.name_a)
    df_b = _prep_df(_read_table(args.b, usecols=needed), cols, tag=args.name_b)
    df_c = _prep_df(_read_table(args.c, usecols=needed), cols, tag=args.name_c)

    # Basic summary
    print("\n==============================")
    print("THREE-WAY MISCLASSIFIED COMPARISON")
    print("==============================")
    print(f"{args.name_a}: {args.a} | n={len(df_a)} | thr={df_a['threshold'].iloc[0]:.9f}")
    print(f"{args.name_b}: {args.b} | n={len(df_b)} | thr={df_b['threshold'].iloc[0]:.9f}")
    print(f"{args.name_c}: {args.c} | n={len(df_c)} | thr={df_c['threshold'].iloc[0]:.9f}")

    # Sets (overall)
    sa = set(df_a["key"].tolist())
    sb = set(df_b["key"].tolist())
    sc = set(df_c["key"].tolist())

    overall = _breakdown(sa, sb, sc)
    _print_breakdown("ALL errors", overall)

    # FN/FP sets separately
    sa_fn = set(df_a[df_a["error_type"] == "FN"]["key"].tolist())
    sb_fn = set(df_b[df_b["error_type"] == "FN"]["key"].tolist())
    sc_fn = set(df_c[df_c["error_type"] == "FN"]["key"].tolist())

    sa_fp = set(df_a[df_a["error_type"] == "FP"]["key"].tolist())
    sb_fp = set(df_b[df_b["error_type"] == "FP"]["key"].tolist())
    sc_fp = set(df_c[df_c["error_type"] == "FP"]["key"].tolist())

    fn_bd = _breakdown(sa_fn, sb_fn, sc_fn)
    fp_bd = _breakdown(sa_fp, sb_fp, sc_fp)

    _print_breakdown("FN only", fn_bd)
    _print_breakdown("FP only", fp_bd)

    # Jaccard similarities
    print("\n--- Jaccard similarities ---")
    print(
        f"ALL: J({args.name_a},{args.name_b})={_jaccard(sa, sb):.4f} | "
        f"J({args.name_a},{args.name_c})={_jaccard(sa, sc):.4f} | "
        f"J({args.name_b},{args.name_c})={_jaccard(sb, sc):.4f}"
    )
    print(
        f" FN: J({args.name_a},{args.name_b})={_jaccard(sa_fn, sb_fn):.4f} | "
        f"J({args.name_a},{args.name_c})={_jaccard(sa_fn, sc_fn):.4f} | "
        f"J({args.name_b},{args.name_c})={_jaccard(sb_fn, sc_fn):.4f}"
    )
    print(
        f" FP: J({args.name_a},{args.name_b})={_jaccard(sa_fp, sb_fp):.4f} | "
        f"J({args.name_a},{args.name_c})={_jaccard(sa_fp, sc_fp):.4f} | "
        f"J({args.name_b},{args.name_c})={_jaccard(sb_fp, sc_fp):.4f}"
    )

    # Pairwise score similarity on shared keys
    def score_corr(d1: pd.DataFrame, d2: pd.DataFrame, s1: Set[Tuple], s2: Set[Tuple], n1: str, n2: str) -> None:
        shared = s1 & s2
        if not shared:
            print(f"{n1} vs {n2}: no shared keys")
            return
        x = _slice_by_keys(d1, shared).set_index("key")["score"].astype(float)
        y = _slice_by_keys(d2, shared).set_index("key")["score"].astype(float)
        m = x.to_frame("x").join(y.to_frame("y"), how="inner")
        px = m["x"].to_numpy()
        py = m["y"].to_numpy()
        print(
            f"{n1} vs {n2}: shared={len(m)} | "
            f"pearson(score)={_pearson(px, py):.4f} | spearman(score)={_spearman(px, py):.4f}"
        )

    print("\n--- Score similarity on shared errors (pairwise) ---")
    score_corr(df_a, df_b, sa, sb, args.name_a, args.name_b)
    score_corr(df_a, df_c, sa, sc, args.name_a, args.name_c)
    score_corr(df_b, df_c, sb, sc, args.name_b, args.name_c)

    # Flag-rate comparisons on key groups
    all3_keys = overall["all3"]
    only_a = overall["only_a"]
    only_b = overall["only_b"]
    only_c = overall["only_c"]

    print("\n--- Flag rates: shared vs unique (ALL errors) ---")
    _print_flag_block("ALL-3 shared (rates computed in A's rows)", _flag_rates(_slice_by_keys(df_a, all3_keys)))
    _print_flag_block(f"Unique to {args.name_a}", _flag_rates(_slice_by_keys(df_a, only_a)))
    _print_flag_block(f"Unique to {args.name_b}", _flag_rates(_slice_by_keys(df_b, only_b)))
    _print_flag_block(f"Unique to {args.name_c}", _flag_rates(_slice_by_keys(df_c, only_c)))

    # -------------------------
    # NEW: Margin diagnostics
    # -------------------------
    print("\n--- Margin diagnostics (each model uses its OWN threshold) ---")
    # all3
    _print_margin_block("Group: all-3 shared errors", _slice_by_keys(df_a, all3_keys), args.name_a)
    _print_margin_block("Group: all-3 shared errors", _slice_by_keys(df_b, all3_keys), args.name_b)
    _print_margin_block("Group: all-3 shared errors", _slice_by_keys(df_c, all3_keys), args.name_c)

    # uniques
    _print_margin_block(f"Group: unique-to-{args.name_a}", _slice_by_keys(df_a, only_a), args.name_a)
    _print_margin_block(f"Group: unique-to-{args.name_b}", _slice_by_keys(df_b, only_b), args.name_b)
    _print_margin_block(f"Group: unique-to-{args.name_c}", _slice_by_keys(df_c, only_c), args.name_c)

    # pairwise-only groups
    ab_only = overall["ab_only"]
    ac_only = overall["ac_only"]
    bc_only = overall["bc_only"]

    _print_margin_block(f"Group: {args.name_a}∩{args.name_b} (not {args.name_c})", _slice_by_keys(df_a, ab_only), args.name_a)
    _print_margin_block(f"Group: {args.name_a}∩{args.name_b} (not {args.name_c})", _slice_by_keys(df_b, ab_only), args.name_b)

    _print_margin_block(f"Group: {args.name_a}∩{args.name_c} (not {args.name_b})", _slice_by_keys(df_a, ac_only), args.name_a)
    _print_margin_block(f"Group: {args.name_a}∩{args.name_c} (not {args.name_b})", _slice_by_keys(df_c, ac_only), args.name_c)

    _print_margin_block(f"Group: {args.name_b}∩{args.name_c} (not {args.name_a})", _slice_by_keys(df_b, bc_only), args.name_b)
    _print_margin_block(f"Group: {args.name_b}∩{args.name_c} (not {args.name_a})", _slice_by_keys(df_c, bc_only), args.name_c)

    # On all-3 shared keys: which model is most confident wrong?
    m_all3 = _merge_all3(df_a, df_b, df_c, all3_keys)
    winners_all = _winner_counts(m_all3, args.name_a, args.name_b, args.name_c)

    print("\n--- On ALL-3 shared errors: which model is MOST confident wrong? (largest |margin|) ---")
    _print_winner_dict("Overall:", winners_all)

    m_fn = m_all3[m_all3["error_type"] == "FN"]
    m_fp = m_all3[m_all3["error_type"] == "FP"]
    _print_winner_dict("FN only:", _winner_counts(m_fn, args.name_a, args.name_b, args.name_c))
    _print_winner_dict("FP only:", _winner_counts(m_fp, args.name_a, args.name_b, args.name_c))

    # Representative examples:
    hard_shared, border_shared = _rank_shared_examples(df_a, df_b, df_c, all3_keys, topk=args.topk)

    if len(hard_shared):
        hs_view = hard_shared[
            ["fraudulent_name", "real_name", "y_true", "error_type",
             "score_a", "margin_a", "score_b", "margin_b", "score_c", "margin_c", "mean_abs_margin"]
        ]
        bs_view = border_shared[
            ["fraudulent_name", "real_name", "y_true", "error_type",
             "score_a", "margin_a", "score_b", "margin_b", "score_c", "margin_c", "mean_abs_margin"]
        ]
        _print_examples_block(
            f"All-3 shared HARD errors (top {args.topk} by mean |margin|)",
            [
                f"fraud={r.fraudulent_name!r} | real={r.real_name!r} | y={int(r.y_true)} | type={r.error_type} | "
                f"{args.name_a} score={r.score_a:.6f} m={r.margin_a:+.6f} | "
                f"{args.name_b} score={r.score_b:.6f} m={r.margin_b:+.6f} | "
                f"{args.name_c} score={r.score_c:.6f} m={r.margin_c:+.6f} | "
                f"mean|m|={r.mean_abs_margin:.6f}"
                for r in hs_view.itertuples(index=False)
            ],
        )
        _print_examples_block(
            f"All-3 shared BORDERLINE errors (top {args.topk} by smallest mean |margin|)",
            [
                f"fraud={r.fraudulent_name!r} | real={r.real_name!r} | y={int(r.y_true)} | type={r.error_type} | "
                f"{args.name_a} score={r.score_a:.6f} m={r.margin_a:+.6f} | "
                f"{args.name_b} score={r.score_b:.6f} m={r.margin_b:+.6f} | "
                f"{args.name_c} score={r.score_c:.6f} m={r.margin_c:+.6f} | "
                f"mean|m|={r.mean_abs_margin:.6f}"
                for r in bs_view.itertuples(index=False)
            ],
        )
    else:
        _print_examples_block("All-3 shared HARD errors", [])
        _print_examples_block("All-3 shared BORDERLINE errors", [])

    # Unique hard per model (largest |margin| within that model among only_* keys)
    def unique_hard(df: pd.DataFrame, keys: Set[Tuple], name: str) -> None:
        d = _slice_by_keys(df, keys)
        if len(d) == 0:
            _print_examples_block(f"Unique HARD errors for {name}", [])
            return
        d = d.assign(abs_margin=d["margin"].abs()).sort_values("abs_margin", ascending=False).head(args.topk)
        _print_examples_block(f"Unique HARD errors for {name} (top {args.topk} by |margin|)", _example_lines(d, args.topk))

    unique_hard(df_a, only_a, args.name_a)
    unique_hard(df_b, only_b, args.name_b)
    unique_hard(df_c, only_c, args.name_c)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
USAGE EXAMPLE:

python text_to_image/compare_failure.py \
  --a text_to_image/Golden_and_Text/deja_misclassified.parquet --name-a deja \
  --b text_to_image/Golden_and_Text/source_misclassified.parquet --name-b source \
  --c text_to_image/Golden_and_Text/pacifico_misclassified.parquet --name-c pacifico \
  --topk 10
"""
