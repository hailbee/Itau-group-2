#!/usr/bin/env python3
"""
Failure-case deliverable for ONE big results file: deja_results.parquet

Assumed schema (no edge-case handling):
  - fraudulent_name (str)
  - real_name (str)
  - y_true (int 0/1)
  - score (float)
  - pred (int 0/1)
  - threshold (float; same value for all rows)

Behavior:
  - Reads deja_results.parquet
  - Computes margin = score - threshold
  - Splits into FN/FP/TP/TN
  - Prints:
      1) Core counts + FNR/FPR
      2) Quantiles for score/margin/length for FN vs FP
      3) Row-level “failure mode” flag rates (non-ASCII, punycode, hyphen-change, digit-change)
      4) Script-bucket presence rates (sampled)
      5) Alignment-based edit patterns (sampled): top substitutions, top inserts, top deletes
      6) Representative examples (borderline + confident + special buckets)

No files are saved.

Usage:
  python text_to_image/failure.py --input text_to_image/Golden_and_Text/deja_misclassified.parquet
"""

from __future__ import annotations

import argparse
import difflib
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: RapidFuzz for fast string metrics (nice-to-have, not required)
try:
    from rapidfuzz.fuzz import token_set_ratio as _rf_tsr
    from rapidfuzz.distance import Levenshtein as _rf_lev
except Exception:
    _rf_tsr = None
    _rf_lev = None


# -------------------------
# Small helpers
# -------------------------
def unicode_name(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except Exception:
        return "UNKNOWN"

def token_set_ratio(a: str, b: str) -> float:
    """TSR in [0,100]. Higher = more similar."""
    if _rf_tsr is not None:
        return float(_rf_tsr(a, b))
    # crude fallback (token overlap F1-like)
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 100.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    prec = inter / max(1, len(ta))
    rec = inter / max(1, len(tb))
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return 100.0 * f1

def levenshtein(a: str, b: str) -> int:
    """Edit distance. Lower = more similar."""
    if _rf_lev is not None:
        return int(_rf_lev.distance(a, b))
    # DP fallback (OK for a few examples; do NOT use for large loops)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def script_bucket_char(ch: str) -> str:
    if ord(ch) < 128:
        return "ASCII"
    name = unicode_name(ch).upper()
    cat = unicodedata.category(ch)
    if cat.startswith("S") or "SYMBOL" in name or "SIGN" in name or "MATHEMATICAL" in name:
        return "SYMBOL"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "LATIN" in name:
        return "LATIN"
    if "GREEK" in name:
        return "GREEK"
    if "HEBREW" in name:
        return "HEBREW"
    if "ARABIC" in name:
        return "ARABIC"
    return "OTHER"

def row_script_presence(a: str, b: str) -> Counter:
    buckets = set()
    for s in (a, b):
        for ch in s:
            buckets.add(script_bucket_char(ch))
    return Counter(buckets)

def quantiles(s: pd.Series) -> Dict[str, float]:
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    q = s.quantile(qs).to_dict()
    return {f"q{int(k*100):02d}": float(v) for k, v in q.items()}

def fmt_q(q: Dict[str, float]) -> str:
    return ", ".join([f"{k}={v:.6g}" for k, v in q.items()])


# -------------------------
# Edit-op patterns (sampled)
# -------------------------
def collect_edit_signals(
    a: str,
    b: str,
    sub_ctr: Counter,
    ins_ctr: Counter,
    del_ctr: Counter,
) -> None:
    """
    Uses SequenceMatcher opcodes:
      - counts single-char substitutions (x->y) when replace spans have equal length
      - counts inserted spans
      - counts deleted spans
    """
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "replace":
            sa = a[i1:i2]
            sb = b[j1:j2]
            if len(sa) == len(sb):
                for ca, cb in zip(sa, sb):
                    if ca != cb:
                        sub_ctr[(ca, cb)] += 1
        elif tag == "insert":
            ins_ctr[b[j1:j2]] += 1
        elif tag == "delete":
            del_ctr[a[i1:i2]] += 1

def summarize_top_subs(sub_ctr: Counter, topk: int) -> List[str]:
    lines = []
    for (a, b), n in sub_ctr.most_common(topk):
        extra = ""
        if ord(a) >= 128 or ord(b) >= 128:
            extra = f"  [{unicode_name(a) if ord(a)>=128 else 'ASCII'} -> {unicode_name(b) if ord(b)>=128 else 'ASCII'}]"
        lines.append(f"{a} -> {b} : {n}{extra}")
    return lines


# -------------------------
# Examples
# -------------------------
def format_example(row: pd.Series) -> str:
    a = str(row["fraudulent_name"])
    b = str(row["real_name"])
    score = float(row["score"])
    margin = float(row["margin"])

    tsr = token_set_ratio(a, b)
    lev = levenshtein(a, b)

    # small substitution preview
    sub_ctr = Counter()
    ins_ctr = Counter()
    del_ctr = Counter()
    collect_edit_signals(a, b, sub_ctr, ins_ctr, del_ctr)
    top_subs = [f"{x}->{y}" for (x, y), _ in sub_ctr.most_common(6)]

    flags = {
        "non_ascii": bool(row["has_non_ascii"]),
        "punycode": bool(row["has_punycode"]),
        "hyphen_change": bool(row["hyphen_change"]),
        "digit_change": bool(row["digit_change"]),
    }

    return (
        f"fraud={a!r} | real={b!r}\n"
        f"  score={score:.6f}, margin={margin:+.6f}, lens=({len(a)},{len(b)}), TSR={tsr:.1f}, Lev={lev}\n"
        f"  flags={flags}, subs={top_subs}"
    )


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="deja_results.parquet")
    ap.add_argument("--script_sample", type=int, default=50_000)
    ap.add_argument("--ops_sample", type=int, default=50_000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Read only what we need (fast)
    cols = ["fraudulent_name", "real_name", "y_true", "score", "pred", "threshold"]
    df = pd.read_parquet(args.input, columns=cols)

    # Hard assumptions: perfect schema
    df["fraudulent_name"] = df["fraudulent_name"].astype(str)
    df["real_name"] = df["real_name"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["score"] = df["score"].astype(float)

    thr_vals = pd.unique(df["threshold"])
    if len(thr_vals) != 1:
        raise ValueError("threshold must be a single constant across the file.")
    thr = float(thr_vals[0])

    df["margin"] = df["score"] - thr
    df["fraud_len"] = df["fraudulent_name"].str.len()
    df["real_len"] = df["real_name"].str.len()
    df["len_diff"] = (df["fraud_len"] - df["real_len"]).abs()

    # Row-level flags (vectorized; cheap)
    df["has_non_ascii"] = (
        df["fraudulent_name"].str.contains(r"[^\x00-\x7F]", regex=True)
        | df["real_name"].str.contains(r"[^\x00-\x7F]", regex=True)
    )
    df["has_punycode"] = df["fraudulent_name"].str.contains("xn--", regex=False) | df["real_name"].str.contains("xn--", regex=False)
    df["hyphen_change"] = df["fraudulent_name"].str.contains("-", regex=False) ^ df["real_name"].str.contains("-", regex=False)
    df["digit_change"] = df["fraudulent_name"].str.contains(r"\d", regex=True) ^ df["real_name"].str.contains(r"\d", regex=True)

    y = df["y_true"].to_numpy()
    p = df["pred"].to_numpy()

    fn = df[(y == 1) & (p == 0)].copy()
    fp = df[(y == 0) & (p == 1)].copy()
    tp = df[(y == 1) & (p == 1)].copy()
    tn = df[(y == 0) & (p == 0)].copy()

    # Core counts
    n = len(df)
    pos = len(tp) + len(fn)
    neg = len(tn) + len(fp)
    fnr = len(fn) / max(1, pos)
    fpr = len(fp) / max(1, neg)

    print("\n========================")
    print("FAILURE-CASE DELIVERABLE")
    print("========================")
    print(f"file: {args.input}")
    print(f"threshold: {thr:.9f}")
    print(f"n_total={n} | TP={len(tp)} TN={len(tn)} FN={len(fn)} FP={len(fp)}")
    print(f"FNR={fnr:.4f}  (FN/(TP+FN))")
    print(f"FPR={fpr:.4f}  (FP/(TN+FP))")

    print("\n--- Quantiles (FN vs FP) ---")
    print("FN margin:", fmt_q(quantiles(fn["margin"])))
    print("FP margin:", fmt_q(quantiles(fp["margin"])))
    print("FN score: ", fmt_q(quantiles(fn["score"])))
    print("FP score: ", fmt_q(quantiles(fp["score"])))
    print("FN fraud_len:", fmt_q(quantiles(fn["fraud_len"])))
    print("FP fraud_len:", fmt_q(quantiles(fp["fraud_len"])))

    print("\n--- Failure-mode flag rates (row-level) ---")
    for name, d in [("FN", fn), ("FP", fp)]:
        if len(d) == 0:
            continue
        print(
            f"{name}: non_ascii={d['has_non_ascii'].mean():.4f}, "
            f"punycode={d['has_punycode'].mean():.4f}, "
            f"hyphen_change={d['hyphen_change'].mean():.4f}, "
            f"digit_change={d['digit_change'].mean():.4f}"
        )

    # Script bucket presence on sample (row-level presence)
    def script_presence_rates(d: pd.DataFrame, sample_n: int) -> Dict[str, float]:
        if len(d) == 0:
            return {}
        d2 = d.sample(n=min(sample_n, len(d)), random_state=args.seed)
        ctr = Counter()
        for a_str, b_str in zip(d2["fraudulent_name"].to_numpy(), d2["real_name"].to_numpy()):
            ctr.update(row_script_presence(a_str, b_str))
        denom = max(1, len(d2))
        rates = {k: float(v / denom) for k, v in ctr.items()}
        return dict(sorted(rates.items(), key=lambda kv: (-kv[1], kv[0])))

    print("\n--- Script bucket presence (sampled) ---")
    print(f"FN sample_n={min(args.script_sample, len(fn))}: {script_presence_rates(fn, args.script_sample)}")
    print(f"FP sample_n={min(args.script_sample, len(fp))}: {script_presence_rates(fp, args.script_sample)}")

    # Edit patterns on sample
    def edit_patterns(d: pd.DataFrame, sample_n: int, topk: int) -> Tuple[List[str], List[str], List[str]]:
        if len(d) == 0:
            return [], [], []
        d2 = d.sample(n=min(sample_n, len(d)), random_state=args.seed)
        sub_ctr = Counter()
        ins_ctr = Counter()
        del_ctr = Counter()
        for a_str, b_str in zip(d2["fraudulent_name"].to_numpy(), d2["real_name"].to_numpy()):
            collect_edit_signals(a_str, b_str, sub_ctr, ins_ctr, del_ctr)
        top_subs = summarize_top_subs(sub_ctr, topk)
        top_ins = [f"{s!r}: {n}" for s, n in ins_ctr.most_common(topk)]
        top_del = [f"{s!r}: {n}" for s, n in del_ctr.most_common(topk)]
        return top_subs, top_ins, top_del

    print("\n--- Alignment-based edit patterns (sampled) ---")
    fn_subs, fn_ins, fn_del = edit_patterns(fn, args.ops_sample, args.topk)
    fp_subs, fp_ins, fp_del = edit_patterns(fp, args.ops_sample, args.topk)

    print(f"\nFN top substitutions (n={min(args.ops_sample, len(fn))}):")
    for line in fn_subs:
        print("  ", line)

    print(f"\nFP top substitutions (n={min(args.ops_sample, len(fp))}):")
    for line in fp_subs:
        print("  ", line)

    print("\nFN top inserts:")
    for line in fn_ins[:10]:
        print("  ", line)

    print("\nFP top inserts:")
    for line in fp_ins[:10]:
        print("  ", line)

    # Representative examples
    def show_block(title: str, d: pd.DataFrame) -> None:
        print("\n" + title)
        print("-" * len(title))
        if len(d) == 0:
            print("(none)")
            return
        for i, (_, r) in enumerate(d.iterrows(), 1):
            print(f"[{i}] {format_example(r)}\n")

    k = args.examples

    # Borderline / confident by margin
    show_block("FN borderline (closest to threshold)", fn.sort_values("margin", ascending=False).head(k))
    show_block("FN confident (far below threshold)", fn.sort_values("margin", ascending=True).head(k))
    show_block("FP borderline (closest to threshold)", fp.sort_values("margin", ascending=True).head(k))
    show_block("FP confident (far above threshold)", fp.sort_values("margin", ascending=False).head(k))

    # Special buckets (use near-threshold first so they are believable)
    show_block("FN non-ASCII examples (often the core miss mode)", fn[fn["has_non_ascii"]].sort_values("margin", ascending=False).head(k))
    show_block("FP punycode examples (xn--)", fp[fp["has_punycode"]].sort_values("margin", ascending=True).head(k))
    show_block("FP hyphen-change examples", fp[fp["hyphen_change"]].sort_values("margin", ascending=True).head(k))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
