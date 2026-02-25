#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mechanism_onehot_audit.py

Compute multi-label (multi-hot) mechanism flags for (fraudulent_name, real_name) pairs and export:

1) An output parquet with added flag columns + combo id/label + an all-zero indicator
2) A text file listing ALL rows where every mechanism flag is 0 (if any), so you can manually
   inspect and classify them.

This script is intentionally self-contained (no dependency on for_paper/mechanisms.py).

Example:
  python for_paper/mechanism_onehot_audit.py \
    --data ../Downloads/vate_test.parquet \
    --out-parquet for_paper/deja_onehot.parquet \
    --font-tag deja
"""

from __future__ import annotations

import argparse
import difflib
import os
import unicodedata
from typing import Dict, List, Sequence, Tuple

import pandas as pd


# -------------------------
# Multi-hot mechanism flags
# -------------------------

COMMON_TLDS: set[str] = {
    "com", "net", "org", "io", "co", "gov", "edu", "us", "uk", "de", "fr", "jp", "br", "ru", "cn", "in",
    "info", "biz", "app", "dev", "ai", "me", "tv",
}

# Order matters for combo_id bit positions.
FLAG_KEYS: List[str] = [
    "identical",
    "extension",
    "punycode",
    "non_ascii",
    "unicode_marks_only",
    "unicode_homoglyph",
    "case_change_only",
    "whitespace_change",
    "hyphen_change",
    "digit_change",
    "transposition",
    "insertion",
    "deletion",
    "substitution",
    "digit_substitution",
]


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).strip()


def _strip_marks(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) >= 128 for ch in s)


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


def _is_extension_like(f: str, r: str) -> bool:
    # domain-ish: one is the other plus ".tld" or removed ".tld"
    def split_tld(x: str) -> Tuple[str, str]:
        if "." not in x:
            return x, ""
        base, tld = x.rsplit(".", 1)
        return base, tld.lower()

    fb, ft = split_tld(f)
    rb, rt = split_tld(r)

    if ft in COMMON_TLDS and fb == r:
        return True
    if rt in COMMON_TLDS and rb == f:
        return True
    if ft in COMMON_TLDS and rt in COMMON_TLDS and fb == rb and ft != rt:
        return True
    return False


def _diff_ops(real_s: str, fraud_s: str) -> Tuple[int, int, int]:
    """
    Return (n_insert, n_delete, n_replace) using SequenceMatcher opcodes,
    where we treat "real" as the reference and "fraud" as the edited string.

    insert  => fraud has extra characters (insertion)
    delete  => fraud is missing characters (deletion)
    replace => substitution/replace
    """
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


def mechanism_flags(fraudulent: str, real: str) -> Dict[str, int]:
    """
    Multi-label mechanism flags (0/1) for a single (fraud, real) pair.
    Multiple flags may be 1 for a single pair.
    """
    out: Dict[str, int] = {k: 0 for k in FLAG_KEYS}

    f = _nfkc(fraudulent)
    r = _nfkc(real)
    
    if f == r:
        out["identical"] = 1

    if not f and not r:
        return out

    # cheap string-level flags
    out["punycode"] = int(("xn--" in f) or ("xn--" in r))
    out["non_ascii"] = int(_has_non_ascii(f) or _has_non_ascii(r))
    out["hyphen_change"] = int(("-" in f) ^ ("-" in r))
    out["whitespace_change"] = int((any(ch.isspace() for ch in f)) ^ (any(ch.isspace() for ch in r)))
    out["digit_change"] = int((any(ch.isdigit() for ch in f)) ^ (any(ch.isdigit() for ch in r)))

    # extension-like (domain/tld)
    out["extension"] = int(_is_extension_like(f, r))

    # unicode marks / diacritics relationship
    f_stripped = _strip_marks(f)
    r_stripped = _strip_marks(r)
    if (f_stripped.casefold() == r_stripped.casefold()) and (f.casefold() != r.casefold()):
        out["unicode_marks_only"] = 1
        if out["non_ascii"]:
            out["unicode_homoglyph"] = 1

    # case-only changes
    if (f.casefold() == r.casefold()) and (f != r):
        out["case_change_only"] = 1

    # adjacent transposition (single swap)
    out["transposition"] = int(_is_single_adjacent_swap(r, f) or _is_single_adjacent_swap(f, r))

    # edit operations (insertion/deletion/substitution)
    # Use casefold to avoid case-only turning into "replace"
    n_ins, n_del, n_rep = _diff_ops(r.casefold(), f.casefold())
    out["insertion"] = int(n_ins > 0)
    out["deletion"] = int(n_del > 0)
    out["substitution"] = int(n_rep > 0)

    # digit_substitution: replacements where diffs involve digits (simple heuristic)
    if len(f) == len(r) and f != r:
        diffs = [(a, b) for a, b in zip(f, r) if a != b]
        if diffs:
            digitish = 0
            for a, b in diffs:
                if (a.isdigit() and b.isdigit()) or (a.isdigit() and not b.isdigit()) or (b.isdigit() and not a.isdigit()):
                    digitish += 1
            if digitish >= 1 and digitish == len(diffs) and len(diffs) <= 8:
                out["digit_substitution"] = 1

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


# -------------------------
# Reporting helpers
# -------------------------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in --data: {missing}")


def write_all_zero_text(
    df: pd.DataFrame,
    fraud_col: str,
    real_col: str,
    label_col: str | None,
    font_col: str,
    out_txt: str,
    max_rows: int | None,
) -> int:
    flag_cols = [c for c in FLAG_KEYS if c in df.columns]
    all_zero = (df[flag_cols].sum(axis=1) == 0)
    keep_cols = [font_col, fraud_col, real_col] + ([label_col] if (label_col and label_col in df.columns) else [])
    sub = df.loc[all_zero, keep_cols].copy()
    n = int(len(sub))

    header_lines: List[str] = []
    header_lines.append(f"n_all_zero={n}")
    header_lines.append(f"flags={','.join(flag_cols)}")
    header_lines.append("")
    header = "\n".join(header_lines)

    if n == 0:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(header + "\n(No all-zero rows.)\n")
        return 0

    if max_rows is not None and max_rows > 0:
        sub = sub.head(int(max_rows))

    lines: List[str] = [header]
    for i, row in sub.iterrows():
        lab = ""
        if label_col and label_col in df.columns:
            lab = f" | label={row[label_col]}"
        lines.append(f"[row={int(i)}] font={row[font_col]}{lab}")
        lines.append(f"  fraud: {str(row[fraud_col])}")
        lines.append(f"  real : {str(row[real_col])}")
        lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return n


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True, help="Input parquet containing fraud/real name columns.")
    ap.add_argument("--out-parquet", required=True, help="Output parquet with multi-hot flags appended.")
    ap.add_argument("--out-all-zero-txt", default=None, help="Text file listing rows where all flags are 0.")
    ap.add_argument("--out-summary-csv", default=None, help="Optional CSV summary by (font, combo).")

    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")
    ap.add_argument("--label-col", default="label", help="Optional label column (included in all-zero txt if present).")
    ap.add_argument("--font-tag", default=None, help="If dataset is one font, store this tag in a '_font' column.")
    ap.add_argument("--font-col", default=None, help="If present, use this column as font instead of --font-tag.")

    ap.add_argument("--max-all-zero-rows", type=int, default=5000, help="Cap rows written to the all-zero text file (0 = no cap).")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)

    _ensure_cols(df, [args.fraud_name_col, args.real_name_col])

    # Determine font column
    if args.font_col is not None:
        if args.font_col not in df.columns:
            raise ValueError(f"--font-col {args.font_col!r} not found in data.")
        df["_font"] = df[args.font_col].astype(str)
    else:
        df["_font"] = str(args.font_tag) if args.font_tag is not None else "NA"

    fraud = df[args.fraud_name_col].astype(str).to_numpy()
    real = df[args.real_name_col].astype(str).to_numpy()

    flags = mechanism_flags_df(fraud, real)
    df = pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)

    df["mech_combo_id"] = [combo_id(r) for _, r in flags.iterrows()]
    df["mech_combo"] = [combo_label(r) for _, r in flags.iterrows()]
    df["mech_all_zero"] = (flags.sum(axis=1) == 0).astype(int)

    # Write parquet
    df.to_parquet(args.out_parquet, index=False)
    print(f"[OK] wrote parquet: {args.out_parquet}")

    # Write all-zero text file
    base = os.path.splitext(args.out_parquet)[0]
    out_txt = args.out_all_zero_txt or (base + "_all_zero.txt")

    max_rows = None
    if args.max_all_zero_rows is not None:
        mr = int(args.max_all_zero_rows)
        if mr > 0:
            max_rows = mr

    label_col = args.label_col if args.label_col in df.columns else None
    n0 = write_all_zero_text(
        df=df,
        fraud_col=args.fraud_name_col,
        real_col=args.real_name_col,
        label_col=label_col,
        font_col="_font",
        out_txt=out_txt,
        max_rows=max_rows,
    )
    if n0 == 0:
        print(f"[OK] wrote all-zero audit (none found): {out_txt}")
    else:
        print(f"[OK] wrote all-zero audit: {out_txt} (n_all_zero={n0:,})")

    # Optional summary by (font, combo)
    if args.out_summary_csv is not None:
        grp_cols = ["_font", "mech_combo"]
        if label_col is not None:
            summary = df.groupby(grp_cols, as_index=False).agg(
                n=("mech_combo", "size"),
                mean_label=(label_col, "mean"),
                all_zero_rate=("mech_all_zero", "mean"),
            )
        else:
            summary = df.groupby(grp_cols, as_index=False).agg(
                n=("mech_combo", "size"),
                all_zero_rate=("mech_all_zero", "mean"),
            )
        summary = summary.sort_values(["_font", "n"], ascending=[True, False])
        summary.to_csv(args.out_summary_csv, index=False)
        print(f"[OK] wrote summary csv: {args.out_summary_csv}")


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES

# Single-font parquet (use a tag)
python for_paper/mechanism_onehot_audit.py \
  --data ../Downloads/vate_test.parquet \
  --out-parquet for_paper/vate_onehot.parquet \
  --font-tag vate

# If your parquet already contains a font column (e.g., 'font')
python for_paper/mechanism_onehot_audit.py \
  --data ../Downloads/vate_all_fonts.parquet \
  --out-parquet for_paper/allfonts_onehot.parquet \
  --font-col font \
  --out-summary-csv for_paper/allfonts_onehot_combo_summary.csv
"""