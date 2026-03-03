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
  python for_paper/mechanism_multihot_audit.py \
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

# Domain TLDs + common file extensions (Woodbridge covers process names too).
COMMON_SUFFIXES: set[str] = {
    # common TLDs
    "com", "net", "org", "io", "co", "gov", "edu", "us", "uk", "de", "fr", "jp", "br", "ru", "cn", "in",
    "info", "biz", "app", "dev", "ai", "me", "tv",
    # common file extensions
    "exe", "dll", "sys", "scr", "bat", "cmd", "ps1", "vbs", "js", "jar", "msi",
}

# Characters that often act as separators in spoof strings.
SEPARATORS: set[str] = set(" \t\r\n-_./\\:·•—–‐-‒―")

# Invisible / format-ish characters frequently used to hide differences.
INVISIBLE_CODEPOINTS: set[str] = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad",  # SOFT HYPHEN
    "\u2060",  # WORD JOINER
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR (deprecated but still appears)
}

# Lightweight confusable mapping (not a full UTS#39 implementation, but catches common stuff).
CONFUSABLE_CHAR_MAP: Dict[str, str] = {
    # Latin extensions that look like ASCII
    "ł": "l", "Ł": "l",
    "ø": "o", "Ø": "o",
    "đ": "d", "Đ": "d",
    "ı": "i",  # dotless i
    # Cyrillic lookalikes (common)
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
    # Greek lookalikes (common)
    "α": "a", "Α": "a",
    "ο": "o", "Ο": "o",
    "ρ": "p", "Ρ": "p",
    "χ": "x", "Χ": "x",
    "ν": "v", "Ν": "v",
}

# Multi-char visual confusables commonly cited in homoglyph/name spoofing.
MULTICHAR_CONFUSABLES: List[Tuple[str, str]] = [
    ("rn", "m"),
    ("cl", "d"),
    ("vv", "w"),
    ("l1", "h"),   # sometimes looks like h in some fonts
    ("1l", "h"),
]

# STRICT leetspeak substitutions: conservative, unambiguous.
# (This is intentional: avoids exploding combos and keeps the bucket interpretable.)
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

# Order matters for combo_id bit positions.
# Existing keys first (unchanged), then new keys appended at the end.
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
    # NEW (appended to preserve existing bit positions)
    "zero_width_or_format",
    "mixed_script",
    "separator_change",
    "repeat_char",
    "affix",
    "numeric_affix",
    "visual_confusable",
    "multichar_confusable",
    # NEW NEW
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
    """
    Very lightweight script classifier based on Unicode character names.
    Good enough to flag obvious mixed-script (Latin+Cyrillic/Greek/etc.) attacks.
    """
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


def _is_extension_like(f: str, r: str) -> bool:
    def split_suffix(x: str) -> Tuple[str, str]:
        if "." not in x:
            return x, ""
        base, suf = x.rsplit(".", 1)
        return base, suf.lower()

    fb, ft = split_suffix(f)
    rb, rt = split_suffix(r)

    if ft in COMMON_SUFFIXES and fb == r:
        return True
    if rt in COMMON_SUFFIXES and rb == f:
        return True
    if ft in COMMON_SUFFIXES and rt in COMMON_SUFFIXES and fb == rb and ft != rt:
        return True
    return False


def _diff_ops(real_s: str, fraud_s: str) -> Tuple[int, int, int]:
    """
    Return (n_insert, n_delete, n_replace) using SequenceMatcher opcodes,
    where we treat "real" as the reference and "fraud" as the edited string.
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
    """
    Return (is_affix, extra) where extra is the added prefix/suffix when one string
    is the other plus a short affix. We treat casefolded strings.
    """
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
    """
    Lightweight 'visual' skeleton:
      1) apply common multichar confusables on casefolded text
      2) map common single-char Unicode lookalikes into ASCII-ish forms
    """
    t = _multichar_skeleton(s)
    out_chars: List[str] = []
    for ch in t:
        out_chars.append(CONFUSABLE_CHAR_MAP.get(ch, ch))
    return "".join(out_chars)


def _is_strict_leet_pair(f: str, r: str) -> bool:
    """
    Strict definition:
      - operate position-by-position (so require same length)
      - any mismatch must be explainable by a leet substitution in LEET_MAP
      - require at least one leet substitution was actually used
    """
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

    if f == r:
        out["identical"] = 1

    if not f and not r:
        return out

    # cheap string-level flags (preserved)
    out["punycode"] = int(("xn--" in f) or ("xn--" in r))
    out["non_ascii"] = int(_has_non_ascii(f) or _has_non_ascii(r))
    out["hyphen_change"] = int(("-" in f) ^ ("-" in r))
    out["whitespace_change"] = int((any(ch.isspace() for ch in f)) ^ (any(ch.isspace() for ch in r)))
    out["digit_change"] = int((any(ch.isdigit() for ch in f)) ^ (any(ch.isdigit() for ch in r)))

    # extension-like (domain/tld or file extension)
    out["extension"] = int(_is_extension_like(f, r))

    # unicode marks / diacritics relationship (preserved)
    f_stripped = _strip_marks(f)
    r_stripped = _strip_marks(r)
    if (f_stripped.casefold() == r_stripped.casefold()) and (f.casefold() != r.casefold()):
        out["unicode_marks_only"] = 1
        if out["non_ascii"]:
            out["unicode_homoglyph"] = 1  # legacy behavior (name is imperfect but preserved)

    # case-only changes
    if (f.casefold() == r.casefold()) and (f != r):
        out["case_change_only"] = 1

    # adjacent transposition (single swap)
    out["transposition"] = int(_is_single_adjacent_swap(r, f) or _is_single_adjacent_swap(f, r))

    # edit operations (insertion/deletion/substitution)
    n_ins, n_del, n_rep = _diff_ops(r.casefold(), f.casefold())
    out["insertion"] = int(n_ins > 0)
    out["deletion"] = int(n_del > 0)
    out["substitution"] = int(n_rep > 0)

    # digit_substitution (preserved heuristic)
    if len(f) == len(r) and f != r:
        diffs = [(a, b) for a, b in zip(f, r) if a != b]
        if diffs:
            digitish = 0
            for a, b in diffs:
                if (a.isdigit() and b.isdigit()) or (a.isdigit() and not b.isdigit()) or (b.isdigit() and not a.isdigit()):
                    digitish += 1
            if digitish >= 1 and digitish == len(diffs) and len(diffs) <= 8:
                out["digit_substitution"] = 1

    # -------------------------
    # New buckets (visual spoofing style)
    # -------------------------

    out["zero_width_or_format"] = int(_contains_invisible_or_format(f_raw) or _contains_invisible_or_format(r_raw))
    out["mixed_script"] = int(_has_mixed_script(f) or _has_mixed_script(r))

    # separator-only changes
    f_nosep = _remove_separators(f).casefold()
    r_nosep = _remove_separators(r).casefold()
    if f != r and f_nosep and (f_nosep == r_nosep) and (
        any(ch in SEPARATORS or ch.isspace() for ch in f) or any(ch in SEPARATORS or ch.isspace() for ch in r)
    ):
        out["separator_change"] = 1

    # repeated-char stretching
    f_coll = _collapse_runs(f).casefold()
    r_coll = _collapse_runs(r).casefold()
    if f != r and f_coll and (f_coll == r_coll) and (len(f) != len(r)):
        out["repeat_char"] = 1

    # short prefix/suffix append/remove
    is_aff, extra = _affix_extra(f.casefold(), r.casefold())
    out["affix"] = int(is_aff)
    if is_aff:
        extra_compact = _remove_separators(extra)
        out["numeric_affix"] = int(extra_compact.isdigit() and len(extra_compact) > 0)

    # visual confusable skeleton (Unicode lookalikes + multichar swaps)
    if f != r and (f.casefold() != r.casefold()):
        sk_f = _confusable_skeleton(f)
        sk_r = _confusable_skeleton(r)
        if sk_f == sk_r and sk_f != "":
            out["visual_confusable"] = 1
        if _multichar_skeleton(f) == _multichar_skeleton(r) and _multichar_skeleton(f) != "":
            out["multichar_confusable"] = 1

    # STRICT leet_pair:
    # - require fraud actually uses a leet character
    # - require real does NOT use leet characters (directional: "fraud is leeted version of real")
    # - require every mismatch is explainable by LEET_MAP, position-by-position
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
    ap.add_argument("--add-op-counts", action="store_true", help="Also add mech_n_ins/mech_n_del/mech_n_rep/mech_n_ops integer columns.")
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

    # Optional: add edit-op counts as numeric features (not part of FLAG_KEYS)
    if args.add_op_counts:
        n_ins_list: List[int] = []
        n_del_list: List[int] = []
        n_rep_list: List[int] = []
        for f, r in zip(fraud, real):
            f2 = _nfkc(f).casefold()
            r2 = _nfkc(r).casefold()
            ni, nd, nr = _diff_ops(r2, f2)
            n_ins_list.append(int(ni))
            n_del_list.append(int(nd))
            n_rep_list.append(int(nr))
        df["mech_n_ins"] = n_ins_list
        df["mech_n_del"] = n_del_list
        df["mech_n_rep"] = n_rep_list
        df["mech_n_ops"] = [int(a + b + c) for a, b, c in zip(n_ins_list, n_del_list, n_rep_list)]

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
  --font-tag vate \
  --add-op-counts

# If your parquet already contains a font column (e.g., 'font')
python for_paper/mechanism_onehot_audit.py \
  --data ../Downloads/vate_all_fonts.parquet \
  --out-parquet for_paper/allfonts_onehot.parquet \
  --font-col font \
  --out-summary-csv for_paper/allfonts_onehot_combo_summary.csv \
  --add-op-counts
"""