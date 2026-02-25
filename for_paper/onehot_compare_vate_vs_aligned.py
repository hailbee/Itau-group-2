#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onehot_compare_vate_vs_aligned.py

Summarize how "alignment" (student models: deja/source/pacifico) changes performance
ACROSS multi-hot mechanism flags (one-hots) already present in a parquet.

This version:
  - outputs BOTH error COUNTS and error RATES (percentages as fractions in [0,1])
  - does NOT treat `label` as a flag
  - excludes embedding columns (base + aligned) from flag detection
  - excludes `identical` entirely (not in CSV, not in report)
  - supports binary flags stored as bool/int OR float {0.0,1.0}

Outputs
-------
1) --out-summary-csv: per-flag summary with counts + rates
2) --out-report-txt (optional): prints EVERY flag for each aligned model vs vate

Example
-------
python for_paper/onehot_compare_vate_vs_aligned.py \
  --data for_paper/vate_onehot_with_3aligned.parquet \
  --aligned-tags deja,source,pacifico \
  --out-summary-csv for_paper/onehot_delta_summary.csv \
  --out-report-txt for_paper/onehot_delta_report.txt \
  --exclude-identical-spoof \
  --device cuda
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


# -------------------------
# Utilities
# -------------------------
def pick_device(override: Optional[str]) -> torch.device:
    if override:
        d = torch.device(override)
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    return s.strip().casefold()


def has_both_classes(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    u = np.unique(y)
    return (len(u) >= 2) and (0 in u) and (1 in u)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if not has_both_classes(y_true):
        raise ValueError("Cannot compute Youden threshold on one-class set.")
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


_INT_SUFFIX_RE = re.compile(r"-?\d+$")


def prefixed_numeric_cols(all_cols: Sequence[str], prefix: str) -> List[str]:
    cols: List[str] = []
    for c in all_cols:
        if isinstance(c, str) and c.startswith(prefix):
            suf = c[len(prefix) :]
            if _INT_SUFFIX_RE.fullmatch(suf):
                cols.append(c)
    cols.sort(key=lambda c: int(c[len(prefix) :]))
    return cols


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def is_binary_series(s: pd.Series) -> bool:
    """
    Binary if (after dropping NA) unique values are subset of {0,1}.
    Supports: bool, integer dtypes (incl pandas nullable), float dtypes with exact {0.0,1.0}.
    """
    if pd.api.types.is_bool_dtype(s):
        return True

    x = s.dropna()
    if len(x) == 0:
        return False

    if pd.api.types.is_integer_dtype(x):
        vals = x.unique()
        vals_set = set(int(v) for v in vals.tolist())
        return vals_set.issubset({0, 1})

    if pd.api.types.is_float_dtype(x):
        vals = x.unique()
        for v in vals.tolist():
            if v not in (0.0, 1.0):
                return False
        return True

    return False


# -------------------------
# Cosine scoring
# -------------------------
@torch.inference_mode()
def paired_cosine_from_arrays(
    fraud: np.ndarray,
    real: np.ndarray,
    device: torch.device,
    batch_rows: int,
) -> np.ndarray:
    if fraud.shape != real.shape:
        raise ValueError(f"shape mismatch: {fraud.shape} vs {real.shape}")

    n = int(fraud.shape[0])
    bs = int(max(1, batch_rows))
    out = np.empty((n,), dtype=np.float32)

    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        f = torch.from_numpy(fraud[i0:i1]).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(real[i0:i1]).to(device=device, dtype=torch.float32)
        f = F.normalize(f, dim=1)
        r = F.normalize(r, dim=1)
        out[i0:i1] = torch.sum(f * r, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)

    return np.clip(out, -1.0, 1.0)


@torch.inference_mode()
def paired_cosine_from_parquet(
    path: str,
    fraud_cols: List[str],
    real_cols: List[str],
    device: torch.device,
    batch_rows: int,
) -> np.ndarray:
    df = pd.read_parquet(path, columns=fraud_cols + real_cols)
    fraud = df[fraud_cols].to_numpy(dtype=np.float32, copy=True)
    real = df[real_cols].to_numpy(dtype=np.float32, copy=True)
    del df
    return paired_cosine_from_arrays(fraud, real, device, batch_rows)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--aligned-tags", required=True)

    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")
    ap.add_argument("--label-col", default="label")

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")
    ap.add_argument("--aligned-fraud-base", default="fraud_txt_emb_aligned_")
    ap.add_argument("--aligned-real-base", default="real_txt_emb_aligned_")

    ap.add_argument("--exclude-identical-spoof", action="store_true")
    ap.add_argument("--score-batch-rows", type=int, default=8192)
    ap.add_argument("--device", default=None)

    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-report-txt", default=None)

    args = ap.parse_args()
    tags = parse_csv_list(args.aligned_tags)
    if not tags:
        raise ValueError("--aligned-tags parsed empty")

    device = pick_device(args.device)
    print(f"[INFO] device={device}")
    print(f"[INFO] tags={tags}")

    df = pd.read_parquet(args.data)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col!r}")

    y = (df[args.label_col].to_numpy(dtype=np.float32, copy=False) >= 0.5).astype(np.int32)

    # Eval mask for fitting thresholds
    if args.exclude_identical_spoof:
        f_norm = df[args.fraud_name_col].astype(str).map(normalize_name).to_numpy()
        r_norm = df[args.real_name_col].astype(str).map(normalize_name).to_numpy()
        eval_mask = ~((y == 1) & (f_norm == r_norm))
    else:
        eval_mask = np.ones(len(df), dtype=bool)

    if not has_both_classes(y[eval_mask]):
        raise RuntimeError("Threshold eval set became one-class; cannot compute Youden thresholds.")

    all_cols = list(df.columns)

    # Base embedding columns
    base_fraud_cols = prefixed_numeric_cols(all_cols, args.fraud_prefix)
    base_real_cols = prefixed_numeric_cols(all_cols, args.real_prefix)
    if not base_fraud_cols or not base_real_cols:
        raise ValueError("Could not find base embedding columns; check --fraud-prefix/--real-prefix.")
    if len(base_fraud_cols) != len(base_real_cols):
        raise ValueError(f"Base dim mismatch: fraud={len(base_fraud_cols)} real={len(base_real_cols)}")

    # Aligned embedding columns (for exclusion + sanity)
    aligned_embed_cols: List[str] = []
    for t in tags:
        fcols = prefixed_numeric_cols(all_cols, f"{args.aligned_fraud_base}{t}_")
        rcols = prefixed_numeric_cols(all_cols, f"{args.aligned_real_base}{t}_")
        if not fcols or not rcols:
            raise ValueError(f"Missing aligned embedding cols for tag={t!r}")
        if len(fcols) != len(base_fraud_cols) or len(rcols) != len(base_real_cols):
            raise ValueError(
                f"[{t}] aligned dim mismatch: fraud={len(fcols)} real={len(rcols)} vs base={len(base_fraud_cols)}"
            )
        aligned_embed_cols.extend(fcols)
        aligned_embed_cols.extend(rcols)

    # Detect flag columns (exclude label/names/embeddings; exclude 'identical')
    excluded = set(base_fraud_cols) | set(base_real_cols) | set(aligned_embed_cols)
    excluded |= {args.fraud_name_col, args.real_name_col, args.label_col}
    excluded |= {"identical"}  # remove entirely

    flag_cols: List[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if is_binary_series(df[c]):
            flag_cols.append(c)

    print(f"[INFO] n_rows={len(df):,}")
    print(f"[INFO] n_flags_detected={len(flag_cols)}")
    if len(flag_cols) == 0:
        raise RuntimeError("No binary flag columns detected (after exclusions).")

    # Compute thresholds + error masks per model
    thr: Dict[str, float] = {}
    err: Dict[str, np.ndarray] = {}

    print("[INFO] computing vate scores...")
    sc_base = paired_cosine_from_parquet(
        args.data, base_fraud_cols, base_real_cols, device=device, batch_rows=int(args.score_batch_rows)
    )
    thr_base = youden_threshold(y[eval_mask], sc_base[eval_mask])
    pred_base = (sc_base >= thr_base).astype(np.int32)
    err_base = (pred_base != y)

    thr["vate"] = float(thr_base)
    err["vate"] = err_base
    print(f"[DIAG] vate thr={thr_base:.6f} | total_errors={int(err_base.sum()):,}")

    for t in tags:
        fcols = prefixed_numeric_cols(all_cols, f"{args.aligned_fraud_base}{t}_")
        rcols = prefixed_numeric_cols(all_cols, f"{args.aligned_real_base}{t}_")

        print(f"[INFO] computing aligned_{t} scores...")
        sc = paired_cosine_from_parquet(
            args.data, fcols, rcols, device=device, batch_rows=int(args.score_batch_rows)
        )
        th = youden_threshold(y[eval_mask], sc[eval_mask])
        pred = (sc >= th).astype(np.int32)
        er = (pred != y)

        thr[f"aligned_{t}"] = float(th)
        err[f"aligned_{t}"] = er
        print(f"[DIAG] aligned_{t} thr={th:.6f} | total_errors={int(er.sum()):,}")

    # Per-flag counts + rates
    rows: List[Dict[str, object]] = []
    N = int(len(df))

    for flag in flag_cols:
        in_flag = df[flag].fillna(0).to_numpy()
        # accepts bool/int/float 0/1
        in_flag = (in_flag.astype(np.float32) >= 0.5)
        n_flag = int(in_flag.sum())
        if n_flag == 0:
            continue

        base_err_count = int(err["vate"][in_flag].sum())
        base_err_rate = float(base_err_count / n_flag)

        row: Dict[str, object] = {
            "flag": flag,
            "n_flag": n_flag,
            "flag_rate": float(n_flag / N),
            "vate_error_count": base_err_count,
            "vate_error_rate": base_err_rate,
        }

        for t in tags:
            m = f"aligned_{t}"
            aligned_err_count = int(err[m][in_flag].sum())
            aligned_err_rate = float(aligned_err_count / n_flag)
            row[f"{m}_error_count"] = aligned_err_count
            row[f"{m}_error_rate"] = aligned_err_rate
            row[f"{m}_delta_error_count"] = int(aligned_err_count - base_err_count)
            row[f"{m}_delta_error_rate"] = float(aligned_err_rate - base_err_rate)

        rows.append(row)

    out_summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary_csv)), exist_ok=True)
    out_summary.to_csv(args.out_summary_csv, index=False)
    print(f"[OK] wrote summary: {args.out_summary_csv} (rows={len(out_summary):,})")

    # Full text report (EVERY flag)
    if args.out_report_txt:
        lines: List[str] = []
        lines.append(f"data={args.data}")
        lines.append(f"tags={tags}")
        lines.append(f"n_rows={len(df):,}")
        lines.append(f"n_flags={len(out_summary):,}")
        lines.append(f"exclude_identical_spoof={bool(args.exclude_identical_spoof)}")
        lines.append("")
        lines.append("Per-model thresholds (Youden):")
        lines.append(f"  vate: {thr['vate']:.6f}")
        for t in tags:
            lines.append(f"  aligned_{t}: {thr[f'aligned_{t}']:.6f}")
        lines.append("")

        # Keep stable order: as produced in CSV (df column order filtered), no sorting.
        for t in tags:
            m = f"aligned_{t}"
            lines.append(f"=== {m} vs vate ===")
            for _, r in out_summary.iterrows():
                # r is a Series; cast carefully
                flag = str(r["flag"])
                n_flag = int(r["n_flag"])
                v_ec = int(r["vate_error_count"])
                v_er = float(r["vate_error_rate"])
                a_ec = int(r[f"{m}_error_count"])
                a_er = float(r[f"{m}_error_rate"])
                d_ec = int(r[f"{m}_delta_error_count"])
                d_er = float(r[f"{m}_delta_error_rate"])

                lines.append(
                    f"  {flag:<30} "
                    f"n={n_flag:>7d} "
                    f"vate_err={v_ec:>7d} ({v_er*100:>6.2f}%) "
                    f"{m}_err={a_ec:>7d} ({a_er*100:>6.2f}%) "
                    f"delta={d_ec:+7d} ({d_er*100:+7.2f}%)"
                )
            lines.append("")

        os.makedirs(os.path.dirname(os.path.abspath(args.out_report_txt)), exist_ok=True)
        with open(args.out_report_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[OK] wrote report: {args.out_report_txt}")

    print("[DONE]")


if __name__ == "__main__":
    main()