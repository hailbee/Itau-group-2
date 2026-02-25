#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onehot_compare_vate_vs_aligned.py

Summarize how "alignment" (student models: deja/source/pacifico) changes performance
ACROSS multi-hot mechanism flags (one-hots) already present in a parquet.

This is designed to replace clustering when rows can have multiple mechanism flags = 1.

Core idea
---------
For each model (VATE base + each aligned tag), we:
  1) compute a per-model Youden threshold on an eval mask (optionally excluding identical spoof rows)
  2) define predictions: pred = (paired_cosine >= thr)
  3) define errors = (pred != y_true)
  4) define confidence = |margin| where margin = score - thr
     - "more confidently wrong" => larger |margin| on error rows

Then we summarize, for each flag column (multi-hot allowed):
  - error_rate within flag
  - FP/FN rates within flag
  - confidence-weighted error: mean(|margin| * 1[error]) within flag
  - fixed/broken vs VATE within flag (where "fixed" means base wrong -> aligned correct; "broken" is base correct -> aligned wrong)
  - optional confidence-percentile breakdowns

IMPORTANT: Mechanism flags do NOT change across models (they are string-diff features).
What changes is: which mechanism-flag groups have higher/lower error rates after alignment.

Input parquet assumptions
------------------------
Required columns:
  - fraudulent_name, real_name, label (1=spoof)
  - VATE base embeddings:
        fraud_txt_emb_0.., real_txt_emb_0..
  - Aligned embeddings for each tag:
        fraud_txt_emb_aligned_<tag>_0.., real_txt_emb_aligned_<tag>_0..
  - One-hot / multi-hot mechanism columns already present (0/1)

This script reads only the needed columns; it does NOT modify the parquet.

Outputs
-------
1) --out-summary-csv: per-flag summary (one row per flag)
2) --out-bins-csv (optional): per-flag * confidence-bin summary (long format)
3) --out-report-txt (optional): human-readable top improvements / regressions

Example
-------
python for_paper/onehot_compare_vate_vs_aligned.py \
  --data for_paper/vate_onehot_with_3aligned.parquet \
  --aligned-tags deja,source,pacifico \
  --out-summary-csv for_paper/onehot_delta_summary.csv \
  --out-bins-csv for_paper/onehot_delta_by_conf_bin.csv \
  --out-report-txt for_paper/onehot_delta_report.txt \
  --exclude-identical-spoof \
  --confidence-quantiles 0,0.25,0.5,0.75,0.9,0.95,0.99,1 \
  --device cuda
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

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
            suf = c[len(prefix):]
            if _INT_SUFFIX_RE.fullmatch(suf):
                cols.append(c)
    if not cols:
        raise KeyError(f"No numeric-suffix columns found with prefix '{prefix}'")
    cols.sort(key=lambda c: int(c[len(prefix):]))
    return cols


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_quantiles(s: str) -> List[float]:
    q = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
    if len(q) < 2:
        raise ValueError("--confidence-quantiles must have at least 2 values (e.g., 0,0.5,1)")
    if abs(q[0] - 0.0) > 1e-9 or abs(q[-1] - 1.0) > 1e-9:
        raise ValueError("--confidence-quantiles must start at 0 and end at 1")
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            raise ValueError("--confidence-quantiles must be strictly increasing")
    return q


def is_binary_series(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    if not np.issubdtype(s.dtype, np.integer):
        return False
    vals = s.dropna().unique()
    if len(vals) == 0:
        return False
    vals_set = set(int(v) for v in vals.tolist())
    return vals_set.issubset({0, 1})


# -------------------------
# Score computation (paired cosine) with low memory
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
    """Row-batched parquet read (pyarrow if available) -> paired cosine scores."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None  # type: ignore

    if pq is None:
        # Fallback: load columns via pandas (may be memory-heavy)
        df = pd.read_parquet(path, columns=fraud_cols + real_cols)
        fraud = df[fraud_cols].to_numpy(dtype=np.float32, copy=True)
        real = df[real_cols].to_numpy(dtype=np.float32, copy=True)
        del df
        return paired_cosine_from_arrays(fraud, real, device=device, batch_rows=batch_rows)

    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    out = np.empty((n,), dtype=np.float32)

    cols = fraud_cols + real_cols
    offset = 0
    bs = int(max(1, batch_rows))

    for batch in pf.iter_batches(columns=cols, batch_size=bs):
        bdf = batch.to_pandas(split_blocks=True, self_destruct=True)
        f = bdf[fraud_cols].to_numpy(dtype=np.float32, copy=False)
        r = bdf[real_cols].to_numpy(dtype=np.float32, copy=False)
        scores = paired_cosine_from_arrays(f, r, device=device, batch_rows=bs)
        out[offset:offset + scores.shape[0]] = scores
        offset += scores.shape[0]
        del bdf, f, r, scores

    if offset != n:
        raise RuntimeError(f"Row count mismatch while reading parquet: got {offset}, expected {n}")
    return out


# -------------------------
# Main summarization
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True, help="Parquet containing base+aligned embeddings + one-hot flags.")
    ap.add_argument("--aligned-tags", required=True, help="Comma-separated: deja,source,pacifico")

    ap.add_argument("--fraud-name-col", default="fraudulent_name")
    ap.add_argument("--real-name-col", default="real_name")
    ap.add_argument("--label-col", default="label")

    ap.add_argument("--fraud-prefix", default="fraud_txt_emb_")
    ap.add_argument("--real-prefix", default="real_txt_emb_")
    ap.add_argument("--aligned-fraud-base", default="fraud_txt_emb_aligned_")
    ap.add_argument("--aligned-real-base", default="real_txt_emb_aligned_")

    ap.add_argument("--flags", default=None, help="Optional comma-separated list of flag columns. If omitted, auto-detect binary columns.")
    ap.add_argument("--exclude-flags", default=None, help="Optional comma-separated list of flags to exclude after auto-detect.")
    ap.add_argument("--exclude-identical-spoof", action="store_true", help="Exclude rows with label=1 and normalized fraud==real when fitting thresholds.")

    ap.add_argument("--score-batch-rows", type=int, default=8192)
    ap.add_argument("--device", default=None)

    ap.add_argument("--confidence-quantiles", default="0,0.25,0.5,0.75,0.9,0.95,0.99,1",
                    help="Comma-separated quantiles for |margin| bins (must start 0 end 1).")

    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-bins-csv", default=None)
    ap.add_argument("--out-report-txt", default=None)

    args = ap.parse_args()
    tags = parse_csv_list(args.aligned_tags)
    if not tags:
        raise ValueError("--aligned-tags parsed empty")

    device = pick_device(args.device)
    print(f"[INFO] device={device}")
    print(f"[INFO] tags={tags}")

    # Column names without loading the whole file when possible
    try:
        import pyarrow.parquet as pq  # type: ignore
        all_cols = list(pq.ParquetFile(args.data).schema.names)
    except Exception:
        # Fallback: load just the label column to force pandas to read metadata
        df_tmp = pd.read_parquet(args.data, columns=[args.label_col])
        all_cols = list(df_tmp.columns)
        del df_tmp

    # Embedding column lists
    base_fraud_cols = prefixed_numeric_cols(all_cols, args.fraud_prefix)
    base_real_cols = prefixed_numeric_cols(all_cols, args.real_prefix)
    D = len(base_fraud_cols)
    if len(base_real_cols) != D:
        raise ValueError(f"Base fraud cols={len(base_fraud_cols)} but base real cols={len(base_real_cols)}")

    aligned_cols: Dict[str, Tuple[List[str], List[str]]] = {}
    for t in tags:
        fpre = f"{args.aligned_fraud_base}{t}_"
        rpre = f"{args.aligned_real_base}{t}_"
        fcols = prefixed_numeric_cols(all_cols, fpre)
        rcols = prefixed_numeric_cols(all_cols, rpre)
        if len(fcols) != D or len(rcols) != D:
            raise ValueError(f"[{t}] aligned dim mismatch: fraud={len(fcols)} real={len(rcols)} vs base={D}")
        aligned_cols[t] = (fcols, rcols)

    # Load small metadata (names/label)
    small_cols = [args.fraud_name_col, args.real_name_col, args.label_col]

    # Determine flags
    if args.flags is not None:
        flag_cols = parse_csv_list(args.flags)
    else:
        # auto-detect: read non-embedding candidates and check binary-ness
        exclude_prefixes = [args.fraud_prefix, args.real_prefix, args.aligned_fraud_base, args.aligned_real_base]
        excluded_names = {
            args.fraud_name_col, args.real_name_col, args.label_col,
            "y_true", "score", "pred", "thr", "margin",
            "fraud_len", "real_len", "len_diff", "combo_id", "combo_label", "primary_mechanism",
        }
        candidates: List[str] = []
        for c in all_cols:
            if c in excluded_names:
                continue
            if any(str(c).startswith(p) for p in exclude_prefixes):
                continue
            if str(c).endswith("_len") or str(c) == "len_diff":
                continue
            candidates.append(c)

        df_cand = pd.read_parquet(args.data, columns=small_cols + candidates)
        flag_cols = [c for c in candidates if c in df_cand.columns and is_binary_series(df_cand[c])]
        del df_cand

    if args.exclude_flags is not None:
        excl = set(parse_csv_list(args.exclude_flags))
        flag_cols = [c for c in flag_cols if c not in excl]

    if not flag_cols:
        raise ValueError("No flag columns found. Provide --flags explicitly or check your parquet columns.")

    df = pd.read_parquet(args.data, columns=small_cols + flag_cols)
    y = (df[args.label_col].to_numpy(dtype=np.float32, copy=False) >= 0.5).astype(np.int32)

    # Eval mask for thresholds
    if args.exclude_identical_spoof:
        f_norm = df[args.fraud_name_col].astype(str).map(normalize_name).to_numpy()
        r_norm = df[args.real_name_col].astype(str).map(normalize_name).to_numpy()
        eval_mask = ~((y == 1) & (f_norm == r_norm))
    else:
        eval_mask = np.ones(len(df), dtype=bool)

    if not has_both_classes(y[eval_mask]):
        raise RuntimeError("Threshold eval set became one-class; cannot compute Youden thresholds.")

    # Compute scores + thresholds per model
    model_names = ["vate"] + [f"aligned_{t}" for t in tags]
    thr: Dict[str, float] = {}
    pred: Dict[str, np.ndarray] = {}
    margin: Dict[str, np.ndarray] = {}
    err: Dict[str, np.ndarray] = {}
    fp: Dict[str, np.ndarray] = {}
    fn: Dict[str, np.ndarray] = {}

    print("[INFO] computing vate paired cosine...")
    sc_base = paired_cosine_from_parquet(args.data, base_fraud_cols, base_real_cols, device=device, batch_rows=int(args.score_batch_rows))
    thr_base = youden_threshold(y[eval_mask], sc_base[eval_mask])
    pr_base = (sc_base >= thr_base).astype(np.int32)
    mg_base = (sc_base - float(thr_base)).astype(np.float32, copy=False)
    er_base = (pr_base != y)

    thr["vate"] = float(thr_base)
    pred["vate"] = pr_base
    margin["vate"] = mg_base
    err["vate"] = er_base
    fp["vate"] = ((y == 0) & (pr_base == 1))
    fn["vate"] = ((y == 1) & (pr_base == 0))
    print(f"[DIAG] vate thr={thr_base:.6f} | errors={int(er_base.sum()):,}")

    for t in tags:
        m = f"aligned_{t}"
        fcols, rcols = aligned_cols[t]
        print(f"[INFO] computing {m} paired cosine...")
        sc = paired_cosine_from_parquet(args.data, fcols, rcols, device=device, batch_rows=int(args.score_batch_rows))
        th = youden_threshold(y[eval_mask], sc[eval_mask])
        pr = (sc >= th).astype(np.int32)
        mg = (sc - float(th)).astype(np.float32, copy=False)
        er = (pr != y)

        thr[m] = float(th)
        pred[m] = pr
        margin[m] = mg
        err[m] = er
        fp[m] = ((y == 0) & (pr == 1))
        fn[m] = ((y == 1) & (pr == 0))
        print(f"[DIAG] {m} thr={th:.6f} | errors={int(er.sum()):,}")

    # Fixed/broken vs base
    fixed: Dict[str, np.ndarray] = {}
    broken: Dict[str, np.ndarray] = {}
    for t in tags:
        m = f"aligned_{t}"
        fixed[m] = (err["vate"] & (~err[m]))
        broken[m] = ((~err["vate"]) & err[m])

    # Confidence quantile bins per model (based on |margin| over all rows)
    q = parse_quantiles(args.confidence_quantiles)
    conf_edges: Dict[str, np.ndarray] = {}
    for m in model_names:
        conf = np.abs(margin[m]).astype(np.float64)
        edges = np.quantile(conf, q).astype(np.float64)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        conf_edges[m] = edges

    # Flag matrix
    flag_mat = np.vstack([df[c].astype(int).to_numpy() for c in flag_cols]).T  # (N, F)

    # Per-flag summary
    rows = []
    for j, flag in enumerate(flag_cols):
        in_flag = (flag_mat[:, j] == 1)
        n_flag = int(in_flag.sum())
        if n_flag == 0:
            continue

        row: Dict[str, object] = {"flag": flag, "n_flag": n_flag, "flag_rate": float(n_flag / len(df))}

        for m in model_names:
            er = err[m]
            mg = margin[m]
            row[f"{m}_error_rate"] = float(er[in_flag].mean())
            row[f"{m}_fp_rate"] = float(fp[m][in_flag].mean())
            row[f"{m}_fn_rate"] = float(fn[m][in_flag].mean())
            row[f"{m}_werr"] = float((np.abs(mg[in_flag]) * er[in_flag].astype(np.float32)).mean())

            bad = in_flag & er
            if int(bad.sum()) >= 5:
                ae = np.abs(mg[bad]).astype(np.float64)
                row[f"{m}_err_conf_p50"] = float(np.quantile(ae, 0.50))
                row[f"{m}_err_conf_p90"] = float(np.quantile(ae, 0.90))
                row[f"{m}_err_conf_p95"] = float(np.quantile(ae, 0.95))
            else:
                row[f"{m}_err_conf_p50"] = float("nan")
                row[f"{m}_err_conf_p90"] = float("nan")
                row[f"{m}_err_conf_p95"] = float("nan")

        for t in tags:
            m = f"aligned_{t}"
            row[f"{m}_delta_error_rate"] = float(row[f"{m}_error_rate"] - row["vate_error_rate"])
            row[f"{m}_delta_werr"] = float(row[f"{m}_werr"] - row["vate_werr"])
            row[f"{m}_fixed_rate"] = float(fixed[m][in_flag].mean())
            row[f"{m}_broken_rate"] = float(broken[m][in_flag].mean())
            row[f"{m}_net_fix"] = float(row[f"{m}_fixed_rate"] - row[f"{m}_broken_rate"])

        rows.append(row)

    out_summary = pd.DataFrame(rows).sort_values("n_flag", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary_csv)), exist_ok=True)
    out_summary.to_csv(args.out_summary_csv, index=False)
    print(f"[OK] wrote summary: {args.out_summary_csv} (rows={len(out_summary):,})")

    # Confidence-bin breakdown (optional; long format)
    if args.out_bins_csv:
        bin_rows = []
        for j, flag in enumerate(flag_cols):
            in_flag = (flag_mat[:, j] == 1)
            if int(in_flag.sum()) == 0:
                continue

            for m in model_names:
                conf = np.abs(margin[m]).astype(np.float64)
                edges = conf_edges[m]
                for b in range(len(edges) - 1):
                    lo, hi = float(edges[b]), float(edges[b + 1])
                    in_bin = (conf >= lo) & (conf < hi) if b < (len(edges) - 2) else (conf >= lo) & (conf <= hi)
                    mask = in_flag & in_bin
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    bin_rows.append({
                        "flag": flag,
                        "model": m,
                        "bin_idx": b,
                        "bin_lo": lo,
                        "bin_hi": hi,
                        "n": n,
                        "error_rate": float(err[m][mask].mean()),
                        "werr": float((conf[mask] * err[m][mask].astype(np.float32)).mean()),
                    })

        out_bins = pd.DataFrame(bin_rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_bins_csv)), exist_ok=True)
        out_bins.to_csv(args.out_bins_csv, index=False)
        print(f"[OK] wrote bins: {args.out_bins_csv} (rows={len(out_bins):,})")

    # Text report (optional)
    if args.out_report_txt:
        lines: List[str] = []
        lines.append(f"data={args.data}")
        lines.append(f"tags={tags}")
        lines.append(f"n_rows={len(df):,}")
        lines.append(f"n_flags={len(flag_cols):,}")
        lines.append(f"exclude_identical_spoof={bool(args.exclude_identical_spoof)}")
        lines.append("")
        lines.append("Per-model thresholds (Youden):")
        lines.append(f"  vate: {thr['vate']:.6f}")
        for t in tags:
            lines.append(f"  aligned_{t}: {thr[f'aligned_{t}']:.6f}")
        lines.append("")

        for t in tags:
            m = f"aligned_{t}"
            lines.append(f"=== {m} vs vate ===")
            if len(out_summary) == 0:
                lines.append("  (no flags)")
                lines.append("")
                continue

            tmp = out_summary[["flag", "n_flag", f"{m}_delta_error_rate", f"{m}_delta_werr", f"{m}_net_fix"]].copy()
            tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

            lines.append("  Top improvements (lowest delta_error_rate):")
            best = tmp.sort_values(f"{m}_delta_error_rate", ascending=True).head(10)
            for _, r in best.iterrows():
                lines.append(
                    f"    {r['flag']:<30} n={int(r['n_flag']):>7d} "
                    f"d_err={float(r[f'{m}_delta_error_rate']):+.4f} "
                    f"d_werr={float(r[f'{m}_delta_werr']):+.4f} "
                    f"net_fix={float(r[f'{m}_net_fix']):+.4f}"
                )

            lines.append("  Top regressions (highest delta_error_rate):")
            worst = tmp.sort_values(f"{m}_delta_error_rate", ascending=False).head(10)
            for _, r in worst.iterrows():
                lines.append(
                    f"    {r['flag']:<30} n={int(r['n_flag']):>7d} "
                    f"d_err={float(r[f'{m}_delta_error_rate']):+.4f} "
                    f"d_werr={float(r[f'{m}_delta_werr']):+.4f} "
                    f"net_fix={float(r[f'{m}_net_fix']):+.4f}"
                )
            lines.append("")

        os.makedirs(os.path.dirname(os.path.abspath(args.out_report_txt)), exist_ok=True)
        with open(args.out_report_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[OK] wrote report: {args.out_report_txt}")

    print("[DONE]")


if __name__ == "__main__":
    main()
