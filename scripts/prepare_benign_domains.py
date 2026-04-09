#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


def resolve_input_csv(path: Path) -> tuple[Path, dict[str, object]]:
    lowered = path.name.lower()
    if lowered.endswith(".csv"):
        return path, {}

    if lowered.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise FileNotFoundError(f"No CSV file found inside {path}")
            if len(csv_members) > 1:
                raise RuntimeError(
                    f"Expected exactly one CSV inside {path}, found {len(csv_members)}: {', '.join(csv_members)}"
                )
        return path, {"compression": "zip"}

    raise ValueError(f"Unsupported input type for {path}. Use a .csv or a .zip download containing one CSV.")


def detect_source_format(
    source_path: Path,
    read_csv_kwargs: dict[str, object],
    *,
    domain_col: str,
    label_col: str,
) -> str:
    preview = pd.read_csv(source_path, nrows=5, **read_csv_kwargs)
    columns = {str(column) for column in preview.columns}
    if domain_col in columns and label_col in columns:
        return "kaggle_labeled"

    preview_no_header = pd.read_csv(source_path, header=None, nrows=5, **read_csv_kwargs)
    if preview_no_header.shape[1] >= 2:
        rank_values = preview_no_header.iloc[:, 0].astype(str).str.fullmatch(r"\d+")
        domain_values = preview_no_header.iloc[:, 1].astype(str).str.contains(r"\.", regex=True)
        if bool(rank_values.all() and domain_values.all()):
            return "rank_domain"

    raise RuntimeError(
        f"Could not infer dataset format for {source_path}. "
        f"Expected either columns {domain_col!r}/{label_col!r} or a two-column rank/domain CSV."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a one-column domain CSV from either the Kaggle benign export or a rank/domain source such as Alexa top-1m.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source CSV or ZIP download.",
    )
    parser.add_argument(
        "--output",
        default="data/benign_domains.csv",
        help="Path to write the one-column domain CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows to read per chunk while filtering.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column name containing the class label.",
    )
    parser.add_argument(
        "--domain-col",
        default="domain",
        help="Column name containing the domain string.",
    )
    parser.add_argument(
        "--benign-label",
        default="benign",
        help="Label value to keep in the output dataset.",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "kaggle_labeled", "rank_domain"),
        default="auto",
        help="Override auto-detection when needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path, read_csv_kwargs = resolve_input_csv(input_path)
    source_format = (
        str(args.source_format)
        if str(args.source_format) != "auto"
        else detect_source_format(
            source_path,
            read_csv_kwargs,
            domain_col=str(args.domain_col),
            label_col=str(args.label_col),
        )
    )

    first_chunk = True
    if source_format == "kaggle_labeled":
        usecols = [str(args.domain_col), str(args.label_col)]
        chunk_iter = pd.read_csv(
            source_path,
            usecols=usecols,
            chunksize=max(1, int(args.chunksize)),
            **read_csv_kwargs,
        )
        for chunk in chunk_iter:
            domains = chunk.loc[
                chunk[str(args.label_col)].astype(str) == str(args.benign_label),
                [str(args.domain_col)],
            ]
            domains.columns = ["domain"]
            domains.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False
    elif source_format == "rank_domain":
        chunk_iter = pd.read_csv(
            source_path,
            header=None,
            names=["rank", "domain"],
            usecols=["domain"],
            chunksize=max(1, int(args.chunksize)),
            **read_csv_kwargs,
        )
        for chunk in chunk_iter:
            domains = chunk[["domain"]].copy()
            domains["domain"] = domains["domain"].astype(str).str.strip()
            domains = domains.loc[domains["domain"] != ""]
            domains.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False
    else:
        raise RuntimeError(f"Unsupported source format: {source_format}")

    print(f"Wrote {output_path} from {input_path} using {source_format}")


if __name__ == "__main__":
    main()
