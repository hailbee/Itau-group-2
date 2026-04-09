#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain_matcher import (
    DEFAULT_DOMAIN_DATASET,
    DEFAULT_PRECOMPUTED_STORE_DIR,
    FONT_FILES,
    PROJECTOR_PATHS,
    SiglipBackbone,
    normalize_domain_string,
)
from main import (
    STRING_FEATURES,
    choose_device,
    feature_source_for,
    load_model_spec,
    load_projector,
    resolve_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute projected candidate-side inputs for the saved matching models.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DOMAIN_DATASET),
        help="One-column CSV of candidate domains.",
    )
    parser.add_argument(
        "--model-path",
        default="saved_models/total_5f_model.joblib",
        help="Saved model whose candidate-side inputs should be cached.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PRECOMPUTED_STORE_DIR),
        help="Directory to hold the precomputed store.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for projected embeddings.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke tests or partial builds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing files in the output directory.",
    )
    return parser.parse_args()


def count_rows(csv_path: Path, max_rows: int | None = None) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        total = max(0, sum(1 for _ in handle) - 1)
    return min(total, int(max_rows)) if max_rows is not None else total


def required_sources(bundle) -> list[str]:
    sources = [
        feature_source_for(bundle, feature_name)
        for feature_name in bundle.feature_names
        if feature_name not in STRING_FEATURES
    ]
    return list(dict.fromkeys(sources))


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in output_dir.iterdir()]
    if existing_files and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Re-run with --overwrite to rebuild it.")
    if overwrite:
        for path in existing_files:
            if path.is_file():
                path.unlink()


def projected_embedding(
    embeddings: np.ndarray,
    projector: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(device)
    with torch.inference_mode():
        projected = projector.encode(tensor)
        projected = F.normalize(projected, dim=1, eps=1e-8).float().cpu().numpy()
    return projected.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = resolve_model_path(args.model_path, REPO_ROOT / "saved_models")
    model_spec = load_model_spec(model_path)
    needed_sources = required_sources(model_spec)
    device = choose_device(None)
    store_dtype = np.float16 if args.dtype == "float16" else np.float32
    batch_size = max(1, int(args.batch_size))

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    loaded_projectors: dict[str, tuple[torch.nn.Module, int]] = {}
    output_dims: dict[str, int] = {}
    for source_key in needed_sources:
        projector_path = PROJECTOR_PATHS[source_key].resolve()
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing projector for {source_key}: {projector_path}")
        projector, input_dim = load_projector(projector_path, device)
        loaded_projectors[source_key] = (projector, input_dim)
        output_dims[source_key] = int(projector.head[-1].out_features)
        if source_key != "text" and not FONT_FILES[source_key].exists():
            raise FileNotFoundError(f"Missing font for {source_key}: {FONT_FILES[source_key]}")

    row_count = count_rows(dataset_path, max_rows=args.max_rows)
    if row_count <= 0:
        raise RuntimeError(f"No rows found in {dataset_path}")

    print(f"[INFO] Model: {model_path.name}")
    print(f"[INFO] Sources: {', '.join(needed_sources)}")
    print(f"[INFO] Rows: {row_count:,}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output dir: {output_dir}")

    backbone = SiglipBackbone(model_name="google/siglip-base-patch16-224", device=device)
    memmaps = {
        source_key: open_memmap(
            output_dir / f"{source_key}_projected.npy",
            mode="w+",
            dtype=store_dtype,
            shape=(row_count, output_dims[source_key]),
        )
        for source_key in needed_sources
    }

    domains_output_path = output_dir / "domains.csv"
    write_start = 0
    first_chunk = True

    for chunk in pd.read_csv(dataset_path, usecols=["domain"], chunksize=batch_size):
        if args.max_rows is not None and write_start >= int(args.max_rows):
            break

        if args.max_rows is not None:
            remaining = int(args.max_rows) - write_start
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        raw_domains = chunk["domain"].fillna("").astype(str).tolist()
        normalized_domains = [normalize_domain_string(domain) for domain in raw_domains]
        write_end = write_start + len(normalized_domains)

        pd.DataFrame(
            {
                "domain": raw_domains,
                "normalized_domain": normalized_domains,
            }
        ).to_csv(
            domains_output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

        if "text" in needed_sources:
            text_projector, _ = loaded_projectors["text"]
            text_embeddings = backbone.encode_texts(normalized_domains, batch_size=batch_size)
            memmaps["text"][write_start:write_end] = projected_embedding(
                text_embeddings,
                text_projector,
                device,
            ).astype(store_dtype)

        for source_key in needed_sources:
            if source_key == "text":
                continue
            projector, _ = loaded_projectors[source_key]
            glyph_embeddings = backbone.encode_glyphs(
                normalized_domains,
                font_path=FONT_FILES[source_key],
                batch_size=batch_size,
            )
            memmaps[source_key][write_start:write_end] = projected_embedding(
                glyph_embeddings,
                projector,
                device,
            ).astype(store_dtype)

        write_start = write_end
        print(f"[INFO] Processed {write_start:,}/{row_count:,} rows", flush=True)

    for memmap in memmaps.values():
        memmap.flush()

    metadata = {
        "store_type": "projected_candidate_inputs_v1",
        "model_name": model_path.name,
        "feature_names": model_spec.feature_names,
        "rows": write_start,
        "domains_file": domains_output_path.name,
        "sources": {
            source_key: {
                "file": f"{source_key}_projected.npy",
                "dim": output_dims[source_key],
                "dtype": np.dtype(store_dtype).name,
            }
            for source_key in needed_sources
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote precomputed store to {output_dir}")


if __name__ == "__main__":
    main()
