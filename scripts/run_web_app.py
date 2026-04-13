#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain_matcher import DEFAULT_DOMAIN_DATASET, DEFAULT_MODEL_PATH, default_precomputed_store_dir
from web_app import app, get_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Total 5F web app with a quick prerequisite check.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = DEFAULT_DOMAIN_DATASET.resolve()
    expected_precomputed_dir = default_precomputed_store_dir(DEFAULT_MODEL_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Missing benign dataset at "
            f"{dataset_path}. Build it first with:\n"
            "python3 scripts/prepare_benign_domains.py --input <kaggle_zip_or_csv> "
            "--output data/benign_domains.csv"
        )

    print(f"[INFO] Dataset found: {dataset_path}")
    matcher = get_matcher()
    if matcher.precomputed_store is not None:
        print(f"[INFO] Precomputed store found: {matcher.precomputed_store.store_dir}")
        print("[INFO] Searches will use cached candidate inputs when possible.")
        for notice in matcher.precomputed_store_notices:
            print(f"[WARN] {notice}")
    else:
        for notice in matcher.precomputed_store_notices:
            print(f"[WARN] {notice}")
        print(f"[WARN] No compatible precomputed store found at {expected_precomputed_dir}")
        print("[WARN] The app will still start, but full searches will be much slower.")
        print("[WARN] Build it with:")
        print(
            "python3 scripts/precompute_model_inputs.py "
            "--dataset data/benign_domains.csv "
            "--model-path saved_models/total_5f_img_model.joblib "
            "--output-dir precomputed/benign_total5f_img"
        )

    print(f"[INFO] Starting web app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
