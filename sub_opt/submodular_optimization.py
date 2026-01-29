#!/usr/bin/env python3
"""
Run submodular optimization over font subsets.

Objective:
  F(S) = sum_{x in X} w_x * max_{f in S} Delta_f(x)

Where:
  - X is the validation set (rows of a Parquet file)
  - w_x = 0.1 if baseline correct on x, else 1.0
  - Delta_f(x) is provided by an external implementation

Inputs:
  - Validation Parquet file (validate_pairs_ref_10k.parquet)

Outputs:
  - Printed best font subset and objective value

"""
#sbatch run_submodular_selection.sh

from __future__ import annotations

import argparse
from itertools import combinations
import pandas as pd
import random


# NOT IMPLEMENTED FOR NOW
def delta_f_x(font: str, x: pd.Series):
    """
    Args:
        font: one of the font identifiers
        x: one row from the validation parquet

    Returns:
        (baseline_correct: bool, contribution: float)
    """
    bool_val = random.choice([True, False])
    contribution = random.choice([-1, 0, 1])
    return bool_val, contribution


def submodular_optimization(fonts, examples, k, delta_f_x):
    best_S = None
    best_value = float("-inf")

    for S in combinations(fonts, k):
        print("Evaluating subset:", S)
        total_value = 0.0

        for x in examples:
            best_gain = float("-inf")
            weight = None

            for f in S:
                baseline_correct, value = delta_f_x(f, x)
                w_x = 0.1 if baseline_correct else 1.0

                if value > best_gain:
                    best_gain = value
                    weight = w_x

            total_value += weight * best_gain

        if total_value > best_value:
            best_value = total_value
            best_S = set(S)

    return best_S, best_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact submodular font selection (non-greedy)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Validation parquet file (e.g. validate_pairs_ref_10k.parquet)",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of fonts to select",
    )
    args = parser.parse_args()

    fonts = [
        "tahoma",
        "roboto condensed",
        "century gothic",
        "helvetica condensed",
        "silom",
        "calibri",
        "caveat",
        "pacifico",
        "nanum brush script",
        "Source Code Pro",
    ]

    df = pd.read_parquet(args.input)

    examples = [row for _, row in df.iterrows()]

    best_fonts, score = submodular_optimization(
        fonts=fonts,
        examples=examples,
        k=args.k,
        delta_f_x=delta_f_x,
    )

    print("Best font subset:")
    for f in sorted(best_fonts):
        print(" ", f)
    print("Objective value:", score)


if __name__ == "__main__":
    main()
