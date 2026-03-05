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

python sub_opt/submodular_optimization.py \
  --input "sub_opt/validate_pairs_ref_10k.parquet" \
  --k 5
"""
#sbatch run_submodular_selection.sh

from __future__ import annotations

import argparse
from itertools import combinations
import pandas as pd
import random
import os
import numpy as np

def read_parquet(font: str) -> pd.DataFrame:
    """
    Args:
        font: one of the font identifiers

    Returns:
        DataFrame with columns: ['fraudulent_name', 'real_name', 'label', 'cosine_sim']
    """
    filename = f"sub_opt/cosine_sim/{font}_validate_pairs_ref_10k.parquet"
    df = pd.read_parquet(filename)
    return df


def load_baseline_parquet() -> pd.DataFrame:
    """
    Load DejaVu Sans parquet file with precomputed cosine similarities.
    
    Returns:
        DataFrame with columns: ['fraudulent_name', 'real_name', 'label', 'cosine_sim']
    """
    baseline_path = "sub_opt/cosine_sim/dejavusans_validate_pairs_ref_10k.parquet"
    df = pd.read_parquet(baseline_path)
    return df


def get_baseline_correctness(x: pd.Series, baseline_df: pd.DataFrame, baseline_threshold: float) -> bool:
    """
    Determine if the baseline (DejaVu Sans) correctly classifies the pair.
    
    Args:
        x: one row from the validation parquet (contains: fraudulent_name, real_name, label)
        baseline_df: DataFrame with precomputed DejaVu Sans cosine similarities
        baseline_threshold: DejaVu Sans threshold
        
    Returns:
        True if baseline correctly classifies, False otherwise
    """
    label = x['label']
    
    # Get the corresponding row from baseline_df using index
    baseline_row = baseline_df.iloc[x.name] if isinstance(x.name, int) else baseline_df.iloc[int(x.name)]
    cosine_sim = baseline_row['cosine_sim']
    
    # Check if baseline correctly classifies
    baseline_correct = (label == 1 and cosine_sim >= baseline_threshold) or (label == 0 and cosine_sim < baseline_threshold)
    return baseline_correct

# NOT IMPLEMENTED FOR NOW
def delta_f_x(font: str, x: pd.Series, t: float):
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

def delta_f_x_discreet(font: str, x: pd.Series, t: float, baseline_correct: bool):
    """
    Args:
        font: one of the font identifiers
        x: one row from the validation parquet (contains: fraudulent_name, real_name, label, cosine_sim)
        t: threshold for the font
        baseline_correct: whether DejaVu Sans got it correct

    Returns:
        contribution: +1 if font improves over baseline, 0 if same, -1 if worse
    """
    # Check if the font correctly classifies the pair
    label = x['label']
    cosine_sim = x['cosine_sim']
    
    # Font is correct if: (label == 1 and cosine_sim >= t) or (label == 0 and cosine_sim < t)
    font_correct = (label == 1 and cosine_sim >= t) or (label == 0 and cosine_sim < t)
    
    # Compare to baseline
    if font_correct and baseline_correct:
        return 0  # Both correct
    elif font_correct and not baseline_correct:
        return 1  # Font improves
    elif not font_correct and baseline_correct:
        return -1  # Font is worse
    else:
        return 0  # Both incorrect


def delta_f_x_margin(font: str, x: pd.Series, t: float, baseline_correct: bool, baseline_cosine_sim: float):
    """
    Compute the improvement of a font over baseline using signed confidence (margin).
    
    Args:
        font: one of the font identifiers
        x: one row from the validation parquet (contains: fraudulent_name, real_name, label, cosine_sim)
        t: threshold for the font
        baseline_correct: whether DejaVu Sans got it correct
        baseline_cosine_sim: cosine similarity from baseline (DejaVu Sans)

    Returns:
        improvement: max(0, signed_confidence_font - signed_confidence_baseline)
        
    Where signed_confidence = confidence * sign, where:
        - confidence = |cosine_sim - threshold| (absolute distance from threshold)
        - sign = +1 if correct, -1 if incorrect
    """
    label = x['label']
    font_cosine_sim = x['cosine_sim']
    
    # Compute font correctness and signed confidence
    font_correct = (label == 1 and font_cosine_sim >= t) or (label == 0 and font_cosine_sim < t)
    font_sign = 1 if font_correct else -1
    font_confidence = abs(font_cosine_sim - t)
    font_signed_confidence = font_sign * font_confidence
    
    # Compute baseline signed confidence
    baseline_sign = 1 if baseline_correct else -1
    baseline_confidence = abs(baseline_cosine_sim - t)
    baseline_signed_confidence = baseline_sign * baseline_confidence
    
    # Compute improvement: max(0, delta)
    improvement = max(0, font_signed_confidence - baseline_signed_confidence)
    
    return improvement


def submodular_optimization(fonts, examples, k, delta_f_x, thresholds, baseline_threshold, baseline_df, to_print=True):
    best_S = None
    best_value = float("-inf")

    # Load all font parquet files once
    font_dfs = {f: read_parquet(f) for f in fonts}

    all_combinations = list(combinations(fonts, k))
    print(f"Evaluating {len(all_combinations)} subsets of size {k} from {len(fonts)} fonts.")
    for S in all_combinations:
        print("Evaluating subset:", S)
        total_value = 0.0
        count_positive = 0  # Count of +1s
        count_zero = 0      # Count of 0s
        count_negative = 0  # Count of -1s

        indices = list(range(len(examples)))
        for i in indices:
            best_gain = float("-inf")
            weight = None
            best_contribution = None
            
            # Get baseline correctness for this example
            x = examples[i]
            baseline_correct = get_baseline_correctness(x, baseline_df, baseline_threshold)
            baseline_cosine_sim = baseline_df.iloc[x.name]['cosine_sim'] if isinstance(x.name, int) else baseline_df.iloc[int(x.name)]['cosine_sim']
            # w_x = 0.1 if baseline_correct else 1.0
            w_x = 1.0

            for f in S:
                df_font = font_dfs[f]
                threshold = thresholds[f]
                x_font = df_font.iloc[i]
                # For margin-based functions, pass baseline_cosine_sim; for discrete, it's ignored
                if delta_f_x.__name__ == 'delta_f_x_margin':
                    value = delta_f_x(f, x_font, threshold, baseline_correct, baseline_cosine_sim)
                else:
                    value = delta_f_x(f, x_font, threshold, baseline_correct)

                if value > best_gain:
                    best_gain = value
                    weight = w_x
                    best_contribution = value

            total_value += weight * best_gain
            
            # Track the counts
            if best_contribution == 1:
                count_positive += 1
            elif best_contribution == 0:
                count_zero += 1
            elif best_contribution == -1:
                count_negative += 1

        print(f" Total value: {total_value} ({delta_f_x.__name__})")
        if to_print:
            print(f"  Contributions: +1s={count_positive}, 0s={count_zero}, -1s={count_negative}")
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
    parser.add_argument(
        "--function",
        type=str,
        default="discreet",
        help="Function to use for delta f_x (e.g. random, discreet)",
    )
    parser.add_argument(
        "--print",
        type=str,
        default="True",
        help="Whether to print detailed output",
    )
    args = parser.parse_args()

    font_thresholds = {
        "arimo": 0.8132318854331970,
        "charissil": 0.8223531246185303,
        "cousine": 0.8044030666351318,
        "dejavusans": 0.8149772286415100,
        "doulossil": 0.8242787122726440,
        "exo2": 0.8193700313568115,
        "freeserif": 0.8264714479446411,
        "gentiumplus": 0.8164674639701843,
        "librefranklin": 0.7957530021667480,
        "notosans": 0.8146908283233643,
        "unifont": 0.8102402091026306,
        "vollkorn": 0.8139773607254028,
    }

    baseline_threshold = font_thresholds["dejavusans"]
    fonts = list(font_thresholds.keys()) 
    fonts.remove("dejavusans")

    df = pd.read_parquet(args.input)

    examples = [row for _, row in df.iterrows()]
    reward_functions = {
        "random": delta_f_x,
        "discreet": delta_f_x_discreet,
        "margin": delta_f_x_margin,
    }
    
    # Load baseline parquet file with precomputed cosine similarities
    baseline_df = load_baseline_parquet()

    best_fonts, score = submodular_optimization(
        fonts=fonts,
        examples=examples,
        k=args.k,
        delta_f_x=reward_functions[args.function],
        thresholds=font_thresholds,
        baseline_threshold=baseline_threshold,
        baseline_df=baseline_df,
        to_print=args.print.lower() == "true",
    )

    print("Best font subset:")
    if best_fonts:
        for f in sorted(best_fonts):
            print(" ", f)
        print("Objective value:", score)
    else:
        print("No fonts selected.")


if __name__ == "__main__":
    main()

    # fonts = [
    #     "tahoma",
    #     "robotocondensed",
    #     "centurygothic",
    #     "helvetica",
    #     "silom",
    #     "calibri",
    #     "caveat",
    #     "pacifico",
    #     "nanumbrush",
    #     "sourcecodepro",
    # ]

    # for font in fonts:
    #     df_font = read_parquet(font)
    #     print(f"Font: {font}")
    #     print(df_font.head())