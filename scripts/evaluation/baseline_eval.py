import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from evaluator import Evaluator

"""
run:
python3 ../scripts/evaluation/baseline_eval.py --test_filepath ../../Downloads/golden_embeddings_test.parquet --out_dir ../../Downloads --name_prefix baseline_test --plot
"""

def baseline_eval(
    test_filepath: str,
    out_dir: str = "images",
    name_prefix: str = "baseline_test",
    plot: bool = True,
):
    # Load
    if test_filepath.endswith(".csv"):
        df = pd.read_csv(test_filepath)
    else:
        df = pd.read_parquet(test_filepath)

    # Extract original embeddings (same slices you use elsewhere)
    fraud_np = df.iloc[:, 3:771].to_numpy(dtype=np.float32, copy=False)
    real_np  = df.iloc[:, 771:1539].to_numpy(dtype=np.float32, copy=False)

    fraud = torch.from_numpy(fraud_np)
    real  = torch.from_numpy(real_np)

    # Baseline similarity: cosine on ORIGINAL embeddings
    fraud = F.normalize(fraud, dim=1)
    real  = F.normalize(real, dim=1)
    sims = F.cosine_similarity(fraud, real, dim=1).cpu().numpy()

    results_df = pd.DataFrame({
        "fraudulent_name": df["fraudulent_name"].astype(str).tolist(),
        "real_name": df["real_name"].astype(str).tolist(),
        "label": df["label"].astype(int).tolist(),
        "similarity": sims,
    })

    # Reuse your metric/plot logic
    dummy_model = torch.nn.Identity()  # not used, but Evaluator wants a model
    evaluator = Evaluator(model=dummy_model, batch_size=32, model_type="pair")

    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, f"{name_prefix}_roc.png")
    acc_curve_path = os.path.join(out_dir, f"{name_prefix}_acc_vs_threshold.png")
    cm_path = os.path.join(out_dir, f"{name_prefix}_confusion_matrix_youden.png")

    metrics = evaluator.compute_metrics(
        results_df,
        plot=plot,
        roc_png_path=roc_path,
        acc_curve_png_path=acc_curve_path,
        cm_png_path=cm_path,
        title_prefix="Baseline",
    )

    return results_df, metrics


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--test_filepath", required=True)
    p.add_argument("--out_dir", default="images")
    p.add_argument("--name_prefix", default="baseline_test")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    df, metrics = baseline_eval(
        test_filepath=args.test_filepath,
        out_dir=args.out_dir,
        name_prefix=args.name_prefix,
        plot=args.plot,
    )

    print("\n--- BASELINE METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
