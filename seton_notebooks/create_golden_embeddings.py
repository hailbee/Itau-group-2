import argparse
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import EmbeddingPairDataset

"""
python seton_notebooks/create_golden_embeddings.py \
  --model_ckpt saved_models/best_model_by_val_trial_1_single_run.pt \
  --input_filepath ../Downloads/test_pairs_with_siglip_embeddings.parquet \
  --internal_layer_size 512 \
  --output_dim 128 \
  --output_filepath text_to_image/Golden/golden_embeddings_test.parquet
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate aligned (golden) embeddings from a trained SiameseEmbeddingModel"
    )
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--input_filepath", type=str, required=True)
    parser.add_argument("--output_filepath", type=str, default="golden_embeddings.parquet")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--internal_layer_size", type=int, required=True)
    parser.add_argument("--output_dim", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # Load data
    # -------------------------
    if args.input_filepath.endswith(".parquet"):
        df = pd.read_parquet(args.input_filepath)
    elif args.input_filepath.endswith(".csv"):
        df = pd.read_csv(args.input_filepath)
    else:
        raise ValueError("Input file must be .parquet or .csv")

    dataset = EmbeddingPairDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------
    # Build model
    # -------------------------
    model = SiameseEmbeddingModel(
        embedding_dim=768,
        hidden_dim=args.internal_layer_size,
        out_dim=args.output_dim,
    ).to(device)

    state = torch.load(args.model_ckpt, map_location=device)
    # support either raw state_dict or {"model_state": ...}
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded model from {args.model_ckpt}")

    # -------------------------
    # Generate ALIGNED embeddings only (wide output)
    # -------------------------
    labels_all = []
    z1_all, z2_all = [], []

    with torch.no_grad():
        for (x1, x2, y) in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1, z2 = model(x1, x2)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            labels_all.append(y.detach().cpu().to(torch.int64))
            z1_all.append(z1.detach().cpu())
            z2_all.append(z2.detach().cpu())

    labels = torch.cat(labels_all, dim=0).numpy()
    z1 = torch.cat(z1_all, dim=0).numpy().astype(np.float32)
    z2 = torch.cat(z2_all, dim=0).numpy().astype(np.float32)

    # -------------------------
    # Build output table (KEEP names + label)
    # -------------------------
    out = pd.DataFrame({
        "fraudulent_name": df["fraudulent_name"].astype(str).to_numpy(),
        "real_name": df["real_name"].astype(str).to_numpy(),
        "label": labels,
    })

    D_out = z1.shape[1]
    for i in range(D_out):
        out[f"fraud_aligned_{i}"] = z1[:, i]
        out[f"real_aligned_{i}"] = z2[:, i]

    # -------------------------
    # Save parquet
    # -------------------------
    os.makedirs(os.path.dirname(args.output_filepath) or ".", exist_ok=True)
    out.to_parquet(args.output_filepath, index=False)

    print(f"[INFO] Saved golden embeddings → {args.output_filepath}")
    print(f"[INFO] Rows: {len(out)} | aligned_dim: {D_out} | cols: {out.shape[1]}")


if __name__ == "__main__":
    main()
