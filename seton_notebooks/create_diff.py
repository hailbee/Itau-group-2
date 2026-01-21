import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "train_pairs_with_siglip_embeddings.parquet"

FAKE_EMB_START = 3
FAKE_EMB_END = 771          # exclusive
REAL_EMB_START = 771
REAL_EMB_END = 1539         # exclusive

EASY_PERCENTILE = 20
HARD_PERCENTILE = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(PARQUET_PATH)

# Split positives / negatives
df_pos = df[df["label"] == 1].copy()
df_neg = df[df["label"] == 0].copy()

print(f"Positives: {len(df_pos)}")
print(f"Negatives: {len(df_neg)}")

# -----------------------------
# Extract embeddings
# -----------------------------
fake_emb = torch.tensor(
    df_neg.iloc[:, FAKE_EMB_START:FAKE_EMB_END].values,
    dtype=torch.float32,
    device=DEVICE
)

real_emb = torch.tensor(
    df_neg.iloc[:, REAL_EMB_START:REAL_EMB_END].values,
    dtype=torch.float32,
    device=DEVICE
)

# -----------------------------
# Normalize (important!)
# -----------------------------
fake_emb = torch.nn.functional.normalize(fake_emb, dim=1)
real_emb = torch.nn.functional.normalize(real_emb, dim=1)

# -----------------------------
# Cosine similarity
# -----------------------------
with torch.no_grad():
    cosine_sim = (fake_emb * real_emb).sum(dim=1)

cosine_sim_cpu = cosine_sim.cpu().numpy()
df_neg["cosine_similarity"] = cosine_sim_cpu

# -----------------------------
# Global thresholds
# -----------------------------
low_thresh = np.percentile(cosine_sim_cpu, EASY_PERCENTILE)
high_thresh = np.percentile(cosine_sim_cpu, HARD_PERCENTILE)

print(f"Easy threshold  (< {low_thresh:.4f})")
print(f"Hard threshold  (>= {high_thresh:.4f})")

# -----------------------------
# Split negatives
# -----------------------------
df_neg_easy = df_neg[df_neg["cosine_similarity"] < low_thresh]
df_neg_medium = df_neg[
    (df_neg["cosine_similarity"] >= low_thresh) &
    (df_neg["cosine_similarity"] < high_thresh)
]
df_neg_hard = df_neg[df_neg["cosine_similarity"] >= high_thresh]

print("Negative splits:")
print(f"  Easy:   {len(df_neg_easy)}")
print(f"  Medium: {len(df_neg_medium)}")
print(f"  Hard:   {len(df_neg_hard)}")

# -----------------------------
# Combine with positives
# -----------------------------
easy_dataset = pd.concat([df_pos, df_neg_easy], ignore_index=True)
medium_dataset = pd.concat([df_pos, df_neg_medium], ignore_index=True)
hard_dataset = pd.concat([df_pos, df_neg_hard], ignore_index=True)

# -----------------------------
# Save
# -----------------------------
easy_dataset.to_parquet("train_easy.parquet", index=False)
medium_dataset.to_parquet("train_medium.parquet", index=False)
hard_dataset.to_parquet("train_hard.parquet", index=False)

print("Saved:")
print("  train_easy.parquet")
print("  train_medium.parquet")
print("  train_hard.parquet")
