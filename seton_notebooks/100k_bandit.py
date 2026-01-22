import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "train_pairs_with_siglip_embeddings.parquet"

FAKE_EMB_START = 3
FAKE_EMB_END = 771          # exclusive
REAL_EMB_START = 771
REAL_EMB_END = 1539         # exclusive

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

RANDOM_SEED = 42
SAMPLE_SIZE = 100_000
OUTPUT_PATH = "train_shuffled_100k_with_neg_cosine.parquet"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(PARQUET_PATH)
df["label"] = df["label"].astype(int)

df_pos = df[df["label"] == 1].copy()
df_neg = df[df["label"] == 0].copy()

print(f"Positives: {len(df_pos)}")
print(f"Negatives (raw): {len(df_neg)}")

# -----------------------------
# Extract embeddings (negatives only)
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
# Detect corrupted rows (negatives)
# -----------------------------
fake_norms = fake_emb.norm(dim=1)
real_norms = real_emb.norm(dim=1)

valid_mask = (
    (fake_norms > 0) &
    (real_norms > 0) &
    ~torch.isnan(fake_emb).any(dim=1) &
    ~torch.isnan(real_emb).any(dim=1)
)

num_dropped = (~valid_mask).sum().item()
print(f"Dropping {num_dropped} corrupted negative rows")

# Apply mask to negatives only
fake_emb = fake_emb[valid_mask]
real_emb = real_emb[valid_mask]
df_neg = df_neg.loc[valid_mask.cpu().numpy()].copy()

print(f"Negatives (clean): {len(df_neg)}")

# -----------------------------
# Normalize safely + cosine similarity (negatives)
# -----------------------------
fake_emb = F.normalize(fake_emb, dim=1, eps=1e-8)
real_emb = F.normalize(real_emb, dim=1, eps=1e-8)

with torch.no_grad():
    cosine_sim = F.cosine_similarity(fake_emb, real_emb, dim=1)

cosine_sim_cpu = cosine_sim.cpu().numpy()
assert not np.isnan(cosine_sim_cpu).any(), "NaNs still present in negative cosine similarities!"

df_neg["cosine_similarity"] = cosine_sim_cpu

# -----------------------------
# Combine back (natural distribution)
# -----------------------------
df_out = pd.concat([df_pos, df_neg], ignore_index=True)

# Shuffle (preserve natural label distribution in expectation)
df_out = df_out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# Optional: print label distribution before sampling
label_counts = df_out["label"].value_counts(normalize=True).sort_index()
print("Label distribution (after cleaning negatives):")
print(label_counts)

# -----------------------------
# Sample 100k (if possible)
# -----------------------------
if SAMPLE_SIZE is not None:
    if len(df_out) < SAMPLE_SIZE:
        raise ValueError(f"Not enough rows to sample {SAMPLE_SIZE}. Only have {len(df_out)}.")
    df_out = df_out.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Final rows: {len(df_out)}")
print("NaNs in cosine_similarity:", df_out["cosine_similarity"].isna().sum())

# -----------------------------
# Save
# -----------------------------
df_out.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
