import pandas as pd
import os

# -----------------------------
# Config
# -----------------------------
EASY_PATH = "train_easy.parquet"
MEDIUM_PATH = "train_medium.parquet"
HARD_PATH = "train_hard.parquet"

OUTPUT_DIR = "balanced_100k"
POS_SIZE = 50_000
NEG_SIZE = 50_000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load files
# -----------------------------
df_easy = pd.read_parquet(EASY_PATH)
df_medium = pd.read_parquet(MEDIUM_PATH)
df_hard = pd.read_parquet(HARD_PATH)

# Ensure label is int
for df in [df_easy, df_medium, df_hard]:
    df["label"] = df["label"].astype(int)

# -----------------------------
# Extract positives (from ANY file — they are the same)
# -----------------------------
df_pos = df_easy[df_easy["label"] == 1]

assert len(df_pos) >= POS_SIZE, "Not enough positives"

df_pos_shared = df_pos.sample(
    n=POS_SIZE,
    random_state=RANDOM_SEED,
    replace=False
)

# -----------------------------
# Sample negatives per file
# -----------------------------
def sample_negatives(df, name):
    df_neg = df[df["label"] == 0]
    assert len(df_neg) >= NEG_SIZE, f"Not enough negatives in {name}"
    return df_neg.sample(
        n=NEG_SIZE,
        random_state=RANDOM_SEED,
        replace=False
    )

df_easy_neg = sample_negatives(df_easy, "easy")
df_medium_neg = sample_negatives(df_medium, "medium")
df_hard_neg = sample_negatives(df_hard, "hard")

# -----------------------------
# Combine
# -----------------------------
easy_100k = pd.concat([df_pos_shared, df_easy_neg], ignore_index=True)
medium_100k = pd.concat([df_pos_shared, df_medium_neg], ignore_index=True)
hard_100k = pd.concat([df_pos_shared, df_hard_neg], ignore_index=True)

# -----------------------------
# Sanity checks
# -----------------------------
for name, df in [
    ("easy", easy_100k),
    ("medium", medium_100k),
    ("hard", hard_100k),
]:
    assert len(df) == 100_000
    assert (df["label"] == 1).sum() == POS_SIZE
    print(f"{name}: ✔ 100k total, 50k positives")

# -----------------------------
# Save
# -----------------------------
easy_100k.to_parquet(f"{OUTPUT_DIR}/train_easy_100k.parquet", index=False)
medium_100k.to_parquet(f"{OUTPUT_DIR}/train_medium_100k.parquet", index=False)
hard_100k.to_parquet(f"{OUTPUT_DIR}/train_hard_100k.parquet", index=False)

print("✔ Saved balanced 100k parquet files")
