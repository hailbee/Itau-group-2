#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/myjob-%A-%a.log
#SBATCH -a 0-9  # 10 tasks, one per font

module load miniforge
source ~/.bashrc
conda activate itau

# Input file (same for all tasks)
input_file="data/processed/validate_pairs_ref_10k.parquet"
input_base=$(basename "$input_file" .parquet)  # strip directory & extension

# Define arrays for fonts and outputs (without timestamp)
fonts=("tahoma" "roboto condensed" "century gothic" "helvetica condensed" "silom" \
       "calibri" "caveat" "pacifico" "nanum brush script" "Source Code Pro")

output_parquets_base=("tahoma_validate_pairs_ref_10k.parquet" \
                 "robotocondensed_validate_pairs_ref_10k.parquet" \
                 "centurygothic_validate_pairs_ref_10k.parquet" \
                 "helvetica_validate_pairs_ref_10k.parquet" \
                 "silom_validate_pairs_ref_10k.parquet" \
                 "calibri_validate_pairs_ref_10k.parquet" \
                 "caveat_validate_pairs_ref_10k.parquet" \
                 "pacifico_validate_pairs_ref_10k.parquet" \
                 "nanumbrush_validate_pairs_ref_10k.parquet" \
                 "sourcecodepro_validate_pairs_ref_10k.parquet")

output_embeddings_base=("tahoma_embed_validate_pairs_ref_10k.npz" \
                   "robotocondensed_embed_validate_pairs_ref_10k.npz" \
                   "centurygothic_embed_validate_pairs_ref_10k.npz" \
                   "helvetica_embed_validate_pairs_ref_10k.npz" \
                   "silom_embed_validate_pairs_ref_10k.npz" \
                   "calibri_embed_validate_pairs_ref_10k.npz" \
                   "caveat_embed_validate_pairs_ref_10k.npz" \
                   "pacifico_embed_validate_pairs_ref_10k.npz" \
                   "nanumbrush_embed_validate_pairs_ref_10k.npz" \
                   "sourcecodepro_embed_validate_pairs_ref_10k.npz")

# Pick the index for this task
idx=$SLURM_ARRAY_TASK_ID

# Append timestamp and input file name to output names
output_file="${output_parquets_base[$idx]%.parquet}"
output_embed="${output_embeddings_base[$idx]%.npz}"

# Print info for logs
echo "[INFO] Task $idx"
echo "[INFO] Font: ${fonts[$idx]}"
echo "[INFO] Input: $input_file"
echo "[INFO] Output parquet: $output_file"
echo "[INFO] Output embeddings: $output_embed"
echo "[INFO] Timestamp: $ts"

# Run Python
python fonts/create_font_embeddings_script.py \
  --input "$input_file" \
  --output "$output_file" \
  --output-embeddings "$output_embed" \
  --font "${fonts[$idx]}" \
  --batch-size 128 \
  --device cuda
