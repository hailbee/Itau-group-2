#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/submodular-%j.log

module load miniforge
source ~/.bashrc
conda activate itau

input_file="data/processed/validate_pairs_ref_10k.parquet"
k=5

echo "[INFO] Running submodular font selection"
echo "[INFO] Input: $input_file"
echo "[INFO] k: $k"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Start time: $(date)"

# Run the submodular font selection script
python sub_opt/submodular_font_selection.py \
  --input "$input_file" \
  --k "$k"

#windowed version for testing
#python sub_opt/submodular_font_selection.py --input data/processed/validate_pairs_ref_10k.parquet --k 5


echo "[INFO] Finished at: $(date)"
