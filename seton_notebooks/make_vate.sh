#!/bin/bash
set -e  # exit immediately if any command fails

echo "===== STEP 1: Create Vate Embeddings (test/train/validate) ====="

echo "

chmod +x seton_notebooks/make_vate.sh
./seton_notebooks/make_vate.sh

"

python seton_notebooks/create_VATE.py \
  --input ../Downloads/validate_pairs_with_siglip_embeddings.parquet \
  --vate-only-output ../Downloads/vate_validate.parquet \
  --vate-include-keys fraudulent_name real_name label \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256 \
  --overwrite-cols
  
  python seton_notebooks/create_VATE.py \
  --input ../Downloads/train_pairs_with_siglip_embeddings.parquet \
  --vate-only-output ../Downloads/vate_train.parquet \
  --vate-include-keys fraudulent_name real_name label \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256 \
  --overwrite-cols
  
  python seton_notebooks/create_VATE.py \
  --input ../Downloads/test_pairs_with_siglip_embeddings.parquet \
  --vate-only-output ../Downloads/vate_test.parquet \
  --vate-include-keys fraudulent_name real_name label \
  --backbone siglip \
  --model-weights weights/best_model_siglip_pair.pt \
  --batch-size 256 \
  --overwrite-cols
