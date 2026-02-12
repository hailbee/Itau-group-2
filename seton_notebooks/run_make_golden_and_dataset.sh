#!/bin/bash
set -e  # exit immediately if any command fails

echo "===== STEP 1: Create Golden Embeddings (test/train/validate) ====="

echo "

chmod +x seton_notebooks/run_make_golden_and_dataset.sh
./seton_notebooks/run_make_golden_and_dataset.sh

"

python seton_notebooks/create_golden_embeddings.py \
  --model_ckpt saved_models/best_model_by_val_trial_1_single_run.pt \
  --input_filepath ../Downloads/test_pairs_with_siglip_embeddings.parquet \
  --internal_layer_size 1024 \
  --output_dim 768 \
  --output_filepath text_to_image/Golden/golden_embeddings_test.parquet

python seton_notebooks/create_golden_embeddings.py \
  --model_ckpt saved_models/best_model_by_val_trial_1_single_run.pt \
  --input_filepath ../Downloads/train_pairs_with_siglip_embeddings.parquet \
  --internal_layer_size 1024 \
  --output_dim 768 \
  --output_filepath text_to_image/Golden/golden_embeddings_train.parquet

python seton_notebooks/create_golden_embeddings.py \
  --model_ckpt saved_models/best_model_by_val_trial_1_single_run.pt \
  --input_filepath ../Downloads/validate_pairs_with_siglip_embeddings.parquet \
  --internal_layer_size 1024 \
  --output_dim 768 \
  --output_filepath text_to_image/Golden/golden_embeddings_validate.parquet


echo "===== STEP 2: Merge Golden + VATE Text Embeddings (test/train/validate) ====="

python seton_notebooks/create_golden_with_VATE_embeddings_precomputed.py \
  --input text_to_image/Golden/golden_embeddings_test.parquet \
  --vate-parquet ../Downloads/vate_test.parquet \
  --output text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --overwrite

python seton_notebooks/create_golden_with_VATE_embeddings_precomputed.py \
  --input text_to_image/Golden/golden_embeddings_train.parquet \
  --vate-parquet ../Downloads/vate_train.parquet \
  --output text_to_image/Golden_and_Text/train_pairs_with_img_and_vate_txt_embs.parquet \
  --overwrite

python seton_notebooks/create_golden_with_VATE_embeddings_precomputed.py \
  --input text_to_image/Golden/golden_embeddings_validate.parquet \
  --vate-parquet ../Downloads/vate_validate.parquet \
  --output text_to_image/Golden_and_Text/validate_pairs_with_img_and_vate_txt_embs.parquet \
  --overwrite


echo "===== STEP 3: Build Positive-Only Dataset (test/train/validate) ====="

python seton_notebooks/build_t2i_positive_only_dataset.py \
  --input text_to_image/Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/Golden_and_Text/test.parquet

python seton_notebooks/build_t2i_positive_only_dataset.py \
  --input text_to_image/Golden_and_Text/train_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/Golden_and_Text/train.parquet

python seton_notebooks/build_t2i_positive_only_dataset.py \
  --input text_to_image/Golden_and_Text/validate_pairs_with_img_and_vate_txt_embs.parquet \
  --output text_to_image/Golden_and_Text/validate.parquet


echo "===== DONE: All outputs generated successfully ====="
