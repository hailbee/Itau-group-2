#!/usr/bin/env python3
"""
Append SigLIP TEXT embeddings to an existing file that already contains IMAGE embeddings.

SAFE VERSION:
- Preserves all existing columns exactly
- Appends fraud_txt_emb_* and real_txt_emb_* as float32
- Hard-fails on ANY column collision
- No index reset, no reordering, no dtype pollution

Example:
python seton_notebooks/create_golden_with_text_embeddings.py \
  --input text_to_image/Golden/golden_embeddings_validate.parquet \
  --output text_to_image/Golden_and_Text/validate_pairs_with_img_and_txt_embs.parquet \
  --batch-size 256
  
FIX THIS IT IS PROBABLY WRONG
"""
