Run `scripts/precompute_model_inputs.py` to build a precomputed candidate-feature store here.

Example:

```bash
python3 scripts/precompute_model_inputs.py \
  --dataset data/benign_domains.csv \
  --model-path saved_models/total_5f_img_model.joblib \
  --output-dir precomputed/benign_total5f_img
```

The generated files can be large. For `total_5f_img_model.joblib`, a full benign-domain cache is still multi-GB even
with `float16` storage because it stores projected embeddings for every required candidate-side font source.
