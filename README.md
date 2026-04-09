# Itau Group 2 Models

This repository contains saved classifiers for business-name matching and a command-line script for applying them to new data.

## Files

`main.py`
Lists the available models, explains the inputs for each one, and runs predictions.

`saved_models/*.joblib`
Pretrained classifiers.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

List the available models:

```bash
python3 main.py --list-models
```

See what a model expects:

```bash
python3 main.py --describe-model total_5f_model.joblib
```

Run a model:

```bash
python3 main.py --model-path metrics_model.joblib \
  --data your_pairs.csv
```

Predictions are written to `outputs/<model>_predictions.csv` unless you set `--output`. If your input includes a label column, the script also prints evaluation metrics.

## Input Format

At minimum, text-based features use these columns:

```text
fraudulent_name
real_name
```

If your label column is present, the default name is:

```text
label
```

You can override the column names with:

```bash
--fraud-col
--real-col
--label-col
--positive-label
```

For cosine-based features, there are two ways to supply inputs:

1. Put the required feature columns directly in `--data`.
2. Pass source tables such as `--deja-data`, `--text-data`, or `--unifont-data`, and add the matching `--*-projector` file if the cosine feature needs to be built from embeddings.

## Model Guide

| Model | Required inputs |
| --- | --- |
| `metrics_model.joblib` | raw text columns only |
| `deja_model.joblib` | Deja cosine input |
| `sigliptext_model.joblib` | text cosine input |
| `image_model.joblib` | raw text columns and Deja cosine input |
| `text_model.joblib` | raw text columns and text cosine input |
| `small_model.joblib` | text cosine and Deja cosine |
| `medium_model.joblib` | text cosine, Deja cosine, Unifont cosine, Gentium cosine |
| `large_model.joblib` | text cosine, Deja cosine, Unifont cosine, Libre cosine, Exo2 cosine, Doulos cosine, Cousine cosine |
| `total_1f_model.joblib` | raw text columns, text cosine, Deja cosine |
| `total_3f_model.joblib` | raw text columns, text cosine, Deja cosine, Unifont cosine, Gentium cosine |
| `total_5f_model.joblib` | raw text columns, text cosine, Deja cosine, Unifont cosine, Libre cosine, Doulos cosine, Cousine cosine |
| `total_5f_mod_model.joblib` | same inputs as `total_5f_model.joblib` |
| `total_5f_img_model.joblib` | raw text columns, Deja cosine, Unifont cosine, Libre cosine, Doulos cosine, Cousine cosine |

For the exact flags for a given model, run:

```bash
python3 main.py --describe-model <model_name>
```

## Examples

Apply the text-metrics model:

```bash
python3 main.py --model-path metrics_model.joblib \
  --data your_pairs.csv
```

Apply a model that needs a separate Deja cosine source:

```bash
python3 main.py --model-path image_model.joblib \
  --data your_pairs.csv \
  --deja-data your_deja_features.parquet
```

Apply a model with a merged feature table:

```bash
python3 main.py --model-path total_5f_model.joblib \
  --data your_feature_table.parquet
```

Apply a model when you have embeddings and projector checkpoints:

```bash
python3 main.py --model-path total_5f_model.joblib \
  --data your_pairs.csv \
  --text-data your_text_embeddings.parquet \
  --text-projector your_text_projector.pt \
  --deja-data your_deja_embeddings.parquet \
  --deja-projector your_deja_projector.pt
```

Add the other sources the same way for `unifont`, `gentium`, `libre`, `exo2`, `doulos`, and `cousine` when the selected model requires them.
