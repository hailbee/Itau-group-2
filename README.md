# Itau Group 2 Models

This repository contains saved classifiers for business-name matching, a command-line script for applying them to new data, and a Flask web app for scanning a benign-domain dataset with the `total_5f_model.joblib` pipeline.

## Files

`main.py`
Lists the available models, explains the inputs for each one, and runs predictions.

`saved_models/*.joblib`
Pretrained classifiers.

`data/benign_domains.csv`
Bundled one-column domain dataset, currently sourced from the Alexa top 1M list.

`assets/fonts/*.ttf`
Local font files used to render glyph images for the font-based similarity features.

`web_app.py`
Flask entrypoint for the interactive matcher UI.

`domain_matcher.py`
Shared runtime for text metrics, rendered-font embeddings, and chunked dataset search.

`scripts/precompute_model_inputs.py`
Offline builder for precomputing candidate-side projected embeddings so the web app only needs to compare a query against cached model inputs.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the project virtual environment for any command that actually scores a saved `.joblib` model. The bundled model files were saved with the pinned `scikit-learn` version from `requirements.txt`, so running them from a system Python with a different sklearn version can fail during unpickling. Metadata-only commands such as `python3 main.py --list-models`, `python3 main.py --describe-model ...`, and `scripts/precompute_model_inputs.py` no longer need to unpickle the estimator itself.

## Web App

Shortest path if `data/benign_domains.csv` already exists:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_web_app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Fast local test from scratch:

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build the one-column domain dataset if you do not already have `data/benign_domains.csv`:

```bash
python3 scripts/prepare_benign_domains.py \
  --input /path/to/archive.zip \
  --output data/benign_domains.csv
```

3. Precompute the projected candidate-side inputs:

```bash
python3 scripts/precompute_model_inputs.py \
  --dataset data/benign_domains.csv \
  --model-path saved_models/total_5f_model.joblib \
  --output-dir precomputed/benign_total5f
```

4. Start the web app:

```bash
python3 scripts/run_web_app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Full exact flow for the best runtime during searches:

```bash
python3 scripts/precompute_model_inputs.py \
  --dataset data/benign_domains.csv \
  --model-path saved_models/total_5f_model.joblib \
  --output-dir precomputed/benign_total5f
```

Run the Flask app:

```bash
python3 scripts/run_web_app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The app:

1. Loads the precomputed projected candidate-side inputs from `precomputed/benign_total5f`
2. Normalizes the user input
3. Computes the query-side embeddings and text metrics
4. Compares the query against the cached candidate vectors
5. Builds the `total_5f_model.joblib` feature vector and ranks the best matches

Important:

- The exact projected-cosine path requires projector checkpoints in `projectors/README.md`.
- With the bundled 1M-row dataset, the precompute store is about `6.68 GiB` with `float16` output and about `13.35 GiB` with `float32`. Larger datasets scale roughly linearly.
- If the precomputed store is missing, the matcher falls back to runtime candidate embedding generation, which is much slower.
- The UI still exposes `chunk_size` and `max_rows` so you can preview on a smaller slice before scanning all `1,000,000` bundled domains.
- The web UI shows live scan progress while a search is running, including rows scanned, percent complete, predicted positives so far, and elapsed time.
- For a quick sanity check, start with a small `max_rows` such as `50000` or `100000`, then remove the cap once everything looks right.

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

Create the bundled one-column domain CSV from the Alexa archive:

```bash
python3 scripts/prepare_benign_domains.py \
  --input archive.zip \
  --output data/benign_domains.csv
```

The same script also supports the Kaggle benign/DGA source:

```bash
python3 scripts/prepare_benign_domains.py \
  --input dga-or-benign-domain-names.zip \
  --output data/benign_domains.csv
```

Precompute candidate-side model inputs for faster web searches:

```bash
python3 scripts/precompute_model_inputs.py \
  --dataset data/benign_domains.csv \
  --model-path saved_models/total_5f_model.joblib \
  --output-dir precomputed/benign_total5f
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
