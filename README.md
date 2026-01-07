# Itau Group 2 – Text Similarity Evaluation

This repository contains code for evaluating text similarity models for matching
fraudulent (spoofed) company names against real company names using Siamese-style
embedding models and baseline vision-language models.

The project supports:

* Baseline model evaluation (CLIP, CoCa, FLAVA, SigLIP)
* Evaluation of trained Siamese models (embedding + similarity scoring)
* ROC/AUC, accuracy, precision, recall metrics
* Optional plots (ROC, confusion matrix) and exporting misclassified samples

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All experiments are run through `main.py`.

### Modes

`main.py` supports two modes:

* `baseline`
  Runs baseline VLM embedding similarity evaluation using one or more pretrained models.

* `evaluate_saved`
  Evaluates a trained Siamese model checkpoint on a labeled test dataset.

---

## CLI Parameters (main.py)

Below are the arguments supported by `main.py`:

* `--mode` (required)
  Choices: `baseline`, `evaluate_saved`

  * `baseline`: test baseline pretrained models
  * `evaluate_saved`: evaluate a trained Siamese model

* `--test_filepath` (required)
  Path to test data (**CSV or Parquet**) containing:

  * `fraudulent_name`
  * `real_name`
  * `label`

* `--baseline_model` (baseline mode)
  Choices: `clip`, `coca`, `flava`, `siglip`, `all`
  Default: `clip`
  Selects which baseline model(s) to evaluate.

* `--backbone` (siamese backbone selection)
  Choices: `clip`, `coca`, `flava`, `siglip`
  Default: `clip`
  Vision-language backbone used by the Siamese model during evaluation.

* `--batch_size`
  Default: `32`
  Batch size used during embedding generation / evaluation.

* `--plot_roc`
  Default: `False`
  If `True`, plots ROC curve.

* `--model_weights` (evaluate_saved mode)
  Path to the trained Siamese model weights checkpoint.

* `--device`
  Default: auto-detected (e.g., `cuda` if available, else `cpu`)
  Device used to run inference.

* `--plot`
  Default: `False`
  If `True`, plots ROC and confusion matrix.

* `--save_misclassified`
  Default: `False`
  If set, exports misclassified samples to a CSV (path or enabled flag depending on implementation).

---

## Example Commands

### 1) Baseline evaluation (single model)

```bash
python main.py \
  --mode baseline \
  --test_filepath data/processed/german_merged_dataset.csv \
  --baseline_model clip \
  --batch_size 32 \
  --device cuda \
  --plot True
```

### 2) Baseline evaluation (all models)

```bash
python main.py \
  --mode baseline \
  --test_filepath data/processed/german_merged_dataset.csv \
  --baseline_model all \
  --batch_size 32 \
  --device cuda \
  --plot_roc True
```

### 3) Evaluate a trained Siamese model

```bash
python main.py \
  --mode evaluate_saved \
  --test_filepath data/processed/german_merged_dataset.csv \
  --model_weights weights/siamese_checkpoint.pt \
  --backbone clip \
  --batch_size 32 \
  --device cuda \
  --plot True \
  --save_misclassified misclassified.csv
```

---

## Evaluation Outputs

* Metrics printed to stdout:

  * accuracy
  * precision
  * recall
  * ROC AUC
  * optimal thresholds (if enabled)
* Optional plots:

  * ROC curve
  * confusion matrix
* Misclassified samples (if enabled):

  * CSV export of incorrectly predicted pairs

---

## Requirements

* Python 3.12.5
  See `requirements.txt` for the full list.
