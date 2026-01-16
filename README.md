# Itau Group 2 – Text Similarity Evaluation

This repository contains code for evaluating text similarity models for matching
fraudulent (spoofed) names against real names using Siamese-style embedding models
and baseline vision-language models (VLMs).

The project supports:

* Baseline model evaluation (CLIP, CoCa, FLAVA, SigLIP, InternVL)
* Evaluation of trained Siamese models using embedding similarity
* ROC/AUC, accuracy, precision, recall metrics
* Optional plots (ROC, confusion matrix)
* Exporting misclassified samples for error analysis

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All evaluations are run through `main.py`.

### Supported Modes

`main.py` supports two execution modes:

* **`baseline`**
  Evaluates pretrained vision-language models by computing similarity between name pairs.

* **`evaluate_saved`**
  Loads a trained Siamese model checkpoint and evaluates it on a labeled test dataset.

---

## CLI Parameters (`main.py`)

Below are the arguments supported by `main.py` and how they are used:

* **`--mode`** (required)
  Choices: `baseline`, `evaluate_saved`

  * `baseline`: evaluate baseline VLM models
  * `evaluate_saved`: evaluate a trained Siamese model

* **`--test_filepath`** (required)
  Path to test data (**CSV or Parquet**) containing the columns:

  * `fraudulent_name`
  * `real_name`
  * `label`

* **`--baseline_model`** (baseline mode only)
  Choices: `clip`, `coca`, `flava`, `siglip`, `internvl`, `all`
  Default: `clip`
  Selects which baseline model(s) to evaluate.

* **`--backbone`** (evaluate_saved mode)
  Choices: `clip`, `coca`, `flava`, `siglip`, `internvl`
  Default: `clip`
  Backbone used when instantiating the Siamese model for evaluation.

* **`--model_weights`** (evaluate_saved mode, required)
  Path to the trained Siamese model checkpoint (`.pt`).

* **`--batch_size`**
  Default: `32`
  Batch size used during embedding generation and evaluation.

* **`--device`**
  Default: automatically set to `cuda` if available, otherwise `cpu`.

* **`--plot_roc`**
  Default: `False`
  If `True`, plots the ROC curve (baseline mode).

* **`--plot`**
  Default: `False`
  If `True`, plots ROC curve and confusion matrix (Siamese evaluation).

* **`--save_misclassified`**
  Default: `False`
  If enabled, extracts misclassified samples after evaluation and saves them to
  `misclassified_samples.csv`.

---

## Example Commands

### 1) Baseline evaluation (single model)

```bash
python main.py \
  --mode baseline \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --baseline_model clip \
  --batch_size 32 \
  --plot_roc True
```

---

### 2) Baseline evaluation (all models)

```bash
python main.py \
  --mode baseline \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --baseline_model all \
  --batch_size 32
```

---

### 3) Evaluate a trained Siamese model

```bash
python main.py \
  --mode evaluate_saved \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --backbone siglip \
  --model_weights weights/best_model_siglip_pair.pt \
  --plot False \
  --save_misclassified True
```

---

## Evaluation Outputs

* Metrics printed to stdout:

  * accuracy
  * precision
  * recall
  * ROC AUC
  * optimal threshold
* Optional plots:

  * ROC curve
  * confusion matrix
* Error analysis:

  * `misclassified_samples.csv` containing incorrectly predicted pairs

---

## Requirements

* Python 3.12.5

See `requirements.txt` for the full list.
