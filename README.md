# Itau Group 2 – Text Similarity Evaluation

This repository contains code for evaluating text similarity models for matching
fraudulent names against real names using Siamese-style embedding models and
baseline vision-language models.

The project supports:
- Baseline model evaluation (CLIP, CoCa, FLAVA, SigLIP)
- Evaluation of trained Siamese models
- ROC/AUC, accuracy, precision, recall metrics

---

## Repository Structure

Itau-group-2/
├── data/                     # Datasets (CSV / Parquet)
├── main.py                   # Entry point for evaluation
├── model_utils/              # Model definitions and wrappers
├── scripts/
│   ├── baseline/             # Baseline model evaluation
│   └── evaluation/           # Evaluator and error analysis
├── utils/                    # Embeddings, metrics, helpers
├── weights/                  # Trained model checkpoints
├── requirements.txt          # Python dependencies
├── README.md

---

## Install dependencies

pip install -r requirements.txt

---

## Usage

All experiments are run through `main.py`.

---

## Evaluation Outputs

- Metrics printed to stdout:
  - accuracy
  - precision
  - recall
  - ROC AUC
  - optimal thresholds
- Misclassified samples (if enabled):

---

## Requirements

- Python 3.12.5
See `requirements.txt` for the full list.
