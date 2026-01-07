# Itau Group 2 – Text Similarity Evaluation

This repository contains code for evaluating text similarity models for matching
fraudulent names against real names using Siamese-style embedding models and
baseline vision-language models.

The project supports:
- Baseline model evaluation (CLIP, CoCa, FLAVA, SigLIP)
- Evaluation of trained Siamese models
- ROC/AUC, accuracy, precision, recall metrics

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
