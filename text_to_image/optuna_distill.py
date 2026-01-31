#!/usr/bin/env python3
"""
optuna_distill.py  (UPDATED for NEW TASK)

Optuna tuning for:
  - Training on NEW 4-pairs-per-row TEXT->IMAGE binary dataset
      (left_txt_emb_*, right_img_emb_*, label, pair_kind optional)
  - Validation objective: ROC AUC on cosine similarity between
      translated fraud text and translated real text
    using Evaluator2 on an EVAL parquet with:
      (fraud_txt_emb_*, real_txt_emb_*, label or spoof_attempt)

Notes:
  - This script does NOT require a text->image validation set.
  - It returns the *student* ROC AUC from Evaluator2 ("STUDENT" metric).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import TextImageBinaryPairDataset
from siamese import SiameseEmbeddingModel
from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig
from distill_losses import BinaryCosineMarginLoss


# -------------------------
# Utils
# -------------------------
def _pick_device(device_override: Optional[str] = None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _infer_dim_from_prefix(df: pd.DataFrame, prefix: str) -> int:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns with prefix '{prefix}'")
    return len(cols)


def _build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


# -------------------------
# Config
# -------------------------
@dataclass
class OptunaConfig:
    # ---- TRAIN (text->image pairs parquet) ----
    left_txt_prefix: str = "left_txt_emb_"
    right_img_prefix: str = "right_img_emb_"
    train_label_col: str = "label"
    pair_kind_col: str = "pair_kind"
    return_pair_kind: bool = True

    # ---- EVAL (text-text pairs parquet) ----
    eval_fraud_txt_prefix: str = "fraud_txt_emb_"
    eval_real_txt_prefix: str = "real_txt_emb_"
    eval_label_col: str = "label"

    # ---- Optuna / training schedule ----
    n_trials: int = 50
    short_epochs: int = 4

    lr_low: float = 1e-5
    lr_high: float = 3e-4
    batch_sizes: Tuple[int, ...] = (128, 256, 512)

    hidden_dims: Tuple[int, ...] = (256, 512, 768, 1024)

    optimizers: Tuple[str, ...] = ("adamw", "adam")
    weight_decay_low: float = 1e-7
    weight_decay_high: float = 3e-4

    # ---- Margin-only tuning ----
    pos_margin_low: float = 0.3
    pos_margin_high: float = 0.85

    # Fixed negative threshold (do NOT tune). 0.0 is the “don’t be positively correlated” rule.
    fixed_neg_margin: float = 0.0

    # architecture switches
    share_text_heads_choices: Tuple[bool, ...] = (False, True)
    share_teacher_heads_choices: Tuple[bool, ...] = (True,)  # usually keep True

    # training misc
    use_amp: bool = True
    grad_clip_norm: Optional[float] = None

    # evaluation
    eval_batch_size: int = 4096
    eval_max_rows: Optional[int] = None


# -------------------------
# Eval helper (objective metric)
# -------------------------
@torch.inference_mode()
def _eval_student_auc(model, eval_filepath: str, cfg: OptunaConfig) -> float:
    model.eval()
    evaluator = Evaluator2(
        model,
        EvalConfig(
            batch_size=cfg.eval_batch_size,
            fraud_txt_prefix=cfg.eval_fraud_txt_prefix,
            real_txt_prefix=cfg.eval_real_txt_prefix,
            label_col=cfg.eval_label_col,
            compute_raw_text=False,   # keep optuna fast
            compute_teacher=False,    # keep optuna fast
        ),
    )
    _df, metrics = evaluator.evaluate(eval_filepath, max_rows=cfg.eval_max_rows)
    return float(metrics["student"]["roc_auc"])


# -------------------------
# Optuna runner
# -------------------------
def run_optuna(
    *,
    train_t2i_filepath: str,
    val_eval_filepath: str,
    device: Optional[str] = None,
    cfg: Optional[OptunaConfig] = None,
):
    import optuna

    cfg = cfg or OptunaConfig()
    dev = _pick_device(device)
    print(f"[OPTUNA] device={dev}")
    print(f"[OPTUNA] train_t2i={train_t2i_filepath}")
    print(f"[OPTUNA] val_eval={val_eval_filepath}")

    train_df = _load_table(train_t2i_filepath)
    eval_df = _load_table(val_eval_filepath)

    # Train dims from NEW dataset
    text_dim = _infer_dim_from_prefix(train_df, cfg.left_txt_prefix)
    teacher_dim = _infer_dim_from_prefix(train_df, cfg.right_img_prefix)

    # Eval dims sanity check
    eval_text_dim_f = _infer_dim_from_prefix(eval_df, cfg.eval_fraud_txt_prefix)
    eval_text_dim_r = _infer_dim_from_prefix(eval_df, cfg.eval_real_txt_prefix)
    if eval_text_dim_f != text_dim or eval_text_dim_r != text_dim:
        print(
            f"[WARN] Eval text dims ({eval_text_dim_f},{eval_text_dim_r}) != train left_txt dim ({text_dim}). "
            f"This is OK only if you intentionally changed embeddings; otherwise fix prefixes/files."
        )

    train_ds = TextImageBinaryPairDataset(
        train_df,
        left_txt_prefix=cfg.left_txt_prefix,
        right_img_prefix=cfg.right_img_prefix,
        label_col=cfg.train_label_col,
        pair_kind_col=cfg.pair_kind_col,
        return_pair_kind=cfg.return_pair_kind,
        return_orig_row_id=False,
    )

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", cfg.lr_low, cfg.lr_high, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)
        hidden_dim = trial.suggest_categorical("hidden_dim", cfg.hidden_dims)
        optimizer_name = trial.suggest_categorical("optimizer", cfg.optimizers)
        weight_decay = trial.suggest_float("weight_decay", cfg.weight_decay_low, cfg.weight_decay_high, log=True)

        share_text_heads = trial.suggest_categorical("share_text_heads", cfg.share_text_heads_choices)
        share_teacher_heads = trial.suggest_categorical("share_teacher_heads", cfg.share_teacher_heads_choices)

        # Keep out_dim matched to image/teacher dim (your current setup).
        out_dim = int(teacher_dim)

        model = SiameseEmbeddingModel(
            embedding_dim=int(text_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            teacher_dim=int(teacher_dim),
            share_text_heads=bool(share_text_heads),
            share_teacher_heads=bool(share_teacher_heads),
            dropout=0.0,
            activation="relu",
        ).to(dev)

        # One-margin behavior: tune only pos_margin; keep neg fixed
        pos_margin = trial.suggest_float("pos_margin", cfg.pos_margin_low, cfg.pos_margin_high)
        criterion = BinaryCosineMarginLoss(
            pos_margin=float(pos_margin),
            neg_margin=float(cfg.fixed_neg_margin),
            normalize_inputs=False,  # Trainer normalizes
            squared=True,
        ).to(dev)

        optimizer = _build_optimizer(
            optimizer_name,
            model.parameters(),  # margin loss has no parameters; keep optimizer clean
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=True,
            pin_memory=(dev.type == "cuda"),
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=dev,
            use_amp=cfg.use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        trainer.train(
            dataloader=train_loader,
            validate_dataloader=None,  # keep optuna fast
            test_filepath=None,        # do NOT eval each epoch in optuna
            trial_number=trial.number,
            epochs=int(cfg.short_epochs),
            early_stopping=False,
            save_best=False,
        )

        val_auc = _eval_student_auc(model, val_eval_filepath, cfg)
        trial.set_user_attr("val_auc", val_auc)
        return float(val_auc)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(cfg.n_trials))

    print("\n[OPTUNA] best params:", study.best_trial.params)
    print("[OPTUNA] best value:", study.best_value)
    return study


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-t2i", required=True, help="Training parquet: NEW text->image 4-pairs dataset.")
    ap.add_argument("--val-eval", required=True, help="Validation parquet: text-text pairs for ROC AUC (Evaluator2).")
    ap.add_argument("--device", default=None)

    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--short-epochs", type=int, default=4)

    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--eval-max-rows", type=int, default=None)

    args = ap.parse_args()

    cfg = OptunaConfig(
        n_trials=int(args.n_trials),
        short_epochs=int(args.short_epochs),
        eval_batch_size=int(args.eval_batch_size),
        eval_max_rows=None if args.eval_max_rows is None else int(args.eval_max_rows),
    )

    run_optuna(
        train_t2i_filepath=args.train_t2i,
        val_eval_filepath=args.val_eval,
        device=args.device,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
