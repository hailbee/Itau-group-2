#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from trainer import Trainer
from siamese import SiameseEmbeddingModel
from data import Text2TeacherDistillDataset
from distill_losses import ThesisMarginOnlyWithTeacherProj


# -------------------------
# Utils
# -------------------------
def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(callable_obj)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


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
# Validation metric
# -------------------------
@torch.inference_mode()
def _val_student_auc(model, val_loader, device):
    model.eval()
    sims_all, y_all = [], []

    for fraud_txt, real_txt, *_t, y in val_loader:
        fraud_txt = fraud_txt.to(device)
        real_txt = real_txt.to(device)

        z_fraud, z_real = model(fraud_txt, real_txt)
        z_fraud = F.normalize(z_fraud, dim=1)
        z_real = F.normalize(z_real, dim=1)

        sims = (z_fraud * z_real).sum(dim=1)
        sims_all.append(sims.cpu())
        y_all.append(y.cpu())

    sims_np = torch.cat(sims_all).numpy()
    y_np = torch.cat(y_all).numpy().astype(np.int32)
    return float(roc_auc_score(y_np, sims_np))


# -------------------------
# Config
# -------------------------
@dataclass
class OptunaConfig:
    # Dataset-related (ONLY these go to the dataset)
    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"
    fraud_teacher_prefix: str = "fraud_aligned_"
    real_teacher_prefix: str = "real_aligned_"
    label_col: str = "label"

    # Optuna / training
    n_trials: int = 50
    short_epochs: int = 5

    lr_low: float = 1e-5
    lr_high: float = 3e-4
    batch_sizes: Tuple[int, ...] = (64, 128, 256)

    hidden_dims: Tuple[int, ...] = (256, 512, 768, 1024)
    out_dims: Tuple[int, ...] = (128, 256, 512, 768)

    optimizers: Tuple[str, ...] = ("adamw", "adam")
    weight_decay_low: float = 1e-7
    weight_decay_high: float = 1e-4

    margin_low: float = 0.2
    margin_high: float = 1.6


# -------------------------
# Optuna runner
# -------------------------
def run_optuna(
    *,
    training_filepath: str,
    validate_filepath: str,
    device: Optional[str] = None,
    cfg: Optional[OptunaConfig] = None,
):
    import optuna

    cfg = cfg or OptunaConfig()
    dev = _pick_device(device)
    print(f"[OPTUNA] device={dev}")

    train_df = _load_table(training_filepath)
    val_df = _load_table(validate_filepath)

    text_dim = _infer_dim_from_prefix(train_df, cfg.fraud_txt_prefix)
    teacher_dim = _infer_dim_from_prefix(train_df, cfg.fraud_teacher_prefix)

    # ✅ PASS ONLY DATASET ARGS
    train_ds = Text2TeacherDistillDataset(
        train_df,
        fraud_txt_prefix=cfg.fraud_txt_prefix,
        real_txt_prefix=cfg.real_txt_prefix,
        fraud_teacher_prefix=cfg.fraud_teacher_prefix,
        real_teacher_prefix=cfg.real_teacher_prefix,
        label_col=cfg.label_col,
    )
    val_ds = Text2TeacherDistillDataset(
        val_df,
        fraud_txt_prefix=cfg.fraud_txt_prefix,
        real_txt_prefix=cfg.real_txt_prefix,
        fraud_teacher_prefix=cfg.fraud_teacher_prefix,
        real_teacher_prefix=cfg.real_teacher_prefix,
        label_col=cfg.label_col,
    )

    def objective(trial):
        lr = trial.suggest_float("lr", cfg.lr_low, cfg.lr_high, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)
        hidden_dim = trial.suggest_categorical("hidden_dim", cfg.hidden_dims)
        out_dim = trial.suggest_categorical("out_dim", cfg.out_dims)
        optimizer_name = trial.suggest_categorical("optimizer", cfg.optimizers)
        weight_decay = trial.suggest_float(
            "weight_decay", cfg.weight_decay_low, cfg.weight_decay_high, log=True
        )
        margin = trial.suggest_float("margin", cfg.margin_low, cfg.margin_high)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            teacher_dim=teacher_dim,
        ).to(dev)

        criterion = ThesisMarginOnlyWithTeacherProj(margin=margin).to(dev)

        optimizer = _build_optimizer(
            optimizer_name,
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=dev,
        )

        trainer.train(
            dataloader=train_loader,
            validate_dataloader=None,
            trial_number=trial.number,
            epochs=cfg.short_epochs,
            early_stopping=False,
            save_best=False,
        )

        val_auc = _val_student_auc(model, val_loader, dev)
        return val_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.n_trials)

    print("[OPTUNA] best params:", study.best_trial.params)
    print("[OPTUNA] best value:", study.best_value)


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--short-epochs", type=int, default=5)
    args = ap.parse_args()

    cfg = OptunaConfig(
        n_trials=args.n_trials,
        short_epochs=args.short_epochs,
    )

    run_optuna(
        training_filepath=args.train,
        validate_filepath=args.val,
        device=args.device,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
