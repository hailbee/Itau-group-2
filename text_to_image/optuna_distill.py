#!/usr/bin/env python3
"""
Optuna runner for Text -> Teacher distillation.

Key design goals:
- Optuna is isolated here (main file doesn't need to import optuna directly).
- Robust to Trainer API drift: we filter kwargs based on function signatures.
- We DO NOT depend on Evaluator2 API during Optuna; we compute validation AUC directly
  from the val dataloader to avoid "unexpected keyword" issues.

Expected dataset columns (prefix-based):
  - label
  - fraud_txt_*
  - real_txt_*
  - fraud_aligned_*   (teacher)
  - real_aligned_*    (teacher)
"""

from __future__ import annotations

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
from distill_losses import EmbeddingCosineDistillLoss

def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by callable_obj (function or method)."""
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
        raise ValueError(f"Could not infer dim: no columns with prefix '{prefix}'")
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


@torch.inference_mode()
def _val_student_auc(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Student score = cosine(pred_fraud, pred_real).
    AUC computed against label (0/1).
    """
    model.eval()
    sims_all = []
    y_all = []

    for batch in val_loader:
        fraud_txt, real_txt, _t_fraud, _t_real, y = batch
        fraud_txt = fraud_txt.to(device, non_blocking=True)
        real_txt = real_txt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred_fraud, pred_real = model(fraud_txt, real_txt)
        pred_fraud = F.normalize(pred_fraud, dim=1)
        pred_real = F.normalize(pred_real, dim=1)
        sims = F.cosine_similarity(pred_fraud, pred_real, dim=1)

        sims_all.append(sims.detach().cpu())
        y_all.append(y.detach().cpu())

    sims_np = torch.cat(sims_all, dim=0).numpy()
    y_np = torch.cat(y_all, dim=0).numpy().astype(np.int32)

    # roc_auc_score expects higher score = more likely class 1
    return float(roc_auc_score(y_np, sims_np))


@dataclass
class OptunaConfig:
    # dataset prefixes
    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"
    fraud_teacher_prefix: str = "fraud_aligned_"
    real_teacher_prefix: str = "real_aligned_"
    label_col: str = "label"

    # optuna control
    n_trials: int = 50
    short_epochs: int = 5  # keep optuna cheap

    # search space (defaults mirror your style)
    lr_low: float = 1e-5
    lr_high: float = 3e-4
    batch_sizes: Tuple[int, ...] = (64, 128, 256)
    hidden_dims: Tuple[int, ...] = (256, 512, 768)
    optimizers: Tuple[str, ...] = ("adamw", "adam")
    weight_decay_low: float = 1e-7
    weight_decay_high: float = 1e-4


def run_optuna(
    *,
    training_filepath: str,
    validate_filepath: str,
    device: Optional[str] = None,
    cfg: Optional[OptunaConfig] = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - best_params
      - best_value (val AUC)
    """
    try:
        import optuna  # intentionally local import
    except Exception as e:
        raise RuntimeError(
            "Optuna is not installed in this environment. "
            "Install it (pip install optuna) or run with --optuna False."
        ) from e

    cfg = cfg or OptunaConfig()
    dev = _pick_device(device)
    print(f"[OPTUNA] device={dev}")

    # Load once (fast enough, avoids repeated parquet reads)
    train_df = _load_table(training_filepath)
    val_df = _load_table(validate_filepath)

    text_dim = _infer_dim_from_prefix(train_df, cfg.fraud_txt_prefix)
    teacher_dim = _infer_dim_from_prefix(train_df, cfg.fraud_teacher_prefix)
    print(f"[OPTUNA] inferred dims: text_dim={text_dim} | teacher_dim={teacher_dim}")

    # Build datasets once; dataloaders will be re-created per trial (batch size varies)
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
        batch_size = trial.suggest_categorical("batch_size", list(cfg.batch_sizes))
        internal_layer_size = trial.suggest_categorical("internal_layer_size", list(cfg.hidden_dims))
        optimizer_name = trial.suggest_categorical("optimizer", list(cfg.optimizers))
        weight_decay = trial.suggest_float("weight_decay", cfg.weight_decay_low, cfg.weight_decay_high, log=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=True,
            pin_memory=(dev.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            pin_memory=(dev.type == "cuda"),
        )

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=int(internal_layer_size),
            out_dim=teacher_dim,  # IMPORTANT: must match teacher (128)
        ).to(dev)

        criterion = EmbeddingCosineDistillLoss().to(dev)

        optim_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = _build_optimizer(optimizer_name, optim_params, lr=float(lr), weight_decay=float(weight_decay))

        # Trainer ctor can vary across your copies — filter kwargs
        trainer_kwargs = dict(model=model, criterion=criterion, optimizer=optimizer, device=dev, model_type=None)
        trainer = Trainer(**_filter_kwargs(Trainer.__init__, trainer_kwargs))

        # Trainer.train API can vary — filter kwargs and DO NOT pass mode
        train_kwargs = dict(
            dataloader=train_loader,
            validate_dataloader=None,
            trial_number=int(trial.number),
            test_filepath=None,
            string="_optuna",
            epochs=int(cfg.short_epochs),
            plot_losses=False,
            early_stopping=False,
            save_best=False,
        )
        trainer.train(**_filter_kwargs(trainer.train, train_kwargs))

        # Metric for Optuna: student AUC on val (stable, doesn’t depend on Evaluator2 API)
        val_auc = _val_student_auc(model, val_loader, dev)
        return val_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(cfg.n_trials))

    print("[OPTUNA] best params:", study.best_trial.params)
    print("[OPTUNA] best value (val AUC):", float(study.best_value))

    return {"best_params": dict(study.best_trial.params), "best_value": float(study.best_value)}
