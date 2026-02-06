#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainer import Trainer
from siamese import SiameseEmbeddingModel
from data import TextTeacherPairDataset
from distill_losses import ThesisMarginOnlyWithTeacherProj


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
    raise ValueError(f"Unknown optimizer: {name}")


# -------------------------
# Validation loss (CORRECT OBJECTIVE)
# -------------------------
@torch.inference_mode()
def _val_loss(model, val_loader, criterion, device):
    model.eval()
    total = 0.0

    for txt, img, y in val_loader:
        txt = txt.to(device)
        img = img.to(device)
        y = y.to(device)

        z_txt = F.normalize(model.encode_text(txt), dim=1)
        z_img = F.normalize(img, dim=1)

        loss = criterion(z_txt, z_img, y)
        total += loss.item()

    return total / max(len(val_loader), 1)


# -------------------------
# Config
# -------------------------
@dataclass
class OptunaConfig:
    # Dataset
    txt_prefix: str = "left_txt_emb_"
    img_prefix: str = "right_img_emb_"
    label_col: str = "label"

    # Optuna
    n_trials: int = 50
    short_epochs: int = 5

    lr_low: float = 1e-5
    lr_high: float = 3e-4
    batch_sizes: Tuple[int, ...] = (64, 128, 256)

    hidden_dims: Tuple[int, ...] = (256, 512, 768, 1024)

    optimizers: Tuple[str, ...] = ("adamw", "adam")
    weight_decay_low: float = 1e-7
    weight_decay_high: float = 1e-4


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

    text_dim = _infer_dim_from_prefix(train_df, cfg.txt_prefix)
    img_dim = _infer_dim_from_prefix(train_df, cfg.img_prefix)

    train_ds = TextTeacherPairDataset(
        train_df,
        txt_prefix=cfg.txt_prefix,
        img_prefix=cfg.img_prefix,
        label_col=cfg.label_col,
    )
    val_ds = TextTeacherPairDataset(
        val_df,
        txt_prefix=cfg.txt_prefix,
        img_prefix=cfg.img_prefix,
        label_col=cfg.label_col,
    )

    def objective(trial):
        lr = trial.suggest_float("lr", cfg.lr_low, cfg.lr_high, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)
        hidden_dim = trial.suggest_categorical("hidden_dim", cfg.hidden_dims)
        optimizer_name = trial.suggest_categorical("optimizer", cfg.optimizers)
        weight_decay = trial.suggest_float(
            "weight_decay", cfg.weight_decay_low, cfg.weight_decay_high, log=True
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SiameseEmbeddingModel(
            hidden_dim=hidden_dim
        ).to(dev)

        criterion = ThesisMarginOnlyWithTeacherProj().to(dev)

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
            epochs=cfg.short_epochs,
        )

        # ✅ CORRECT OPTUNA OBJECTIVE
        return _val_loss(model, val_loader, criterion, dev)

    # ✅ CORRECT DIRECTION
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials)

    print("[OPTUNA] best params:", study.best_trial.params)
    print("[OPTUNA] best value (val loss):", study.best_value)


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
