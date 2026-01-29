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


# -------------------------
# Loss (keep whatever you're using)
# -------------------------
class AUCBestHybridLoss(torch.nn.Module):
    def __init__(self, tau=0.05, lam_diag=0.1, lam_mat=0.0, eps=1e-8):
        super().__init__()
        self.tau = float(tau)
        self.lam_diag = float(lam_diag)
        self.lam_mat = float(lam_mat)
        self.eps = float(eps)

    def _pairwise_logistic_rank(self, sim: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pos = sim[y > 0.5]
        neg = sim[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return sim.new_tensor(0.0)
        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
        return F.softplus(-diffs / self.tau).mean()

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label):
        y = label.float()

        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        sim_s = (S_f * S_r).sum(dim=1)

        rank_loss = self._pairwise_logistic_rank(sim_s, y)

        if (self.lam_diag <= 0.0) and (self.lam_mat <= 0.0):
            return rank_loss

        T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps).detach()
        T_r = F.normalize(real_teacher,  dim=1, eps=self.eps).detach()
        sim_t = (T_f * T_r).sum(dim=1)

        reg_diag = (sim_s - sim_t).pow(2).mean() if self.lam_diag > 0.0 else sim_s.new_tensor(0.0)

        reg_mat = sim_s.new_tensor(0.0)
        if self.lam_mat > 0.0:
            M_s = S_f @ S_r.T
            M_t = T_f @ T_r.T
            B = M_s.size(0)
            mask = ~torch.eye(B, dtype=torch.bool, device=M_s.device)
            reg_mat = (M_s[mask] - M_t[mask]).pow(2).mean()

        return rank_loss + self.lam_diag * reg_diag + self.lam_mat * reg_mat


# -------------------------
# Validation metric (student AUC)
# -------------------------
@torch.inference_mode()
def _val_student_auc(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sims_all = []
    y_all = []

    for batch in val_loader:
        fraud_txt, real_txt, _t_fraud, _t_real, y = batch
        fraud_txt = fraud_txt.to(device, non_blocking=True)
        real_txt = real_txt.to(device, non_blocking=True)

        pred_fraud, pred_real = model(fraud_txt, real_txt)
        pred_fraud = F.normalize(pred_fraud, dim=1)
        pred_real  = F.normalize(pred_real,  dim=1)
        sims = (pred_fraud * pred_real).sum(dim=1)

        sims_all.append(sims.detach().cpu())
        y_all.append(y.detach().cpu())

    sims_np = torch.cat(sims_all, dim=0).numpy()
    y_np = torch.cat(y_all, dim=0).numpy().astype(np.int32)
    return float(roc_auc_score(y_np, sims_np))


# -------------------------
# Config
# -------------------------
@dataclass
class OptunaConfig:
    fraud_txt_prefix: str = "fraud_txt_emb_"
    real_txt_prefix: str = "real_txt_emb_"
    fraud_teacher_prefix: str = "fraud_aligned_"
    real_teacher_prefix: str = "real_aligned_"
    label_col: str = "label"

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

    tau_low: float = 0.01
    tau_high: float = 0.2
    lam_diag_low: float = 0.0
    lam_diag_high: float = 1.0
    lam_mat_low: float = 0.0
    lam_mat_high: float = 0.3


def run_optuna(
    *,
    training_filepath: str,
    validate_filepath: str,
    device: Optional[str] = None,
    cfg: Optional[OptunaConfig] = None,
) -> Dict[str, Any]:
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna not installed. `pip install optuna`") from e

    cfg = cfg or OptunaConfig()
    dev = _pick_device(device)
    print(f"[OPTUNA] device={dev}")

    train_df = _load_table(training_filepath)
    val_df = _load_table(validate_filepath)

    text_dim = _infer_dim_from_prefix(train_df, cfg.fraud_txt_prefix)
    _teacher_dim = _infer_dim_from_prefix(train_df, cfg.fraud_teacher_prefix)
    print(f"[OPTUNA] inferred dims: text_dim={text_dim}")

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
        hidden_dim = trial.suggest_categorical("hidden_dim", list(cfg.hidden_dims))
        out_dim = trial.suggest_categorical("out_dim", list(cfg.out_dims))
        optimizer_name = trial.suggest_categorical("optimizer", list(cfg.optimizers))
        weight_decay = trial.suggest_float("weight_decay", cfg.weight_decay_low, cfg.weight_decay_high, log=True)

        tau = trial.suggest_float("tau", cfg.tau_low, cfg.tau_high, log=True)
        lam_diag = trial.suggest_float("lam_diag", cfg.lam_diag_low, cfg.lam_diag_high)
        lam_mat = trial.suggest_float("lam_mat", cfg.lam_mat_low, cfg.lam_mat_high)

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
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
        ).to(dev)

        criterion = AUCBestHybridLoss(tau=tau, lam_diag=lam_diag, lam_mat=lam_mat).to(dev)

        # criterion has no params -> model params only
        optimizer = _build_optimizer(
            optimizer_name,
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        trainer_kwargs = dict(model=model, criterion=criterion, optimizer=optimizer, device=dev)
        trainer = Trainer(**_filter_kwargs(Trainer.__init__, trainer_kwargs))

        train_kwargs = dict(
            dataloader=train_loader,
            validate_dataloader=None,   # keep optuna cheap
            trial_number=int(trial.number),
            test_filepath=None,
            string="_optuna",
            epochs=int(cfg.short_epochs),
            plot_losses=False,
            early_stopping=False,
            save_best=False,
        )
        trainer.train(**_filter_kwargs(trainer.train, train_kwargs))

        val_auc = _val_student_auc(model, val_loader, dev)
        trial.report(val_auc, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(val_auc)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=int(cfg.n_trials))

    print("[OPTUNA] best params:", study.best_trial.params)
    print("[OPTUNA] best value:", float(study.best_value))

    return {
        "best_params": dict(study.best_trial.params),
        "best_value": float(study.best_value),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--short-epochs", type=int, default=5)
    args = ap.parse_args()

    cfg = OptunaConfig(n_trials=int(args.n_trials), short_epochs=int(args.short_epochs))
    run_optuna(
        training_filepath=args.train,
        validate_filepath=args.val,
        device=args.device,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
