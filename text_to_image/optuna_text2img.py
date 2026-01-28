# scripts/optimization/optuna_text2img.py

import optuna
import torch
import pandas as pd
from torch.utils.data import DataLoader

from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator2 import Evaluator2, EvalConfig

from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import Text2ImgDistillDataset
from model_utils.loss.distill_losses import TeacherScoreDistillBCELoss

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def objective(trial):
    device = pick_device()

    # --------------------
    # Hyperparameters
    # --------------------
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])

    # --------------------
    # Load data
    # --------------------
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    train_ds = Text2ImgDistillDataset(train_df)
    val_ds   = Text2ImgDistillDataset(val_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------
    # Model
    # --------------------
    model = SiameseEmbeddingModel(
        embedding_dim=768,
        hidden_dim=hidden_dim,
        out_dim=768,
    ).to(device)

    # --------------------
    # Loss (FIXED)
    # --------------------
    criterion = TeacherScoreDistillBCELoss().to(device)

    # --------------------
    # Optimizer
    # --------------------
    params = list(model.parameters()) + list(criterion.parameters())

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    # --------------------
    # Trainer
    # --------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_type="text2img",
    )

    # --------------------
    # Train (SHORT!)
    # --------------------
    trainer.train(
        dataloader=train_loader,
        trial_number=trial.number,
        test_filepath=None,
        string="_optuna",
        mode="text2img",
        epochs=3,              # 🔴 SHORT ON PURPOSE
        validate_dataloader=val_loader,
        plot_losses=False,
        early_stopping=False,
        save_best=False,
    )

    # --------------------
    # Evaluate
    # --------------------
    evaluator = Evaluator2(
        model,
        EvalConfig(batch_size=2048),
    )

    _, metrics = evaluator.evaluate(VAL_PATH)
    val_auc = metrics["aligned_text_space"]["roc_auc"]

    return val_auc

if __name__ == "__main__":
    TRAIN_PATH = "Golden_and_Text/train_pairs_with_img_and_txt_embs.parquet"
    VAL_PATH   = "Golden_and_Text/validate_pairs_with_img_and_txt_embs.parquet"

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial.params)
    print("Best AUC:", study.best_value)
