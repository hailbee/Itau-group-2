# scripts/optimization/base_optimizer.py

import os
from datetime import datetime

import torch
import pandas as pd
import numpy as np

from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import EmbeddingPairDataset

# -----------------------------
# Loss import (with safe fallback)
# -----------------------------
try:
    # Preferred: your updated two-margin cosine-hinge ContrastiveLoss lives here
    # class ContrastiveLoss(m_pos, m_neg, w_pos=1.0, w_neg=1.0, reduction="mean", enforce_gap=True)
    from model_utils.loss.pair_losses import ContrastiveLoss as CosineContrastiveTwoMargin
except Exception:
    import torch.nn as nn
    import torch.nn.functional as F

    class CosineContrastiveTwoMargin(nn.Module):
        """
        Two-margin cosine contrastive hinge loss.
    
        y=1 (positive): penalize if cos < m_pos
        y=0 (negative): penalize if cos > m_neg
    
        REQUIRE: m_pos > m_neg
        """
        def __init__(self, m_pos: float, m_neg: float):
            super().__init__()
            self.m_pos = float(m_pos)
            self.m_neg = float(m_neg)
            if not (self.m_pos > self.m_neg):
                raise ValueError(f"Need m_pos > m_neg, got m_pos={self.m_pos}, m_neg={self.m_neg}")
    
        def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y.float()
    
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            c = (z1 * z2).sum(dim=1)  # cosine similarity
    
            pos_loss = y * F.relu(self.m_pos - c).pow(2)
            neg_loss = (1.0 - y) * F.relu(c - self.m_neg).pow(2)
    
            return (pos_loss + neg_loss).mean()


# ============================================================
# BaseOptimizer (PRECOMPUTED EMBEDDINGS VERSION)
# ============================================================

class BaseOptimizer:
    """
    Hyperparameter optimization using PRECOMPUTED EMBEDDINGS.

    Updated for TWO-MARGIN cosine contrastive:
      - Uses CosineContrastiveTwoMargin(m_pos, m_neg, w_pos, w_neg)
      - Samples/tunes: m_neg + gap (so m_pos = m_neg + gap), w_pos, w_neg
      - Saves best model by validation ROC-AUC when validate_filepath is provided
        (fallback: best val loss when no validation set)
    """

    def __init__(
        self,
        model_type,
        model_name=None,
        device=None,
        log_dir="optimization_results",
        embedding_dim=768,
        fake_start=3,
        fake_end=771,
        real_start=771,
        real_end=1539,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.log_dir = log_dir

        # embedding info (kept for compatibility even if unused elsewhere)
        self.embedding_dim = embedding_dim
        self.fake_slice = slice(fake_start, fake_end)
        self.real_slice = slice(real_start, real_end)

        self.results = []

        # Best-tracking
        self.best_val_loss = float("inf")
        self.best_val_auc = float("-inf")

        # Normalize log_dir if running in Colab paths
        if "/content" in self.log_dir:
            self.log_dir = "optimization_results"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)

        print(f"[INFO] Using precomputed embeddings (dim={self.embedding_dim})")
        print(f"[INFO] Logs: {self.log_dir}")

    # ------------------------------------------------------------
    # MODEL CREATION
    # ------------------------------------------------------------
    def create_siamese_model(self, mode, hidden_dim=512, out_dim=128):
        """
        Create Siamese model where hidden_dim and out_dim can be tuned independently.
        """
        if mode == "pair":
            return SiameseEmbeddingModel(
                embedding_dim=self.embedding_dim,
                hidden_dim=int(hidden_dim),
                out_dim=int(out_dim),
            )
        raise ValueError(f"Unsupported mode: {mode}")

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------
    def create_dataloader(self, dataframe, batch_size, mode, shuffle):
        if mode != "pair":
            raise ValueError("Only pair mode supported with embeddings")

        dataset = EmbeddingPairDataset(dataframe)

        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    def create_optimizer(self, model, params):
        opt = str(params.get("optimizer", "adamw")).lower()
        lr = float(params["lr"])
        wd = float(params.get("weight_decay", 0.0))

        if opt == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if opt == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        raise ValueError(f"Unknown optimizer: {params.get('optimizer')}")

    # ------------------------------------------------------------
    # HYPERPARAMETER SAMPLING (non-Optuna path)
    # ------------------------------------------------------------
    def sample_hyperparameters(self, mode, n_samples):
        """
        Random-sample hyperparameters (non-Optuna path).

        Two-margin cosine hinge sampling (based on your histogram):
          lr: 1e-5 to 1e-3
          batch_size: [64, 128, 256, 512, 1024]
          internal_layer_size: [256, 512, 768, 1024]
          output_dim: [128, 256, 512]
          weight_decay: 1e-6 to 1e-3

          m_neg: [0.78, 0.88]
          gap:   [0.04, 0.14]  => m_pos = min(m_neg + gap, 0.99)
          w_pos: [0.5, 2.0]
          w_neg: [1.0, 12.0] (log-uniform-ish)
        """
        if mode != "pair":
            raise ValueError("Only pair mode supported in this embeddings optimizer")

        samples = []
        for _ in range(int(n_samples)):
            lr = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))))
            batch_size = int(np.random.choice([64, 128, 256, 512, 1024]))
            internal_layer_size = int(np.random.choice([256, 512, 768, 1024]))
            output_dim = int(np.random.choice([128, 256, 512]))
            optimizer_name = str(np.random.choice(["adam", "adamw", "sgd"]))
            weight_decay = float(np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3))))

            m_neg = float(np.random.uniform(0.78, 0.88))
            gap = float(np.random.uniform(0.04, 0.14))
            m_pos = float(min(m_neg + gap, 0.99))

            samples.append({
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "output_dim": output_dim,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "m_neg": m_neg,
                "gap": gap,       # keep for interpretability
                "m_pos": m_pos   # explicit; ensures m_pos > m_neg
            })

        return samples

    def sample_initial_hyperparameters(self, mode, n_samples):
        return self.sample_hyperparameters(mode, n_samples)

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------
    def evaluate_trial(
        self,
        params,
        training_filepath,
        test_filepath,
        mode,
        loss_type,
        trial_number,
        epochs=5,
        validate_filepath=None,
        save_best_model=True,
        want_test=False
    ):
        """
        Train a model for one trial.

        Returns a dict containing (at least):
          - best_train_loss, final_train_loss, final_val_loss
          - val_roc_auc (if validate_filepath provided)
          - optional test metrics (if want_test=True)
        """
        import json

        def convert_np(obj):
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            if hasattr(obj, "item") and callable(obj.item):
                return obj.item()
            return obj

        try:
            batch_size = int(params["batch_size"])
            hidden_dim = int(params["internal_layer_size"])
            out_dim = int(params.get("output_dim", 128))
            lr = float(params["lr"])

            # Accept either:
            #   (m_pos, m_neg) directly
            # or (m_neg, gap) and compute m_pos
            if "m_pos" in params and "m_neg" in params:
                m_pos = float(params["m_pos"])
                m_neg = float(params["m_neg"])
            elif "m_neg" in params and "gap" in params:
                m_neg = float(params["m_neg"])
                gap = float(params["gap"])
                m_pos = float(min(m_neg + gap, 0.99))
                params["m_pos"] = m_pos  # ensure it gets saved
            else:
                raise ValueError("Two-margin loss requires params['m_pos'] and params['m_neg'] (or params['m_neg'] + params['gap']).")

            if not (m_pos > m_neg):
                raise ValueError(f"Need m_pos > m_neg. Got m_pos={m_pos}, m_neg={m_neg}")

            print(
                f"Testing params: "
                f"LR={lr:.6f}, Batch={batch_size}, "
                f"Hidden={hidden_dim}, OutDim={out_dim}, "
                f"Opt={params.get('optimizer')}, WD={float(params.get('weight_decay', 0.0)):.2e}, "
                f"m_pos={m_pos:.4f}, m_neg={m_neg:.4f}"
            )

            run_tag = (
                f"_{self.model_type}"
                f"_Mode={mode}"
                f"_Loss={loss_type}"
                f"_LR={lr:.6f}"
                f"_WD={float(params.get('weight_decay', 0.0)):.2e}"
                f"_Batch={batch_size}"
                f"_Hidden={hidden_dim}"
                f"_OutDim={out_dim}"
                f"_Opt={params.get('optimizer')}"
                f"_mPos={m_pos:.3f}"
                f"_mNeg={m_neg:.3f}"
                f"_wNeg={w_neg:.2f}"
                f"_Ep={int(epochs)}"
            )

            # Ensure output dirs exist
            os.makedirs("images", exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)

            # --- TRAIN LOADER ---
            train_df = pd.read_parquet(training_filepath)
            train_loader = self.create_dataloader(train_df, batch_size, mode, shuffle=True)

            # --- VAL LOADER (loss-only) ---
            val_loader = None
            if validate_filepath is not None:
                val_df = pd.read_parquet(validate_filepath)
                val_loader = self.create_dataloader(val_df, batch_size, mode, shuffle=False)

            # --- MODEL / OPT / LOSS ---
            model = self.create_siamese_model(mode, hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)
            optimizer = self.create_optimizer(model, params)

            criterion = CosineContrastiveTwoMargin(
                m_pos=m_pos,
                m_neg=m_neg
            )

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                model_type=mode,
            )

            # --- TRAIN ---
            train_metrics = trainer.train(
                dataloader=train_loader,
                trial_number=trial_number,
                test_filepath=test_filepath,
                string=run_tag,
                mode=mode,
                epochs=int(epochs),
                validate_filepath=validate_filepath,
                validate_dataloader=val_loader,
                save_best=False,
                plot_losses=False,
                want_test=False,  # no test inside trial unless explicitly asked
            )

            result = {
                "timestamp": datetime.now(),
                "trial_number": int(trial_number),

                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": hidden_dim,
                "output_dim": out_dim,
                "optimizer": params.get("optimizer"),
                "weight_decay": float(params.get("weight_decay", 0.0)),
                "mode": mode,
                "loss_type": loss_type,

                # TWO-MARGIN params
                "m_pos": m_pos,
                "m_neg": m_neg,

                # losses
                "best_train_loss": train_metrics.get("best_train_loss"),
                "final_train_loss": train_metrics.get("final_train_loss"),
                "final_val_loss": train_metrics.get("final_val_loss"),
            }

            # --- Validation ROC-AUC (preferred "best model" metric) ---
            val_auc = None
            if validate_filepath is not None:
                evaluator = Evaluator(model=model, batch_size=batch_size, model_type=mode)
                _, val_metrics = evaluator.evaluate(validate_filepath)
                val_auc = float(val_metrics.get("roc_auc", 0.0))
                result["val_roc_auc"] = val_auc
                result["val_accuracy"] = float(val_metrics.get("accuracy", 0.0))
                result["val_youden_threshold"] = float(val_metrics.get("youden_threshold", 0.0))
                result["val_youden_j"] = float(val_metrics.get("youden_j", 0.0))

            # --- Optional test eval (only when requested) ---
            if want_test and test_filepath is not None:
                evaluator = Evaluator(model=model, batch_size=batch_size, model_type=mode)
                _, test_metrics = evaluator.evaluate(test_filepath)
                for k, v in test_metrics.items():
                    if k == "roc_curve":
                        continue
                    result[f"test_{k}"] = v

            # --- Save best model ---
            if save_best_model:
                model_id = f"{self.model_type}_{mode}"
                best_model_path = os.path.join(self.log_dir, "results", f"best_model_{model_id}.pt")
                best_hparams_path = os.path.join(self.log_dir, "results", f"best_hparams_{model_id}.json")

                # Prefer saving by validation ROC-AUC when available
                if validate_filepath is not None and val_auc is not None:
                    if val_auc > self.best_val_auc:
                        self.best_val_auc = val_auc
                        torch.save(model.state_dict(), best_model_path)
                        with open(best_hparams_path, "w", encoding="utf-8") as f:
                            json.dump(convert_np(params), f, indent=2)
                        print(f"[DEBUG] Saved best checkpoint by val_roc_auc={val_auc:.6f} -> {best_model_path}")
                else:
                    # Fallback: save by val loss if available
                    val_loss = result.get("final_val_loss", None)
                    if val_loss is not None and float(val_loss) < self.best_val_loss:
                        self.best_val_loss = float(val_loss)
                        torch.save(model.state_dict(), best_model_path)
                        with open(best_hparams_path, "w", encoding="utf-8") as f:
                            json.dump(convert_np(params), f, indent=2)
                        print(f"[DEBUG] Saved best checkpoint by val_loss={val_loss:.6f} -> {best_model_path}")

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now(),
                "trial_number": int(trial_number) if trial_number is not None else None,
                "error": str(e),
                "val_roc_auc": 0.0,
                "val_accuracy": 0.0,
            }


"""
USAGE EXAMPLE

from scripts.optimization.base_optimizer import BaseOptimizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = BaseOptimizer(model_type="pairwise_contrastive", device=device, log_dir="optimization_results")

params = {
    "lr": 1e-4,
    "batch_size": 128,
    "internal_layer_size": 512,
    "output_dim": 128,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "m_pos": 0.92,
    "m_neg": 0.84,
    "w_pos": 1.0,
    "w_neg": 3.0,
}

result = opt.evaluate_trial(
    params=params,
    training_filepath="/path/to/train.parquet",
    validate_filepath="/path/to/val.parquet",
    test_filepath="/path/to/test.parquet",
    mode="pair",
    loss_type="cosine",
    trial_number=1,
    epochs=5,
    want_test=False,
)
print(result)
"""
