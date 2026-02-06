# scripts/optimization/optuna.py

import os
import json
from datetime import datetime

import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner

from scripts.evaluation.evaluator import Evaluator
from scripts.training.trainer import Trainer
from .base_optimizer import BaseOptimizer

# Suppress Optuna's internal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# -----------------------------
# Loss import (with safe fallback)
# -----------------------------
try:
    # If you updated model_utils/loss/pair_losses.py to your 2-margin cosine ContrastiveLoss,
    # this import will use it directly.
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

class OptunaOptimizer(BaseOptimizer):
    """
    Optuna-based hyperparameter optimization.

    FIXED for TWO-MARGIN cosine contrastive:
      - Uses CosineContrastiveTwoMargin(m_pos, m_neg, w_pos, w_neg)
      - Tunes: m_neg + gap (so m_pos = m_neg + gap, guaranteeing m_pos > m_neg)
      - Objective: VALIDATION ROC-AUC (no test leakage during trials)
      - Saves global best model by VAL ROC-AUC
    """

    def __init__(self, model_type, model_name=None, device=None, log_dir=None):
        log_dir = "./" if log_dir is None else log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)

        super().__init__(model_type, model_name, device, log_dir)

        # Track best model by validation ROC-AUC (maximize)
        self.best_val_auc = float("-inf")

    # ------------------------------------------------------------
    # OVERRIDE: evaluate_trial (TWO-MARGIN loss + val ROC-AUC objective)
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
        want_test=False,
    ):
        """
        Train one trial and return a result dict.

        Key behavior:
          - Train with two-margin cosine contrastive
          - Compute VALIDATION ROC-AUC using Evaluator on validate_filepath (if provided)
          - Save global best model by validation ROC-AUC
        """

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

            m_pos = float(params["m_pos"])
            m_neg = float(params["m_neg"])

            if not (m_pos > m_neg):
                raise ValueError(f"Invalid margins: need m_pos > m_neg, got m_pos={m_pos}, m_neg={m_neg}")

            print(
                f"Testing params: "
                f"LR={lr:.6f}, Batch={batch_size}, Hidden={hidden_dim}, OutDim={out_dim}, "
                f"Opt={params['optimizer']}, WD={float(params['weight_decay']):.2e}, "
                f"m_pos={m_pos:.4f}, m_neg={m_neg:.4f}"
            )

            run_tag = (
                f"_{self.model_type}"
                f"_Mode={mode}"
                f"_Loss={loss_type}"
                f"_LR={lr:.6f}"
                f"_WD={float(params['weight_decay']):.2e}"
                f"_Batch={batch_size}"
                f"_Hidden={hidden_dim}"
                f"_OutDim={out_dim}"
                f"_Opt={params['optimizer']}"
                f"_mPos={m_pos:.3f}"
                f"_mNeg={m_neg:.3f}"
                f"_Ep={int(epochs)}"
            )

            # --- TRAIN LOADER ---
            train_df = pd.read_parquet(training_filepath)
            train_loader = self.create_dataloader(train_df, batch_size, mode, shuffle=True)

            # --- VAL LOADER (loss only; we compute ROC-AUC separately below) ---
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

            # --- TRAIN (FAST: loss only; no test eval in trials) ---
            train_metrics = trainer.train(
                dataloader=train_loader,
                trial_number=trial_number,
                test_filepath=test_filepath,   # passed through, but want_test=False prevents leakage
                string=run_tag,
                mode=mode,
                epochs=int(epochs),
                validate_filepath=validate_filepath,
                validate_dataloader=val_loader,
                save_best=False,
                plot_losses=False,
                want_test=False,
            )

            result = {
                "timestamp": datetime.now(),
                "trial_number": int(trial_number),

                # params
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": hidden_dim,
                "output_dim": out_dim,
                "optimizer": params["optimizer"],
                "weight_decay": float(params["weight_decay"]),
                "mode": mode,
                "loss_type": loss_type,
                "m_pos": m_pos,
                "m_neg": m_neg,

                # losses
                "best_train_loss": train_metrics.get("best_train_loss"),
                "final_train_loss": train_metrics.get("final_train_loss"),
                "final_val_loss": train_metrics.get("final_val_loss"),
            }

            # --- Compute VALIDATION ROC-AUC for objective (no test leakage) ---
            val_auc = None
            if validate_filepath is not None:
                evaluator = Evaluator(model=model, batch_size=batch_size, model_type=mode)
                _, val_metrics = evaluator.evaluate(validate_filepath)
                val_auc = float(val_metrics.get("roc_auc", 0.0))
                result["val_roc_auc"] = val_auc
                result["val_accuracy"] = float(val_metrics.get("accuracy", 0.0))
                result["val_youden_threshold"] = float(val_metrics.get("youden_threshold", 0.0))
                result["val_youden_j"] = float(val_metrics.get("youden_j", 0.0))

            # --- Optional TEST eval (only if explicitly requested) ---
            if want_test and test_filepath is not None:
                evaluator = Evaluator(model=model, batch_size=batch_size, model_type=mode)
                _, test_metrics = evaluator.evaluate(test_filepath)
                for k, v in test_metrics.items():
                    if k == "roc_curve":
                        continue
                    result[f"test_{k}"] = v

            # --- Save global best model by VAL ROC-AUC ---
            if save_best_model and validate_filepath is not None and val_auc is not None:
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    model_id = f"{self.model_type}_{mode}"

                    best_model_path = os.path.join(self.log_dir, "results", f"best_model_{model_id}.pt")
                    best_hparams_path = os.path.join(self.log_dir, "results", f"best_hparams_{model_id}.json")

                    torch.save(model.state_dict(), best_model_path)
                    with open(best_hparams_path, "w", encoding="utf-8") as f:
                        json.dump(convert_np(params), f, indent=2)

                    print(
                        f"[DEBUG] Saved NEW best checkpoint by val_roc_auc={val_auc:.6f} -> {best_model_path}"
                    )

            # If no validation set: fall back to best_val_loss (rare in your setup)
            if save_best_model and validate_filepath is None:
                val_loss = result.get("final_val_loss", None)
                if val_loss is not None and float(val_loss) < getattr(self, "best_val_loss", float("inf")):
                    self.best_val_loss = float(val_loss)
                    model_id = f"{self.model_type}_{mode}"
                    best_model_path = os.path.join(self.log_dir, "results", f"best_model_{model_id}.pt")
                    best_hparams_path = os.path.join(self.log_dir, "results", f"best_hparams_{model_id}.json")
                    torch.save(model.state_dict(), best_model_path)
                    with open(best_hparams_path, "w", encoding="utf-8") as f:
                        json.dump(convert_np(params), f, indent=2)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now(),
                "trial_number": int(trial_number),
                "error": str(e),
                "val_roc_auc": 0.0,
            }

    # ------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------
    def objective(self, trial, training_filepath, test_filepath, mode, loss_type, epochs=5, validate_filepath=None):
        """
        Objective returns VALIDATION ROC-AUC (preferred) to avoid test leakage.
        """

        # Core hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        internal_layer_size = trial.suggest_categorical("internal_layer_size", [256, 512, 768, 1024])
        output_dim = trial.suggest_categorical("output_dim", [128, 256, 768])

        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        params = {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "internal_layer_size": int(internal_layer_size),
            "output_dim": int(output_dim),
            "optimizer": str(optimizer_name),
            "weight_decay": float(weight_decay),
        }

        # Keep temperature for other modes if you still run them
        if mode in ["supcon", "infonce"]:
            temperature = trial.suggest_float("temperature", 0.01, 0.1, log=True)
            params["temperature"] = float(temperature)
        else:
            # TWO-MARGIN cosine hinge
            #
            # We parametrize as: m_neg + gap => m_pos = m_neg + gap
            # This guarantees m_pos > m_neg and avoids "bad deadzone" configs.
            #
            # Given your histogram overlap, reasonable bands:
            #   m_neg ~ 0.78–0.88, gap ~ 0.04–0.14  => m_pos ~ 0.82–1.02 (we clamp)
            m_neg = trial.suggest_float("m_neg", 0.78, 0.88)
            gap = trial.suggest_float("gap", 0.04, 0.14)
            m_pos = min(float(m_neg + gap), 0.99)

            params["m_neg"] = float(m_neg)
            params["m_pos"] = float(m_pos)
            params["gap"] = float(gap)          # logged for interpretability

        try:
            print(f"\n{'='*50}")
            print(f"Starting Trial {trial.number + 1}")
            print(f"{'='*50}")

            result = self.evaluate_trial(
                params=params,
                trial_number=trial.number + 1,
                training_filepath=training_filepath,
                test_filepath=test_filepath,
                mode=mode,
                loss_type=loss_type,
                epochs=int(epochs),
                validate_filepath=validate_filepath,
                want_test=False,       # avoid test leakage during optimization
                save_best_model=True,
            )

            # Store trial row (for CSV)
            row = {
                "trial_number": trial.number + 1,
                "best_train_loss": result.get("best_train_loss"),
                "final_train_loss": result.get("final_train_loss"),
                "final_val_loss": result.get("final_val_loss"),
                "val_roc_auc": result.get("val_roc_auc"),
            }
            row.update(trial.params)
            self.results.append(row)

            print(f"\nTrial {trial.number + 1} completed.")

            # Objective: maximize validation ROC-AUC if available
            if validate_filepath is not None:
                val_auc = float(result.get("val_roc_auc", 0.0))
                return val_auc

            # Fallback (no validation): maximize negative loss (smaller loss is better)
            val_loss = result.get("final_val_loss", None)
            if val_loss is not None:
                return -float(val_loss)

            train_loss = result.get("final_train_loss", 1e9)
            return -float(train_loss)

        except Exception as e:
            print(f"\nTrial {trial.number + 1} failed with error: {e}")
            return 0.0

    # ------------------------------------------------------------
    # Run Optuna optimization
    # ------------------------------------------------------------
    def optimize(
        self,
        training_filepath,
        test_filepath,
        mode="pair",
        loss_type="cosine",
        epochs=5,
        n_trials=50,
        sampler="tpe",
        pruner="median",
        study_name=None,
        validate_filepath=None,
    ):
        """
        Run Optuna optimization.

        During trials:
          - objective uses VALIDATION ROC-AUC (if validate_filepath provided)

        After study:
          - evaluate the globally-best checkpoint on TEST exactly once
        """
        print(f"Starting Optuna optimization for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        print(f"Sampler: {sampler}, Pruner: {pruner}")
        print(f"Will run {n_trials} trials")

        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)

        if study_name is None:
            study_name = f"{self.model_type}_{mode}_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Sampler
        if sampler == "tpe":
            sampler_obj = TPESampler(seed=42)
        elif sampler == "random":
            sampler_obj = RandomSampler(seed=42)
        elif sampler == "cmaes":
            sampler_obj = CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # Pruner
        if pruner is None or pruner == "none":
            pruner_obj = NopPruner()
        elif pruner == "median":
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            pruner_obj = HyperbandPruner(min_resource=1, max_resource=int(epochs))
        else:
            raise ValueError(f"Unknown pruner: {pruner}")

        # Study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler_obj,
            pruner=pruner_obj,
            study_name=study_name,
        )

        def objective_wrapper(trial):
            return self.objective(
                trial=trial,
                training_filepath=training_filepath,
                test_filepath=test_filepath,
                mode=mode,
                loss_type=loss_type,
                epochs=int(epochs),
                validate_filepath=validate_filepath,
            )

        study.optimize(objective_wrapper, n_trials=int(n_trials))

        # Save trial table
        self._save_results(study)

        # Final: evaluate global best model on test set ONCE
        print("\n" + "=" * 60)
        print("[DEBUG] FINAL COMPARISON: Evaluating best model on test set after all Optuna trials...")
        print("=" * 60 + "\n")

        model_id = f"{self.model_type}_{mode}"
        best_model_path = os.path.join(self.log_dir, "results", f"best_model_{model_id}.pt")
        best_hparams_path = os.path.join(self.log_dir, "results", f"best_hparams_{model_id}.json")

        if os.path.exists(best_model_path) and os.path.exists(best_hparams_path):
            with open(best_hparams_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)

            rounded_best_params = {
                k: (round(v, 4) if isinstance(v, (float, int)) else v)
                for k, v in best_params.items()
            }
            print(f"[DEBUG] Best hyperparameters (global): {rounded_best_params}")

            hidden_dim = int(best_params.get("internal_layer_size", 512))
            out_dim = int(best_params.get("output_dim", 128))
            bs = int(best_params.get("batch_size", 32))

            model = self.create_siamese_model(mode, hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            model.eval()

            evaluator = Evaluator(model, batch_size=bs, model_type=mode)
            _, test_metrics = evaluator.evaluate(test_filepath)

            metrics_to_print = {
                k: (round(v, 4) if isinstance(v, (float, int)) else v)
                for k, v in test_metrics.items()
                if k != "roc_curve"
            }

            print("\n--- FINAL TEST SET METRICS ---")
            if "youden_j" in metrics_to_print:
                print(f"Youden's J statistic:         {metrics_to_print['youden_j']:.4f}")
            if "youden_threshold" in metrics_to_print:
                print(f"Youden Threshold:             {metrics_to_print['youden_threshold']:.4f}")
            if "accuracy" in metrics_to_print:
                print(f"Accuracy:                     {metrics_to_print['accuracy']:.4f}")
            if "roc_auc" in metrics_to_print:
                print(f"ROC AUC:                      {metrics_to_print['roc_auc']:.4f}")
            if "f1" in metrics_to_print:
                print(f"F1 Score:                     {metrics_to_print['f1']:.4f}")
            if "precision" in metrics_to_print:
                print(f"Precision:                    {metrics_to_print['precision']:.4f}")
            if "recall" in metrics_to_print:
                print(f"Recall:                       {metrics_to_print['recall']:.4f}")
            print("-----------------------------\n")

            for k, v in metrics_to_print.items():
                if k not in {"youden_j", "youden_threshold", "accuracy", "roc_auc", "f1", "precision", "recall"}:
                    print(f"{k}: {v}")

            return test_metrics

        print("[DEBUG] No best model found for final test set evaluation.")
        return self.results

    def _save_results(self, study):
        """Save optimization results to CSV."""
        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)

        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, "results", f"optuna_results_{timestamp}.csv")
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")


"""
USAGE EXAMPLE

from scripts.optimization.optuna import OptunaOptimizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = OptunaOptimizer(model_type="pairwise_contrastive", device=device, log_dir="./optimization_results/optuna")
opt.optimize(
    training_filepath="/path/to/train.parquet",
    validate_filepath="/path/to/val.parquet",   # enables val ROC-AUC objective
    test_filepath="/path/to/test.parquet",
    mode="pair",
    loss_type="cosine",
    epochs=5,
    n_trials=50,
    sampler="tpe",
    pruner="median",   # or "hyperband" or "none"
)
"""
