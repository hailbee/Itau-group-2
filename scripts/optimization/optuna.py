import torch
import pandas as pd
import os
from datetime import datetime
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

from .base_optimizer import BaseOptimizer

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer(BaseOptimizer):
    """
    Optuna optimizer. Supports:
      - mode="pair"
      - mode="text2img" (output_dim fixed to 768)
    """
    def __init__(self, model_type, model_name=None, device=None, log_dir=None):
        log_dir = log_dir or "./"
        os.makedirs(log_dir, exist_ok=True)
        super().__init__(model_type, model_name, device, log_dir)

    def objective(self, trial, training_filepath, test_filepath, mode, loss_type, epochs=5, validate_filepath=None):
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        internal_layer_size = trial.suggest_categorical("internal_layer_size", [128, 256, 512, 768, 1024])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        params = {
            "lr": lr,
            "batch_size": batch_size,
            "internal_layer_size": internal_layer_size,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
        }

        if mode == "pair":
            params["output_dim"] = trial.suggest_categorical("output_dim", [128, 256, 512])
            params["margin"] = trial.suggest_float("margin", 0.05, 0.7)
        elif mode == "text2img":
            params["output_dim"] = self.embedding_dim  # fixed 768
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        result = self.evaluate_trial(
            params=params,
            training_filepath=training_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            trial_number=trial.number + 1,
            epochs=epochs,
            validate_filepath=validate_filepath,
            want_test=False,
        )

        val = result.get("best_val_loss") or result.get("final_val_loss") or result.get("final_train_loss")
        if val is None:
            return float("-inf")
        return -float(val)  # maximize negative loss

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
        if study_name is None:
            study_name = f"{self.model_type}_{mode}_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if sampler == "tpe":
            sampler_obj = TPESampler(seed=42)
        elif sampler == "random":
            sampler_obj = RandomSampler(seed=42)
        elif sampler == "cmaes":
            sampler_obj = CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        if pruner is None or pruner == "none":
            pruner_obj = None
        elif pruner == "median":
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            pruner_obj = HyperbandPruner(min_resource=1, max_resource=epochs)
        else:
            raise ValueError(f"Unknown pruner: {pruner}")

        study = optuna.create_study(direction="maximize", sampler=sampler_obj, pruner=pruner_obj, study_name=study_name)

        def wrapper(trial):
            return self.objective(trial, training_filepath, test_filepath, mode, loss_type, epochs, validate_filepath)

        study.optimize(wrapper, n_trials=int(n_trials))
        self._save_results(study, mode, loss_type)
        return study.best_params

    def _save_results(self, study, mode, loss_type):
        trials = []
        for t in study.trials:
            row = dict(t.params)
            row["value"] = t.value
            row["state"] = str(t.state)
            trials.append(row)

        df = pd.DataFrame(trials)
        out_dir = os.path.join(self.log_dir, "results")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(
            out_dir,
            f"optuna_results_{self.model_type}_{mode}_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(filename, index=False)
        print(f"[INFO] Results saved to {filename}")
