# scripts/optimization/unified_optimizer.py

import os
import copy

import torch
import numpy as np

from scripts.optimization.optuna import OptunaOptimizer


class UnifiedHyperparameterOptimizer:
    """
    Unified interface for different hyperparameter optimization methods.

    FIXED for TWO-MARGIN cosine contrastive:
      - For pairwise modes: uses m_neg + gap -> m_pos, plus w_pos/w_neg
      - Keeps supcon/infonce temperature paths intact
      - Delegates Optuna tuning to OptunaOptimizer (which should now tune m_neg+gap / m_pos/m_neg, w_pos, w_neg)
    """

    def __init__(self, model_type, model_name=None, device=None, log_dir="optimization_results"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.log_dir = log_dir
        self.results = []

        # Informational tracking (not required for correctness)
        self.best_auc = 0.0
        self.best_accuracy = 0.0

        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize Optuna optimizer
        self.optuna_optimizer = OptunaOptimizer(
            model_type=model_type,
            model_name=model_name,
            device=self.device,
            log_dir=os.path.join(self.log_dir, "optuna"),
        )

    # ------------------------------------------------------------
    # Optional evolutionary helpers (not used for optuna path)
    # ------------------------------------------------------------
    def sample_initial_hyperparameters(self, mode, population_size):
        """
        Sample initial hyperparameters for a population.

        For pairwise TWO-MARGIN cosine hinge:
          - m_neg in [0.78, 0.88]
          - gap in [0.04, 0.14] => m_pos = min(m_neg + gap, 0.99)
          - w_neg > w_pos often helps
        """
        np.random.seed(42)

        population = []
        for _ in range(int(population_size)):
            lr = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))))
            batch_size = int(np.random.choice([16, 32, 64, 128]))
            internal_layer_size = int(np.random.choice([64, 128, 256, 512]))
            output_dim = int(np.random.choice([128, 256, 768]))
            optimizer_name = str(np.random.choice(["adam", "adamw", "sgd"]))
            weight_decay = float(np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3))))

            if mode in ["supcon", "infonce"]:
                temperature = float(np.exp(np.random.uniform(np.log(0.01), np.log(0.1))))
                params = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "internal_layer_size": internal_layer_size,
                    "output_dim": output_dim,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                    "temperature": temperature,
                }
            else:
                m_neg = float(np.random.uniform(0.78, 0.88))
                gap = float(np.random.uniform(0.04, 0.14))
                m_pos = float(min(m_neg + gap, 0.99))

                w_pos = float(np.random.uniform(0.5, 2.0))
                w_neg = float(np.exp(np.random.uniform(np.log(1.0), np.log(12.0))))

                params = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "internal_layer_size": internal_layer_size,
                    "output_dim": output_dim,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                    "m_neg": m_neg,
                    "gap": gap,      # keep for interpretability
                    "m_pos": m_pos,  # explicit for downstream code paths
                    "w_pos": w_pos,
                    "w_neg": w_neg,
                }

            population.append(params)

        return population

    def mutate_hyperparameters(self, params, mode, mutation_rate=0.2):
        """
        Mutate hyperparameters for evolution.

        For pairwise TWO-MARGIN cosine hinge, mutate:
          - m_neg and gap (then recompute m_pos)
          - w_pos, w_neg
        """
        new_params = copy.deepcopy(params)

        # Mutate learning rate
        if np.random.random() < mutation_rate:
            new_params["lr"] *= float(np.exp(np.random.normal(0, 0.5)))
            new_params["lr"] = float(np.clip(new_params["lr"], 1e-5, 1e-3))

        # Mutate batch size
        if np.random.random() < mutation_rate:
            new_params["batch_size"] = int(np.random.choice([16, 32, 64, 128]))

        # Mutate internal layer size
        if np.random.random() < mutation_rate:
            new_params["internal_layer_size"] = int(np.random.choice([64, 128, 256, 512]))

        # Mutate output dim
        if np.random.random() < mutation_rate:
            new_params["output_dim"] = int(np.random.choice([128, 256, 768]))

        # Mutate optimizer
        if np.random.random() < mutation_rate:
            new_params["optimizer"] = str(np.random.choice(["adam", "adamw", "sgd"]))

        # Mutate weight decay
        if np.random.random() < mutation_rate:
            new_params["weight_decay"] *= float(np.exp(np.random.normal(0, 0.5)))
            new_params["weight_decay"] = float(np.clip(new_params["weight_decay"], 1e-6, 1e-3))

        # Mutate mode-specific parameters
        if mode in ["supcon", "infonce"]:
            if "temperature" not in new_params:
                new_params["temperature"] = 0.07
            if np.random.random() < mutation_rate:
                new_params["temperature"] *= float(np.exp(np.random.normal(0, 0.5)))
                new_params["temperature"] = float(np.clip(new_params["temperature"], 0.01, 0.1))
        else:
            # Ensure required keys exist
            if "m_neg" not in new_params:
                new_params["m_neg"] = 0.84
            if "gap" not in new_params:
                new_params["gap"] = 0.08
            if "w_pos" not in new_params:
                new_params["w_pos"] = 1.0
            if "w_neg" not in new_params:
                new_params["w_neg"] = 3.0

            if np.random.random() < mutation_rate:
                new_params["m_neg"] *= float(np.exp(np.random.normal(0, 0.02)))
                new_params["m_neg"] = float(np.clip(new_params["m_neg"], 0.70, 0.92))

            if np.random.random() < mutation_rate:
                new_params["gap"] *= float(np.exp(np.random.normal(0, 0.25)))
                new_params["gap"] = float(np.clip(new_params["gap"], 0.02, 0.20))

            # Recompute m_pos every time to maintain ordering
            new_params["m_pos"] = float(min(new_params["m_neg"] + new_params["gap"], 0.99))
            # (Optional safety: guarantee strict inequality)
            if not (new_params["m_pos"] > new_params["m_neg"]):
                new_params["m_pos"] = float(min(new_params["m_neg"] + 1e-3, 0.99))

            if np.random.random() < mutation_rate:
                new_params["w_pos"] *= float(np.exp(np.random.normal(0, 0.25)))
                new_params["w_pos"] = float(np.clip(new_params["w_pos"], 0.25, 4.0))

            if np.random.random() < mutation_rate:
                new_params["w_neg"] *= float(np.exp(np.random.normal(0, 0.35)))
                new_params["w_neg"] = float(np.clip(new_params["w_neg"], 0.5, 20.0))

        return new_params

    # ------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------
    def optimize(
        self,
        method,
        training_filepath,
        test_filepath,
        mode="pair",
        loss_type="cosine",
        validate_filepath=None,
        **kwargs
    ):
        """
        Run hyperparameter optimization using the specified method.

        For Optuna, we forward args. OptunaOptimizer is responsible for
        tuning TWO-MARGIN params (m_neg+gap -> m_pos), plus w_pos/w_neg.
        """
        if method == "optuna":
            allowed = ["n_trials", "sampler", "pruner", "study_name", "epochs"]
            filtered = {k: kwargs[k] for k in allowed if k in kwargs}
            return self._run_optuna_optimization(
                training_filepath=training_filepath,
                test_filepath=test_filepath,
                mode=mode,
                loss_type=loss_type,
                validate_filepath=validate_filepath,
                **filtered,
            )

        raise ValueError(f"Unknown optimization method: {method}")

    def _run_optuna_optimization(
        self,
        training_filepath,
        test_filepath,
        mode,
        loss_type,
        validate_filepath=None,
        **kwargs
    ):
        """Run Optuna optimization."""
        return self.optuna_optimizer.optimize(
            training_filepath=training_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            validate_filepath=validate_filepath,
            **kwargs,
        )

    # ------------------------------------------------------------
    # Convenience recommendations
    # ------------------------------------------------------------
    def get_recommended_settings(self, mode, loss_type, dataset_size=None):
        """
        Get recommended hyperparameter settings based on the model type and mode.

        For pairwise TWO-MARGIN cosine hinge:
          - use an "overlap band" window: e.g., m_neg=0.84, m_pos=0.92 (gap=0.08)
          - w_neg > w_pos helps if hard negatives are a problem
        """
        recommendations = {
            "pairwise_contrastive": {
                "pair": {
                    "lr": 1e-4,
                    "batch_size": 32,
                    "internal_layer_size": 256,
                    "output_dim": 128,
                    "m_neg": 0.84,
                    "gap": 0.08,
                    "m_pos": 0.92,
                    "w_pos": 1.0,
                    "w_neg": 3.0,
                },
                "triplet": {
                    "lr": 1e-4,
                    "batch_size": 64,
                    "internal_layer_size": 256,
                    "output_dim": 128,
                },
                "supcon": {
                    "lr": 1e-4,
                    "batch_size": 32,
                    "internal_layer_size": 256,
                    "output_dim": 128,
                    "temperature": 0.07,
                },
                "infonce": {
                    "lr": 1e-4,
                    "batch_size": 32,
                    "internal_layer_size": 256,
                    "output_dim": 128,
                    "temperature": 0.07,
                },
            }
        }

        if self.model_type in recommendations and mode in recommendations[self.model_type]:
            base_recs = recommendations[self.model_type][mode].copy()

            if dataset_size:
                if dataset_size < 1000:
                    base_recs["batch_size"] = min(int(base_recs.get("batch_size", 32)), 16)
                    base_recs["lr"] = float(base_recs.get("lr", 1e-4)) * 2.0
                elif dataset_size > 10000:
                    base_recs["batch_size"] = min(int(base_recs.get("batch_size", 32)) * 2, 128)
                    base_recs["lr"] = float(base_recs.get("lr", 1e-4)) * 0.5

            base_recs.update({
                "optimizer": "adamw",
                "weight_decay": 1e-4,
            })

            # Ensure m_pos consistent with m_neg + gap if present
            if "m_neg" in base_recs and "gap" in base_recs:
                base_recs["m_pos"] = float(min(float(base_recs["m_neg"]) + float(base_recs["gap"]), 0.99))

            return base_recs

        # Fallback
        return {
            "lr": 1e-4,
            "batch_size": 32,
            "internal_layer_size": 256,
            "output_dim": 128,
            "m_neg": 0.84,
            "gap": 0.08,
            "m_pos": 0.92,
            "w_pos": 1.0,
            "w_neg": 3.0,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
        }


"""
USAGE EXAMPLE

from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = UnifiedHyperparameterOptimizer(
    model_type="pairwise_contrastive",
    device=device,
    log_dir="optimization_results",
)

# Optuna search (recommended)
opt.optimize(
    method="optuna",
    training_filepath="/path/to/train.parquet",
    validate_filepath="/path/to/val.parquet",
    test_filepath="/path/to/test.parquet",
    mode="pair",
    loss_type="cosine",
    epochs=5,
    n_trials=50,
    sampler="tpe",
    pruner="median",
)
"""
