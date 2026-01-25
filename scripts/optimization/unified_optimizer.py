import torch
from scripts.optimization.optuna import OptunaOptimizer


class UnifiedHyperparameterOptimizer:
    """
    Unified interface for optimizers.

    For CASE 2 text2img: NO teacher args exist in this pipeline.
    """

    def __init__(self, model_type, model_name=None, device=None, log_dir="optimization_results"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.log_dir = log_dir

        self.optuna_optimizer = OptunaOptimizer(model_type, model_name, device, f"{log_dir}/optuna")

    def optimize(self, method, training_filepath, test_filepath, mode="pair", loss_type="cosine", validate_filepath=None, **kwargs):
        if method != "optuna":
            raise ValueError(f"Unknown optimization method: {method}")

        allowed = ["n_trials", "sampler", "pruner", "study_name", "epochs"]
        filtered = {k: kwargs[k] for k in allowed if k in kwargs}

        return self.optuna_optimizer.optimize(
            training_filepath=training_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            validate_filepath=validate_filepath,
            **filtered,
        )
