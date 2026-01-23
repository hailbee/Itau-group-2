import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime

from scripts.training.trainer import Trainer
from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import EmbeddingPairDataset
from scripts.evaluation.evaluator import Evaluator

# ============================================================
# BaseOptimizer (PRECOMPUTED EMBEDDINGS VERSION)
# ============================================================

class BaseOptimizer:
    """
    Hyperparameter optimization using PRECOMPUTED EMBEDDINGS.
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

        # embedding info
        self.embedding_dim = embedding_dim
        self.fake_slice = slice(fake_start, fake_end)
        self.real_slice = slice(real_start, real_end)

        self.results = []
        self.best_val_loss = float("inf")

        if "/content" in self.log_dir:
            self.log_dir = "optimization_results"
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"[INFO] Using precomputed embeddings (dim={self.embedding_dim})")

    # ------------------------------------------------------------
    # MODEL CREATION
    # ------------------------------------------------------------

    def create_siamese_model(self, mode, hidden_dim=512, out_dim=128):
        """
        Create Siamese model where hidden_dim and out_dim can be tuned independently.
        """
        if mode == "pair":
            return SiameseEmbeddingModel(
                embedding_dim=self.embedding_dim,  # 768
                hidden_dim=hidden_dim,             # internal layer size
                out_dim=out_dim                    # output projection dim
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
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    def create_optimizer(self, model, params):
        if params["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )
        elif params["optimizer"] == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )
        else:
            return torch.optim.SGD(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )

    # ------------------------------------------------------------
    # HYPERPARAMETER SAMPLING
    # ------------------------------------------------------------

    def sample_hyperparameters(self, mode, n_samples):
        """
        Random-sample hyperparameters (non-Optuna path).
        Target ranges:
          lr: 1e-5 to 1e-3
          batch_size: [64, 128, 256, 512, 1024]
          internal_layer_size (hidden_dim): [256, 512, 768, 1024]
          output_dim (out_dim): [128, 256, 512]
          margin: 0.05 to 0.7
          weight_decay: 1e-5 to 1e-3
        """
        samples = []
        for _ in range(n_samples):
            samples.append({
                "lr": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))),
                "batch_size": int(np.random.choice([64, 128, 256, 512, 1024])),
                "internal_layer_size": int(np.random.choice([256, 512, 768, 1024])),
                "output_dim": int(np.random.choice([128, 256, 512])),
                "optimizer": np.random.choice(["adam", "adamw", "sgd"]),
                "weight_decay": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))),
                "margin": float(np.random.uniform(0.05, 0.7)),
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
        import json

        def convert_np(obj):
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(v) for v in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        try:
            batch_size = int(params["batch_size"])
            hidden_dim = int(params["internal_layer_size"])
            out_dim = int(params.get("output_dim", 128))
            lr = float(params["lr"])

            print(
                f"Testing params: "
                f"LR={lr:.6f}, Batch={batch_size}, "
                f"Hidden={hidden_dim}, OutDim={out_dim}, "
                f"Opt={params['optimizer']}"
            )

            string = (
                f"_{self.model_type}"
                f"_Mode={mode}"
                f"_Loss={loss_type}"
                f"_LR={lr:.6f}"
                f"_WD={float(params['weight_decay']):.2e}"
                f"_Batch={batch_size}"
                f"_Hidden={hidden_dim}"
                f"_OutDim={out_dim}"
                f"_Opt={params['optimizer']}"
                f"_Margin={float(params['margin']):.3f}"
                f"_Ep={epochs}"
            )

            # Ensure output dir exists for plots
            os.makedirs("images", exist_ok=True)

            # --- MAIN TRAIN LOADER (HARD) ---
            train_df = pd.read_parquet(training_filepath)
            dataloader = self.create_dataloader(train_df, batch_size, mode, shuffle=True)

            # --- VALIDATION LOADER ---
            val_dataloader = None
            if validate_filepath is not None:
                val_df = pd.read_parquet(validate_filepath)
                val_dataloader = self.create_dataloader(val_df, batch_size, mode, shuffle=False)

            # --- MODEL / OPT / LOSS ---
            model = self.create_siamese_model(mode, hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)
            optimizer = self.create_optimizer(model, params)

            from model_utils.loss.pair_losses import ContrastiveLoss
            criterion = ContrastiveLoss(margin=float(params["margin"]))

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                model_type=mode,
            )

            # --- TRAIN (PASSES LOADERS) ---
            best_metrics = trainer.train(
                dataloader=dataloader,
                trial_number=trial_number,
                test_filepath=test_filepath,
                string=string,
                mode=mode,
                epochs=epochs,
                validate_filepath=validate_filepath,
                validate_dataloader=val_dataloader,
                save_best=False,
                plot_losses=False
            )

            if want_test and test_filepath is not None:
                evaluator = Evaluator(model=model, batch_size=batch_size, model_type=mode)
                _, eval_metrics = evaluator.evaluate(test_filepath)
                best_metrics.update(eval_metrics)

            result = {
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": hidden_dim,
                "output_dim": out_dim,
                "optimizer": params["optimizer"],
                "weight_decay": params["weight_decay"],
                "margin": float(params["margin"]),
                "mode": mode,
                "loss_type": loss_type,

                # losses (always present)
                "best_train_loss": best_metrics.get("best_train_loss"),
                "final_train_loss": best_metrics.get("final_train_loss"),
                "final_val_loss": best_metrics.get("final_val_loss"),
            }

            # add test metrics ONLY when requested
            if want_test:
                result.update({
                    "test_roc_auc": best_metrics.get("roc_auc"),
                    "test_accuracy": best_metrics.get("accuracy"),
                    "youden_j": best_metrics.get("youden_j"),
                    "youden_threshold": best_metrics.get("youden_threshold"),
                })

            val_loss = result.get("final_val_loss", None)

            if save_best_model and val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                model_id = f"{self.model_type}_{mode}"

                torch.save(
                    model.state_dict(),
                    os.path.join(self.log_dir, f"results/best_model_{model_id}.pt"),
                )
                with open(
                    os.path.join(self.log_dir, f"results/best_hparams_{model_id}.json"),
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(convert_np(params), f)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "test_roc_auc": 0.0,
                "test_accuracy": 0.0,
            }
