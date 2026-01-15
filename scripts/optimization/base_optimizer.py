import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime

from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.learning.siamese import SiameseEmbeddingModel

class EmbeddingPairDataset(torch.utils.data.Dataset):
    def __init__(self, df, fake_slice, real_slice):
        self.fake = df.iloc[:, fake_slice].values.astype("float32")
        self.real = df.iloc[:, real_slice].values.astype("float32")

    def __len__(self):
        return len(self.fake)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.fake[idx]),
            torch.from_numpy(self.real[idx]),
            torch.tensor(1.0)
        )

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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir

        # embedding info
        self.embedding_dim = embedding_dim
        self.fake_slice = slice(fake_start, fake_end)
        self.real_slice = slice(real_start, real_end)

        self.results = []
        self.best_auc = 0.0
        self.best_accuracy = 0.0

        if "/content" in self.log_dir:
            self.log_dir = "optimization_results"
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"[INFO] Using precomputed embeddings (dim={self.embedding_dim})")

    # ------------------------------------------------------------
    # MODEL CREATION
    # ------------------------------------------------------------

    def create_siamese_model(self, mode, projection_dim=128):
        if mode == "pair":
            return SiameseEmbeddingModel(
                embedding_dim=self.embedding_dim,     # 768
                hidden_dim=projection_dim,             # internal layer size
                out_dim=projection_dim                 # final projected dim
            )

        raise ValueError(f"Unsupported mode: {mode}")

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    def create_dataloader(self, dataframe, batch_size, mode):
        """
        Returns embedding tensors instead of text.
        """
        if mode != "pair":
            raise ValueError("Only pair mode supported with embeddings")

        dataset = EmbeddingPairDataset(
            dataframe,
            self.fake_slice,
            self.real_slice
        )

        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    def create_optimizer(self, model, params):
        if params['optimizer'] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        else:
            return torch.optim.SGD(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )

    # ------------------------------------------------------------
    # HYPERPARAMETER SAMPLING
    # ------------------------------------------------------------

    def sample_hyperparameters(self, mode, n_samples):
        samples = []

        for _ in range(n_samples):
            samples.append({
                "lr": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))),
                "batch_size": int(np.random.choice([16, 32, 64, 128])),
                "internal_layer_size": int(np.random.choice([64, 128, 256, 512])),
                "optimizer": np.random.choice(["adam", "adamw", "sgd"]),
                "weight_decay": float(np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3)))),
                "margin": float(np.random.uniform(0.1, 2.0)),
            })

        return samples

    def sample_initial_hyperparameters(self, mode, n_samples):
        return self.sample_hyperparameters(mode, n_samples)

    # ------------------------------------------------------------
    # EVALUATION (UNCHANGED)
    # ------------------------------------------------------------

    def evaluate_trial(
        self,
        params,
        training_filepath,
        test_filepath,
        mode,
        loss_type,
        medium_filepath=None,
        easy_filepath=None,
        epochs=5,
        validate_filepath=None,
        save_best_model=True,
        curriculum=None,
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
            internal_layer_size = int(params["internal_layer_size"])
            lr = float(params["lr"])

            print(
                f"Testing params: "
                f"LR={lr:.6f}, Batch={batch_size}, "
                f"Layer={internal_layer_size}, "
                f"Opt={params['optimizer']}"
            )

            dataframe = pd.read_parquet(training_filepath)
            dataloader = self.create_dataloader(dataframe, batch_size, mode)

            model = self.create_siamese_model(mode, internal_layer_size).to(self.device)
            optimizer = self.create_optimizer(model, params)

            from model_utils.loss.pair_losses import ContrastiveLoss
            criterion = ContrastiveLoss(margin=float(params["margin"]))

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                model_type=mode,
            )

            best_metrics = trainer.train(
                dataloader=dataloader,
                test_filepath=test_filepath,
                mode=mode,
                epochs=epochs,
                validate_filepath=validate_filepath
            )

            result = {
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "optimizer": params["optimizer"],
                "weight_decay": params["weight_decay"],
                "mode": mode,
                "loss_type": loss_type,
                "test_auc": best_metrics.get("roc_auc", 0),
                "test_accuracy": best_metrics.get("accuracy", 0),
                **best_metrics,
            }

            self.results.append(result)

            if save_best_model and result["test_auc"] > self.best_auc:
                self.best_auc = result["test_auc"]
                self.best_accuracy = result["test_accuracy"]

                model_id = f"{self.model_type}_{mode}"
                torch.save(
                    model.state_dict(),
                    os.path.join(self.log_dir, f"best_model_{model_id}.pt"),
                )
                with open(
                    os.path.join(self.log_dir, f"best_hparams_{model_id}.json"), "w"
                ) as f:
                    json.dump(convert_np(params), f)

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "test_auc": 0.0,
                "test_accuracy": 0.0,
            }
