import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime

from scripts.training.trainer import Trainer
from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import EmbeddingPairDataset, Text2ImgDistillDataset


class BaseOptimizer:
    """
    Hyperparameter optimization using PRECOMPUTED EMBEDDINGS.
    Supports:
      - mode="pair"
      - mode="text2img" (case 2 distillation; output_dim fixed to 768)
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

        fraud_text_start=1539,
        fraud_text_end=2307,
        real_text_start=2307,
        real_text_end=3075,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.log_dir = log_dir
        self.embedding_dim = int(embedding_dim)

        self.fake_slice = slice(int(fake_start), int(fake_end))
        self.real_slice = slice(int(real_start), int(real_end))
        self.fraud_text_slice = slice(int(fraud_text_start), int(fraud_text_end))
        self.real_text_slice = slice(int(real_text_start), int(real_text_end))

        self.best_val_loss = float("inf")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)
        os.makedirs("images", exist_ok=True)

        print(f"[INFO] Using precomputed embeddings (dim={self.embedding_dim})")
        print(f"[INFO] fake_slice={self.fake_slice}, real_slice={self.real_slice}")
        print(f"[INFO] fraud_text_slice={self.fraud_text_slice}, real_text_slice={self.real_text_slice}")

    def create_siamese_model(self, mode, hidden_dim=512, out_dim=None):
        hidden_dim = int(hidden_dim)

        if mode == "pair":
            out_dim = int(out_dim if out_dim is not None else 128)
            return SiameseEmbeddingModel(embedding_dim=self.embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim)

        if mode == "text2img":
            out_dim = int(out_dim if out_dim is not None else self.embedding_dim)
            if out_dim != self.embedding_dim:
                raise ValueError(f"text2img requires out_dim == {self.embedding_dim}, got {out_dim}")
            return SiameseEmbeddingModel(embedding_dim=self.embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim)

        raise ValueError(f"Unsupported mode: {mode}")

    def create_dataloader(self, dataframe, batch_size, mode, shuffle):
        from torch.utils.data import DataLoader

        batch_size = int(batch_size)

        if mode == "pair":
            dataset = EmbeddingPairDataset(dataframe, fraud_slice=self.fake_slice, real_slice=self.real_slice, label_col=2)
        elif mode == "text2img":
            dataset = Text2ImgDistillDataset(
                dataframe,
                fraud_img_slice=self.fake_slice,
                real_img_slice=self.real_slice,
                fraud_txt_slice=self.fraud_text_slice,
                real_txt_slice=self.real_text_slice,
                label_col=2,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def create_optimizer(self, model, params):
        opt = str(params.get("optimizer", "adamw")).lower()
        lr = float(params["lr"])
        wd = float(params.get("weight_decay", 0.0))

        if opt == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if opt == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

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
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        try:
            batch_size = int(params["batch_size"])
            hidden_dim = int(params["internal_layer_size"])
            lr = float(params["lr"])

            if mode == "pair":
                out_dim = int(params.get("output_dim", 128))
            elif mode == "text2img":
                out_dim = self.embedding_dim
                params["output_dim"] = out_dim
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            print(
                f"Testing params: LR={lr:.6f}, Batch={batch_size}, Hidden={hidden_dim}, "
                f"OutDim={out_dim}, Opt={params.get('optimizer')}"
            )

            string = (
                f"_{self.model_type}"
                f"_Mode={mode}"
                f"_Loss={loss_type}"
                f"_LR={lr:.6f}"
                f"_WD={float(params.get('weight_decay', 0.0)):.2e}"
                f"_Batch={batch_size}"
                f"_Hidden={hidden_dim}"
                f"_OutDim={out_dim}"
                f"_Opt={params.get('optimizer')}"
                f"_Ep={int(epochs)}"
            )
            if mode == "pair":
                string += f"_Margin={float(params.get('margin', 0.0)):.3f}"

            train_df = pd.read_parquet(training_filepath)
            dataloader = self.create_dataloader(train_df, batch_size, mode, shuffle=True)

            val_dataloader = None
            if validate_filepath is not None:
                val_df = pd.read_parquet(validate_filepath)
                val_dataloader = self.create_dataloader(val_df, batch_size, mode, shuffle=False)

            model = self.create_siamese_model(mode, hidden_dim=hidden_dim, out_dim=out_dim).to(self.device)
            optimizer = self.create_optimizer(model, params)

            if mode == "pair":
                from model_utils.loss.pair_losses import ContrastiveLoss
                criterion = ContrastiveLoss(margin=float(params.get("margin", 1.0)))
            else:
                from model_utils.loss.distill_losses import CosineDistillLoss
                criterion = CosineDistillLoss()

            trainer = Trainer(model, criterion, optimizer, self.device, model_type=mode)

            best_metrics = trainer.train(
                dataloader=dataloader,
                trial_number=trial_number,
                test_filepath=test_filepath,
                string=string,
                mode=mode,
                epochs=int(epochs),
                validate_filepath=validate_filepath,
                validate_dataloader=val_dataloader,
                save_best=False,
                plot_losses=False,
                want_test=False,
            )

            result = {
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": hidden_dim,
                "output_dim": out_dim,
                "optimizer": params.get("optimizer"),
                "weight_decay": float(params.get("weight_decay", 0.0)),
                "mode": mode,
                "loss_type": loss_type,
                "best_train_loss": best_metrics.get("best_train_loss"),
                "final_train_loss": best_metrics.get("final_train_loss"),
                "final_val_loss": best_metrics.get("final_val_loss"),
                "best_val_loss": best_metrics.get("best_val_loss"),
            }
            if mode == "pair":
                result["margin"] = float(params.get("margin", 0.0))

            val_loss = result.get("best_val_loss") or result.get("final_val_loss")
            if save_best_model and val_loss is not None and float(val_loss) < float(self.best_val_loss):
                self.best_val_loss = float(val_loss)

                model_id = f"{self.model_type}_{mode}"
                results_dir = os.path.join(self.log_dir, "results")
                os.makedirs(results_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(results_dir, f"best_model_{model_id}.pt"))
                with open(os.path.join(results_dir, f"best_hparams_{model_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(convert_np(params), f, indent=2)

                print(f"[DEBUG] Saved new best model (val_loss={self.best_val_loss:.6f}) -> best_model_{model_id}.pt")

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"timestamp": datetime.now(), "error": str(e)}
