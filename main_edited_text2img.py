#!/usr/bin/env python3
import argparse
import torch
import pandas as pd

from torch.utils.data import DataLoader

from scripts.training.trainer import Trainer
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer

from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.utils.data import Text2ImgDistillDataset
from model_utils.loss.distill_losses import CosineDistillLoss


# -------------------------
# Utils
# -------------------------
def pick_device(device_override=None):
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_optimizer(name, model, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Text → spoof-aware image embedding distillation (CASE 2)"
    )

    # mode
    parser.add_argument("--mode", type=str, choices=["train", "evaluate_saved"], required=True)
    parser.add_argument("--optuna", type=str, choices=["True", "False"], default="True")

    # data
    parser.add_argument("--training_filepath", type=str)
    parser.add_argument("--validate_filepath", type=str, default=None)
    parser.add_argument("--test_filepath", type=str, required=True)

    # saved model eval
    parser.add_argument("--model_path", type=str, default=None)

    # embedding slices (your known layout)
    parser.add_argument("--fake_start", type=int, default=3)
    parser.add_argument("--fake_end", type=int, default=771)
    parser.add_argument("--real_start", type=int, default=771)
    parser.add_argument("--real_end", type=int, default=1539)

    parser.add_argument("--fraud_text_start", type=int, default=1539)
    parser.add_argument("--fraud_text_end", type=int, default=2307)
    parser.add_argument("--real_text_start", type=int, default=2307)
    parser.add_argument("--real_text_end", type=int, default=3075)

    # training hyperparams (single-run)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--internal_layer_size", type=int, default=512)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer for single-run mode (ignored when --optuna True)",
    )

    # optuna
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random", "cmaes"])
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "hyperband", "none"])
    parser.add_argument("--study_name", type=str, default=None)

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--log_dir", type=str, default="optimization_results")
    parser.add_argument("--plot_accuracy", type=str, choices=["True", "False"], default="False")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        if args.training_filepath is None:
            raise ValueError("--training_filepath is required for training")

        # ---------- OPTUNA ----------
        if args.optuna == "True":
            optimizer = UnifiedHyperparameterOptimizer(
                model_type="text2img_distill",
                device=device,
                log_dir=args.log_dir,
            )

            best = optimizer.optimize(
                method="optuna",
                training_filepath=args.training_filepath,
                test_filepath=args.test_filepath,
                validate_filepath=args.validate_filepath,
                mode="text2img",
                loss_type="cosine_distill",
                epochs=args.epochs,
                n_trials=args.n_trials,
                sampler=args.sampler,
                pruner=args.pruner,
                study_name=args.study_name,
            )

            print("[INFO] Optuna finished. Best params:")
            print(best)
            return

        # ---------- SINGLE RUN ----------
        train_df = load_table(args.training_filepath)
        val_df = load_table(args.validate_filepath) if args.validate_filepath else None

        train_ds = Text2ImgDistillDataset(
            train_df,
            fraud_img_slice=slice(args.fake_start, args.fake_end),
            real_img_slice=slice(args.real_start, args.real_end),
            fraud_txt_slice=slice(args.fraud_text_start, args.fraud_text_end),
            real_txt_slice=slice(args.real_text_start, args.real_text_end),
            label_col=2,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        val_loader = None
        if val_df is not None:
            val_ds = Text2ImgDistillDataset(
                val_df,
                fraud_img_slice=slice(args.fake_start, args.fake_end),
                real_img_slice=slice(args.real_start, args.real_end),
                fraud_txt_slice=slice(args.fraud_text_start, args.fraud_text_end),
                real_txt_slice=slice(args.real_text_start, args.real_text_end),
                label_col=2,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=768,
        ).to(device)

        criterion = CosineDistillLoss()
        optimizer = build_optimizer(
            args.optimizer, model, lr=args.lr, weight_decay=args.weight_decay
        )

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device,
            model_type="text2img",
        )

        run_string = (
            f"_text2img_lr{args.lr:.2e}"
            f"_bs{args.batch_size}"
            f"_h{args.internal_layer_size}"
            f"_opt{args.optimizer}"
            f"_ep{args.epochs}"
        )

        metrics = trainer.train(
            dataloader=train_loader,
            trial_number=0,
            test_filepath=args.test_filepath,
            string=run_string,
            mode="text2img",
            epochs=args.epochs,
            validate_dataloader=val_loader,
            plot_losses=True,
            plot_accuracy=(args.plot_accuracy == "True"),
            save_best=True,
            save_dir=args.save_dir,
            early_stopping=False,
            min_epochs=1,
        )

        print("[INFO] Training finished")
        print(metrics)
        return

    # -------------------------
    # EVALUATE SAVED
    # -------------------------
    if args.mode == "evaluate_saved":
        if args.model_path is None:
            raise ValueError("--model_path is required for evaluate_saved")

        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=768,
        ).to(device)

        state = torch.load(args.model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        model.eval()

        dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = Trainer(
            model,
            CosineDistillLoss(),
            dummy_opt,
            device,
            model_type="text2img",
        )

        _, metrics = trainer.evaluate(
            args.test_filepath,
            plot=True,
            roc_png_path="images/text2img_roc.png",
            cm_png_path="images/text2img_cm.png",
        )

        print("[INFO] Evaluation metrics:")
        print(metrics)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
