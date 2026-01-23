import argparse
import torch
import os
import json
from datetime import datetime

from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer

from model_utils.utils.data import EmbeddingPairDataset
from model_utils.models.learning.siamese import SiameseEmbeddingModel
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a Siamese model for business name matching.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate_saved"],
        required=True,
        help="Mode to run: train or evaluate_saved",
    )
    parser.add_argument(
        "--optuna",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Optuna hyperparameter optimization (True/False)",
    )

    parser.add_argument("--training_filepath", type=str, help="Path to training data (for training modes)")
    parser.add_argument(
        "--test_filepath",
        type=str,
        required=True,
        help="Path to test data (CSV or Parquet with fraudulent_name, real_name, label)",
    )
    parser.add_argument("--validate_filepath", type=str, default=None, help="Optional validation file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")

    # Eval plotting
    parser.add_argument("--plot", action="store_true", help="If set, plot ROC + confusion matrix during evaluation")
    parser.add_argument(
        "--eval_name_prefix",
        type=str,
        default="final_model_test",
        help="Filename prefix for evaluation images",
    )

    # Training parameters (single-run path)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--log_dir", type=str, default="saved_models", help="Directory to save results")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--internal_layer_size", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--optimizer_name", type=str, choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for optimization methods")
    parser.add_argument("--sampler", type=str, choices=["tpe", "random", "cmaes"], default="tpe")
    parser.add_argument("--pruner", type=str, choices=["median", "hyperband", "none"], default="median")
    parser.add_argument("--study_name", type=str, help="Study name for Optuna optimization")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Directories next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "saved_models")
    images_dir = os.path.join(script_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # -----------------------------
    # MODE: evaluate_saved
    # -----------------------------
    if args.mode == "evaluate_saved":
        print("Loading saved model for evaluation...")

        # Build model architecture
        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=args.output_dim,
        ).to(device)

        # Load weights
        ckpt_path = os.path.join(save_dir, "single_run_model.pt")  # change if needed
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[INFO] Loaded model from: {ckpt_path}")

        evaluator = Evaluator(model, batch_size=args.batch_size, model_type="pair")

        roc_path = os.path.join(images_dir, f"{args.eval_name_prefix}_roc.png")
        cm_path = os.path.join(images_dir, f"{args.eval_name_prefix}_confusion_matrix_youden.png")

        results_df, metrics = evaluator.evaluate(
            args.test_filepath,
            plot=args.plot,
            roc_png_path=roc_path,
            cm_png_path=cm_path,
        )

        print("\nEvaluation complete. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_csv_path = os.path.join(save_dir, f"eval_results_{timestamp}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"[INFO] Saved eval results CSV to: {results_csv_path}")

        # Convert numpy types for JSON (defensive)
        def convert_np(obj):
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            if hasattr(obj, "item") and callable(obj.item):
                return obj.item()
            return obj

        metrics_json_path = os.path.join(save_dir, f"eval_metrics_{timestamp}.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(convert_np(metrics), f, indent=2)
        print(f"[INFO] Saved eval metrics JSON to: {metrics_json_path}")
        return

    # -----------------------------
    # MODE: train
    # -----------------------------
    if args.mode == "train":
        if args.optuna == "True":
            optimizer = UnifiedHyperparameterOptimizer(
                "pairwise_contrastive",
                device=device,
                log_dir=args.log_dir,
            )

            _results = optimizer.optimize(
                method="optuna",
                training_filepath=args.training_filepath,
                test_filepath=args.test_filepath,
                mode="pair",
                loss_type="contrastive",
                epochs=args.epochs,
                n_trials=args.n_trials,
                sampler=args.sampler,
                pruner=args.pruner if args.pruner != "none" else None,
                validate_filepath=args.validate_filepath,
            )
            return

        # -----------------------------
        # Single-run training (non-optuna)
        # -----------------------------
        import pandas as pd
        from model_utils.loss.pair_losses import ContrastiveLoss

        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=args.output_dim,
        ).to(device)

        criterion = ContrastiveLoss(margin=args.margin)

        if args.optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_df = pd.read_parquet(args.training_filepath)
        train_dataset = EmbeddingPairDataset(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if args.validate_filepath is not None:
            val_df = pd.read_parquet(args.validate_filepath)
            val_dataset = EmbeddingPairDataset(val_df)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        trainer = Trainer(model, criterion, optimizer, device, model_type="pair")

        metrics = trainer.train(
            dataloader=train_loader,
            trial_number=1,
            test_filepath=args.test_filepath,
            string="_single_run",
            mode="pair",
            epochs=args.epochs,
            validate_dataloader=val_loader,
            want_test=False,  # don't test every run
        )
        print("Training done. Returned metrics:", metrics)

        # Final test eval (optional)
        evaluator = Evaluator(model, batch_size=args.batch_size, model_type="pair")
        roc_path = os.path.join(images_dir, "single_run_test_roc.png")
        cm_path = os.path.join(images_dir, "single_run_test_confusion_matrix_youden.png")

        _, test_metrics = evaluator.evaluate(
            args.test_filepath,
            plot=args.plot,
            roc_png_path=roc_path,
            cm_png_path=cm_path,
        )
        print("\n--- FINAL TEST METRICS ---")
        for k, v in test_metrics.items():
            print(f"{k}: {v}")

        # Save model + hparams
        torch.save(model.state_dict(), os.path.join(save_dir, "single_run_model.pt"))
        print(f"Saved trained model to {os.path.join(save_dir, 'single_run_model.pt')}")

        hparams = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "internal_layer_size": args.internal_layer_size,
            "output_dim": args.output_dim,
            "optimizer": args.optimizer_name,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "epochs": args.epochs,
        }
        with open(os.path.join(save_dir, "single_run_hparams.json"), "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2)
        print(f"Saved hyperparameters to {os.path.join(save_dir, 'single_run_hparams.json')}")


if __name__ == "__main__":
    main()
