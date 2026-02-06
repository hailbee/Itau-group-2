# main_edited.py

import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer

from model_utils.utils.data import EmbeddingPairDataset
from model_utils.models.learning.siamese import SiameseEmbeddingModel


def _resolve_under_script_dir(script_dir: str, path: str) -> str:
    """
    If `path` is relative, resolve it under the directory containing this script.
    This preserves your prior behavior of writing outputs next to the script.
    """
    if path is None:
        return script_dir
    return path if os.path.isabs(path) else os.path.join(script_dir, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a Siamese model for business name matching."
    )
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

    parser.add_argument(
        "--training_filepath",
        type=str,
        default=None,
        help="Path to training data (Parquet) (required for train mode)",
    )
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
    parser.add_argument(
        "--log_dir",
        type=str,
        default="saved_models",
        help="Directory (relative to this script unless absolute) to save models/results",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--internal_layer_size", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--optimizer_name", type=str, choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # -----------------------------
    # TWO-MARGIN COSINE CONTRASTIVE (hinge)
    # -----------------------------
    parser.add_argument(
        "--m_pos",
        type=float,
        default=0.92,
        help="Positive cosine margin: positives want cos >= m_pos",
    )
    parser.add_argument(
        "--m_neg",
        type=float,
        default=0.84,
        help="Negative cosine margin: negatives want cos <= m_neg",
    )
    parser.add_argument("--w_pos", type=float, default=1.0, help="Weight on positive term")
    parser.add_argument("--w_neg", type=float, default=3.0, help="Weight on negative term (often > w_pos)")

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for Optuna")
    parser.add_argument("--sampler", type=str, choices=["tpe", "random", "cmaes"], default="tpe")
    parser.add_argument("--pruner", type=str, choices=["median", "hyperband", "none"], default="median")
    parser.add_argument("--study_name", type=str, default=None, help="Study name for Optuna optimization")

    args = parser.parse_args()

    # Guardrail: two-margin requires m_pos > m_neg (only relevant for single-run)
    if not (args.m_pos > args.m_neg):
        raise ValueError(
            f"Need m_pos > m_neg for two-margin cosine hinge. Got m_pos={args.m_pos}, m_neg={args.m_neg}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Directories next to this script (unless user passes absolute paths)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = _resolve_under_script_dir(script_dir, args.log_dir)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # -----------------------------
    # MODE: evaluate_saved
    # -----------------------------
    if args.mode == "evaluate_saved":
        print("Loading saved model for evaluation...")

        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=args.output_dim,
        ).to(device)

        ckpt_path = os.path.join(save_dir, "single_run_model.pt")  # default path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_csv_path = os.path.join(save_dir, f"eval_results_{timestamp}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"[INFO] Saved eval results CSV to: {results_csv_path}")

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
        if args.training_filepath is None:
            raise ValueError("--training_filepath is required when --mode train")

        # -----------------------------
        # Optuna training
        # -----------------------------
        if args.optuna == "True":
            optimizer = UnifiedHyperparameterOptimizer(
                model_type="pairwise_contrastive",
                device=device,
                log_dir=save_dir,
            )

            _results = optimizer.optimize(
                method="optuna",
                training_filepath=args.training_filepath,
                test_filepath=args.test_filepath,
                mode="pair",
                loss_type="contrastive",  # informational label; your optimizer defines the actual loss
                epochs=args.epochs,
                n_trials=args.n_trials,
                sampler=args.sampler,
                pruner=(args.pruner if args.pruner != "none" else None),
                study_name=args.study_name,
                validate_filepath=args.validate_filepath,
            )
            return

        # -----------------------------
        # Single-run training (non-optuna)
        # -----------------------------
        import pandas as pd

        # Your UPDATED two-margin cosine hinge loss should live here and accept:
        # ContrastiveLoss(m_pos, m_neg, w_pos=..., w_neg=..., reduction="mean")
        from model_utils.loss.pair_losses import ContrastiveLoss

        model = SiameseEmbeddingModel(
            embedding_dim=768,
            hidden_dim=args.internal_layer_size,
            out_dim=args.output_dim,
        ).to(device)

        criterion = ContrastiveLoss(
            m_pos=args.m_pos,
            m_neg=args.m_neg,
            w_pos=args.w_pos,
            w_neg=args.w_neg,
        )

        if args.optimizer_name == "adam":
            optimizer_t = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer_name == "adamw":
            optimizer_t = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer_t = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

        trainer = Trainer(model, criterion, optimizer_t, device, model_type="pair")

        metrics = trainer.train(
            dataloader=train_loader,
            trial_number=1,
            test_filepath=args.test_filepath,
            string="_single_run",
            mode="pair",
            epochs=args.epochs,
            validate_dataloader=val_loader,
            want_test=False,
        )
        print("Training done. Returned metrics:", metrics)

        # Final test eval
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
        model_path = os.path.join(save_dir, "single_run_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved trained model to {model_path}")

        hparams = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "internal_layer_size": args.internal_layer_size,
            "output_dim": args.output_dim,
            "optimizer": args.optimizer_name,
            "weight_decay": args.weight_decay,
            "m_pos": args.m_pos,
            "m_neg": args.m_neg,
            "w_pos": args.w_pos,
            "w_neg": args.w_neg,
            "epochs": args.epochs,
        }
        hparams_path = os.path.join(save_dir, "single_run_hparams.json")
        with open(hparams_path, "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2)
        print(f"Saved hyperparameters to {hparams_path}")


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES

1) Single-run training (no Optuna):
python main_edited.py \
  --mode train \
  --optuna False \
  --training_filepath /path/to/train.parquet \
  --validate_filepath /path/to/val.parquet \
  --test_filepath /path/to/test.parquet \
  --m_pos 0.92 \
  --m_neg 0.84 \
  --w_neg 3.0 \
  --epochs 10 \
  --batch_size 32 \
  --plot

2) Evaluate a saved model:
python main_edited.py \
  --mode evaluate_saved \
  --test_filepath /path/to/test.parquet \
  --batch_size 32 \
  --plot
"""
