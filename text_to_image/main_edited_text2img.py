#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig

from siamese import SiameseEmbeddingModel
from data import Text2TeacherDistillDataset
from distill_losses import TeacherScoreDistillBCELoss


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


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def infer_dim_from_prefix(df: pd.DataFrame, prefix: str) -> int:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise ValueError(f"Could not infer dim: no columns with prefix '{prefix}'")
    return len(cols)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Text → spoof-aware (golden) embedding distillation (teacher-score BCE, single-term)"
    )

    # mode
    parser.add_argument("--mode", type=str, choices=["train", "evaluate_saved"], required=True)

    # NOTE: Optuna flow is legacy in your codebase; keep it but warn
    parser.add_argument("--optuna", type=str, choices=["True", "False"], default="False")

    # data
    parser.add_argument("--training_filepath", type=str)
    parser.add_argument("--validate_filepath", type=str, default=None)
    parser.add_argument("--test_filepath", type=str, required=True)

    # saved model eval
    parser.add_argument("--model_path", type=str, default=None)
    
    # training hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--internal_layer_size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--log_dir", type=str, default="optimization_results")

    # eval speed
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=None)

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        if args.training_filepath is None:
            raise ValueError("--training_filepath is required for training")

        if args.optuna == "True":
            raise RuntimeError(
                "Optuna path is legacy and not wired to the new prefix-based datasets/evaluator. "
                "Run with --optuna False."
            )

        train_df = load_table(args.training_filepath)
        val_df = load_table(args.validate_filepath) if args.validate_filepath else None

        # Prefix-based dataset (your schema)
        train_ds = Text2TeacherDistillDataset(
            train_df,
            fraud_txt_prefix="fraud_txt_",
            real_txt_prefix="real_txt_",
            fraud_teacher_prefix="fraud_aligned_",
            real_teacher_prefix="real_aligned_",
            label_col="label",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        val_loader = None
        if val_df is not None:
            val_ds = Text2TeacherDistillDataset(
                val_df,
                fraud_txt_prefix="fraud_txt_",
                real_txt_prefix="real_txt_",
                fraud_teacher_prefix="fraud_aligned_",
                real_teacher_prefix="real_aligned_",
                label_col="label",
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

        # ✅ Infer dims from dataframe to avoid teacher-dim mistakes
        text_dim = infer_dim_from_prefix(train_df, "fraud_txt_")
        teacher_dim = infer_dim_from_prefix(train_df, "fraud_aligned_")

        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim}")

        # ✅ IMPORTANT FIX:
        # student outputs MUST match teacher_dim (128), not 768
        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=args.internal_layer_size,
            out_dim=teacher_dim,
        ).to(device)

        criterion = TeacherScoreDistillBCELoss().to(device)
        optim_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = build_optimizer(args.optimizer, optim_params, args.lr, args.weight_decay)

        trainer = Trainer(model, criterion, optimizer, device)

        trainer.train(
            dataloader=train_loader,
            validate_dataloader=val_loader,
            test_filepath=args.test_filepath,
            string="_distill",
            trial_number=1,
            epochs=args.epochs,
            eval_every=1,
            save_dir=args.save_dir,
        )

        # -------- FINAL TEST EVAL --------
        print("\n[INFO] Running final test evaluation...")

        evaluator = Evaluator2(
            model,
            EvalConfig(
                batch_size=args.eval_batch_size,
                fraud_txt_prefix="fraud_txt_",
                real_txt_prefix="real_txt_",
                fraud_teacher_prefix="fraud_aligned_",
                real_teacher_prefix="real_aligned_",
                label_col="label",
            ),
        )

        results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

        print("\n[INFO] Final test metrics:")
        print("\nAlignment debug:")
        print(test_metrics["alignment_debug"])

        print("\nSpoof decision (TEACHER / GOLDEN space):")
        print(test_metrics["teacher"])

        print("\nSpoof decision (RAW TEXT space):")
        print(test_metrics["raw_text"])

        print("\nSpoof decision (STUDENT space):")
        print(test_metrics["student"])

        print("\nDeltas:")
        print(test_metrics["deltas"])

        return

    # -------------------------
    # EVALUATE SAVED
    # -------------------------
    if args.mode == "evaluate_saved":
        if args.model_path is None:
            raise ValueError("--model_path is required for evaluate_saved")

        # We infer dims from test file so you don't hardcode 768/128 incorrectly
        test_df = load_table(args.test_filepath)
        text_dim = infer_dim_from_prefix(test_df, "fraud_txt_")
        teacher_dim = infer_dim_from_prefix(test_df, "fraud_aligned_")

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=args.internal_layer_size,
            out_dim=teacher_dim,
        ).to(device)

        state = torch.load(args.model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        model.eval()

        evaluator = Evaluator2(
            model,
            EvalConfig(
                batch_size=args.eval_batch_size,
                fraud_txt_prefix="fraud_txt_",
                real_txt_prefix="real_txt_",
                fraud_teacher_prefix="fraud_aligned_",
                real_teacher_prefix="real_aligned_",
                label_col="label",
            ),
        )

        print("\n[INFO] Running final test evaluation...")
        results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

        print("\n[INFO] Final test metrics:")
        print("\nAlignment debug:")
        print(test_metrics["alignment_debug"])

        print("\nSpoof decision (TEACHER / GOLDEN space):")
        print(test_metrics["teacher"])

        print("\nSpoof decision (RAW TEXT space):")
        print(test_metrics["raw_text"])

        print("\nSpoof decision (STUDENT space):")
        print(test_metrics["student"])

        print("\nDeltas:")
        print(test_metrics["deltas"])

        return


if __name__ == "__main__":
    main()
