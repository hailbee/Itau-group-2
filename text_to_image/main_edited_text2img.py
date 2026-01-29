#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig

from siamese import SiameseEmbeddingModel
from data import Text2TeacherDistillDataset
from distill_losses import EmbeddingCosineDistillLoss


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
        description="Text → teacher (golden) embedding distillation (teacher-score BCE)"
    )

    # mode
    parser.add_argument("--mode", type=str, choices=["train", "evaluate_saved"], required=True)
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
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # optuna controls (kept simple)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--optuna_short_epochs", type=int, default=5)

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")

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
        if args.validate_filepath is None:
            raise ValueError("--validate_filepath is required for training (needed for early stopping / sanity eval)")

        # ---------- OPTUNA ----------
        if args.optuna == "True":
            from optuna_distill import run_optuna, OptunaConfig

            best = run_optuna(
                training_filepath=args.training_filepath,
                validate_filepath=args.validate_filepath,
                device=args.device,
                cfg=OptunaConfig(n_trials=int(args.n_trials), short_epochs=int(args.optuna_short_epochs)),
            )

            print("\n[OPTUNA RESULT]")
            print("best_value:", best["best_value"])
            print("best_params:", best["best_params"])
            return

        # ---------- SINGLE RUN ----------
        train_df = load_table(args.training_filepath)
        val_df = load_table(args.validate_filepath)

        train_ds = Text2TeacherDistillDataset(
            train_df,
            fraud_txt_prefix="fraud_txt_emb_",
            real_txt_prefix="real_txt_emb_",
            fraud_teacher_prefix="fraud_aligned_",
            real_teacher_prefix="real_aligned_",
            label_col="label",  
        )

        val_ds = Text2TeacherDistillDataset(
            val_df,
            fraud_txt_prefix="fraud_txt_emb_",
            real_txt_prefix="real_txt_emb_",
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
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        # Infer dims (prevents 768/128 mismatches)
        text_dim = infer_dim_from_prefix(train_df, "fraud_txt_emb_")
        teacher_dim = infer_dim_from_prefix(train_df, "fraud_aligned_")
        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim}")

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=args.internal_layer_size,
            out_dim=teacher_dim,   # IMPORTANT: match teacher (128)
        ).to(device)

        criterion = EmbeddingCosineDistillLoss().to(device)

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
            grad_clip=args.grad_clip,
            save_dir=args.save_dir,
        )

        # -------- FINAL TEST EVAL --------
        print("\n[INFO] Running final test evaluation...")

        evaluator = Evaluator2(
            model,
            EvalConfig(batch_size=args.eval_batch_size),
        )
        results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

        print("\n[INFO] Final test metrics:")
        print("\nAlignment debug:")
        print(test_metrics.get("alignment_debug"))
        print("\nTeacher:")
        print(test_metrics.get("teacher_image_space") or test_metrics.get("teacher"))
        print("\nRaw text:")
        print(test_metrics.get("raw_text_space") or test_metrics.get("raw_text"))
        print("\nStudent:")
        print(test_metrics.get("aligned_text_space") or test_metrics.get("student"))
        print("\nDeltas:")
        print(test_metrics.get("deltas"))
        return

    # -------------------------
    # EVALUATE SAVED
    # -------------------------
    if args.mode == "evaluate_saved":
        if args.model_path is None:
            raise ValueError("--model_path is required for evaluate_saved")

        test_df = load_table(args.test_filepath)
        text_dim = infer_dim_from_prefix(test_df, "fraud_txt_emb_")
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

        evaluator = Evaluator2(model, EvalConfig(batch_size=args.eval_batch_size))
        print("\n[INFO] Running final test evaluation...")
        results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

        print("\n[INFO] Final test metrics:")
        print(test_metrics)
        return


if __name__ == "__main__":
    main()
