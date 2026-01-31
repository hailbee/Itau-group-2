#!/usr/bin/env python3
import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig

from siamese import SiameseEmbeddingModel
from data import Text2TeacherDistillDataset

# ✅ use the margin-only thesis loss with learned teacher projection
from distill_losses import ThesisMarginOnlyWithTeacherProj


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


def safe_torch_load(path: str, map_location: torch.device):
    """
    Torch 2.6+ supports weights_only=True for safer loading.
    Older torch will throw TypeError (unexpected kwarg), so we fall back.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Text → teacher (golden) embedding distillation (thesis-style margin contrastive + learned teacher projection)"
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

    # ✅ allow out_dim != teacher_dim
    parser.add_argument(
        "--out_dim",
        type=int,
        default=None,
        help="Student output embedding dim. Default=None -> match teacher_dim.",
    )

    # ✅ thesis loss knob (ONLY knob)
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for thesis-style contrastive loss")

    # optional: projection bias (keep default False unless you want to test it)
    parser.add_argument("--proj_bias", type=str, choices=["True", "False"], default="False")

    # optuna controls
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

    proj_bias = True if args.proj_bias == "True" else False

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

        # Infer dims
        text_dim = infer_dim_from_prefix(train_df, "fraud_txt_emb_")
        teacher_dim = infer_dim_from_prefix(train_df, "fraud_aligned_")
        out_dim = int(args.out_dim) if args.out_dim is not None else int(teacher_dim)

        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim} | out_dim={out_dim}")
        print(f"[INFO] thesis loss hyperparam: margin={args.margin} | proj_bias={proj_bias}")

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=int(args.internal_layer_size),
            out_dim=out_dim,
            teacher_dim=int(teacher_dim),
        ).to(device)


        # ✅ instantiate thesis loss (has a learnable teacher projection)
        criterion = ThesisMarginOnlyWithTeacherProj(
            margin=float(args.margin)
        ).to(device)

        # ✅ include criterion parameters (teacher_proj) in optimizer
        optimizer = build_optimizer(
            args.optimizer,
            list(model.parameters()),
            args.lr,
            args.weight_decay,
        )


        trainer = Trainer(model, criterion, optimizer, device)

        train_result = trainer.train(
            dataloader=train_loader,
            validate_dataloader=val_loader,
            test_filepath=args.test_filepath,
            string="_distill",
            trial_number=1,
            epochs=args.epochs,
            save_dir=args.save_dir,
        )

        if isinstance(train_result, dict) and train_result.get("best_model_path"):
            print(f"[INFO] Best checkpoint: {train_result['best_model_path']}")

        # -------- FINAL TEST EVAL --------
        print("\n[INFO] Running final test evaluation...")

        evaluator = Evaluator2(
            model,
            EvalConfig(batch_size=int(args.eval_batch_size)),
        )
        _results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

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
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

        test_df = load_table(args.test_filepath)
        text_dim = infer_dim_from_prefix(test_df, "fraud_txt_emb_")
        teacher_dim = infer_dim_from_prefix(test_df, "fraud_aligned_")
        out_dim = int(args.out_dim) if args.out_dim is not None else int(teacher_dim)

        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim} | out_dim={out_dim}")

        hidden_dim = int(args.internal_layer_size)
        
        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            teacher_dim=int(teacher_dim),
        ).to(device)
        

        # We also reconstruct the criterion so we can restore teacher_proj if it was saved.
        criterion = ThesisMarginOnlyWithTeacherProj(
            margin=float(args.margin)
        ).to(device)


        state = safe_torch_load(args.model_path, map_location=device)

        # Trainer saves {"model_state", "criterion_state", ...}
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            if "criterion_state" in state:
                try:
                    criterion.load_state_dict(state["criterion_state"])
                except Exception as e:
                    print(f"[WARN] Could not load criterion_state (ok if not present/changed): {e}")
        else:
            # also support raw state_dict checkpoints
            model.load_state_dict(state)

        model.eval()

        evaluator = Evaluator2(model, EvalConfig(batch_size=int(args.eval_batch_size)))
        print("\n[INFO] Running final test evaluation...")
        _results_df, test_metrics = evaluator.evaluate(args.test_filepath, max_rows=args.eval_max_rows)

        print("\n[INFO] Final test metrics:")
        print(test_metrics)
        return


if __name__ == "__main__":
    main()
