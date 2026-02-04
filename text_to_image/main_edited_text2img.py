#!/usr/bin/env python3
import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig
from siamese import SiameseEmbeddingModel
from data import TextTeacherPairDataset
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
        raise ValueError(f"No columns with prefix '{prefix}'")
    return len(cols)


def safe_torch_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Positive-only text → image embedding distillation"
    )

    # mode
    parser.add_argument("--mode", choices=["train", "evaluate_saved"], required=True)

    # data
    parser.add_argument("--training_filepath", type=str)
    parser.add_argument("--validate_filepath", type=str)
    parser.add_argument("--test_filepath", type=str, required=True)

    # saved model eval
    parser.add_argument("--model_path", type=str)

    # training hyperparams
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
    )

    # embedding dims
    parser.add_argument("--out_dim", type=int, default=None)

    # loss
    parser.add_argument("--margin", type=float, default=1.0)

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")

    # eval
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=None)

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        if args.training_filepath is None or args.validate_filepath is None:
            raise ValueError(
                "Training requires --training_filepath and --validate_filepath"
            )

        train_df = load_table(args.training_filepath)
        val_df = load_table(args.validate_filepath)

        train_ds = TextTeacherPairDataset(
            train_df,
            txt_prefix="left_txt_emb_",
            img_prefix="right_img_emb_",
            label_col="label",
        )
        val_ds = TextTeacherPairDataset(
            val_df,
            txt_prefix="left_txt_emb_",
            img_prefix="right_img_emb_",
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
        text_dim = infer_dim_from_prefix(train_df, "left_txt_emb_")
        teacher_dim = infer_dim_from_prefix(train_df, "right_img_emb_")
        out_dim = args.out_dim or teacher_dim

        print(
            f"[INFO] text_dim={text_dim} | "
            f"teacher_dim={teacher_dim} | out_dim={out_dim}"
        )

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=args.internal_layer_size,
            out_dim=out_dim,
            teacher_dim=teacher_dim,
        ).to(device)

        criterion = ThesisMarginOnlyWithTeacherProj(
            margin=args.margin
        ).to(device)

        optimizer = build_optimizer(
            args.optimizer,
            model.parameters(),
            args.lr,
            args.weight_decay,
        )

        trainer = Trainer(model, criterion, optimizer, device)

        trainer.train(
            dataloader=train_loader,
            validate_dataloader=val_loader,
            epochs=args.epochs,
            save_dir=args.save_dir,
            string="_distill",
        )

        print("\n[INFO] Running final test evaluation...")
        evaluator = Evaluator2(
            model,
            EvalConfig(batch_size=args.eval_batch_size),
        )
        _, test_metrics = evaluator.evaluate(
            args.test_filepath,
            max_rows=args.eval_max_rows,
        )

        print("\n[INFO] Final test metrics:")
        print(test_metrics)
        return

    # -------------------------
    # EVALUATE SAVED
    # -------------------------
    if args.mode == "evaluate_saved":
        if args.model_path is None:
            raise ValueError("--model_path is required for evaluate_saved")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path not found: {args.model_path}")

        test_df = load_table(args.test_filepath)

        text_dim = infer_dim_from_prefix(test_df, "left_txt_emb_")
        teacher_dim = infer_dim_from_prefix(test_df, "right_img_emb_")
        out_dim = args.out_dim or teacher_dim

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=args.internal_layer_size,
            out_dim=out_dim,
            teacher_dim=teacher_dim,
        ).to(device)

        criterion = ThesisMarginOnlyWithTeacherProj(
            margin=args.margin
        ).to(device)

        state = safe_torch_load(args.model_path, map_location=device)

        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            if "criterion_state" in state:
                try:
                    criterion.load_state_dict(state["criterion_state"])
                except Exception:
                    pass
        else:
            model.load_state_dict(state)

        model.eval()

        evaluator = Evaluator2(
            model,
            EvalConfig(batch_size=args.eval_batch_size),
        )
        _, test_metrics = evaluator.evaluate(
            args.test_filepath,
            max_rows=args.eval_max_rows,
        )

        print("\n[INFO] Final test metrics:")
        print(test_metrics)
        return


if __name__ == "__main__":
    main()
