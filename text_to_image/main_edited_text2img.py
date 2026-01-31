#!/usr/bin/env python3
"""
main_edited_text2img.py

NEW TASK: Train on the 4-pairs-per-row TEXT->IMAGE binary dataset
         (left_txt_emb_* , right_img_emb_* , label , pair_kind, ...)

Goal: learn projections so:
  - Training uses text->image binary supervision (close/far)
  - Evaluation (Evaluator2) computes ROC AUC on cosine similarity between
    translated fraud text and translated real text separately:
        score = cos( normalize(Pf(fraud_txt)), normalize(Pr(real_txt)) )

This script assumes you will update evaluator2.py to use the new "translation" model behavior.

Expected training parquet columns (defaults):
  - left_txt_emb_0..D-1
  - right_img_emb_0..K-1
  - label (0/1)
  - pair_kind (optional but recommended; from the builder script)

Expected evaluation parquet columns (defaults like your old setup):
  - fraud_txt_emb_0..D-1
  - real_txt_emb_0..D-1
  - label (0/1)
  (and optionally fraud_aligned_/real_aligned_ if you want extra debugging)

Run (train):
  python3 main_edited_text2img.py \
    --mode train \
    --training_filepath Golden_and_Text/t2i_4pairs_train.parquet \
    --validate_filepath Golden_and_Text/t2i_4pairs_val.parquet \
    --test_filepath Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet \
    --batch_size 256 \
    --epochs 30 \
    --lr 1e-4 \
    --internal_layer_size 512 \
    --loss bce \
    --scale 10.0

Run (evaluate saved):
  python3 main_edited_text2img.py \
    --mode evaluate_saved \
    --model_path saved_models/best_model_trial_1_t2i.pt \
    --test_filepath Golden_and_Text/test_pairs_with_img_and_vate_txt_embs.parquet
"""

from __future__ import annotations

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
from evaluator2 import Evaluator2, EvalConfig

from siamese import SiameseEmbeddingModel
from data import TextImageBinaryPairDataset

from distill_losses import (
    BinaryCosineBCEWithLogits,
    BinaryCosineMarginLoss,
)


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
        description="Train TEXT->IMAGE binary contrastive dataset (4 pairs per original row) and eval via text->text cosine ROC AUC."
    )

    # mode
    parser.add_argument("--mode", type=str, choices=["train", "evaluate_saved"], required=True)

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

    # model dims
    parser.add_argument(
        "--out_dim",
        type=int,
        default=None,
        help="Projection output dim. Default=None -> match image/teacher dim inferred from training data.",
    )

    # head sharing
    parser.add_argument("--share_text_heads", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--share_teacher_heads", type=str, choices=["True", "False"], default="True")

    # loss
    parser.add_argument("--loss", type=str, choices=["bce", "margin"], default="bce")
    parser.add_argument("--scale", type=float, default=10.0, help="For BCE-on-cosine: logits = scale * cosine")
    parser.add_argument("--pos_weight", type=float, default=None, help="Optional BCE pos_weight for class imbalance")
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--pos_margin", type=float, default=0.5)
    parser.add_argument("--neg_margin", type=float, default=0.2)

    # NEW dataset prefixes/cols
    parser.add_argument("--left_txt_prefix", type=str, default="left_txt_emb_")
    parser.add_argument("--right_img_prefix", type=str, default="right_img_emb_")
    parser.add_argument("--train_label_col", type=str, default="label")
    parser.add_argument("--pair_kind_col", type=str, default="pair_kind")
    parser.add_argument("--return_pair_kind", type=str, choices=["True", "False"], default="True")

    # eval config (Evaluator2)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=None)

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--use_amp", type=str, choices=["True", "False"], default="True")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    share_text_heads = True if args.share_text_heads == "True" else False
    share_teacher_heads = True if args.share_teacher_heads == "True" else False
    return_pair_kind = True if args.return_pair_kind == "True" else False
    use_amp = True if args.use_amp == "True" else False

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        if args.training_filepath is None:
            raise ValueError("--training_filepath is required for training")
        if args.validate_filepath is None:
            raise ValueError("--validate_filepath is required for training")

        train_df = load_table(args.training_filepath)
        val_df = load_table(args.validate_filepath)

        # NEW dataset loaders
        train_ds = TextImageBinaryPairDataset(
            train_df,
            left_txt_prefix=args.left_txt_prefix,
            right_img_prefix=args.right_img_prefix,
            label_col=args.train_label_col,
            pair_kind_col=args.pair_kind_col,
            return_pair_kind=return_pair_kind,
            return_orig_row_id=False,
        )
        val_ds = TextImageBinaryPairDataset(
            val_df,
            left_txt_prefix=args.left_txt_prefix,
            right_img_prefix=args.right_img_prefix,
            label_col=args.train_label_col,
            pair_kind_col=args.pair_kind_col,
            return_pair_kind=return_pair_kind,
            return_orig_row_id=False,
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

        # Infer dims from TRAINING parquet (new dataset)
        text_dim = infer_dim_from_prefix(train_df, args.left_txt_prefix)
        teacher_dim = infer_dim_from_prefix(train_df, args.right_img_prefix)
        out_dim = int(args.out_dim) if args.out_dim is not None else int(teacher_dim)

        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim} | out_dim={out_dim}")
        print(f"[INFO] share_text_heads={share_text_heads} | share_teacher_heads={share_teacher_heads}")
        print(f"[INFO] loss={args.loss}")

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=int(args.internal_layer_size),
            out_dim=out_dim,
            teacher_dim=int(teacher_dim),
            share_text_heads=share_text_heads,
            share_teacher_heads=share_teacher_heads,
            dropout=0.0,
            activation="relu",
        ).to(device)

        if args.loss == "bce":
            criterion = BinaryCosineBCEWithLogits(
                scale=float(args.scale),
                normalize_inputs=False,  # Trainer normalizes
                pos_weight=None if args.pos_weight is None else float(args.pos_weight),
                label_smoothing=float(args.label_smoothing),
            ).to(device)
        else:
            criterion = BinaryCosineMarginLoss(
                pos_margin=float(args.pos_margin),
                neg_margin=float(args.neg_margin),
                normalize_inputs=False,  # Trainer normalizes
                squared=True,
            ).to(device)

        optimizer = build_optimizer(
            args.optimizer,
            list(model.parameters()) + list(criterion.parameters()),
            args.lr,
            args.weight_decay,
        )

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
        )

        train_result = trainer.train(
            dataloader=train_loader,
            validate_dataloader=val_loader,
            test_filepath=args.test_filepath,   # Evaluator2 should use this (text->text cosine AUC)
            string="_t2i",
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
        print(test_metrics)
        return

    # -------------------------
    # EVALUATE SAVED
    # -------------------------
    if args.mode == "evaluate_saved":
        if args.model_path is None:
            raise ValueError("--model_path is required for evaluate_saved")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

        # Build model skeleton by inferring dims from TEST file:
        test_df = load_table(args.test_filepath)

        # text_dim inferred from eval parquet (fraud_txt_emb_)
        # (Evaluator2 defaults to these prefixes; keep consistent)
        text_dim = infer_dim_from_prefix(test_df, "fraud_txt_emb_")

        # teacher_dim is only needed to rebuild the same architecture.
        # If the eval parquet has fraud_aligned_ columns, infer; otherwise require --out_dim and set teacher_dim=out_dim.
        if any(isinstance(c, str) and c.startswith("fraud_aligned_") for c in test_df.columns):
            teacher_dim = infer_dim_from_prefix(test_df, "fraud_aligned_")
        else:
            teacher_dim = None

        out_dim = int(args.out_dim) if args.out_dim is not None else int(teacher_dim) if teacher_dim is not None else None
        if out_dim is None:
            raise ValueError(
                "Could not infer out_dim for evaluate_saved. Provide --out_dim (must match training)."
            )

        print(f"[INFO] text_dim={text_dim} | teacher_dim={teacher_dim} | out_dim={out_dim}")

        model = SiameseEmbeddingModel(
            embedding_dim=text_dim,
            hidden_dim=int(args.internal_layer_size),
            out_dim=int(out_dim),
            teacher_dim=None if teacher_dim is None else int(teacher_dim),
            share_text_heads=share_text_heads,
            share_teacher_heads=share_teacher_heads,
            dropout=0.0,
            activation="relu",
        ).to(device)

        # Recreate criterion so criterion_state can load if present
        if args.loss == "bce":
            criterion = BinaryCosineBCEWithLogits(
                scale=float(args.scale),
                normalize_inputs=False,
                pos_weight=None if args.pos_weight is None else float(args.pos_weight),
                label_smoothing=float(args.label_smoothing),
            ).to(device)
        else:
            criterion = BinaryCosineMarginLoss(
                pos_margin=float(args.pos_margin),
                neg_margin=float(args.neg_margin),
                normalize_inputs=False,
                squared=True,
            ).to(device)

        state = safe_torch_load(args.model_path, map_location=device)

        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            if "criterion_state" in state:
                try:
                    criterion.load_state_dict(state["criterion_state"])
                except Exception as e:
                    print(f"[WARN] Could not load criterion_state (ok if changed): {e}")
        else:
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
