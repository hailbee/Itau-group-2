#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from trainer import Trainer
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


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train text → image embedding model (save best by val loss)"
    )

    # data
    parser.add_argument("--training_filepath", required=True)
    parser.add_argument("--validate_filepath", required=True)

    # prefixes (IMPORTANT: must match your parquet columns)
    parser.add_argument("--fraud_txt_prefix", type=str, default="fraud_txt_emb_")
    parser.add_argument("--real_txt_prefix", type=str, default="real_txt_emb_")
    parser.add_argument("--fraud_img_prefix", type=str, default="fraud_aligned_")
    parser.add_argument("--real_img_prefix", type=str, default="real_aligned_")
    parser.add_argument("--label_col", type=str, default="label")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
    )

    # loss hyperparams (new!)
    parser.add_argument("--align_margin", type=float, default=0.69)
    parser.add_argument("--pos_pair_margin", type=float, default=0.95)
    parser.add_argument("--neg_pair_margin", type=float, default=0.90)
    parser.add_argument("--pair_weight", type=float, default=1.0)

    # misc
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    # -------------------------
    # Load data
    # -------------------------
    train_df = load_table(args.training_filepath)
    val_df = load_table(args.validate_filepath)

    # infer embedding dims
    text_dim = infer_dim_from_prefix(train_df, args.fraud_txt_prefix)
    img_dim = infer_dim_from_prefix(train_df, args.fraud_img_prefix)

    print(f"[INFO] text_dim={text_dim} | img_dim={img_dim}")

    train_ds = TextTeacherPairDataset(
        train_df,
        fraud_txt_prefix=args.fraud_txt_prefix,
        real_txt_prefix=args.real_txt_prefix,
        fraud_img_prefix=args.fraud_img_prefix,
        real_img_prefix=args.real_img_prefix,
        label_col=args.label_col,
    )
    val_ds = TextTeacherPairDataset(
        val_df,
        fraud_txt_prefix=args.fraud_txt_prefix,
        real_txt_prefix=args.real_txt_prefix,
        fraud_img_prefix=args.fraud_img_prefix,
        real_img_prefix=args.real_img_prefix,
        label_col=args.label_col,
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

    # -------------------------
    # Model
    # -------------------------
    model = SiameseEmbeddingModel(
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        image_dim=img_dim,
    ).to(device)

    criterion = ThesisMarginOnlyWithTeacherProj(
        align_margin=args.align_margin,
        pos_pair_margin=args.pos_pair_margin,
        neg_pair_margin=args.neg_pair_margin,
        pair_weight=args.pair_weight,
    ).to(device)

    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        args.lr,
        args.weight_decay,
    )

    trainer = Trainer(model, criterion, optimizer, device)

    # -------------------------
    # TRAIN (SAVE BEST BY VAL LOSS)
    # -------------------------
    trainer.train(
        dataloader=train_loader,
        validate_dataloader=val_loader,
        epochs=args.epochs,
        save_best=True,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
