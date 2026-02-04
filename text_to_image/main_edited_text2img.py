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

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--margin", type=float, default=1.0)

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

    # -------------------------
    # Model
    # -------------------------
    text_dim = infer_dim_from_prefix(train_df, "left_txt_emb_")
    img_dim = infer_dim_from_prefix(train_df, "right_img_emb_")

    print(f"[INFO] text_dim={text_dim} | img_dim={img_dim}")

    model = SiameseEmbeddingModel(
        embedding_dim=text_dim,
        hidden_dim=args.hidden_dim,
        out_dim=img_dim,   # IMPORTANT: must match image dim
    ).to(device)

    criterion = ThesisMarginOnlyWithTeacherProj(margin=args.margin).to(device)

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
        validate_dataloader=val_loader,   # ✅ THIS IS THE KEY LINE
        epochs=args.epochs,
        save_best=True,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
