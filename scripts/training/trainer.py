# scripts/training/trainer.py

import os
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from scripts.evaluation.evaluator import Evaluator


class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type

        self.model.to(device)

        # Keep evaluator consistent with your existing usage.
        # (Evaluator may optionally accept batch_size; we don't rely on it here.)
        self.evaluator = Evaluator(model, model_type=model_type)

        lr = optimizer.param_groups[0].get("lr", None)
        if lr is not None:
            print(f"[DEBUG] Using fixed learning rate: {lr:.6f}")

    # -------------------------
    # Epoch train/val (LOSS ONLY, fast)
    # -------------------------
    def train_epoch(self, dataloader, mode="pair", grad_clip: Optional[float] = 1.0) -> float:
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(dataloader):
            x1, x2, y = batch
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))

            self.optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0].get("lr", None)
                if lr is not None:
                    print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")
                else:
                    print(f"Step {i} / {len(dataloader)}")

        return epoch_loss / max(n_batches, 1)

    def validate_epoch(self, dataloader) -> Optional[float]:
        if dataloader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x1, x2, y = batch
                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)

                epoch_loss += float(loss.item())
                n_batches += 1

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0].get("lr", None)
                    if lr is not None:
                        print(f"Val Step {i} / {len(dataloader)} | LR: {lr:.6f}")
                    else:
                        print(f"Val Step {i} / {len(dataloader)}")

        return epoch_loss / max(n_batches, 1)

    def evaluate(self, test_filepath, plot: bool = False, roc_png_path: Optional[str] = None):
        self.model.eval()
        return self.evaluator.evaluate(test_filepath, plot=plot, roc_png_path=roc_png_path)

    # -------------------------
    # Accuracy evaluation helpers (uses cosine + Evaluator.compute_metrics)
    # -------------------------
    def _scores_from_loader(self, dataloader, max_batches: Optional[int] = None) -> pd.DataFrame:
        """
        Collect cosine similarity scores + labels from a dataloader yielding (x1, x2, y).
        Returns a DataFrame with columns: label, similarity
        """
        self.model.eval()

        sims_all = []
        y_all = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break

                x1, x2, y = batch
                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)

                # Match cosine-contrastive behavior: normalize then cosine (dot)
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                sims = (z1 * z2).sum(dim=1)

                sims_all.append(sims.detach().cpu())
                y_all.append(y.detach().cpu())

        if len(sims_all) == 0:
            return pd.DataFrame({"label": [], "similarity": []})

        similarities = torch.cat(sims_all, dim=0).numpy()
        labels = torch.cat(y_all, dim=0).numpy().astype(int)

        return pd.DataFrame({"label": labels, "similarity": similarities})

    def _accuracy_at_threshold(self, results_df: pd.DataFrame, threshold: float) -> float:
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df["similarity"].astype(float).to_numpy()
        y_pred = (y_scores >= float(threshold)).astype(int)
        return float((y_pred == y_true).mean())

    # -------------------------
    # Main training loop
    # -------------------------
    def train(
        self,
        dataloader,
        trial_number,
        test_filepath,
        string,
        mode="pair",
        epochs: int = 30,
        validate_filepath=None,       # kept for compatibility
        validate_dataloader=None,

        # optional test eval at end
        want_test: bool = False,

        # plotting (loss)
        plot_losses: bool = True,

        # stability knobs
        grad_clip: Optional[float] = 1.0,

        # early stopping + saving
        early_stopping: bool = True,
        patience: int = 5,
        min_epochs: int = 25,
        min_delta: float = 1e-6,
        save_best: bool = True,
        save_dir: str = "saved_models",

        # NEW: make plot output location consistent
        plot_dir: Optional[str] = None,

        # -------------------------
        # QUICK SWITCH (accuracy curves + Youden eval)
        # -------------------------
        plot_accuracy: bool = False,
        accuracy_eval_every: int = 1,
        accuracy_max_train_batches: Optional[int] = 50,
        accuracy_max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        train_loss_history = []
        val_loss_history = []
        best_epoch_loss = float("inf")

        # accuracy histories (only used when plot_accuracy=True)
        acc_epochs = []
        train_acc_history = []
        val_acc_history = []
        youden_thr_history = []
        val_auc_history = []

        # Directories
        os.makedirs(save_dir, exist_ok=True)
        if plot_dir is None:
            plot_dir = os.path.join(save_dir, "images")
        os.makedirs(plot_dir, exist_ok=True)

        # -------------------------
        # Early stopping state + checkpoint path
        # -------------------------
        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None

        if save_best:
            best_model_path = os.path.join(
                save_dir,
                f"best_model_by_val_trial_{trial_number}{string}.pt"
            )

        for epoch in range(int(epochs)):
            # -------------------------
            # TRAIN (loss only)
            # -------------------------
            train_loss = self.train_epoch(dataloader, mode=mode, grad_clip=grad_clip)
            train_loss_history.append(train_loss)
            best_epoch_loss = min(best_epoch_loss, train_loss)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f}")

            # -------------------------
            # VALIDATE (loss only)
            # -------------------------
            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is not None:
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.6f}")

                # Save best model by val loss (default behavior)
                if save_best and best_model_path is not None and val_loss < best_val_loss - float(min_delta):
                    best_val_loss = float(val_loss)
                    bad_epochs = 0

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        best_model_path
                    )
                    print(f"[DEBUG] Saved best checkpoint (val_loss={best_val_loss:.6f}) -> {best_model_path}")
                else:
                    bad_epochs += 1

                # Early stopping only after min_epochs
                if early_stopping and (epoch + 1) >= int(min_epochs) and bad_epochs >= int(patience):
                    print(f"[DEBUG] Early stopping at epoch {epoch + 1} (best_val_loss={best_val_loss:.6f})")
                    break

            # -------------------------
            # OPTIONAL: Accuracy/ROC tracking (slower)
            # Uses Youden threshold from VAL each time,
            # then applies that threshold to TRAIN for a fair comparison.
            # -------------------------
            if (
                plot_accuracy
                and validate_dataloader is not None
                and (epoch + 1) % int(max(1, accuracy_eval_every)) == 0
            ):
                val_df = self._scores_from_loader(validate_dataloader, max_batches=accuracy_max_val_batches)
                if len(val_df) > 0:
                    val_metrics = self.evaluator.compute_metrics(val_df, plot=False)
                    youden_thr = float(val_metrics["youden_threshold"])
                    val_acc = float(val_metrics["accuracy"])
                    val_auc = float(val_metrics["roc_auc"])

                    train_df = self._scores_from_loader(dataloader, max_batches=accuracy_max_train_batches)
                    train_acc = self._accuracy_at_threshold(train_df, youden_thr) if len(train_df) > 0 else None

                    acc_epochs.append(epoch + 1)
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    youden_thr_history.append(youden_thr)
                    val_auc_history.append(val_auc)

                    if train_acc is not None:
                        print(
                            f"[ACC] Epoch {epoch+1} | Train Acc (Youden@val): {train_acc:.4f} | "
                            f"Val Acc (Youden): {val_acc:.4f} | Thr: {youden_thr:.4f} | AUC: {val_auc:.4f}"
                        )
                    else:
                        print(
                            f"[ACC] Epoch {epoch+1} | Val Acc (Youden): {val_acc:.4f} | "
                            f"Thr: {youden_thr:.4f} | AUC: {val_auc:.4f}"
                        )

        # -------------------------
        # Restore best checkpoint weights into memory (if we saved one)
        # -------------------------
        if save_best and best_model_path is not None and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            print(
                f"[DEBUG] Restored best model into memory from {best_model_path} "
                f"(best_val_loss={ckpt.get('best_val_loss')})"
            )

        # -------------------------
        # Plot losses
        # -------------------------
        if plot_losses:
            plot_path = os.path.join(plot_dir, f"loss_curve_trial_{trial_number}{string}.png")
            plt.figure()
            plt.plot(train_loss_history, label="Train Loss")
            if len(val_loss_history) > 0:
                plt.plot(val_loss_history, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training / Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"[DEBUG] Saved loss curve to {plot_path}")

        # -------------------------
        # Plot accuracies (optional)
        # -------------------------
        if plot_accuracy and len(acc_epochs) > 0:
            acc_plot_path = os.path.join(plot_dir, f"acc_curve_trial_{trial_number}{string}.png")
            plt.figure()
            if any(v is not None for v in train_acc_history):
                plt.plot(acc_epochs, train_acc_history, label="Train Acc (Youden@val)")
            plt.plot(acc_epochs, val_acc_history, label="Val Acc (Youden)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training / Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(acc_plot_path)
            plt.close()
            print(f"[DEBUG] Saved accuracy curve to {acc_plot_path}")

        # -------------------------
        # Optional test eval at end
        # -------------------------
        if want_test and test_filepath is not None:
            self.evaluate(
                test_filepath,
                plot=False,
                roc_png_path=os.path.join(plot_dir, f"roc_curve_trial_{trial_number}{string}.png"),
            )

        return {
            "best_train_loss": best_epoch_loss,
            "final_train_loss": (train_loss_history[-1] if train_loss_history else None),
            "final_val_loss": (val_loss_history[-1] if val_loss_history else None),
            "best_val_loss": (best_val_loss if best_val_loss < float("inf") else None),
            "best_model_path": best_model_path,

            # Optional accuracy outputs (only populated if plot_accuracy=True)
            "final_train_acc_youden_at_val": (train_acc_history[-1] if train_acc_history else None),
            "final_val_acc_youden": (val_acc_history[-1] if val_acc_history else None),
            "final_val_youden_threshold": (youden_thr_history[-1] if youden_thr_history else None),
            "final_val_roc_auc": (val_auc_history[-1] if val_auc_history else None),
        }


"""
USAGE EXAMPLE (minimal)

from scripts.training.trainer import Trainer
from model_utils.models.learning.siamese import SiameseEmbeddingModel
from model_utils.loss.pair_losses import ContrastiveLoss  # your cosine-hinge contrastive
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseEmbeddingModel(embedding_dim=768, hidden_dim=256, out_dim=128).to(device)
criterion = ContrastiveLoss(m_pos=0.92, m_neg=0.84, w_pos=1.0, w_neg=3.0)  # or your one-margin variant
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

trainer = Trainer(model, criterion, optimizer, device, model_type="pair")
# trainer.train(dataloader=train_loader, trial_number=1, test_filepath="...", string="_debug", epochs=5, validate_dataloader=val_loader)
"""
