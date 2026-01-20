from scripts.evaluation.evaluator import Evaluator
import torch
import os
import matplotlib.pyplot as plt

import numpy as np
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.curriculum import get_curriculum_ratios


class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type

        self.model.to(device)
        self.evaluator = Evaluator(model, model_type=model_type)

        print(f"[DEBUG] Using fixed learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def train_epoch(self, dataloader, mode="pair", track_pg=False):
        """
        Train for one epoch.
        If track_pg=True, compute a simple "progress gain" signal:
        loss_before_step - loss_after_step, averaged across batches.
        """
        self.model.train()
        epoch_loss = 0.0

        total_pg = 0.0
        pg_count = 0

        for i, batch in enumerate(dataloader):
            x1, x2, y = batch

            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y  = y.to(self.device, non_blocking=True)

            # forward + loss (before update)
            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if track_pg:
                # progress gain: how much the loss decreases after the update
                with torch.no_grad():
                    z1a, z2a = self.model(x1, x2)
                    loss_after = self.criterion(z1a, z2a, y)
                    total_pg += (loss.item() - loss_after.item())
                    pg_count += 1

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        avg_pg = (total_pg / pg_count) if (track_pg and pg_count > 0) else None
        return avg_loss, avg_pg

    def validate_epoch(self, dataloader):
        if dataloader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x1, x2, y = batch

                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                y  = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)
                epoch_loss += loss.item()

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    def evaluate(self, test_filepath):
        self.model.eval()
        _, metrics = self.evaluator.evaluate(test_filepath)
        return metrics

    def _make_loader_from_dataset(self, base_loader, dataset, shuffle=True):
        """Create a DataLoader that matches base_loader settings as closely as possible."""
        collate_fn = getattr(base_loader, "collate_fn", None)
        return DataLoader(
            dataset,
            batch_size=base_loader.batch_size,
            shuffle=shuffle,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            collate_fn=collate_fn
        )

    def train(
        self,
        dataloader,
        trial_number,
        test_filepath,
        string,
        mode="pair",
        epochs=30,
        validate_filepath=None,      # kept for compatibility; not used directly here
        validate_dataloader=None,

        # --- curriculum additions ---
        easy_loader=None,
        medium_loader=None,
        curriculum=None
    ):
        train_loss_history = []
        val_loss_history = []

        best_epoch_loss = float("inf")

        # For bandit curriculum
        rewards = None
        datasets = None
        chosen = None

        if easy_loader is not None and medium_loader is not None:
            datasets = {
                "easy": easy_loader.dataset,
                "medium": medium_loader.dataset,
                "hard": dataloader.dataset
            }
            rewards = {k: [] for k in datasets.keys()}

        for epoch in range(epochs):
            # -------------------------
            # Choose training loader
            # -------------------------
            track_pg = False

            # Self-paced: mix easy/medium/hard each epoch using ratios
            if curriculum == "self" and easy_loader is not None and medium_loader is not None:
                print(f"[DEBUG][Self-Paced] Epoch {epoch+1}")

                ratios = get_curriculum_ratios(epoch, epochs)

                total_samples = len(dataloader.dataset)
                easy_n = int(ratios.get("easy", 0.0) * total_samples)
                medium_n = int(ratios.get("medium", 0.0) * total_samples)
                hard_n = int(ratios.get("hard", 1.0) * total_samples)

                # Clamp to dataset sizes (avoid replace=False errors)
                easy_n = min(easy_n, len(easy_loader.dataset))
                medium_n = min(medium_n, len(medium_loader.dataset))
                hard_n = min(hard_n, len(dataloader.dataset))

                easy_idx = np.random.choice(len(easy_loader.dataset), easy_n, replace=False) if easy_n > 0 else np.array([], dtype=int)
                med_idx  = np.random.choice(len(medium_loader.dataset), medium_n, replace=False) if medium_n > 0 else np.array([], dtype=int)
                hard_idx = np.random.choice(len(dataloader.dataset), hard_n, replace=False) if hard_n > 0 else np.array([], dtype=int)

                mixed = []
                if easy_n > 0:
                    mixed.append(Subset(easy_loader.dataset, easy_idx))
                if medium_n > 0:
                    mixed.append(Subset(medium_loader.dataset, med_idx))
                if hard_n > 0:
                    mixed.append(Subset(dataloader.dataset, hard_idx))

                if len(mixed) == 0:
                    current_loader = dataloader
                else:
                    mixed_dataset = ConcatDataset(mixed)
                    current_loader = self._make_loader_from_dataset(dataloader, mixed_dataset, shuffle=True)

                print(
                    f"[DEBUG][Self-Paced] Epoch {epoch+1}: "
                    f"easy={easy_n} (ratio={ratios.get('easy', 0):.3f}), "
                    f"medium={medium_n} (ratio={ratios.get('medium', 0):.3f}), "
                    f"hard={hard_n} (ratio={ratios.get('hard', 0):.3f}), "
                    f"total={easy_n+medium_n+hard_n}"
                )

            # Bandit: choose one dataset per epoch based on recent progress gain
            elif curriculum == "bandit" and datasets is not None and rewards is not None:
                epsilon = 0.2
                reward_window = 2
                track_pg = True

                avg_rewards = {
                    k: (float(np.mean(v[-reward_window:])) if len(v) > 0 else 0.0)
                    for k, v in rewards.items()
                }

                if random.random() < epsilon:
                    chosen = random.choice(list(datasets.keys()))
                else:
                    chosen = max(avg_rewards, key=avg_rewards.get)

                current_loader = self._make_loader_from_dataset(dataloader, datasets[chosen], shuffle=True)
                print(
                    f"[DEBUG][Bandit] Epoch {epoch+1}: chosen='{chosen}', "
                    + ", ".join([f"{k}={avg_rewards[k]:.4f}" for k in avg_rewards])
                )

            # Manual (staged): easy -> medium -> hard by thirds
            else:
                phase_len = max(epochs // 3, 1)
                if epoch < phase_len and easy_loader is not None:
                    current_loader = easy_loader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using EASY dataset")
                elif epoch < 2 * phase_len and medium_loader is not None:
                    current_loader = medium_loader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using MEDIUM dataset")
                else:
                    current_loader = dataloader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using HARD dataset")

            # -------------------------
            # TRAIN
            # -------------------------
            avg_loss, avg_pg = self.train_epoch(current_loader, mode=mode, track_pg=track_pg)
            train_loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")
            best_epoch_loss = min(best_epoch_loss, avg_loss)

            # Store bandit reward
            if curriculum == "bandit" and avg_pg is not None and rewards is not None and chosen is not None:
                rewards[chosen].append(avg_pg)

            # -------------------------
            # VALIDATE (loss only)
            # -------------------------
            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is not None:
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        # Ensure output dir exists for plots
        os.makedirs("images", exist_ok=True)

        # -------- SAVE LOSS GRAPH --------
        plot_path = f"images/loss_curve_trial_{trial_number}{string}.png"
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

        # -------- EVALUATE + SAVE ROC CURVE --------
        test_metrics = None
        if test_filepath is not None:
            _, test_metrics = self.evaluator.evaluate(
                test_filepath,
                plot=True,
                roc_png_path=f"images/roc_curve_trial_{trial_number}{string}.png"
            )

        # Return small dict for Optuna
        return {
            "best_train_loss": best_epoch_loss,
            "roc_auc": (test_metrics["roc_auc"] if test_metrics and "roc_auc" in test_metrics else None),
        }
