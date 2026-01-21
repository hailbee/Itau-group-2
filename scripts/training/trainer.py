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

    def train_epoch(self, dataloader, mode="pair"):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            x1, x2, y = batch
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y  = y.to(self.device, non_blocking=True)

            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    def validate_epoch(self, dataloader):
        """Validate for one epoch (loss only)."""
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
                    print(f"Val Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    def evaluate(self, test_filepath, plot=False, roc_png_path=None):
        """Run evaluator on a filepath."""
        self.model.eval()
        return self.evaluator.evaluate(test_filepath, plot=plot, roc_png_path=roc_png_path)

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

        # curriculum
        easy_loader=None,
        medium_loader=None,
        curriculum=None,

        # optional test eval at end of training
        want_test=False,

        # plotting
        plot_losses=True
    ):
        train_loss_history = []
        val_loss_history = []
        best_epoch_loss = float("inf")

        # -------------------------
        # Bandit setup (only if requested and feasible)
        # -------------------------
        datasets = None
        rewards = None
        pulls = None
        prev_val_loss_per_arm = None
        chosen = None

        if curriculum == "bandit":
            if validate_dataloader is None:
                print("[WARN][Bandit] validate_dataloader is None -> no reward signal. Falling back to manual.")
                curriculum = "manual"
            elif easy_loader is None or medium_loader is None:
                print("[WARN][Bandit] Bandit needs easy_loader AND medium_loader. Falling back to manual.")
                curriculum = "manual"
            else:
                datasets = {
                    "easy": easy_loader.dataset,
                    "medium": medium_loader.dataset,
                    "hard": dataloader.dataset
                }
                rewards = {k: [] for k in datasets.keys()}
                pulls = {k: 0 for k in datasets.keys()}
                prev_val_loss_per_arm = {k: None for k in datasets.keys()}

        for epoch in range(epochs):
            # -------------------------
            # Choose training loader
            # -------------------------
            if curriculum == "self" and easy_loader is not None and medium_loader is not None:
                print(f"[DEBUG][Self-Paced] Epoch {epoch+1}")

                ratios = get_curriculum_ratios(epoch, epochs)

                total_samples = len(dataloader.dataset)
                easy_n = int(ratios.get("easy", 0.0) * total_samples)
                medium_n = int(ratios.get("medium", 0.0) * total_samples)
                hard_n = int(ratios.get("hard", 1.0) * total_samples)

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

            elif curriculum == "bandit" and datasets is not None and rewards is not None:
                # bandit knobs
                reward_window = 5
                epsilon = max(0.05, 0.20 * (1 - epoch / max(epochs, 1)))  # decay

                # warm-start: try each arm at least once
                untried = [k for k, c in pulls.items() if c == 0]
                if len(untried) > 0:
                    chosen = random.choice(untried)
                    avg_rewards_dbg = {k: float("nan") for k in rewards.keys()}
                    print(f"[DEBUG][Bandit] Epoch {epoch+1}: chosen='{chosen}', eps={epsilon:.3f} (warm-start)")
                else:
                    avg_rewards = {
                        k: (float(np.mean(v[-reward_window:])) if len(v) > 0 else -1e9)
                        for k, v in rewards.items()
                    }

                    if random.random() < epsilon:
                        chosen = random.choice(list(datasets.keys()))
                    else:
                        chosen = max(avg_rewards, key=avg_rewards.get)

                    print(
                        f"[DEBUG][Bandit] Epoch {epoch+1}: chosen='{chosen}', eps={epsilon:.3f}, "
                        + ", ".join([f"{k}={avg_rewards[k]:.2e}" for k in avg_rewards])
                    )

                current_loader = self._make_loader_from_dataset(dataloader, datasets[chosen], shuffle=True)

            else:
                # Manual staged default: easy -> medium -> hard by thirds
                easy_epochs = max(1, int(0.10 * epochs))   # 10%
                med_epochs  = max(1, int(0.70 * epochs))   # 70%

                if epoch < easy_epochs and easy_loader is not None:
                    current_loader = easy_loader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using EASY dataset")
                elif epoch < easy_epochs + med_epochs and medium_loader is not None:
                    current_loader = medium_loader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using MEDIUM dataset")
                else:
                    current_loader = dataloader
                    print(f"[DEBUG][Manual Curriculum] Epoch {epoch+1}: Using HARD dataset")

            # -------------------------
            # TRAIN
            # -------------------------
            train_loss = self.train_epoch(current_loader, mode=mode)
            train_loss_history.append(train_loss)
            best_epoch_loss = min(best_epoch_loss, train_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f}")

            # -------------------------
            # VALIDATE
            # -------------------------
            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is not None:
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch+1} | Val Loss: {val_loss:.6f}")

            # -------------------------
            # BANDIT reward update (per-arm)
            # -------------------------
            if curriculum == "bandit" and rewards is not None and chosen is not None and val_loss is not None:
                prev = prev_val_loss_per_arm[chosen]
                reward = 0.0 if prev is None else float(prev - val_loss)  # positive is good
                rewards[chosen].append(reward)
                prev_val_loss_per_arm[chosen] = val_loss
                pulls[chosen] += 1

        # -------------------------
        # Plot losses
        # -------------------------
        if plot_losses:
            os.makedirs("images", exist_ok=True)
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

        # -------------------------
        # Optional test eval at end
        # -------------------------
        if want_test and test_filepath is not None:
            self.evaluate(
                test_filepath,
                plot=False,
                roc_png_path=f"images/roc_curve_trial_{trial_number}{string}.png"
            )

        return {
            "best_train_loss": best_epoch_loss,
            "final_train_loss": (train_loss_history[-1] if train_loss_history else None),
            "final_val_loss": (val_loss_history[-1] if val_loss_history else None),
        }
