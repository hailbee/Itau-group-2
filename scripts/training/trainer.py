# scripts/training/trainer.py

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

    # -------------------------
    # Epoch train/val
    # -------------------------
    def train_epoch(self, dataloader, mode="pair", grad_clip=1.0):
        self.model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            x1, x2, y = batch
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            # Guard against silent blowups
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

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
                y = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)
                epoch_loss += loss.item()

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Val Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    def evaluate(self, test_filepath, plot=False, roc_png_path=None):
        self.model.eval()
        return self.evaluator.evaluate(test_filepath, plot=plot, roc_png_path=roc_png_path)

    # -------------------------
    # Mixing helper
    # -------------------------
    def _make_loader_from_dataset(self, base_loader, dataset, shuffle=True):
        collate_fn = getattr(base_loader, "collate_fn", None)
        return DataLoader(
            dataset,
            batch_size=base_loader.batch_size,
            shuffle=shuffle,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            collate_fn=collate_fn,
        )

    def _make_mixed_loader(self, base_loader, easy_loader, medium_loader, hard_loader, ratios, total_samples=None):
        """
        Build a ConcatDataset loader using ratios over easy/medium/hard.
        total_samples controls the target dataset size per epoch.
        If None, uses len(base_loader.dataset) as baseline.
        """
        if total_samples is None:
            total_samples = len(base_loader.dataset)

        easy_n = int(ratios.get("easy", 0.0) * total_samples)
        medium_n = int(ratios.get("medium", 0.0) * total_samples)
        hard_n = int(ratios.get("hard", 0.0) * total_samples)

        # Clamp to dataset sizes
        easy_n = min(easy_n, len(easy_loader.dataset)) if easy_loader is not None else 0
        medium_n = min(medium_n, len(medium_loader.dataset)) if medium_loader is not None else 0
        hard_n = min(hard_n, len(hard_loader.dataset)) if hard_loader is not None else 0

        parts = []
        if easy_n > 0 and easy_loader is not None:
            idx = np.random.choice(len(easy_loader.dataset), easy_n, replace=False)
            parts.append(Subset(easy_loader.dataset, idx))
        if medium_n > 0 and medium_loader is not None:
            idx = np.random.choice(len(medium_loader.dataset), medium_n, replace=False)
            parts.append(Subset(medium_loader.dataset, idx))
        if hard_n > 0 and hard_loader is not None:
            idx = np.random.choice(len(hard_loader.dataset), hard_n, replace=False)
            parts.append(Subset(hard_loader.dataset, idx))

        if len(parts) == 0:
            return base_loader, 0, 0, 0

        mixed_dataset = ConcatDataset(parts)
        mixed_loader = self._make_loader_from_dataset(base_loader, mixed_dataset, shuffle=True)
        return mixed_loader, easy_n, medium_n, hard_n

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
        epochs=30,
        validate_filepath=None,       # kept for compatibility
        validate_dataloader=None,

        # curriculum additions
        easy_loader=None,
        medium_loader=None,
        curriculum=None,

        # optional test eval at end
        want_test=False,

        # plotting
        plot_losses=True,

        # stability knobs
        grad_clip=1.0,

        # early stopping + saving
        early_stopping=True,
        patience=5,
        min_epochs=20,          # <- NEW: do not early stop before this
        min_delta=1e-6,         # <- NEW: minimum improvement threshold for "best"
        save_best=True,
        save_dir="saved_models",

        # bandit smoothing
        bandit_ema_alpha=0.3,
    ):
        train_loss_history = []
        val_loss_history = []
        best_epoch_loss = float("inf")

        hard_loader = dataloader  # naming clarity
        epoch_budget = len(dataloader.dataset)  # fixed budget per epoch (keeps mixes comparable)

        # -------------------------
        # Early stopping state + checkpoint path
        # -------------------------
        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None
        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(
                save_dir,
                f"best_model_by_val_trial_{trial_number}{string}.pt"
            )

        # -------------------------
        # Bandit-over-mixtures setup
        # -------------------------
        bandit_enabled = (curriculum == "bandit")
        if bandit_enabled:
            if validate_dataloader is None:
                print("[WARN][Bandit] validate_dataloader is None -> no reward signal. Falling back to self.")
                bandit_enabled = False
                curriculum = "self"
            if easy_loader is None or medium_loader is None:
                print("[WARN][Bandit] Need easy_loader and medium_loader. Falling back to self.")
                bandit_enabled = False
                curriculum = "self"

        # Mixture arms (hard capped; removed 30% hard arm)
        mixture_arms = {
            "A_warm":   {"easy": 0.60, "medium": 0.40, "hard": 0.00},
            "B_main":   {"easy": 0.10, "medium": 0.80, "hard": 0.10},
            "D_medium": {"easy": 0.00, "medium": 1.00, "hard": 0.00},
            "E_mh15":   {"easy": 0.00, "medium": 0.85, "hard": 0.15},
            "F_mh5":    {"easy": 0.00, "medium": 0.95, "hard": 0.05},
        }

        rewards = {k: [] for k in mixture_arms.keys()} if bandit_enabled else None
        pulls = {k: 0 for k in mixture_arms.keys()} if bandit_enabled else None
        ema_val_loss_per_arm = {k: None for k in mixture_arms.keys()} if bandit_enabled else None
        chosen_arm = None

        for epoch in range(epochs):
            # -------------------------
            # Choose loader
            # -------------------------
            if curriculum == "self" and easy_loader is not None and medium_loader is not None:
                ratios = get_curriculum_ratios(epoch, epochs)
                current_loader, easy_n, med_n, hard_n = self._make_mixed_loader(
                    base_loader=dataloader,
                    easy_loader=easy_loader,
                    medium_loader=medium_loader,
                    hard_loader=hard_loader,
                    ratios=ratios,
                    total_samples=epoch_budget
                )
                print(
                    f"[DEBUG][Self] Epoch {epoch+1}: "
                    f"easy={easy_n} (r={ratios.get('easy',0):.2f}), "
                    f"medium={med_n} (r={ratios.get('medium',0):.2f}), "
                    f"hard={hard_n} (r={ratios.get('hard',0):.2f})"
                )

            elif bandit_enabled:
                reward_window = 5
                epsilon = max(0.05, 0.20 * (1 - epoch / max(epochs, 1)))  # decay exploration

                # warm-start: pull each arm once
                untried = [k for k, c in pulls.items() if c == 0]
                if len(untried) > 0:
                    chosen_arm = random.choice(untried)
                    print(f"[DEBUG][BanditMix] Epoch {epoch+1}: chosen='{chosen_arm}', eps={epsilon:.3f} (warm-start)")
                else:
                    avg_rewards = {
                        k: (float(np.mean(v[-reward_window:])) if len(v) > 0 else -1e9)
                        for k, v in rewards.items()
                    }
                    if random.random() < epsilon:
                        chosen_arm = random.choice(list(mixture_arms.keys()))
                    else:
                        chosen_arm = max(avg_rewards, key=avg_rewards.get)

                    print(
                        f"[DEBUG][BanditMix] Epoch {epoch+1}: chosen='{chosen_arm}', eps={epsilon:.3f}, "
                        + ", ".join([f"{k}={avg_rewards[k]:.2e}" for k in avg_rewards])
                    )

                ratios = mixture_arms[chosen_arm]
                current_loader, easy_n, med_n, hard_n = self._make_mixed_loader(
                    base_loader=dataloader,
                    easy_loader=easy_loader,
                    medium_loader=medium_loader,
                    hard_loader=hard_loader,
                    ratios=ratios,
                    total_samples=epoch_budget
                )
                print(
                    f"[DEBUG][BanditMix] Epoch {epoch+1}: arm='{chosen_arm}' mix -> "
                    f"easy={easy_n}, medium={med_n}, hard={hard_n}"
                )

            else:
                # Manual staged: 10% easy, 70% medium, 20% hard
                easy_epochs = max(1, int(0.10 * epochs))
                med_epochs = max(1, int(0.70 * epochs))

                if epoch < easy_epochs and easy_loader is not None:
                    current_loader = easy_loader
                    print(f"[DEBUG][Manual] Epoch {epoch+1}: EASY")
                elif epoch < easy_epochs + med_epochs and medium_loader is not None:
                    current_loader = medium_loader
                    print(f"[DEBUG][Manual] Epoch {epoch+1}: MEDIUM")
                else:
                    current_loader = dataloader
                    print(f"[DEBUG][Manual] Epoch {epoch+1}: HARD")

            # -------------------------
            # TRAIN
            # -------------------------
            train_loss = self.train_epoch(current_loader, mode=mode, grad_clip=grad_clip)
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

                # Save best model by val loss
                if save_best and val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    bad_epochs = 0

                    # Save a full checkpoint (model+optimizer+epoch)
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
                if early_stopping and (epoch + 1) >= min_epochs and bad_epochs >= patience:
                    print(f"[DEBUG] Early stopping at epoch {epoch+1} (best_val_loss={best_val_loss:.6f})")
                    break

            # -------------------------
            # BANDIT reward update (EMA-smoothed)
            # -------------------------
            if bandit_enabled and chosen_arm is not None and val_loss is not None:
                old_ema = ema_val_loss_per_arm[chosen_arm]
                new_ema = val_loss if old_ema is None else (bandit_ema_alpha * val_loss + (1 - bandit_ema_alpha) * old_ema)

                reward = 0.0 if old_ema is None else float(old_ema - new_ema)  # positive = improved
                rewards[chosen_arm].append(reward)
                ema_val_loss_per_arm[chosen_arm] = new_ema
                pulls[chosen_arm] += 1

        # -------------------------
        # Restore best checkpoint weights into memory (important!)
        # -------------------------
        if save_best and best_model_path is not None and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            print(f"[DEBUG] Restored best model into memory from {best_model_path} (best_val_loss={ckpt.get('best_val_loss')})")

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
            "best_val_loss": (best_val_loss if best_val_loss < float("inf") else None),
            "best_model_path": best_model_path,
        }
