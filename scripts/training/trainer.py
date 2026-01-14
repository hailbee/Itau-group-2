import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from scripts.evaluation.evaluator import Evaluator
from utils.curriculum import get_curriculum_ratios


class Trainer:
    def __init__(self, model, criterion, optimizer, device, log_csv_path="training_log.csv", model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model_type = model_type
        self.model.to(device)
        self.evaluator = Evaluator(model, model_type=model_type)

        print(f"[DEBUG] Using fixed learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def train_epoch(self, dataloader, mode="pair", track_pg=False, epoch_num=None):
        self.model.train()
        epoch_loss = 0.0
        total_pg = 0.0
        pg_count = 0

        for i, batch in enumerate(dataloader):

            if mode == "triplet":
                anchor, positive, negative = batch
                z = self.model(anchor, positive, negative)
                loss = self.criterion(*z)

            elif mode in {"supcon", "infonce"}:
                anchor, positive, negative = batch
                z = self.model(anchor, positive, negative)
                loss = self.criterion(*z)

            else:
                x1, x2, y = batch
                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if track_pg:
                with torch.no_grad():
                    if mode == "pair":
                        z1a, z2a = self.model(x1, x2)
                        loss_after = self.criterion(z1a, z2a, y)
                    else:
                        z_after = self.model(*batch)
                        loss_after = self.criterion(*z_after)

                    total_pg += (loss.item() - loss_after.item())
                    pg_count += 1

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        avg_pg = total_pg / pg_count if track_pg and pg_count > 0 else None
        return epoch_loss / len(dataloader), avg_pg

    def evaluate(self, test_filepath):
        self.model.eval()
        _, metrics = self.evaluator.evaluate(test_filepath)
        return metrics

    def train(
        self,
        dataloader,
        test_filepath,
        mode="pair",
        epochs=30,
        medium_loader=None,
        easy_loader=None,
        curriculum=None,
        validate_filepath=None,
    ):
        best_epoch_loss = float("inf")
        best_val_metrics = None
        best_val_epoch = -1
        halfway_epoch = (epochs - 1) // 2

        if medium_loader is not None and easy_loader is not None:
            datasets = {
                "easy": easy_loader.dataset,
                "medium": medium_loader.dataset,
                "hard": dataloader.dataset,
            }
            rewards = {k: [] for k in datasets}
        else:
            datasets = {}
            rewards = {}

        for epoch in range(epochs):

            if curriculum == "self" and medium_loader and easy_loader:
                ratios = get_curriculum_ratios(epoch, epochs)
                total = len(dataloader.dataset)

                easy_n = int(ratios["easy"] * total)
                med_n = int(ratios["medium"] * total)
                hard_n = int(ratios["hard"] * total)

                mixed = ConcatDataset([
                    Subset(easy_loader.dataset, np.random.choice(len(easy_loader.dataset), easy_n, replace=False)),
                    Subset(medium_loader.dataset, np.random.choice(len(medium_loader.dataset), med_n, replace=False)),
                    Subset(dataloader.dataset, np.random.choice(len(dataloader.dataset), hard_n, replace=False)),
                ])

                current_loader = DataLoader(
                    mixed,
                    batch_size=dataloader.batch_size,
                    shuffle=True,
                    collate_fn=getattr(dataloader, "collate_fn", None),
                )

            elif curriculum == "bandit" and medium_loader and easy_loader:
                epsilon = 0.2
                reward_window = 2

                avg_rewards = {
                    k: np.mean(v[-reward_window:]) if v else 0.0
                    for k, v in rewards.items()
                }

                if random.random() < epsilon:
                    chosen = random.choice(list(datasets.keys()))
                else:
                    chosen = max(avg_rewards, key=avg_rewards.get)

                current_loader = DataLoader(
                    datasets[chosen],
                    batch_size=dataloader.batch_size,
                    shuffle=True,
                    collate_fn=getattr(dataloader, "collate_fn", None),
                )

            else:
                phase = epochs // 3
                if epoch < phase and easy_loader:
                    current_loader = easy_loader
                elif epoch < 2 * phase and medium_loader:
                    current_loader = medium_loader
                else:
                    current_loader = dataloader

            avg_loss, avg_pg = self.train_epoch(
                current_loader,
                mode=mode,
                track_pg=(curriculum == "bandit"),
                epoch_num=epoch + 1,
            )

            print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss

            if curriculum == "bandit" and avg_pg is not None:
                rewards[chosen].append(avg_pg)

            if validate_filepath and (epoch == halfway_epoch or epoch == epochs - 1):
                val_metrics = self.evaluator.evaluate(validate_filepath)[1]

                if best_val_metrics is None or val_metrics["roc_auc"] > best_val_metrics["roc_auc"]:
                    best_val_metrics = val_metrics
                    best_val_epoch = epoch + 1

                print(
                    f"Epoch {epoch + 1} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Prec: {val_metrics['precision']:.4f} | "
                    f"Recall: {val_metrics['recall']:.4f} | "
                    f"AUC: {val_metrics['roc_auc']:.4f}"
                )

        results = {
            "loss": best_epoch_loss,
        }

        if best_val_metrics:
            results.update({
                "val_auc": best_val_metrics["roc_auc"],
                "val_accuracy": best_val_metrics["accuracy"],
                "val_precision": best_val_metrics["precision"],
                "val_recall": best_val_metrics["recall"],
                "val_epoch": best_val_epoch,
            })

        return results
