from scripts.evaluation.evaluator import Evaluator
import torch
import os
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, criterion, optimizer, device,
                 log_csv_path="training_log.csv", model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model_type = model_type

        self.model.to(device)
        self.evaluator = Evaluator(model, model_type=model_type)

        print(f"[DEBUG] Using fixed learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def train_epoch(self, dataloader, mode="pair"):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            x1, x2, y = batch
            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def evaluate_loss(self, dataloader, mode="pair"):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x1, x2, y = batch
                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        dataloader,
        test_filepath,
        mode="pair",
        epochs=30,
        validate_filepath=None,
        validate_dataloader=None
    ):
        train_loss_history = []
        val_loss_history = []

        best_epoch_loss = float("inf")

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, mode=mode)
            train_loss_history.append(avg_loss)

            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")

            best_epoch_loss = min(best_epoch_loss, avg_loss)

            val_loss = None

            val_loss = self.evaluate_loss(validate_dataloader, mode)
            val_loss_history.append(val_loss)

        # -------- SAVE LOSS GRAPH --------
        if self.log_csv_path is not None:
            plot_path = self.log_csv_path.replace(".csv", "_loss.png")

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

        return {
            "best_train_loss": best_epoch_loss,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        }
