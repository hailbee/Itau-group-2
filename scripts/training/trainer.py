from scripts.evaluation.evaluator import Evaluator
import torch
import os
import matplotlib.pyplot as plt


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
        self.model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            x1, x2, y = batch
            z1, z2 = self.model(x1, x2)
            loss = self.criterion(z1, z2, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x1, x2, y = batch
                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)
                epoch_loss += loss.item()

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / len(dataloader)


    def train(
        self,
        dataloader,
        trial_number,
        test_filepath,
        string,
        mode="pair",
        epochs=30,
        validate_filepath=None,
        validate_dataloader=None
    ):
        train_loss_history = []
        val_loss_history = []

        best_epoch_loss = float("inf")

        for epoch in range(epochs):
            # ---- TRAIN ----
            avg_loss = self.train_epoch(dataloader, mode=mode)
            train_loss_history.append(avg_loss)

            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")

            best_epoch_loss = min(best_epoch_loss, avg_loss)

            # ---- VALIDATE ----
            val_loss = self.validate_epoch(validate_dataloader)
            val_loss_history.append(val_loss)

            print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

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

        return {
            "best_train_loss": best_epoch_loss,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        }
