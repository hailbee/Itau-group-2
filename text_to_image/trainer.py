# text_to_image/trainer.py

import os
import torch
import torch.nn.functional as F


class Trainer:
    """
    Trainer for 4-embedding dataset format.

    Batch format:
        (fraud_txt, real_txt, fraud_img, real_img, y)

    Model:
        encode_text(x) -> projected embedding

    Criterion signature:
        loss(z_fraud_txt, z_real_txt, fraud_img, real_img, y)
    """

    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        lr = self.optimizer.param_groups[0]["lr"]
        print(f"[DEBUG] Using learning rate: {lr:.6f}")

    # -------------------------
    # Epoch loops
    # -------------------------
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for step, (fraud_txt, real_txt, fraud_img, real_img, y) in enumerate(dataloader):
            fraud_txt = fraud_txt.to(self.device)
            real_txt = real_txt.to(self.device)
            fraud_img = fraud_img.to(self.device)
            real_img = real_img.to(self.device)
            y = y.to(self.device)

            # Encode fraud + real text
            z_fraud = self.model.encode_text(fraud_txt)
            z_real = self.model.encode_text(real_txt)

            # Normalize everything
            z_fraud = F.normalize(z_fraud, dim=1)
            z_real = F.normalize(z_real, dim=1)

            fraud_img = F.normalize(fraud_img, dim=1)
            real_img = F.normalize(real_img, dim=1)

            # Loss
            loss = self.criterion(z_fraud, z_real, fraud_img, real_img, y)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Step {step}/{len(dataloader)} | "
                    f"Train Loss: {loss.item():.8f}"
                )

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        for fraud_txt, real_txt, fraud_img, real_img, y in dataloader:
            fraud_txt = fraud_txt.to(self.device)
            real_txt = real_txt.to(self.device)
            fraud_img = fraud_img.to(self.device)
            real_img = real_img.to(self.device)
            y = y.to(self.device)

            z_fraud = self.model.encode_text(fraud_txt)
            z_real = self.model.encode_text(real_txt)

            z_fraud = F.normalize(z_fraud, dim=1)
            z_real = F.normalize(z_real, dim=1)

            fraud_img = F.normalize(fraud_img, dim=1)
            real_img = F.normalize(real_img, dim=1)

            loss = self.criterion(z_fraud, z_real, fraud_img, real_img, y)
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    # -------------------------
    # Training driver
    # -------------------------
    def train(
        self,
        dataloader,
        *,
        validate_dataloader=None,
        epochs=30,
        save_best=True,
        save_dir="saved_models",
        min_epochs=25,
        patience=5,
    ):
        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None

        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, "best_model.pt")
            print(f"[INFO] Best model will be saved to: {best_model_path}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.8f}")

            if validate_dataloader is None:
                continue

            val_loss = self.validate_epoch(validate_dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | Val Loss:   {val_loss:.8f}")

            # Improvement check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0

                if save_best:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        best_model_path,
                    )
                    print(
                        f"[INFO] Saved new best model "
                        f"(val_loss={best_val_loss:.8f})"
                    )
            else:
                bad_epochs += 1

            # Early stopping
            if (epoch + 1) >= min_epochs and bad_epochs >= patience:
                print(
                    f"[INFO] Early stopping triggered at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        return {
            "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
            "best_model_path": best_model_path,
        }
