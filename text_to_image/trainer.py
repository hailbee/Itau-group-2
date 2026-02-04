import os
import torch
import torch.nn.functional as F


class Trainer:
    """
    Positive-only text → image trainer.

    Uses margin-based contrastive loss:
        L(z_txt, z_img, y)

    Batch format:
        (txt, img, y)

    - Text embeddings are learned
    - Image embeddings are fixed
    - Best model is saved by LOWEST validation loss
    """

    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device

        lr = self.optimizer.param_groups[0]["lr"]
        print(f"[DEBUG] Using fixed learning rate: {lr:.6f}")

    # -------------------------
    # Train one epoch
    # -------------------------
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for step, (txt, img, y) in enumerate(dataloader):
            txt = txt.to(self.device, non_blocking=True)
            img = img.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Encode text
            z_txt = self.model.encode_text(txt)

            # Normalize
            z_txt = F.normalize(z_txt, dim=1)
            img = F.normalize(img, dim=1)

            # Loss
            loss = self.criterion(z_txt, img, y)

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

    # -------------------------
    # Validate one epoch
    # -------------------------
    @torch.no_grad()
    def validate_epoch(self, dataloader):
        if dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        for step, (txt, img, y) in enumerate(dataloader):
            txt = txt.to(self.device, non_blocking=True)
            img = img.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            z_txt = self.model.encode_text(txt)

            z_txt = F.normalize(z_txt, dim=1)
            img = F.normalize(img, dim=1)

            loss = self.criterion(z_txt, img, y)
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    # -------------------------
    # Training driver
    # -------------------------
    def train(
        self,
        dataloader,
        validate_dataloader=None,
        epochs=30,
        save_best=True,
        save_dir="saved_models",
        min_delta=0.0,
    ):
        best_val_loss = float("inf")
        best_model_path = None

        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, "best_model.pt")
            print(f"[DEBUG] Best model path: {os.path.abspath(best_model_path)}")

        for epoch in range(int(epochs)):
            train_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.8f}")

            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is None:
                continue

            print(f"Epoch {epoch + 1}/{epochs} | Val Loss:   {val_loss:.8f}")

            # Check improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss

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
                        f"[DEBUG] Saved best model "
                        f"(val_loss={best_val_loss:.8f})"
                    )

        # Restore best model
        if save_best and best_model_path and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"[DEBUG] Restored best model from {best_model_path}")

        return {
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_model_path": best_model_path,
        }
