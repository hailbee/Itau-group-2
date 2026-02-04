# text_to_image/trainer.py

import os
import torch
import torch.nn.functional as F


class Trainer:
    """
    Positive-only text → image trainer.

    Uses thesis-style margin loss:
        L(z_txt, z_img, y)

    Batch format:
        (txt, img, y)
    """

    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device

        from evaluator2 import Evaluator2, EvalConfig
        self.evaluator = Evaluator2(model, cfg=EvalConfig(batch_size=256))

        lr = self.optimizer.param_groups[0]["lr"]
        print(f"[DEBUG] Using fixed learning rate: {lr:.6f}")

    # -------------------------
    # Epoch loops
    # -------------------------
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            txt, img, y = batch

            txt = txt.to(self.device, non_blocking=True)
            img = img.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # ---- forward ----
            z_txt = self.model.encode_text(txt)
            z_img = self.model.encode_teacher(img)

            # ---- normalize ----
            z_txt = F.normalize(z_txt, dim=1)
            z_img = F.normalize(z_img, dim=1)

            # ---- loss ----
            loss = self.criterion(z_txt, z_img, y)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Step {i}/{len(dataloader)} | "
                    f"LR: {lr:.6f} | Loss: {loss.item():.10f}"
                )

        return epoch_loss / max(len(dataloader), 1)

    def validate_epoch(self, dataloader):
        if dataloader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                txt, img, y = batch

                txt = txt.to(self.device, non_blocking=True)
                img = img.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                z_txt = self.model.encode_text(txt)
                z_img = self.model.encode_teacher(img)

                z_txt = F.normalize(z_txt, dim=1)
                z_img = F.normalize(z_img, dim=1)

                loss = self.criterion(z_txt, z_img, y)
                epoch_loss += loss.item()

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Val Step {i}/{len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, test_filepath):
        self.model.eval()
        return self.evaluator.evaluate(test_filepath)

    # -------------------------
    # Training driver
    # -------------------------
    def train(
        self,
        dataloader,
        test_filepath=None,
        string="",
        epochs=30,
        validate_dataloader=None,
        early_stopping=True,
        patience=5,
        min_epochs=25,
        min_delta=0.0,
        relative_delta=False,
        save_best=True,
        save_dir="saved_models",
    ):
        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None

        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"best_model{string}.pt")
            print(f"[DEBUG] best_model_path={os.path.abspath(best_model_path)}")

        for epoch in range(int(epochs)):
            train_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.10f}")

            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is None:
                continue

            delta = best_val_loss - val_loss
            improved = (
                True if best_val_loss == float("inf")
                else (delta / max(best_val_loss, 1e-12) > min_delta)
                if relative_delta
                else (delta > min_delta)
            )

            if improved:
                best_val_loss = val_loss
                bad_epochs = 0

                if save_best:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "criterion_state": self.criterion.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        best_model_path,
                    )
                    print(
                        f"[DEBUG] Saved best checkpoint "
                        f"(val_loss={best_val_loss:.12f})"
                    )
            else:
                bad_epochs += 1

            if early_stopping and (epoch + 1) >= min_epochs and bad_epochs >= patience:
                print(f"[DEBUG] Early stopping at epoch {epoch + 1}")
                break

        if save_best and best_model_path and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.criterion.load_state_dict(ckpt["criterion_state"])
            print(f"[DEBUG] Restored best model from {best_model_path}")

        return {
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_model_path": best_model_path,
        }
