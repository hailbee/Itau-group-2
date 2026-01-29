# scripts/training/trainer.py

import os
import torch
import matplotlib.pyplot as plt


class Trainer:
    """
    SINGLE MODE TRAINER

    Saves best checkpoint by validation loss (robust to tiny losses like 1e-6).
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
        self.criterion.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            fraud_txt, real_txt, fraud_teacher, real_teacher, y = batch

            fraud_txt = fraud_txt.to(self.device, non_blocking=True)
            real_txt = real_txt.to(self.device, non_blocking=True)
            fraud_teacher = fraud_teacher.to(self.device, non_blocking=True)
            real_teacher = real_teacher.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred_fraud, pred_real = self.model(fraud_txt, real_txt)
            loss = self.criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, y)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            epoch_loss += float(loss.item())

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
        self.criterion.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                fraud_txt, real_txt, fraud_teacher, real_teacher, y = batch

                fraud_txt = fraud_txt.to(self.device, non_blocking=True)
                real_txt = real_txt.to(self.device, non_blocking=True)
                fraud_teacher = fraud_teacher.to(self.device, non_blocking=True)
                real_teacher = real_teacher.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                pred_fraud, pred_real = self.model(fraud_txt, real_txt)
                loss = self.criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, y)

                epoch_loss += float(loss.item())

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
        trial_number,
        test_filepath=None,
        string="",
        epochs=30,
        validate_dataloader=None,
        want_test=False,
        plot_losses=True,
        early_stopping=True,
        patience=5,
        min_epochs=25,
        min_delta=0.0,          # ✅ FIX: default to saving on *any* improvement
        relative_delta=False,   # optional: use relative improvement threshold instead of absolute
        save_best=True,
        save_dir="saved_models",
        eval_every=None,
    ):
        """
        min_delta:
          - if relative_delta=False: absolute improvement needed (best - val_loss > min_delta)
          - if relative_delta=True: relative improvement needed ((best - val_loss)/max(best,eps) > min_delta)

        With tiny losses (~1e-6), you almost always want min_delta=0.0 or ~1e-9.
        """

        train_loss_history = []
        val_loss_history = []

        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None

        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"best_model_trial_{trial_number}{string}.pt")
            print(f"[DEBUG] best_model_path={os.path.abspath(best_model_path)}")

        for epoch in range(int(epochs)):
            train_loss = self.train_epoch(dataloader)
            train_loss_history.append(train_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.10f}")

            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is not None:
                val_loss_history.append(val_loss)

                # ---- robust improvement check ----
                if best_val_loss == float("inf"):
                    delta = float("inf")
                    improved = True
                else:
                    delta = best_val_loss - val_loss
                    if relative_delta:
                        denom = max(best_val_loss, 1e-12)
                        improved = (delta / denom) > float(min_delta)
                    else:
                        improved = delta > float(min_delta)

                print(
                    f"[VAL] epoch={epoch+1} "
                    f"val_loss={val_loss:.12f} "
                    f"best_val_loss={best_val_loss:.12f} "
                    f"delta={delta:.3e} "
                    f"min_delta={float(min_delta):.3e} "
                    f"relative={relative_delta} "
                    f"improved={improved}"
                )

                if save_best and improved:
                    best_val_loss = val_loss
                    bad_epochs = 0

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
                        f"(val_loss={best_val_loss:.12f}) -> {best_model_path}"
                    )
                else:
                    bad_epochs += 1

                if early_stopping and (epoch + 1) >= int(min_epochs) and bad_epochs >= int(patience):
                    print(
                        f"[DEBUG] Early stopping at epoch {epoch+1} "
                        f"(best_val_loss={best_val_loss:.12f})"
                    )
                    break

            # optional periodic evaluation (not required)
            if eval_every is not None and test_filepath is not None:
                if (epoch + 1) % int(eval_every) == 0:
                    print(f"[EVAL] Running evaluation at epoch {epoch+1}")
                    _, metrics = self.evaluate(test_filepath)

                    s_auc = metrics["student"]["roc_auc"]
                    r_auc = metrics["raw_text"]["roc_auc"]
                    t_auc = metrics["teacher"]["roc_auc"]

                    print(
                        f"[EVAL] Epoch {epoch+1} | "
                        f"Student AUC: {s_auc:.4f} | "
                        f"Raw Text AUC: {r_auc:.4f} | "
                        f"Teacher AUC: {t_auc:.4f}"
                    )

        # Restore best model
        if save_best and best_model_path and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            if "criterion_state" in ckpt:
                self.criterion.load_state_dict(ckpt["criterion_state"])
            self.model.to(self.device)
            self.criterion.to(self.device)
            print(f"[DEBUG] Restored best model from {best_model_path}")
        else:
            print("[WARN] No best checkpoint found on disk. Check save_dir/min_delta/validate_dataloader.")

        # Plot losses
        if plot_losses:
            os.makedirs("images", exist_ok=True)
            plot_path = f"images/loss_curve_trial_{trial_number}{string}.png"
            plt.figure()
            plt.plot(train_loss_history, label="Train Loss")
            if val_loss_history:
                plt.plot(val_loss_history, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training / Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"[DEBUG] Saved loss curve to {plot_path}")

        if want_test and test_filepath is not None:
            self.evaluate(test_filepath)

        return {
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "final_train_loss": train_loss_history[-1] if train_loss_history else None,
            "final_val_loss": val_loss_history[-1] if val_loss_history else None,
            "best_model_path": best_model_path if best_model_path and os.path.exists(best_model_path) else None,
        }
