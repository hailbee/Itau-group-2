# scripts/training/trainer.py

from scripts.evaluation.evaluator import Evaluator
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class Trainer:
    """
    Supports:
      - mode="pair":    batch = (x1, x2, y)
                        model(x1,x2)->(z1,z2)
                        loss = criterion(z1,z2,y)

      - mode="text2img": batch = (fraud_txt, real_txt, fraud_teacher, real_teacher, y, brand_id)
                         model(fraud_txt, real_txt)->(pred_fraud,pred_real)

        IMPORTANT: different losses want different extra args:
          - New label-aware losses: criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, y)
          - Some losses want both:  criterion(..., y, brand_id)
          - Old prototype losses:   criterion(..., brand_id)

        So we try these in order:
          (1) y, brand_id
          (2) y
          (3) brand_id
    """

    def __init__(self, model, criterion, optimizer, device, model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type

        self.model.to(device)

        # Old evaluator for old task; Evaluator2 for text2img
        if model_type == "text2img":
            from scripts.evaluation.evaluator2 import Evaluator2, EvalConfig
            self.evaluator = Evaluator2(model, cfg=EvalConfig(batch_size=256))
        else:
            self.evaluator = Evaluator(model, model_type=model_type)

        print(f"[DEBUG] Using fixed learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def _call_text2img_criterion(self, pred_fraud, pred_real, fraud_teacher, real_teacher, y, brand_id):
        """
        Try calling the criterion with progressively simpler signatures.

        This prevents a common silent failure where:
          - the loss expects (.., label)
          - but trainer passes (.., brand_id) and the loss interprets brand_id as label
        """
        # (1) label + brand_id
        try:
            return self.criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, y, brand_id)
        except TypeError:
            pass

        # (2) label only
        try:
            return self.criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, y)
        except TypeError:
            pass

        # (3) brand_id only (older prototype / NCE variants)
        return self.criterion(pred_fraud, pred_real, fraud_teacher, real_teacher, brand_id)

    def train_epoch(self, dataloader, mode="pair", grad_clip=1.0):
        self.model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            if mode == "pair":
                x1, x2, y = batch
                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2, y)

            elif mode == "text2img":
                # Dataset must return 6 items:
                # fraud_txt, real_txt, fraud_teacher, real_teacher, y, brand_id
                fraud_txt, real_txt, fraud_teacher, real_teacher, y, brand_id = batch

                fraud_txt = fraud_txt.to(self.device, non_blocking=True)
                real_txt = real_txt.to(self.device, non_blocking=True)
                fraud_teacher = fraud_teacher.to(self.device, non_blocking=True)
                real_teacher = real_teacher.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                brand_id = brand_id.to(self.device, non_blocking=True)

                pred_fraud, pred_real = self.model(fraud_txt, real_txt)

                loss = self._call_text2img_criterion(
                    pred_fraud, pred_real, fraud_teacher, real_teacher, y, brand_id
                )

            else:
                raise ValueError(f"Unsupported mode: {mode}")

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

    def validate_epoch(self, dataloader, mode="pair"):
        if dataloader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if mode == "pair":
                    x1, x2, y = batch
                    x1 = x1.to(self.device, non_blocking=True)
                    x2 = x2.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    z1, z2 = self.model(x1, x2)
                    loss = self.criterion(z1, z2, y)

                elif mode == "text2img":
                    fraud_txt, real_txt, fraud_teacher, real_teacher, y, brand_id = batch

                    fraud_txt = fraud_txt.to(self.device, non_blocking=True)
                    real_txt = real_txt.to(self.device, non_blocking=True)
                    fraud_teacher = fraud_teacher.to(self.device, non_blocking=True)
                    real_teacher = real_teacher.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    brand_id = brand_id.to(self.device, non_blocking=True)

                    pred_fraud, pred_real = self.model(fraud_txt, real_txt)

                    loss = self._call_text2img_criterion(
                        pred_fraud, pred_real, fraud_teacher, real_teacher, y, brand_id
                    )

                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                epoch_loss += loss.item()

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Val Step {i} / {len(dataloader)} | LR: {lr:.6f}")

        return epoch_loss / max(len(dataloader), 1)

    def evaluate(self, test_filepath, plot=False, roc_png_path=None, cm_png_path=None):
        self.model.eval()
        return self.evaluator.evaluate(
            test_filepath, plot=plot, roc_png_path=roc_png_path, cm_png_path=cm_png_path
        )

    def _scores_from_loader(self, dataloader, max_batches=None, mode="pair"):
        """
        Utility for optional train/val metric logging during training.
        Returns DF with columns: label, similarity where similarity = cos(z1,z2).

        In text2img mode: similarity = cos(pred_fraud, pred_real).
        """
        self.model.eval()
        sims_all = []
        y_all = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break

                if mode == "pair":
                    x1, x2, y = batch

                elif mode == "text2img":
                    # Dataset returns 6 items; we only need fraud_txt, real_txt, y
                    x1, x2, _t1, _t2, y, _brand_id = batch

                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                z1, z2 = self.model(x1, x2)
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                sims = F.cosine_similarity(z1, z2, dim=1)

                sims_all.append(sims.detach().cpu())
                y_all.append(y.detach().cpu())

        if len(sims_all) == 0:
            return pd.DataFrame({"label": [], "similarity": []})

        similarities = torch.cat(sims_all, dim=0).numpy()
        labels = torch.cat(y_all, dim=0).numpy().astype(int)
        return pd.DataFrame({"label": labels, "similarity": similarities})

    def _accuracy_at_threshold(self, results_df, threshold):
        y_true = results_df["label"].astype(int).to_numpy()
        y_scores = results_df["similarity"].astype(float).to_numpy()
        y_pred = (y_scores >= float(threshold)).astype(int)
        return float((y_pred == y_true).mean())

    def train(
        self,
        dataloader,
        trial_number,
        test_filepath,
        string,
        mode="pair",
        epochs=30,
        validate_filepath=None,
        validate_dataloader=None,
        want_test=False,
        plot_losses=True,
        grad_clip=1.0,
        early_stopping=True,
        patience=5,
        min_epochs=50,
        min_delta=1e-6,
        save_best=True,
        save_dir="saved_models",
        plot_accuracy=False,
        accuracy_eval_every=1,
        accuracy_max_train_batches=50,
        accuracy_max_val_batches=None,
    ):
        train_loss_history = []
        val_loss_history = []
        best_epoch_loss = float("inf")

        acc_epochs = []
        train_acc_history = []
        val_acc_history = []
        youden_thr_history = []
        val_auc_history = []

        best_val_loss = float("inf")
        bad_epochs = 0
        best_model_path = None
        if save_best:
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"best_model_by_val_trial_{trial_number}{string}.pt")

        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader, mode=mode, grad_clip=grad_clip)
            train_loss_history.append(train_loss)
            best_epoch_loss = min(best_epoch_loss, train_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f}")

            val_loss = self.validate_epoch(validate_dataloader, mode=mode)
            if val_loss is not None:
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch+1} | Val Loss: {val_loss:.6f}")

                if save_best and val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        best_model_path,
                    )
                    print(f"[DEBUG] Saved best checkpoint (val_loss={best_val_loss:.6f}) -> {best_model_path}")
                else:
                    bad_epochs += 1

                if early_stopping and (epoch + 1) >= min_epochs and bad_epochs >= patience:
                    print(f"[DEBUG] Early stopping at epoch {epoch+1} (best_val_loss={best_val_loss:.6f})")
                    break

            if (
                plot_accuracy
                and validate_dataloader is not None
                and (epoch + 1) % int(max(1, accuracy_eval_every)) == 0
            ):
                val_df = self._scores_from_loader(validate_dataloader, max_batches=accuracy_max_val_batches, mode=mode)
                if len(val_df) > 0:
                    val_metrics = self.evaluator.compute_metrics(val_df, plot=False)
                    youden_thr = val_metrics.get("youden_threshold")
                    val_acc = val_metrics.get("accuracy_youden")
                    val_auc = val_metrics.get("roc_auc")

                    train_df = self._scores_from_loader(dataloader, max_batches=accuracy_max_train_batches, mode=mode)
                    train_acc = self._accuracy_at_threshold(train_df, youden_thr) if len(train_df) > 0 else None

                    acc_epochs.append(epoch + 1)
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    youden_thr_history.append(youden_thr)
                    val_auc_history.append(val_auc)

                    if train_acc is not None:
                        print(
                            f"[ACC] Epoch {epoch+1} | Train Acc (Youden@val): {train_acc:.4f} | "
                            f"Val Acc (Youden): {val_acc:.4f} | Thr: {youden_thr:.4f} | AUC: {val_auc:.4f}"
                        )

        if save_best and best_model_path is not None and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            print(f"[DEBUG] Restored best model into memory from {best_model_path}")

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

        if want_test and test_filepath is not None:
            self.evaluate(test_filepath, plot=False)

        return {
            "best_train_loss": best_epoch_loss,
            "final_train_loss": (train_loss_history[-1] if train_loss_history else None),
            "final_val_loss": (val_loss_history[-1] if val_loss_history else None),
            "best_val_loss": (best_val_loss if best_val_loss < float("inf") else None),
            "best_model_path": best_model_path,
            "final_train_acc_youden_at_val": (train_acc_history[-1] if train_acc_history else None),
            "final_val_acc_youden": (val_acc_history[-1] if val_acc_history else None),
            "final_val_youden_threshold": (youden_thr_history[-1] if youden_thr_history else None),
            "final_val_roc_auc": (val_auc_history[-1] if val_auc_history else None),
        }
