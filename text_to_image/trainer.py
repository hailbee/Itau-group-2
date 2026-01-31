# trainer.py
# UPDATED: fixes AMP dtype mismatch in _encode_new_pairs (Half vs Float) while preserving logic.
# This version supports:
#   - NEW binary text->image pairs batches: (left_txt, right_img, y[, pair_kind[, orig_row_id]])
#   - OLD distillation batches: (fraud_txt, real_txt, fraud_teacher, real_teacher, y)

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

PairKindBatch = Union[torch.Tensor, list, tuple, None]


class Trainer:
    """
    Supports two batch formats:

    (A) NEW binary text->image pairs (your 4-pairs dataset)
        batch = (left_txt, right_img, y)
            or (left_txt, right_img, y, pair_kind)
            or (left_txt, right_img, y, pair_kind, orig_row_id)

        Training:
          z_txt = model.encode_text(left_txt, side=fraud/real)
          z_img = model.encode_image(right_img, side=fraud/real)
          normalize -> cosine-based binary loss

    (B) OLD distillation batch
        batch = (fraud_txt, real_txt, fraud_teacher, real_teacher, y)

        Training:
          z_fraud_txt, z_real_txt = model(fraud_txt, real_txt)
          z_fraud_t = model.encode_teacher(fraud_teacher)
          z_real_t  = model.encode_teacher(real_teacher)
          loss = mean(distill losses)
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device: torch.device,
        *,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = None,
        pair_kind_id_map: Optional[Dict[int, Tuple[str, str]]] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device

        # AMP
        self.use_amp = bool(use_amp) and (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Optional gradient clipping
        self.grad_clip_norm = grad_clip_norm

        # Default mapping for pair_kind_id (stable if you created the parquet with the provided builder):
        # 0: fraud_txt__fraud_img
        # 1: real_txt__real_img
        # 2: fraud_txt__real_img
        # 3: real_txt__fraud_img
        self.pair_kind_id_map = pair_kind_id_map or {
            0: ("fraud", "fraud"),
            1: ("real", "real"),
            2: ("fraud", "real"),
            3: ("real", "fraud"),
        }

        from evaluator2 import Evaluator2, EvalConfig  # local import to avoid cycles
        self.evaluator = Evaluator2(model, cfg=EvalConfig(batch_size=256))

        lr = self.optimizer.param_groups[0]["lr"]
        print(f"[DEBUG] Using fixed learning rate: {lr:.6f}")
        print(f"[DEBUG] AMP enabled: {self.use_amp}")

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _is_tensor(x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    def _to_device(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        return x

    def _infer_mode(self, batch: Tuple[Any, ...]) -> str:
        """
        Returns: "new" or "old"
        """
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"Expected batch to be list/tuple, got {type(batch)}")

        # NEW is usually 3, 4, or 5 items.
        if len(batch) in (3, 4):
            return "new"

        if len(batch) == 5:
            # Old: (fraud_txt, real_txt, fraud_teacher, real_teacher, y)
            # New: (left_txt, right_img, y, pair_kind, orig_row_id)
            _b0, _b1, _b2, b3, _b4 = batch

            # If item 3 is an embedding matrix (ndim=2), it's the old format (real_teacher).
            if self._is_tensor(b3) and b3.ndim == 2:
                return "old"
            return "new"

        raise ValueError(f"Unsupported batch size: {len(batch)}. Expected 3/4/5 items.")

    def _parse_new_batch(
        self,
        batch: Tuple[Any, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, PairKindBatch]:
        """
        Returns: (left_txt, right_img, y, pair_kind)
        """
        if len(batch) == 3:
            left_txt, right_img, y = batch
            pair_kind = None
        elif len(batch) == 4:
            left_txt, right_img, y, pair_kind = batch
        elif len(batch) == 5:
            left_txt, right_img, y, pair_kind, _orig_row_id = batch
        else:
            raise ValueError(f"NEW batch expected 3/4/5 items, got {len(batch)}")

        left_txt = self._to_device(left_txt)
        right_img = self._to_device(right_img)
        y = self._to_device(y).float().view(-1)

        return left_txt, right_img, y, pair_kind

    def _parse_old_batch(
        self,
        batch: Tuple[Any, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fraud_txt, real_txt, fraud_teacher, real_teacher, y = batch
        fraud_txt = self._to_device(fraud_txt)
        real_txt = self._to_device(real_txt)
        fraud_teacher = self._to_device(fraud_teacher)
        real_teacher = self._to_device(real_teacher)
        y = self._to_device(y).float().view(-1)
        return fraud_txt, real_txt, fraud_teacher, real_teacher, y

    def _encode_new_pairs(
        self,
        left_txt: torch.Tensor,
        right_img: torch.Tensor,
        pair_kind: PairKindBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects (text, image) into a shared out_dim using the appropriate heads.

        IMPORTANT AMP FIX:
          Under autocast, model outputs may be float16. We allocate output buffers
          as float32 and cast model outputs to float32 before masked assignment.
          This avoids: "Index put requires source and destination dtypes match".
        """
        B = left_txt.shape[0]

        # Fast path: no per-row routing (works if heads are shared or you don't care)
        if pair_kind is None:
            z_txt = self.model.encode_text(left_txt, side="fraud").float()
            z_img = self.model.encode_image(right_img, side="fraud").float()
            return z_txt, z_img

        # Convert pair_kind -> per-row (left_side, right_side) ∈ {"fraud","real"}
        if isinstance(pair_kind, (list, tuple)) and len(pair_kind) == B and isinstance(pair_kind[0], str):
            kinds = list(pair_kind)
            left_sides: list[str] = []
            right_sides: list[str] = []
            for k in kinds:
                if k == "fraud_txt__fraud_img":
                    left_sides.append("fraud"); right_sides.append("fraud")
                elif k == "real_txt__real_img":
                    left_sides.append("real"); right_sides.append("real")
                elif k == "fraud_txt__real_img":
                    left_sides.append("fraud"); right_sides.append("real")
                elif k == "real_txt__fraud_img":
                    left_sides.append("real"); right_sides.append("fraud")
                else:
                    raise ValueError(f"Unknown pair_kind string: {k!r}")

        elif isinstance(pair_kind, torch.Tensor):
            pk = self._to_device(pair_kind).view(-1).long()
            if pk.numel() != B:
                raise ValueError(f"pair_kind tensor has {pk.numel()} elements but batch size is {B}")

            left_sides = []
            right_sides = []
            for i in pk.tolist():
                if int(i) not in self.pair_kind_id_map:
                    raise ValueError(
                        f"pair_kind id {i} not in pair_kind_id_map keys={sorted(self.pair_kind_id_map.keys())}"
                    )
                ls, rs = self.pair_kind_id_map[int(i)]
                left_sides.append(ls)
                right_sides.append(rs)
        else:
            raise TypeError(
                f"pair_kind must be None, torch.Tensor, or list[str]. Got type={type(pair_kind)}"
            )

        # Masks
        left_sides_t = torch.tensor([0 if s == "fraud" else 1 for s in left_sides], device=left_txt.device)
        right_sides_t = torch.tensor([0 if s == "fraud" else 1 for s in right_sides], device=right_img.device)

        idx_f = (left_sides_t == 0).nonzero(as_tuple=False).view(-1)
        idx_r = (left_sides_t == 1).nonzero(as_tuple=False).view(-1)

        idx_fi = (right_sides_t == 0).nonzero(as_tuple=False).view(-1)
        idx_ri = (right_sides_t == 1).nonzero(as_tuple=False).view(-1)

        # Allocate float32 buffers (AMP-safe for masked assignment)
        out_dim = int(self.model.out_dim)
        z_txt = torch.empty((B, out_dim), device=left_txt.device, dtype=torch.float32)
        z_img = torch.empty((B, out_dim), device=right_img.device, dtype=torch.float32)

        # Text
        if idx_f.numel() > 0:
            tmp = self.model.encode_text(left_txt[idx_f], side="fraud")
            z_txt[idx_f] = tmp.to(z_txt.dtype)
        if idx_r.numel() > 0:
            tmp = self.model.encode_text(left_txt[idx_r], side="real")
            z_txt[idx_r] = tmp.to(z_txt.dtype)

        # Image/teacher
        if idx_fi.numel() > 0:
            tmp = self.model.encode_image(right_img[idx_fi], side="fraud")
            z_img[idx_fi] = tmp.to(z_img.dtype)
        if idx_ri.numel() > 0:
            tmp = self.model.encode_image(right_img[idx_ri], side="real")
            z_img[idx_ri] = tmp.to(z_img.dtype)

        return z_txt, z_img

    # -------------------------
    # Epoch loops
    # -------------------------
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0

        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext

        for i, batch in enumerate(dataloader):
            mode = self._infer_mode(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                if mode == "new":
                    left_txt, right_img, y, pair_kind = self._parse_new_batch(batch)
                    z_txt, z_img = self._encode_new_pairs(left_txt, right_img, pair_kind)

                    # normalize for cosine-based losses
                    z_txt = F.normalize(z_txt, dim=1)
                    z_img = F.normalize(z_img, dim=1)

                    # New criterion expects (z_left, z_right, y)
                    loss = self.criterion(z_txt, z_img, y)

                else:
                    fraud_txt, real_txt, fraud_teacher, real_teacher, y = self._parse_old_batch(batch)

                    # old student forward
                    z_fraud, z_real = self.model(fraud_txt, real_txt)

                    # old teacher projections
                    z_real_teacher = self.model.encode_teacher(real_teacher)
                    z_fraud_teacher = self.model.encode_teacher(fraud_teacher)

                    # normalize
                    z_fraud = F.normalize(z_fraud, dim=1)
                    z_real = F.normalize(z_real, dim=1)
                    z_real_teacher = F.normalize(z_real_teacher, dim=1)
                    z_fraud_teacher = F.normalize(z_fraud_teacher, dim=1)

                    # old criterion typically expects (z_a, z_b)
                    loss_fraud = self.criterion(z_fraud, z_fraud_teacher)
                    loss_real = self.criterion(z_real, z_real_teacher)
                    loss = 0.5 * (loss_fraud + loss_real)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            # backward/step (AMP-safe)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            epoch_loss += float(loss.item())

            if i % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Step {i}/{len(dataloader)} | LR: {lr:.6f} | Loss: {loss.item():.10f} | Mode: {mode}"
                )

        return epoch_loss / max(len(dataloader), 1)

    def validate_epoch(self, dataloader):
        if dataloader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0

        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                mode = self._infer_mode(batch)

                with autocast_ctx():
                    if mode == "new":
                        left_txt, right_img, y, pair_kind = self._parse_new_batch(batch)
                        z_txt, z_img = self._encode_new_pairs(left_txt, right_img, pair_kind)

                        z_txt = F.normalize(z_txt, dim=1)
                        z_img = F.normalize(z_img, dim=1)

                        loss = self.criterion(z_txt, z_img, y)

                    else:
                        fraud_txt, real_txt, fraud_teacher, real_teacher, y = self._parse_old_batch(batch)

                        z_fraud, z_real = self.model(fraud_txt, real_txt)
                        z_real_teacher = self.model.encode_teacher(real_teacher)
                        z_fraud_teacher = self.model.encode_teacher(fraud_teacher)

                        z_fraud = F.normalize(z_fraud, dim=1)
                        z_real = F.normalize(z_real, dim=1)
                        z_real_teacher = F.normalize(z_real_teacher, dim=1)
                        z_fraud_teacher = F.normalize(z_fraud_teacher, dim=1)

                        loss_fraud = self.criterion(z_fraud, z_fraud_teacher)
                        loss_real = self.criterion(z_real, z_real_teacher)
                        loss = 0.5 * (loss_fraud + loss_real)

                epoch_loss += float(loss.item())

                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Val Step {i}/{len(dataloader)} | LR: {lr:.6f} | Mode: {mode}")

        return epoch_loss / max(len(dataloader), 1)

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, test_filepath: str):
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
            best_model_path = os.path.join(
                save_dir, f"best_model_trial_{trial_number}{string}.pt"
            )
            print(f"[DEBUG] best_model_path={os.path.abspath(best_model_path)}")

        for epoch in range(int(epochs)):
            train_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.10f}")

            val_loss = self.validate_epoch(validate_dataloader)
            if val_loss is None:
                continue
            print(f"Epoch {epoch+1} | Val Loss: {val_loss:.10f}")

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
                    print(f"[DEBUG] Saved best checkpoint (val_loss={best_val_loss:.12f})")
            else:
                bad_epochs += 1

            if early_stopping and (epoch + 1) >= min_epochs and bad_epochs >= patience:
                print(f"[DEBUG] Early stopping at epoch {epoch+1}")
                break

            # Optional: run evaluation each epoch
            if test_filepath is not None and isinstance(test_filepath, str) and test_filepath.strip():
                try:
                    self.evaluate(test_filepath)
                except Exception as e:
                    print(f"[WARN] Evaluation failed this epoch: {e}")

        if save_best and best_model_path and os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            if "criterion_state" in ckpt:
                try:
                    self.criterion.load_state_dict(ckpt["criterion_state"])
                except Exception:
                    pass
            print(f"[DEBUG] Restored best model from {best_model_path}")

        return {
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_model_path": best_model_path,
        }
