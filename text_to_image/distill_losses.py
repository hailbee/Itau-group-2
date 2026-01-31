# distill_losses.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Existing loss (kept as-is)
# -----------------------------
class ThesisMarginOnlyWithTeacherProj(nn.Module):
    """
    CLIP-style symmetric InfoNCE contrastive loss for paired data (text ↔ image).

    Positives:
        (text_i, image_i)

    Negatives:
        all other pairs in the batch

    NOTE: This loss does NOT use binary labels. It assumes every (i,i) is positive.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_text: torch.Tensor, z_image: torch.Tensor) -> torch.Tensor:
        """
        z_text:  [B, D] normalized
        z_image: [B, D] normalized
        """
        logits = torch.matmul(z_text, z_image.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_t2i + loss_i2t)


# -----------------------------
# New losses for your NEW task
# (binary label on cosine)
# -----------------------------
class BinaryCosineBCEWithLogits(nn.Module):
    """
    Binary loss directly on cosine similarity:

      cos = cosine(z_left, z_right) in [-1, 1]
      logits = scale * cos
      loss = BCEWithLogits(logits, y)

    This matches your new dataset where label=1 means "close", label=0 means "far".

    Typical usage (recommended):
      - normalize z_left and z_right in the trainer
      - pass y as float tensor in {0,1}
    """

    def __init__(
        self,
        scale: float = 10.0,
        *,
        normalize_inputs: bool = False,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.scale = float(scale)
        self.normalize_inputs = bool(normalize_inputs)

        if pos_weight is None:
            self.register_buffer("_pos_weight", torch.tensor(1.0))
            self.use_pos_weight = False
        else:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
            self.use_pos_weight = True

        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        z_left: torch.Tensor,
        z_right: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_left:  [B, D]
        z_right: [B, D]
        y:       [B] or [B,1] with values in {0,1} (float or int)
        """
        if self.normalize_inputs:
            z_left = F.normalize(z_left, dim=1)
            z_right = F.normalize(z_right, dim=1)

        cos = F.cosine_similarity(z_left, z_right, dim=1)  # [B]
        logits = self.scale * cos

        y = y.float().view(-1)
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            y = y * (1.0 - eps) + 0.5 * eps

        if self.use_pos_weight:
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)

        return loss


class BinaryCosineMarginLoss(nn.Module):
    """
    UPDATED: "one-margin in practice" variant while keeping the SAME class name + call signature.

    Goal:
      - keep the positive hinge: penalize if cos < pos_margin
      - make negatives "do not be positively correlated" (fixed neg threshold)

    Changes:
      - neg_margin is now OPTIONAL / effectively fixed to a constant (default 0.0),
        so you can stop tuning it.
      - We DO NOT enforce pos_margin > neg_margin anymore (because neg_margin is fixed).
      - API remains compatible: __init__(pos_margin=..., neg_margin=..., ...) still works,
        but neg_margin is treated as a fixed threshold and you can just leave it alone.

    Loss:
      pos = relu(pos_margin - cos)
      neg = relu(cos - neg_margin_fixed)
      loss = y * pos^2 + (1-y) * neg^2
    """

    def __init__(
        self,
        pos_margin: float = 0.5,
        neg_margin: float = 0.0,  # <--- default fixed to 0.0 (recommended)
        *,
        normalize_inputs: bool = False,
        squared: bool = True,
    ):
        super().__init__()
        self.pos_margin = float(pos_margin)

        # Treat neg_margin as a fixed threshold; you generally should not tune it.
        # Default is 0.0 meaning negatives should not have positive cosine similarity.
        self.neg_margin = float(neg_margin)

        self.normalize_inputs = bool(normalize_inputs)
        self.squared = bool(squared)

    def forward(
        self,
        z_left: torch.Tensor,
        z_right: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_inputs:
            z_left = F.normalize(z_left, dim=1)
            z_right = F.normalize(z_right, dim=1)

        cos = F.cosine_similarity(z_left, z_right, dim=1)  # [B]
        y = y.float().view(-1)

        pos = F.relu(self.pos_margin - cos)
        neg = F.relu(cos - self.neg_margin)

        if self.squared:
            pos = pos * pos
            neg = neg * neg

        loss = y * pos + (1.0 - y) * neg
        return loss.mean()


class WeightedSumLoss(nn.Module):
    """
    Utility to combine multiple losses.
    """

    def __init__(self, losses: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = {k: float(v) for k, v in weights.items()}
        for k in self.losses.keys():
            if k not in self.weights:
                raise ValueError(f"Missing weight for loss '{k}'")

    def forward(self, **kwargs) -> torch.Tensor:
        total = 0.0
        for name, loss_fn in self.losses.items():
            w = self.weights[name]
            if w == 0.0:
                continue
            total = total + w * loss_fn(**kwargs)
        return total
