import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Two-margin hinge contrastive loss on cosine similarity.

    Let c = cosine(z1, z2) in [-1, 1] (computed after L2-normalization).

    y = 1 (positive pair): want c >= m_pos  -> penalize relu(m_pos - c)^2
    y = 0 (negative pair): want c <= m_neg  -> penalize relu(c - m_neg)^2

    IMPORTANT:
      - For cosine two-margin hinge, you should set m_pos > m_neg (a separation gap).
      - If you accidentally set m_pos <= m_neg, you can create a "bad deadzone" where both
        classes can satisfy constraints without improving separation.

    Weights:
      - w_pos / w_neg weight the positive vs negative terms.

    Typical starting point (based on your histogram):
      - m_pos ~ 0.90–0.95
      - m_neg ~ 0.80–0.86
      - w_neg >= w_pos (often 1–5x) if you want to suppress hard negatives
    """

    def __init__(
        self,
        m_pos: float,
        m_neg: float,
        w_pos: float = 1.0,
        w_neg: float = 1.0,
        reduction: str = "mean",
        enforce_gap: bool = True,
    ):
        super().__init__()
        self.m_pos = float(m_pos)
        self.m_neg = float(m_neg)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.reduction = reduction

        if enforce_gap and not (self.m_pos > self.m_neg):
            raise ValueError(
                f"For cosine two-margin hinge, require m_pos > m_neg. Got m_pos={self.m_pos}, m_neg={self.m_neg}."
            )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y can be {0,1} ints/bools; cast to float
        y = y.float()

        # cosine similarity (safe even if inputs not normalized)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        c = (z1 * z2).sum(dim=1)  # [batch], in [-1, 1]

        # hinge penalties (two margins)
        pos_loss = F.relu(self.m_pos - c).pow(2)  # only if positive similarity is below m_pos
        neg_loss = F.relu(c - self.m_neg).pow(2)  # only if negative similarity is above m_neg

        # weighted combine per example
        loss = self.w_pos * y * pos_loss + self.w_neg * (1.0 - y) * neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss