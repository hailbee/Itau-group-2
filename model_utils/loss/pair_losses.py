import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Two-margin cosine contrastive hinge loss.

    y=1 (positive): penalize if cos < m_pos
    y=0 (negative): penalize if cos > m_neg

    REQUIRE: m_pos > m_neg
    """
    def __init__(self, m_pos: float, m_neg: float):
        super().__init__()
        self.m_pos = float(m_pos)
        self.m_neg = float(m_neg)
        if not (self.m_pos > self.m_neg):
            raise ValueError(f"Need m_pos > m_neg, got m_pos={self.m_pos}, m_neg={self.m_neg}")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float()

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        c = (z1 * z2).sum(dim=1)  # cosine similarity

        pos_loss = y * F.relu(self.m_pos - c).pow(2)
        neg_loss = (1.0 - y) * F.relu(c - self.m_neg).pow(2)

        return (pos_loss + neg_loss).mean()


"""
USAGE EXAMPLE

loss_fn = ContrastiveLoss(m_pos=0.92, m_neg=0.84)

z1 = torch.randn(32, 128)
z2 = torch.randn(32, 128)
y  = torch.randint(0, 2, (32,))

loss = loss_fn(z1, z2, y)
print(loss.item())
"""
