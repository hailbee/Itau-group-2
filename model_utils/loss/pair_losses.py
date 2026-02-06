# ContrastiveLoss: two-sided band
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, m_pos: float, m_neg: float, w_pos: float = 1.0, w_neg: float = 1.0):
        super().__init__()
        self.m_pos = float(m_pos)
        self.m_neg = float(m_neg)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)
        if not (self.m_neg > self.m_pos):
            raise ValueError(f"Need m_neg > m_pos, got m_pos={self.m_pos}, m_neg={self.m_neg}")

    def forward(self, z1, z2, y):
        y = y.float()
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        d = torch.norm(z1 - z2, dim=1)  # in [0,2]

        pos = y * F.relu(d - self.m_pos).pow(2)          # punish positives only if too far
        neg = (1 - y) * F.relu(self.m_neg - d).pow(2)    # punish negatives only if too close

        return (self.w_pos * pos + self.w_neg * neg).mean()
