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

        self.beta = 10.0  # fixed: higher -> closer to ReLU, but still smooth (no dead zone)

    def forward(self, z1, z2, y):
        y = y.float()
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        d = torch.norm(z1 - z2, dim=1)  # [0,2]

        # smooth hinge ~ relu(x)
        hp = F.softplus(d - self.m_pos, beta=self.beta)   # ~ relu(d-m_pos)
        hn = F.softplus(self.m_neg - d, beta=self.beta)   # ~ relu(m_neg-d)

        pos = y * hp.pow(2)
        neg = (1 - y) * hn.pow(2)

        return (self.w_pos * pos + self.w_neg * neg).mean()
