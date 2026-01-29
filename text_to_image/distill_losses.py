# distill_losses.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCosineDistillLoss(nn.Module):
    """
    Distill embeddings directly:
      loss = mean(1 - cos(S_f, T_f)) + mean(1 - cos(S_r, T_r))
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label=None):
        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps).detach()
        T_r = F.normalize(real_teacher,  dim=1, eps=self.eps).detach()

        cos_f = (S_f * T_f).sum(dim=1)
        cos_r = (S_r * T_r).sum(dim=1)

        return (1.0 - cos_f).mean() + (1.0 - cos_r).mean()
