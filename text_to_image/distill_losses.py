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

"""
not needed:

class EmbeddingCosineGeometryDistillLoss(nn.Module):
    """
    Parameter-free teacher geometry distillation.

    Loss =
      mean(1 - cos(S_f, T_f)) + mean(1 - cos(S_r, T_r))
    + mean((G_Sf - G_Tf)^2) + mean((G_Sr - G_Tr)^2)

    where G_X = normalized pairwise cosine similarity matrix.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def _pairwise_cosine_matrix(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, D), assumed normalized
        G = X @ X.T  # (B, B)
        # remove self-similarity to avoid trivial diagonal dominance
        B = G.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=G.device)
        return G[mask]

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label=None):
        # normalize embeddings
        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps).detach()
        T_r = F.normalize(real_teacher,  dim=1, eps=self.eps).detach()

        # ---------- pointwise cosine distillation ----------
        cos_f = (S_f * T_f).sum(dim=1)
        cos_r = (S_r * T_r).sum(dim=1)
        pointwise_loss = (1.0 - cos_f).mean() + (1.0 - cos_r).mean()

        # ---------- relational (pairwise geometry) distillation ----------
        G_Sf = self._pairwise_cosine_matrix(S_f)
        G_Tf = self._pairwise_cosine_matrix(T_f)
        G_Sr = self._pairwise_cosine_matrix(S_r)
        G_Tr = self._pairwise_cosine_matrix(T_r)

        relational_loss = (
            (G_Sf - G_Tf).pow(2).mean()
            + (G_Sr - G_Tr).pow(2).mean()
        )

        return pointwise_loss + relational_loss
"""