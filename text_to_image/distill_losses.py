import torch
import torch.nn as nn


class ThesisMarginOnlyWithTeacherProj(nn.Module):
    """
    Positive-only contrastive alignment loss:

      L = || z_text - z_teacher ||^2

    Assumes embeddings are normalized (optional but consistent with your setup).
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_text, z_teacher, y=None):
        d = torch.norm(z_text - z_teacher, dim=1)
        return d.pow(2).mean()
