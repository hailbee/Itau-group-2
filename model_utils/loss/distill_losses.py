import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDistillLoss(nn.Module):
    """
    Parameter-free cosine distillation:
      L = mean(1 - cos(normalize(student), normalize(teacher)))
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student, dim=1, eps=self.eps)
        teacher = F.normalize(teacher, dim=1, eps=self.eps)
        cos = (student * teacher).sum(dim=1)
        return (1.0 - cos).mean()
