import torch
import torch.nn as nn
import torch.nn.functional as F


class ThesisMarginOnlyWithTeacherProj(nn.Module):
    """
    Symmetric thesis-style contrastive loss (Hadsell):

      L = 0.5 * [ L_margin( pred_fraud , W(real_teacher), y )
                + L_margin( pred_real  , W(fraud_teacher), y ) ]

    where:
      L_margin(z1,z2,y) = y*d^2 + (1-y)*max(0, margin - d)^2
      d = ||normalize(z1) - normalize(z2)||_2

    ONLY knob: margin
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, z_text, z_teacher, y):
        """
        z_text:    [B, D] normalized
        z_teacher: [B, D] normalized
        y:         [B] in {0,1}
        """
        y = y.float()

        d = torch.norm(z_text - z_teacher, dim=1)

        pos = y * d.pow(2)
        neg = (1.0 - y) * torch.clamp(self.margin - d, min=0.0).pow(2)

        return (pos + neg).mean()