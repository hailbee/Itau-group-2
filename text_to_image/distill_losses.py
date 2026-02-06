import torch
import torch.nn as nn
import torch.nn.functional as F


class ThesisMarginOnlyWithTeacherProj(nn.Module):
    """
    Combined multi-task loss for 4-embedding rows.

    Inputs:
      z_fraud_txt: (B, D)
      z_real_txt:  (B, D)
      fraud_img:   (B, D)
      real_img:    (B, D)
      y:           (B,) float labels (0 or 1)

    Loss = alignment_loss + pair_loss

    Alignment:
      enforce cos(z_fraud_txt, fraud_img) >= align_margin
      enforce cos(z_real_txt,  real_img)  >= align_margin

    Pair discrimination:
      s = cos(z_fraud_txt, z_real_txt)

      if y=1: want s >= pos_pair_margin
      if y=0: want s <= neg_pair_margin
    """

    def __init__(
        self,
        align_margin: float = 0.69,
        pos_pair_margin: float = 0.95,
        neg_pair_margin: float = 0.90,
        pair_weight: float = 1.0,
    ):
        super().__init__()
        self.align_margin = float(align_margin)
        self.pos_pair_margin = float(pos_pair_margin)
        self.neg_pair_margin = float(neg_pair_margin)
        self.pair_weight = float(pair_weight)

        if self.neg_pair_margin >= self.pos_pair_margin:
            raise ValueError(
                f"Expected neg_pair_margin < pos_pair_margin, "
                f"got {self.neg_pair_margin} >= {self.pos_pair_margin}"
            )

    def forward(self, z_fraud_txt, z_real_txt, fraud_img, real_img, y):
        # normalize all embeddings
        z_fraud_txt = F.normalize(z_fraud_txt, dim=1)
        z_real_txt = F.normalize(z_real_txt, dim=1)

        fraud_img = F.normalize(fraud_img, dim=1)
        real_img = F.normalize(real_img, dim=1)

        y = y.float().view(-1)

        # -------------------------
        # 1) Alignment loss
        # -------------------------
        cos_f = (z_fraud_txt * fraud_img).sum(dim=1)
        cos_r = (z_real_txt * real_img).sum(dim=1)

        align_loss_f = F.relu(self.align_margin - cos_f)
        align_loss_r = F.relu(self.align_margin - cos_r)

        align_loss = (align_loss_f + align_loss_r).mean()

        # -------------------------
        # 2) Pair push/pull loss
        # -------------------------
        pair_cos = (z_fraud_txt * z_real_txt).sum(dim=1)

        pos_loss = y * F.relu(self.pos_pair_margin - pair_cos)
        neg_loss = (1.0 - y) * F.relu(pair_cos - self.neg_pair_margin)

        pair_loss = (pos_loss + neg_loss).mean()

        return align_loss + self.pair_weight * pair_loss
