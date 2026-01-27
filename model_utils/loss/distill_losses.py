import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillPlusLabelPairLoss(nn.Module):
    """
    Goal: transfer spoof-aware structure from image embeddings into text-mapped embeddings
    while directly optimizing the ROC-AUC score you use at eval time:
        score = cos(pred_fraud, pred_real)

    Terms:
      (A) pointwise distillation: pred_fraud ~ fraud_teacher, pred_real ~ real_teacher
      (B) label-aware pair loss: push score up if label=1, down if label=0 (BCE on cosine)
      (C) optional teacher-pair distill: match cos(pred_fraud,pred_real) to cos(teacher_fraud,teacher_real)

    Minimal hyperparams: weights (can start with all 1.0) + learnable logit_scale.

    If you want the absolute minimum tuning, start with w_distill=1.0, w_bce=1.0, w_pair_teacher=0.0 (skip C), and only add w_pair_teacher if you see alignment improve but ROC-AUC stall.
    """
    def __init__(self, w_distill: float = 1.0, w_bce: float = 1.0, w_pair_teacher: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.w_distill = float(w_distill)
        self.w_bce = float(w_bce)
        self.w_pair_teacher = float(w_pair_teacher)
        self.eps = float(eps)

        # Learnable scale = avoids you tuning temperature; helps calibration
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label):
        # normalize everything
        pred_fraud = F.normalize(pred_fraud, dim=1, eps=self.eps)
        pred_real  = F.normalize(pred_real,  dim=1, eps=self.eps)
        fraud_teacher = F.normalize(fraud_teacher, dim=1, eps=self.eps)
        real_teacher  = F.normalize(real_teacher,  dim=1, eps=self.eps)

        # (A) pointwise cosine distill (transfer image-space info)
        distill_f = 1.0 - (pred_fraud * fraud_teacher).sum(dim=1).mean()
        distill_r = 1.0 - (pred_real  * real_teacher ).sum(dim=1).mean()
        distill = 0.5 * (distill_f + distill_r)

        # (B) label-aware pair similarity objective (matches your ROC-AUC scoring)
        # score = cos(pred_fraud, pred_real)
        score = (pred_fraud * pred_real).sum(dim=1)
        logits = self.logit_scale * score
        bce = F.binary_cross_entropy_with_logits(logits, label.float())

        # (C) optional: distill the teacher's fraud↔real similarity geometry
        teacher_score = (fraud_teacher * real_teacher).sum(dim=1).detach()
        pair_teacher = F.mse_loss(score, teacher_score)

        return self.w_distill * distill + self.w_bce * bce + self.w_pair_teacher * pair_teacher
