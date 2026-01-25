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


class Text2ImgDistillInfoNCELoss(nn.Module):
    """
    Combines:
      - pointwise cosine distillation (your current objective)
      - in-batch InfoNCE (ranking objective): pred_fraud should match real_teacher (same row)
        better than other real_teacher in the batch.

    No extra model parameters. Temperature is a fixed scalar hyperparam.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.alpha = float(alpha)
        self.temperature = float(temperature)

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher):
        # normalize everything (safe, improves stability)
        pred_fraud = F.normalize(pred_fraud, dim=1)
        pred_real  = F.normalize(pred_real, dim=1)
        fraud_teacher = F.normalize(fraud_teacher, dim=1)
        real_teacher  = F.normalize(real_teacher, dim=1)

        # -------------------------
        # (A) pointwise cosine distill (as you had)
        # maximize cosine => minimize (1 - cosine)
        # -------------------------
        distill_f = 1.0 - (pred_fraud * fraud_teacher).sum(dim=1).mean()
        distill_r = 1.0 - (pred_real  * real_teacher ).sum(dim=1).mean()
        distill = 0.5 * (distill_f + distill_r)

        # -------------------------
        # (B) InfoNCE in-batch ranking (diagonal positives)
        # query: pred_fraud
        # keys:  real_teacher
        # positive for row i is key i
        # -------------------------
        logits = (pred_fraud @ real_teacher.T) / self.temperature  # (B,B)
        targets = torch.arange(logits.size(0), device=logits.device)
        nce = F.cross_entropy(logits, targets)

        # combine
        return self.alpha * distill + (1.0 - self.alpha) * nce

class MultiPositiveInfoNCEDistillLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.alpha = float(alpha)      # pointwise distill weight
        self.beta = float(beta)        # prototype NCE weight
        self.temperature = float(temperature)

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, brand_id):
        # normalize
        pred_fraud = F.normalize(pred_fraud, dim=1)
        pred_real  = F.normalize(pred_real,  dim=1)
        fraud_teacher = F.normalize(fraud_teacher, dim=1)
        real_teacher  = F.normalize(real_teacher,  dim=1)

        # (A) pointwise cosine distill (keep this)
        distill_f = 1.0 - (pred_fraud * fraud_teacher).sum(dim=1).mean()
        distill_r = 1.0 - (pred_real  * real_teacher ).sum(dim=1).mean()
        distill = 0.5 * (distill_f + distill_r)

        # (B) prototype InfoNCE (brand prototypes as keys)
        # build prototypes from real_teacher
        brand_id = brand_id.to(real_teacher.device)
        unique_ids = torch.unique(brand_id)

        # prototypes: (M, D)
        protos = []
        for b in unique_ids:
            protos.append(real_teacher[brand_id == b].mean(dim=0))
        protos = torch.stack(protos, dim=0)              # (M,D)
        protos = F.normalize(protos, dim=1)

        # targets: map each sample's brand_id -> prototype index in [0..M-1]
        # build a lookup dict on GPU
        # (cheap because M <= batch size)
        id_to_index = {int(b.item()): i for i, b in enumerate(unique_ids)}
        targets = torch.tensor([id_to_index[int(b.item())] for b in brand_id], device=brand_id.device)

        # logits: (B,M)
        logits = (pred_fraud @ protos.T) / self.temperature
        proto_nce = F.cross_entropy(logits, targets)

        return self.alpha * distill + self.beta * proto_nce
