import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDistillLoss(nn.Module):
    """
    Single-term pointwise cosine distillation:
      L = mean(1 - cos(normalize(student), normalize(teacher)))

    This exists mainly for backwards compatibility with older training/optuna code
    that still imports CosineDistillLoss.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student, dim=1, eps=self.eps)
        teacher = F.normalize(teacher, dim=1, eps=self.eps)
        cos = (student * teacher).sum(dim=1)
        return (1.0 - cos).mean()


class TeacherScoreDistillBCELoss(nn.Module):
    """
    Single-term teacher-score distillation.

    Per row:
      Teacher score: t = cos(T_f, T_r)
      Student score: s = cos(S_f, S_r)

    Convert teacher score into a soft probability:
      p_teacher = sigmoid(scale_t * t + bias_t)

    Train student logits to match it:
      loss = BCEWithLogits(scale_s * s + bias_s, p_teacher)

    Notes:
      - NOT a hybrid loss (single objective).
      - scale/bias are learned scalars (no manual temperature tuning).
      - label is accepted but ignored (so trainer can pass y without caring).
    """

    def __init__(self, eps: float = 1e-8, clamp_log_scale: tuple[float, float] = (-2.0, 4.0)):
        super().__init__()
        self.eps = float(eps)
        self.clamp_log_scale = (float(clamp_log_scale[0]), float(clamp_log_scale[1]))

        # Student calibration: logits_s = scale_s * s + bias_s
        self.log_scale_s = nn.Parameter(torch.tensor(0.0))  # exp(0)=1
        self.bias_s = nn.Parameter(torch.tensor(0.0))

        # Teacher calibration: p_teacher = sigmoid(scale_t * t + bias_t)
        self.log_scale_t = nn.Parameter(torch.tensor(0.0))
        self.bias_t = nn.Parameter(torch.tensor(0.0))

    def _pos_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clamp_log_scale
        return torch.exp(torch.clamp(log_scale, lo, hi))

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label=None):
        # Normalize embeddings
        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps)
        T_r = F.normalize(real_teacher,  dim=1, eps=self.eps)

        # Pair scores (cosines)
        s = (S_f * S_r).sum(dim=1)  # student score in [-1, 1]
        t = (T_f * T_r).sum(dim=1)  # teacher score in [-1, 1]

        # Student logits
        logits_s = self._pos_scale(self.log_scale_s) * s + self.bias_s

        # Teacher soft target probability (detached)
        logits_t = self._pos_scale(self.log_scale_t) * t + self.bias_t
        p_teacher = torch.sigmoid(logits_t).detach()

        return F.binary_cross_entropy_with_logits(logits_s, p_teacher)
