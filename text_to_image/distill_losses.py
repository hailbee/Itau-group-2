import torch
import torch.nn as nn
import torch.nn.functional as F

class AUCBestHybridLoss(nn.Module):
    """
    AUC-focused hybrid loss (keeps lam_diag + lam_mat) + optional improvements:

    Base:
      L = rank_loss(sim_s, y)                              # main AUC driver
        + lam_diag * MSE(sim_s, sim_t)                     # teacher diagonal sim regularizer
        + lam_mat  * MSE( (S_f S_r^T)_off, (T_f T_r^T)_off) # teacher cross-view matrix regularizer

    Add-ons (optional, cheap-ish):
      - teacher-weighted ranking: weight each pos-vs-neg pair by teacher separation
      - moment matching: match mean/std of sim_s to sim_t (stabilizes scale)

    Hyperparams to tune with Optuna:
      tau, lam_diag, lam_mat,
      w_teacher (0 disables teacher-weighted rank),
      w_pow (shape of teacher weights),
      lam_mom (0 disables moment matching)
    """
    def __init__(
        self,
        tau=0.05,
        lam_diag=0.1,
        lam_mat=0.0,
        w_teacher=0.0,     # 0 => original unweighted ranking
        w_pow=1.0,         # exponent on teacher margin weights
        lam_mom=0.0,       # 0 => no moment matching
        eps=1e-8,
    ):
        super().__init__()
        self.tau = float(tau)
        self.lam_diag = float(lam_diag)
        self.lam_mat = float(lam_mat)
        self.w_teacher = float(w_teacher)
        self.w_pow = float(w_pow)
        self.lam_mom = float(lam_mom)
        self.eps = float(eps)

    def _pairwise_logistic_rank(self, sim: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y in {0,1}; enforce sim_pos > sim_neg
        pos = sim[y > 0.5]
        neg = sim[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return sim.new_tensor(0.0)

        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)  # (P, N)
        return F.softplus(-diffs / self.tau).mean()

    def _pairwise_teacher_weighted_rank(self, sim_s: torch.Tensor, sim_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # weight each (pos,neg) pair by teacher margin max(sim_t_pos - sim_t_neg, 0)^w_pow
        pos_s = sim_s[y > 0.5]
        neg_s = sim_s[y <= 0.5]
        pos_t = sim_t[y > 0.5]
        neg_t = sim_t[y <= 0.5]

        if pos_s.numel() == 0 or neg_s.numel() == 0:
            return sim_s.new_tensor(0.0)

        diffs_s = pos_s.unsqueeze(1) - neg_s.unsqueeze(0)  # (P, N)
        diffs_t = pos_t.unsqueeze(1) - neg_t.unsqueeze(0)  # (P, N)

        # weights: only trust teacher where it separates (positive margin)
        w = torch.clamp(diffs_t, min=0.0)
        if self.w_pow != 1.0:
            w = w.pow(self.w_pow)

        # normalize weights so loss scale is stable across batches
        w_mean = w.mean().clamp_min(1e-12)
        w = w / w_mean

        return (w * F.softplus(-diffs_s / self.tau)).mean()

    def _moment_match(self, sim_s: torch.Tensor, sim_t: torch.Tensor) -> torch.Tensor:
        # match mean and std (cheap stabilizer)
        ms = sim_s.mean()
        mt = sim_t.mean()
        ss = sim_s.std(unbiased=False).clamp_min(1e-6)
        st = sim_t.std(unbiased=False).clamp_min(1e-6)
        return (ms - mt).pow(2) + (ss - st).pow(2)

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label):
        y = label.float()

        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        sim_s = (S_f * S_r).sum(dim=1)  # (B,)

        # If no teacher terms used at all, we can skip computing teacher embeddings.
        need_teacher = (self.lam_diag > 0.0) or (self.lam_mat > 0.0) or (self.w_teacher > 0.0) or (self.lam_mom > 0.0)

        sim_t = None
        T_f = None
        T_r = None

        if need_teacher:
            T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps).detach()
            T_r = F.normalize(real_teacher,  dim=1, eps=self.eps).detach()
            sim_t = (T_f * T_r).sum(dim=1)

        # ----- main AUC driver -----
        if self.w_teacher > 0.0 and sim_t is not None:
            rank_loss = self._pairwise_teacher_weighted_rank(sim_s, sim_t, y)
            # blend with plain rank (helps if teacher is noisy)
            rank_loss = (1.0 - self.w_teacher) * self._pairwise_logistic_rank(sim_s, y) + self.w_teacher * rank_loss
        else:
            rank_loss = self._pairwise_logistic_rank(sim_s, y)

        # ----- teacher regularizers (your originals) -----
        reg_diag = sim_s.new_tensor(0.0)
        reg_mat  = sim_s.new_tensor(0.0)

        if self.lam_diag > 0.0 and sim_t is not None:
            reg_diag = (sim_s - sim_t).pow(2).mean()

        if self.lam_mat > 0.0 and (T_f is not None) and (T_r is not None):
            M_s = S_f @ S_r.T
            M_t = T_f @ T_r.T
            B = M_s.size(0)
            mask = ~torch.eye(B, dtype=torch.bool, device=M_s.device)
            reg_mat = (M_s[mask] - M_t[mask]).pow(2).mean()

        # ----- optional moment matching -----
        mom = sim_s.new_tensor(0.0)
        if self.lam_mom > 0.0 and sim_t is not None:
            mom = self._moment_match(sim_s, sim_t)

        return rank_loss + self.lam_diag * reg_diag + self.lam_mat * reg_mat + self.lam_mom * mom