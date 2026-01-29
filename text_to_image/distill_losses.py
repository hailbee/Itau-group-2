import torch
import torch.nn as nn
import torch.nn.functional as F

class AUCBestHybridLoss(nn.Module):
    """
    Best-shot loss for ROC AUC:

    L = rank_loss(sim_s, y)                              # main driver for AUC
      + lam_diag * MSE(sim_s, sim_t_diag)                # teacher regularizer
      + lam_mat  * MSE(S_f S_r^T, T_f T_r^T) (optional)  # stronger teacher structure

    Hyperparams for Optuna:
      tau, lam_diag, lam_mat
    """
    def __init__(self, tau=0.05, lam_diag=0.1, lam_mat=0.0, eps=1e-8):
        super().__init__()
        self.tau = float(tau)
        self.lam_diag = float(lam_diag)
        self.lam_mat = float(lam_mat)
        self.eps = float(eps)

    def _pairwise_logistic_rank(self, sim: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y in {0,1}; enforce sim_pos > sim_neg
        pos = sim[y > 0.5]
        neg = sim[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            # no ranking signal in this batch
            return sim.new_tensor(0.0)

        # all pos-vs-neg differences: (P,N)
        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
        # logistic ranking: softplus(-(pos-neg)/tau)
        return F.softplus(-diffs / self.tau).mean()

    def forward(self, pred_fraud, pred_real, fraud_teacher, real_teacher, label):
        y = label.float()

        S_f = F.normalize(pred_fraud, dim=1, eps=self.eps)
        S_r = F.normalize(pred_real,  dim=1, eps=self.eps)
        sim_s = (S_f * S_r).sum(dim=1)  # (B,)

        # main AUC driver
        rank_loss = self._pairwise_logistic_rank(sim_s, y)

        # teacher regularizer (diagonal sim)
        reg_diag = sim_s.new_tensor(0.0)
        reg_mat  = sim_s.new_tensor(0.0)

        if self.lam_diag > 0.0 or self.lam_mat > 0.0:
            T_f = F.normalize(fraud_teacher, dim=1, eps=self.eps).detach()
            T_r = F.normalize(real_teacher,  dim=1, eps=self.eps).detach()
            sim_t = (T_f * T_r).sum(dim=1)

            if self.lam_diag > 0.0:
                reg_diag = (sim_s - sim_t).pow(2).mean()

            if self.lam_mat > 0.0:
                M_s = S_f @ S_r.T
                M_t = T_f @ T_r.T
                B = M_s.size(0)
                mask = ~torch.eye(B, dtype=torch.bool, device=M_s.device)
                reg_mat = (M_s[mask] - M_t[mask]).pow(2).mean()

        return rank_loss + self.lam_diag * reg_diag + self.lam_mat * reg_mat
