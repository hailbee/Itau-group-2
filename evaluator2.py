# scripts/evaluation/evaluator2.py

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict


@dataclass
class EvalConfig:
    batch_size: int = 512              # model forward batch size
    retrieval_batch_size: int = 512    # query batch size for retrieval
    ks: tuple[int, ...] = (1, 5, 10)   # evaluate recall@K for these K values
    max_k: int = 10                    # must be >= max(ks)

    # column slices
    fraud_img_slice: tuple[int, int] = (3, 771)
    real_img_slice: tuple[int, int] = (771, 1539)
    fraud_txt_slice: tuple[int, int] = (1539, 2307)
    real_txt_slice: tuple[int, int] = (2307, 3075)

    # column name for grouping
    real_name_col: str = "real_name"


class Evaluator2:
    """
    FAST group-aware evaluator for TEXT → IMAGE alignment.

    Measures:
      (1) Pointwise cosine alignment (student vs teacher)
      (2) Group-aware Recall@K (K in cfg.ks)
          - Correct if ANY of top-K retrieved items belongs to the same real_name group.
      (3) Approx MRR@max_k (MRR but only within top max_k, else contributes 0)

    This is scalable:
      - Uses torch.topk instead of full argsort
      - No NxN matrix allocation
      - Complexity ~ O(N * max_k)
    """

    def __init__(self, model, cfg: EvalConfig | None = None):
        self.model = model
        self.cfg = cfg or EvalConfig()
        if self.cfg.max_k < max(self.cfg.ks):
            raise ValueError("EvalConfig.max_k must be >= max(EvalConfig.ks)")

    @torch.no_grad()
    def evaluate(self, filepath: str, max_rows: int | None = None):
        # ------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------
        df = pd.read_parquet(filepath)
        if max_rows is not None:
            df = df.head(int(max_rows))

        device = next(self.model.parameters()).device
        self.model.eval()

        fi, fe = self.cfg.fraud_img_slice
        ri, re = self.cfg.real_img_slice
        ft, fte = self.cfg.fraud_txt_slice
        rt, rte = self.cfg.real_txt_slice

        fraud_img = torch.from_numpy(df.iloc[:, fi:fe].to_numpy(np.float32)).to(device)
        real_img = torch.from_numpy(df.iloc[:, ri:re].to_numpy(np.float32)).to(device)
        fraud_txt = torch.from_numpy(df.iloc[:, ft:fte].to_numpy(np.float32)).to(device)
        real_txt = torch.from_numpy(df.iloc[:, rt:rte].to_numpy(np.float32)).to(device)

        # ------------------------------------------------------------
        # Build many-to-one groups based on real_name
        # ------------------------------------------------------------
        if self.cfg.real_name_col not in df.columns:
            raise ValueError(
                f"Evaluator2 needs column '{self.cfg.real_name_col}' for group-aware retrieval."
            )

        real_names = df[self.cfg.real_name_col].astype(str).tolist()
        brand_to_indices = defaultdict(list)
        for idx, name in enumerate(real_names):
            brand_to_indices[name].append(idx)

        # Convert index lists to torch tensors on CPU (small) for faster checks
        # We'll use Python sets for membership checks in topK which is tiny.
        brand_to_set = {k: set(v) for k, v in brand_to_indices.items()}

        # ------------------------------------------------------------
        # Student forward (TEXT -> predicted image-space embeddings)
        # ------------------------------------------------------------
        z_fraud_txt, z_real_txt = self.model(fraud_txt, real_txt)
        z_fraud_txt = F.normalize(z_fraud_txt, dim=1)
        z_real_txt = F.normalize(z_real_txt, dim=1)

        # Teacher image embeddings (already spoof-aware, precomputed)
        fraud_img = F.normalize(fraud_img, dim=1)
        real_img = F.normalize(real_img, dim=1)

        # ============================================================
        # (1) ALIGNMENT METRICS
        # ============================================================
        cos_fraud = F.cosine_similarity(z_fraud_txt, fraud_img, dim=1).cpu().numpy()
        cos_real = F.cosine_similarity(z_real_txt, real_img, dim=1).cpu().numpy()

        alignment_metrics = {
            "fraud_mean_cosine": float(np.mean(cos_fraud)),
            "fraud_median_cosine": float(np.median(cos_fraud)),
            "fraud_std_cosine": float(np.std(cos_fraud)),
            "real_mean_cosine": float(np.mean(cos_real)),
            "real_median_cosine": float(np.median(cos_real)),
            "real_std_cosine": float(np.std(cos_real)),
        }

        # ============================================================
        # (2) FAST GROUP-AWARE RETRIEVAL: top-K only
        # ============================================================
        ks = tuple(sorted(set(int(k) for k in self.cfg.ks)))
        max_k = int(self.cfg.max_k)

        # counters
        hits_at_k = {k: 0 for k in ks}
        mrr_at_maxk_sum = 0.0

        B = int(self.cfg.retrieval_batch_size)
        N = real_img.shape[0]
        real_img_T = real_img.T  # (D, N)

        for start in tqdm(range(0, z_fraud_txt.shape[0], B), desc="Retrieval topK (fast)"):
            end = min(start + B, z_fraud_txt.shape[0])
            queries = z_fraud_txt[start:end]          # (B, D)
            sims = queries @ real_img_T               # (B, N)

            # only take top max_k indices
            topk_idx = torch.topk(sims, k=max_k, dim=1, largest=True).indices  # (B, max_k)
            topk_idx_cpu = topk_idx.detach().cpu().numpy()

            # evaluate each query row with group-aware correctness
            for i in range(topk_idx_cpu.shape[0]):
                global_idx = start + i
                brand = real_names[global_idx]
                valid = brand_to_set[brand]

                row = topk_idx_cpu[i].tolist()

                # position of first correct (within top max_k)
                first_pos = None
                for pos, idx in enumerate(row):
                    if idx in valid:
                        first_pos = pos + 1  # 1-indexed
                        break

                # Recall@K: hit if any in top-K
                for k in ks:
                    if first_pos is not None and first_pos <= k:
                        hits_at_k[k] += 1

                # Approx MRR@max_k (0 if miss)
                if first_pos is not None:
                    mrr_at_maxk_sum += 1.0 / float(first_pos)

        total = int(z_fraud_txt.shape[0])
        retrieval_metrics = {f"recall@{k}": float(hits_at_k[k] / total) for k in ks}
        retrieval_metrics[f"mrr@{max_k}"] = float(mrr_at_maxk_sum / total)

        return {
            "alignment": alignment_metrics,
            "retrieval": retrieval_metrics,
            "notes": {
                "group_aware": True,
                "topk_only": True,
                "max_k": max_k,
                "evaluated_ks": list(ks),
                "mrr_is_truncated": True,
            },
        }
