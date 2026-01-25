import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class EmbeddingPairDataset(Dataset):
    """
    Original task (image-embedding pair + label).

    Expected layout by COLUMN POSITION:
      0: fraudulent_name
      1: real_name
      2: label
      3..770:     embedding_0..embedding_767              (fraud side)
      771..1538:  real_embedding_0..real_embedding_767    (real side)
    """
    def __init__(self, df,
                 fraud_slice=slice(3, 771),
                 real_slice=slice(771, 1539),
                 label_col=2):
        self.labels = torch.tensor(df.iloc[:, int(label_col)].values, dtype=torch.float32)
        self.fake_emb = torch.tensor(df.iloc[:, fraud_slice].values, dtype=torch.float32)
        self.real_emb = torch.tensor(df.iloc[:, real_slice].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.fake_emb[idx], self.real_emb[idx], self.labels[idx]


class Text2ImgDistillDataset(Dataset):
    """
    CASE 2:
      Train student MLP to map text embeddings -> spoof-aware image embeddings.

    Expected layout by COLUMN POSITION:
      0: fraudulent_name
      1: real_name
      2: label  (optional for monitoring)

      3..770:     fraud_img_spoofaware_0..767    (teacher target)
      771..1538:  real_img_spoofaware_0..767     (teacher target)

      1539..2306: fraud_text_emb_0..767          (student input)
      2307..3074: real_text_emb_0..767           (student input)
    """
    def __init__(self, df,
                 fraud_img_slice=slice(3, 771),
                 real_img_slice=slice(771, 1539),
                 fraud_txt_slice=slice(1539, 2307),
                 real_txt_slice=slice(2307, 3075),
                 label_col=2,
                 real_name_col="real_name"):
        # labels (optional, but you already use them)
        self.labels = torch.tensor(df.iloc[:, int(label_col)].values, dtype=torch.float32)

        # teacher targets
        self.fraud_teacher = torch.from_numpy(
            df.iloc[:, fraud_img_slice].to_numpy(dtype=np.float32, copy=False)
        )
        self.real_teacher = torch.from_numpy(
            df.iloc[:, real_img_slice].to_numpy(dtype=np.float32, copy=False)
        )

        # student inputs
        self.fraud_txt = torch.from_numpy(
            df.iloc[:, fraud_txt_slice].to_numpy(dtype=np.float32, copy=False)
        )
        self.real_txt = torch.from_numpy(
            df.iloc[:, real_txt_slice].to_numpy(dtype=np.float32, copy=False)
        )

        # --- NEW: brand_id for multi-positive contrastive loss ---
        if real_name_col not in df.columns:
            raise ValueError(f"Text2ImgDistillDataset needs column '{real_name_col}' to build brand_id.")
        # factorize gives stable int IDs for identical strings
        brand_ids, _ = pd.factorize(df[real_name_col].astype(str), sort=False)
        self.brand_id = torch.tensor(brand_ids.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (
            self.fraud_txt[idx],
            self.real_txt[idx],
            self.fraud_teacher[idx],
            self.real_teacher[idx],
            self.labels[idx],
            self.brand_id[idx],   # <-- NEW
        )