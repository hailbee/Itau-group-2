import torch
from torch.utils.data import Dataset
import numpy as np


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


class TextPairDataset(Dataset):
    """
    Kept for compatibility (raw text pairs).
    """
    def __init__(self, dataframe):
        if 'fraudulent_name' in dataframe.columns and 'real_name' in dataframe.columns:
            self.name1 = dataframe['fraudulent_name'].tolist()
            self.name2 = dataframe['real_name'].tolist()
        elif 'name1' in dataframe.columns and 'name2' in dataframe.columns:
            self.name1 = dataframe['name1'].tolist()
            self.name2 = dataframe['name2'].tolist()
        else:
            raise ValueError("DataFrame must have either (fraudulent_name, real_name) or (name1, name2) columns")
        if 'label' not in dataframe.columns:
            raise ValueError("DataFrame must have a 'label' column")
        self.label = dataframe['label'].tolist()

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx):
        return self.name1[idx], self.name2[idx], torch.tensor(self.label[idx], dtype=torch.float32)


class Text2ImgDistillDataset(Dataset):
    """
    CASE 2:
      Train student MLP to map text embeddings -> spoof-aware image embeddings.

    Expected layout by COLUMN POSITION:
      0: fraudulent_name
      1: real_name
      2: label  (optional for distill loss, used for ROC/AUC monitoring)

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
                 label_col=2):
        self.labels = torch.tensor(df.iloc[:, int(label_col)].values, dtype=torch.float32)

        self.fraud_teacher = torch.from_numpy(
            df.iloc[:, fraud_img_slice].to_numpy(dtype=np.float32, copy=False)
        )
        self.real_teacher = torch.from_numpy(
            df.iloc[:, real_img_slice].to_numpy(dtype=np.float32, copy=False)
        )

        self.fraud_txt = torch.from_numpy(
            df.iloc[:, fraud_txt_slice].to_numpy(dtype=np.float32, copy=False)
        )
        self.real_txt = torch.from_numpy(
            df.iloc[:, real_txt_slice].to_numpy(dtype=np.float32, copy=False)
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (
            self.fraud_txt[idx],
            self.real_txt[idx],
            self.fraud_teacher[idx],
            self.real_teacher[idx],
            self.labels[idx],
        )
