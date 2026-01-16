import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class EmbeddingPairDataset(Dataset):
    def __init__(self, df):
        self.labels = torch.tensor(
            df.iloc[:, 2].values,
            dtype=torch.float32
        )

        self.fake_emb = torch.tensor(
            df.iloc[:, 3:771].values,
            dtype=torch.float32
        )

        self.real_emb = torch.tensor(
            df.iloc[:, 771:1539].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.fake_emb[idx],
            self.real_emb[idx],
            self.labels[idx]
        )

class TextPairDataset(Dataset):
    def __init__(self, dataframe):
        # Handle both old format (name1, name2, label) and new format (fraudulent_name, real_name, label)
        if 'fraudulent_name' in dataframe.columns and 'real_name' in dataframe.columns:
            self.name1 = dataframe['fraudulent_name'].tolist()
            self.name2 = dataframe['real_name'].tolist()
        elif 'name1' in dataframe.columns and 'name2' in dataframe.columns:
            self.name1 = dataframe['name1'].tolist()
            self.name2 = dataframe['name2'].tolist()
        else:
            raise ValueError("DataFrame must have either (fraudulent_name, real_name) or (name1, name2) columns")
        
        self.label = dataframe['label'].tolist()

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx):
        return self.name1[idx], self.name2[idx], torch.tensor(self.label[idx], dtype=torch.float32)