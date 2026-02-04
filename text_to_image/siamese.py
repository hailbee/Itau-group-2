import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()

        self.text_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, out_dim),
        )

    def encode_text(self, x):
        return self.text_head(x)

    def forward(self, fraud_txt, real_txt):
        z_f = self.encode_text(fraud_txt)
        z_r = self.encode_text(real_txt)
        return z_f, z_r
