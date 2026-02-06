import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEmbeddingModel(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, image_dim=768):
        super().__init__()
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
        )

    def encode_text(self, x):
        return self.text_head(x)

    def forward(self, fraud_txt, real_txt):
        z_f = self.text_head(fraud_txt)
        z_r = self.text_head(real_txt)
        return z_f, z_r
