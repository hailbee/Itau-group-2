import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseEmbeddingModel(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=256, image_dim=768, dropout=0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(text_dim)

        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_dim),
        )

        # residual projection (lets it learn near-identity / near-linear maps)
        self.res = nn.Linear(text_dim, image_dim, bias=False)

    def encode_text(self, x):
        x = self.in_norm(x)
        return self.res(x) + self.mlp(x)

    def forward(self, fraud_txt, real_txt):
        z_f = self.encode_text(fraud_txt)
        z_r = self.encode_text(real_txt)
        return z_f, z_r
