import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, teacher_dim=None):
        super().__init__()

        self.text_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.teacher_proj = None
        if teacher_dim is not None:
            self.teacher_proj = nn.Sequential(
                nn.Linear(teacher_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def encode_text(self, x):
        return self.text_head(x)  # let trainer normalize

    def encode_teacher(self, t):
        if self.teacher_proj is None:
            raise RuntimeError("teacher_proj not initialized (pass teacher_dim to model)")
        return self.teacher_proj(t)

    def forward(self, fraud_txt, real_txt):
        z_f = self.encode_text(fraud_txt)
        z_r = self.encode_text(real_txt)
        return z_f, z_r
