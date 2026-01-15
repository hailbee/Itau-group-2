import torch

class SiameseEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x1, x2):
        z1 = self.head(x1)
        z2 = self.head(x2)
        return z1, z2

    def encode(self, x):
        # dummy encode method for compatibility
        return x
