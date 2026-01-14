from ..base import BaseSiameseModel
import torch

class SiameseModelPairs(BaseSiameseModel):
    """
    Siamese network for pair-wise learning using any vision-language model as backbone.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None):
        super().__init__(embedding_dim, projection_dim, backbone)
    
    def forward(self, text1, text2, label=None):
        z1 = self.encode(text1)
        z2 = self.encode(text2)
        # Move label to the same device as z1 if it's a tensor
        if label is not None and hasattr(label, 'device'):
            label = label.to(z1.device)
        return z1, z2, label

class SiameseModelTriplet(BaseSiameseModel):
    """
    Siamese network for triplet learning using any vision-language model as backbone.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None):
        super().__init__(embedding_dim, projection_dim, backbone)
    
    def forward(self, anchor_texts, positive_texts=None, negative_texts=None):
        z_anchor = self.encode(anchor_texts)
        if positive_texts is None and negative_texts is None:
            # Inference mode: only anchor embeddings needed
            return z_anchor
        z_positive = self.encode(positive_texts)
        z_negative = self.encode(negative_texts)
        return z_anchor, z_positive, z_negative

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
