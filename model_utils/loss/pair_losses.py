import torch.nn as nn
import torch.nn.functional as F
import torch

class CosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cos_sim = F.cosine_similarity(z1, z2)
        cos_dist = 1 - cos_sim
        loss = label * cos_dist.pow(2) + (1 - label) * F.relu(self.margin - cos_dist).pow(2)
        return loss.mean()

class EuclideanLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        euclidean_dist = F.pairwise_distance(z1, z2, p=2)
        loss = label * euclidean_dist.pow(2) + (1 - label) * F.relu(self.margin - euclidean_dist).pow(2)
        return loss.mean() 
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        """
        z1, z2: [B, D] projected embeddings
        y:      [B] in {0,1}  (1 = positive / spoof)
        """

        # Ensure float labels
        y = y.float()

        # L2 normalize (paper requirement)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Euclidean distance
        d = torch.norm(z1 - z2, dim=1)

        pos_loss = y * d.pow(2)
        neg_loss = (1 - y) * torch.clamp(self.margin - d, min=0).pow(2)

        return (pos_loss + neg_loss).mean()