import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def distance(self, a, b):
        return torch.norm(a - b, dim=1)

    def forward(self, embeddings, dists):
        anchor, positive, negative = embeddings
        pos_dist, neg_dist = dists

        pos_embed_dist = self.distance(anchor, positive)
        neg_embed_dist = self.distance(anchor, negative)

        threshold = neg_dist - pos_dist
        loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        return torch.mean(loss)
