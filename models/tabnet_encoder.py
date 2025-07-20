import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_transformer import FeatureTransformer
from models.attentive_transformer import AttentiveTransformer

class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_steps=3):
        """
        hidden_dim: size used inside FeatureTransformer blocks,
        feat_dim = hidden_dim // 2 is the actual output dim of each block.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.steps = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.n_steps = n_steps

        # Each FeatureTransformer(input_dim, hidden_dim) outputs feat_dim = hidden_dim//2
        for _ in range(n_steps):
            self.steps.append(FeatureTransformer(input_dim, hidden_dim))
            self.attentions.append(AttentiveTransformer(input_dim))

        self.feat_dim = hidden_dim // 2
        self.fc = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_representation=False):
        # x: [B, input_dim]
        x = self.bn(x)
        prior = torch.ones_like(x)             # [B, input_dim]
        aggregated = torch.zeros(x.size(0), self.feat_dim, device=x.device)

        for feat_transform, attn_transform in zip(self.steps, self.attentions):
            mask = attn_transform(x, prior)    # [B, input_dim]
            x_masked = x * mask                # [B, input_dim]
            x_trans = feat_transform(x_masked) # [B, feat_dim]
            aggregated += F.relu(x_trans)      # accumulate
            prior = prior * (1 - mask)         # update prior out-of-place

        if return_representation:
            # guaranteed 2D [B, feat_dim]
            return aggregated

        # classification head
        return torch.sigmoid(self.fc(aggregated)).squeeze()
