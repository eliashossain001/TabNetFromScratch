import torch.nn as nn
import torch.nn.functional as F

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GLU()
        )
        self.skip = nn.Linear(input_dim, hidden_dim // 2)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + self.skip(x)
