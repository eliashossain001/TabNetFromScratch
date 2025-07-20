import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x, prior):
        x = self.linear(x)
        x = self.bn(x)
        x = x * prior
        return F.softmax(x, dim=1)
