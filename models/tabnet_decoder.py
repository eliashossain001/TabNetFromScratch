import torch.nn as nn

class TabNetDecoder(nn.Module):
    def __init__(self, input_dim, repr_dim):
        """
        input_dim: number of original features
        repr_dim: feature dimension from encoder (hidden_dim//2)
        This is a minimal decoder: a single linear layer to reconstruct the inputs.
        """
        super().__init__()
        self.fc = nn.Linear(repr_dim, input_dim)

    def forward(self, encoded):
        # encoded: [B, repr_dim]  or [repr_dim] if B=1
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        # reconstructed: [B, input_dim]
        return self.fc(encoded)
