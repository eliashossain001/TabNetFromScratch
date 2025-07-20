import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from models.tabnet_encoder import TabNetEncoder
from models.tabnet_decoder import TabNetDecoder

# 1) Load data
df = pd.read_csv("data/synthetic_data.csv")
feature_names = df.columns.drop("target")
X = torch.tensor(df[feature_names].values, dtype=torch.float32)

# 2) Dataloader
loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)

# 3) Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = TabNetEncoder(input_dim=X.shape[1], hidden_dim=64, n_steps=3).to(device)
decoder = TabNetDecoder(input_dim=X.shape[1], repr_dim=64//2).to(device)

# 4) Load weights
encoder.load_state_dict(torch.load("encoder.pt", map_location=device))
encoder.eval()
decoder.eval()

# 5) Accumulate errors
mse_loss = nn.MSELoss(reduction="none")
total_loss = 0.0
num_samples = 0
all_sample_mse = []
feature_error_sum = torch.zeros(X.shape[1], device=device)

with torch.no_grad():
    for batch, in loader:
        b = batch.to(device)
        rep = encoder(b, return_representation=True)
        rec = decoder(rep)
        errors = mse_loss(rec, b)                 # [B, F]
        
        # total sum-of-squares
        total_loss += errors.sum().item()
        
        # sample-wise MSE
        sample_mse = errors.mean(dim=1).cpu().numpy()
        all_sample_mse.extend(sample_mse.tolist())
        
        # feature-wise sum-of-squares
        feature_error_sum += errors.sum(dim=0)
        
        num_samples += b.size(0)

# 6) Compute summary metrics
mean_mse_per_sample = total_loss / num_samples
rmse = np.sqrt(mean_mse_per_sample)

feature_mse = (feature_error_sum / num_samples).cpu().numpy()
feature_df = pd.DataFrame({"feature": feature_names, "mse": feature_mse})
feature_df = feature_df.sort_values("mse", ascending=False)

sample_df = pd.DataFrame({"sample_index": np.arange(len(all_sample_mse)),
                          "mse": all_sample_mse})
sample_stats = sample_df["mse"].describe()
top3 = sample_df.nlargest(3, "mse")

# 7) Print everything
print(f"\n Total Reconstruction SSE: {total_loss:.4f}")
print(f" Mean MSE per sample: {mean_mse_per_sample:.4f}")
print(f" RMSE: {rmse:.4f}\n")

print(" Feature-wise MSE (top 5):")
print(feature_df.head(5).to_string(index=False), "\n")

print(" Sample MSE statistics:")
print(sample_stats.to_string(), "\n")

print(" Top 3 worst-reconstructed samples:")
print(top3.to_string(index=False))
