import pandas as pd
from models.tabnet_encoder import TabNetEncoder
from utils.train import train_model
import torch

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load synthetic dataset
df = pd.read_csv("data/synthetic_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Define and train TabNet model
model = TabNetEncoder(input_dim=X.shape[1], hidden_dim=64, n_steps=3)
train_model(model, X, y, epochs=20)

# Save the model
torch.save(model.state_dict(), "encoder.pt")
print("Saved encoder weights to encoder.pt")

# Evaluation
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)  # Move to same device

with torch.no_grad():
    preds = model(X_tensor)
    y_pred = (preds > 0.5).int().cpu().numpy()  # Move to CPU before converting to NumPy

# Results
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Classification Report:\n")
print(classification_report(y, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\n Accuracy Score:", accuracy_score(y, y_pred))
