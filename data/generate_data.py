import pandas as pd
from sklearn.datasets import make_classification

def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=10,
                               n_redundant=5,
                               n_classes=n_classes,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("../data/synthetic_data.csv", index=False)
    print("Synthetic data saved to data/synthetic_data.csv")
