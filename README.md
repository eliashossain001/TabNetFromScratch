# TabNetFromScratch

**TabNet** is a deep learning architecture tailored for tabular data. It alternates between feature transformers and attentive transformers to achieve sparsity, interpretability, and high accuracy. This repository provides a **from-scratch PyTorch implementation**, including:

- **TabNet Encoder**: for supervised classification.
- **TabNet Decoder**: for self-supervised reconstruction (autoencoder).

# TabNet Main Architecture
<img width="2792" height="1440" alt="image" src="https://github.com/user-attachments/assets/9c40edde-ed10-409e-888c-3774c16face0" />
Figure: Overall structure of TabNet. (a) is the encoder part that encodes the input data with the transformer manner. (b) indicates a decoder that restores the encoded representation to the original data representation. And (c) and (d) show the structure of the feature transformer and the attentive transformer, respectively. 

## 📂 Folder Structure

bash```

TabNetFromScratch/
├── data/
│   └── generate\_data.py         # Generate synthetic dataset
│
├── models/
│   ├── feature\_transformer.py   # FeatureTransformer block (GLU, batch-norm, skip)
│   ├── attentive\_transformer.py # AttentiveTransformer block (masking & softmax)
│   ├── tabnet\_encoder.py        # TabNetEncoder implementation
│   └── tabnet\_decoder.py        # TabNetDecoder implementation
│
├── utils/
│   └── train.py                 # Training loop for supervised learning
│
├── main.py                      # Train & evaluate TabNet for classification
├── run\_decoder.py               # Run TabNet autoencoder reconstruction
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview

```

## 📄 File Descriptions

### `data/generate_data.py`

* **Purpose**: Creates a synthetic dataset (1000 samples, 20 features, binary labels).
* **Usage**:
  ```bash
  python data/generate_data.py
  # Outputs data/synthetic_data.csv

### `models/feature_transformer.py`

* **Class**: `FeatureTransformer`
* **Role**: Applies GLU layers, batch normalization, and skip connections to transform inputs.
  
### `models/attentive_transformer.py`

* **Class**: `AttentiveTransformer`
* **Role**: Computes sparse feature masks via linear layer, batch norm, and softmax.
  
### `models/tabnet_encoder.py`

* **Class**: `TabNetEncoder`
* **Modes**:

  * **Classification**: `forward(x)` → predicts labels.
  * **Representation**: `forward(x, return_representation=True)` → outputs latent features.
* **Parameters**: `input_dim`, `hidden_dim`, `n_steps`.

### `models/tabnet_decoder.py`

* **Class**: `TabNetDecoder`
* **Purpose**: Reconstructs original inputs from encoder representations via a linear decoder.
  
### `utils/train.py`

* **Function**: `train_model(model, X, y, ...)`
* **Role**: Trains `TabNetEncoder` with a classification head using BCELoss.

### `main.py`

* **Workflow**:

  1. Load synthetic data.
  2. Initialize `TabNetEncoder`.
  3. Call `train_model`.
  4. Save encoder weights to `encoder.pt`.
  5. Print classification metrics (accuracy, precision, recall, F1, confusion matrix).


### `run_decoder.py`

* **Workflow**:

  1. Load `encoder.pt`.
  2. Read raw features.
  3. Run encoder in representation mode.
  4. Reconstruct inputs via `TabNetDecoder`.
  5. Print reconstruction metrics (SSE, MSE, RMSE, feature-wise/sample-wise errors).

### `requirements.txt`

* **Dependencies**:

  ```text
  torch
  pandas
  scikit-learn
  ```

## 🚀 Setup & Usage

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data**:

   ```bash
   python data/generate_data.py
   ```

3. **Train for classification**:

   ```bash
   python main.py
   ```

4. **Run autoencoder reconstruction**:

   ```bash
   python run_decoder.py
   ```

## 🔧 Hyperparameter Tuning

* **`hidden_dim`**: Size inside feature transformers (default: 64).
* **`n_steps`**: Number of decision steps (default: 3).
* **Learning rate & epochs**: Adjust in `utils/train.py`.
* **Feature scaling**: Standardize inputs for lower reconstruction error.

## 📚 References

* Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679-6687. https://doi.org/10.1609/aaai.v35i8.16826.

## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD Student | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
