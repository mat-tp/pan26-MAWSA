"""
PyTorch-based neural network models for author switch detection.

Provides GPU-accelerated deep learning models with flexible architectures
supporting both CPU and GPU training. Includes hyperparameter optimization
for high-dimensional stylometric feature spaces.

Models:
- TorchMLP: Multi-layer perceptron with batch normalization and dropout
- TorchLSTM: Sequence model for capturing temporal dependencies
- TorchEnsemble: Ensemble of multiple neural architectures
"""

import os
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Install with: pip install torch. "
        "GPU support requires: pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Device Management
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Return CUDA device if available, otherwise CPU. Requires PyTorch."""
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Cannot create device.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[pytorch] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[pytorch] CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device("cpu")
        print("[pytorch] GPU not available. Using CPU (will be slower).")

    return device


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network Models (only defined when PyTorch is available)
# ─────────────────────────────────────────────────────────────────────────────

if PYTORCH_AVAILABLE:

    class TorchMLP(nn.Module):
        """
        Multi-layer perceptron for high-dimensional stylometric features.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = None,
            dropout_rate: float = 0.3,
            batch_norm: bool = True,
            activation: str = "relu",
        ):
            super(TorchMLP, self).__init__()

            if hidden_dims is None:
                hidden_dims = [512, 256, 128]

            self.dropout_rate = dropout_rate
            self.batch_norm = batch_norm

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                if activation.lower() == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation.lower() == "gelu":
                    layers.append(nn.GELU())
                elif activation.lower() == "elu":
                    layers.append(nn.ELU(inplace=True))
                else:
                    layers.append(nn.ReLU(inplace=True))

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.squeeze(1)
            return self.network(x)


    class TorchLSTM(nn.Module):
        """
        LSTM-based model for sequence of stylometric features.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 2,
            bidirectional: bool = True,
            dropout_rate: float = 0.3,
            use_attention: bool = True,
        ):
            super(TorchLSTM, self).__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.use_attention = use_attention

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

            if use_attention:
                self.attention = nn.Linear(lstm_output_dim, 1)

            self.fc1 = nn.Linear(lstm_output_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            lstm_out, (hidden, cell) = self.lstm(x)

            if self.use_attention:
                attention_weights = self.attention(torch.tanh(lstm_out))
                attention_weights = torch.softmax(attention_weights, dim=1)
                context = (lstm_out * attention_weights).sum(dim=1)
            else:
                context = lstm_out[:, -1, :]

            x_out = self.fc1(context)
            x_out = self.bn1(x_out)
            x_out = torch.relu(x_out)
            x_out = self.dropout(x_out)

            x_out = self.fc2(x_out)
            x_out = self.bn2(x_out)
            x_out = torch.relu(x_out)
            x_out = self.dropout(x_out)

            x_out = self.fc3(x_out)
            x_out = self.sigmoid(x_out)

            return x_out


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn-compatible Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PyTorchClassifier:
    """
    Sklearn-compatible wrapper for PyTorch neural networks.
    Falls back gracefully when PyTorch is not installed.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        input_dim: Optional[int] = None,
        hidden_dims: List[int] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        device=None,
        verbose: bool = True,
    ):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        self.device = device or get_device()
        self.model = None
        self.optimizer = None
        self.best_model_state = None
        self.training_history = {"train_loss": [], "val_loss": []}

    def _build_model(self, input_dim: int):
        if self.model_type == "mlp":
            model = TorchMLP(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
            )
        elif self.model_type == "lstm":
            model = TorchLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dims[0] if self.hidden_dims else 256,
                dropout_rate=self.dropout_rate,
                use_attention=True,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.model_type == "lstm" and X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        X = self._prepare_input(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        input_dim = X.shape[-1]
        self.input_dim = input_dim
        self.n_features_in_ = input_dim

        self.model = self._build_model(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        train_dataset = TensorDataset(
            torch.from_numpy(X).to(self.device),
            torch.from_numpy(y).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = self._prepare_input(X_val)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            val_dataset = TensorDataset(
                torch.from_numpy(X_val).to(self.device),
                torch.from_numpy(y_val).to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.training_history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.training_history["val_loss"].append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if self.verbose:
                                print(f"[pytorch] Early stopping at epoch {epoch+1}")
                            break

            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"[pytorch] Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        X = self._prepare_input(X)
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[start:start + batch_size]).to(self.device)
                out = self.model(batch).cpu().numpy().flatten()
                all_proba.append(out)
        proba_1 = np.concatenate(all_proba)
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_state_dict'] = self.model.state_dict()
            state['model'] = None
        return state

    def __setstate__(self, state):
        model_state_dict = state.pop('model_state_dict', None)
        self.__dict__.update(state)
        if model_state_dict is not None and self.input_dim is not None:
            self.model = self._build_model(self.input_dim)
            self.model.load_state_dict(model_state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory functions
# ─────────────────────────────────────────────────────────────────────────────

def make_torch_mlp(input_dim: Optional[int] = None, hidden_dims: List[int] = None, **kwargs):
    """Factory function for PyTorch MLP classifier."""
    return PyTorchClassifier(
        model_type="mlp",
        input_dim=input_dim,
        hidden_dims=hidden_dims or [512, 256, 128],
        **kwargs
    )


def make_torch_lstm(input_dim: Optional[int] = None, **kwargs):
    """Factory function for PyTorch LSTM classifier."""
    return PyTorchClassifier(
        model_type="lstm",
        input_dim=input_dim,
        **kwargs
    )