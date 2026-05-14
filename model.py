"""
model.py — PyTorch model definitions, training loop, and evaluation.

build_cnn()        → 1D CNN classifier
build_cnn_bigru()  → CNN + Bidirectional GRU hybrid (IEEE paper architecture)
build_mlp()        → MLP for K-mer frequency inputs
train_model()      → trains the model, streams epoch progress to Streamlit
evaluate_model()   → confusion matrix, classification report, ROC data
"""

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cpu")  # Streamlit Cloud has no GPU


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    """1D Convolutional Neural Network for sequence classification."""

    def __init__(self, input_channels: int, seq_len: int,
                 filters: int, kernel_size: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pooling
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, channels) → conv expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.classifier(x)


class CNNBiGRUClassifier(nn.Module):
    """CNN + Bidirectional GRU hybrid — IEEE paper architecture."""

    def __init__(self, input_channels: int, seq_len: int,
                 filters: int, gru_units: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, filters, 5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.bigru1 = nn.GRU(filters, gru_units, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(gru_units * 2, gru_units // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_units, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (batch, channels, seq_len)
        x = self.conv(x)                 # (batch, filters, seq_len/2)
        x = x.permute(0, 2, 1)          # (batch, seq_len/2, filters)
        x, _ = self.bigru1(x)           # (batch, seq_len/2, gru_units*2)
        x, _ = self.bigru2(x)           # (batch, seq_len/2, gru_units)
        x = x[:, -1, :]                 # last timestep
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """MLP for flat K-mer frequency inputs."""

    def __init__(self, input_dim: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple, filters: int = 64,
              kernel_size: int = 5, dropout: float = 0.3) -> nn.Module:
    seq_len, channels = input_shape
    return CNNClassifier(channels, seq_len, filters, kernel_size, dropout).to(DEVICE)


def build_cnn_bigru(input_shape: tuple, filters: int = 64,
                    gru_units: int = 64, dropout: float = 0.3) -> nn.Module:
    seq_len, channels = input_shape
    return CNNBiGRUClassifier(channels, seq_len, filters, gru_units, dropout).to(DEVICE)


def build_mlp(input_shape: tuple, dropout: float = 0.3) -> nn.Module:
    input_dim = input_shape[0]
    return MLPClassifier(input_dim, dropout).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    progress_bar,
    status_text,
) -> tuple:
    """
    Train model with Adam + CrossEntropyLoss.
    Returns (model, history_dict, X_test, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    def to_tensors(Xa, ya):
        return (
            torch.tensor(Xa, dtype=torch.float32).to(DEVICE),
            torch.tensor(ya, dtype=torch.long).to(DEVICE),
        )

    Xt, yt = to_tensors(X_train, y_train)
    Xv, yv = to_tensors(X_val,   y_val)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    optimizer  = optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
        train_loss = total_loss / total
        train_acc  = correct / total

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out  = model(Xv)
            val_loss = criterion(val_out, yv).item()
            val_acc  = (val_out.argmax(1) == yv).float().mean().item()

        scheduler.step(val_loss)

        history["loss"].append(round(train_loss, 4))
        history["accuracy"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        # ── progress ───────────────────────────────────────────────────────
        progress_bar.progress((epoch + 1) / epochs)
        status_text.markdown(
            f"**Epoch {epoch + 1}/{epochs}** — "
            f"loss: `{train_loss:.4f}` · acc: `{train_acc:.4f}` · "
            f"val_loss: `{val_loss:.4f}` · val_acc: `{val_acc:.4f}`"
        )

        # ── early stopping ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                status_text.markdown(
                    f"**Early stopping** at epoch {epoch + 1} — "
                    f"best val acc: `{best_val_acc:.4f}`"
                )
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, history, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Confusion matrix, classification report, and one-vs-rest ROC curves."""
    model.eval()
    with torch.no_grad():
        Xt    = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(Xt)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = np.argmax(proba, axis=1)

    report = classification_report(
        y_test, y_pred,
        target_names=["Coding", "lncRNA", "Pseudogene"],
        output_dict=True,
    )
    cm    = confusion_matrix(y_test, y_pred)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc   = {}
    for i, cls in enumerate(["Coding", "lncRNA", "Pseudogene"]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc[cls]    = (fpr.tolist(), tpr.tolist(), round(auc(fpr, tpr), 4))

    return {
        "report":           report,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
        "y_pred_proba":     proba,
        "y_test":           y_test,
        "roc":              roc,
    }
