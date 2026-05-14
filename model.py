"""
model.py — Model definitions, training loop, and evaluation.

build_cnn()           → 1D CNN classifier
build_cnn_bigru()     → CNN + Bidirectional GRU hybrid (IEEE paper architecture)
build_mlp()           → Simple MLP for K-mer frequency inputs
train_model()         → trains whichever model is passed, streams progress to Streamlit
evaluate_model()      → confusion matrix, classification report, ROC data
"""

import numpy as np
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


# ── LAZY TENSORFLOW IMPORT ────────────────────────────────────────────────────
# Imported inside functions so the app loads fast even on slow machines.

def _keras():
    import tensorflow as tf
    return tf.keras


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class _ProgressCallback:
    """Wraps a Keras callback that writes epoch results to Streamlit widgets."""

    def __new__(cls, progress_bar, status_text, total_epochs):
        keras = _keras()

        class _CB(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.bar    = progress_bar
                self.text   = status_text
                self.total  = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                self.bar.progress((epoch + 1) / self.total)
                self.text.markdown(
                    f"**Epoch {epoch + 1}/{self.total}** — "
                    f"loss: `{logs.get('loss', 0):.4f}` · "
                    f"acc: `{logs.get('accuracy', 0):.4f}` · "
                    f"val_loss: `{logs.get('val_loss', 0):.4f}` · "
                    f"val_acc: `{logs.get('val_accuracy', 0):.4f}`"
                )

        return _CB()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(
    input_shape: tuple,
    n_classes: int = 3,
    filters: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.3,
):
    """1D Convolutional Neural Network."""
    keras = _keras()
    inp = keras.Input(shape=input_shape)
    x   = keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.MaxPooling1D(2)(x)
    x   = keras.layers.Conv1D(filters * 2, kernel_size, activation="relu", padding="same")(x)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.GlobalAveragePooling1D()(x)
    x   = keras.layers.Dropout(dropout)(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="CNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_bigru(
    input_shape: tuple,
    n_classes: int = 3,
    filters: int = 64,
    gru_units: int = 64,
    dropout: float = 0.3,
):
    """CNN + Bidirectional GRU hybrid — replicates the IEEE paper architecture."""
    keras = _keras()
    inp = keras.Input(shape=input_shape)
    x   = keras.layers.Conv1D(filters, 5, activation="relu", padding="same")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.MaxPooling1D(2)(x)
    x   = keras.layers.Bidirectional(keras.layers.GRU(gru_units, return_sequences=True))(x)
    x   = keras.layers.Bidirectional(keras.layers.GRU(gru_units // 2))(x)
    x   = keras.layers.Dropout(dropout)(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="CNN_BiGRU")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp(
    input_shape: tuple,
    n_classes: int = 3,
    dropout: float = 0.3,
):
    """MLP for flat K-mer frequency inputs (no sequence order preserved)."""
    keras = _keras()
    inp = keras.Input(shape=input_shape)
    x   = keras.layers.Dense(256, activation="relu")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(dropout)(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="MLP_Kmer")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    progress_bar,
    status_text,
) -> tuple:
    """
    Split data, train with early stopping, stream progress.

    Returns (model, history_dict, X_test, y_test)
    """
    keras = _keras()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    callbacks = [
        _ProgressCallback(progress_bar, status_text, epochs),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history.history, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run inference on the test split and return a results dict containing:
        report         — sklearn classification_report as dict
        confusion_matrix — np.ndarray (3x3)
        y_pred         — predicted class indices
        y_pred_proba   — softmax probabilities (N, 3)
        y_test         — true labels
        roc            — {class: (fpr, tpr, auc)} for one-vs-rest ROC
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    report = classification_report(
        y_test, y_pred,
        target_names=["Coding", "lncRNA", "Pseudogene"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    # One-vs-rest ROC
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc   = {}
    for i, cls in enumerate(["Coding", "lncRNA", "Pseudogene"]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc[cls]    = (fpr.tolist(), tpr.tolist(), round(auc(fpr, tpr), 4))

    return {
        "report":           report,
        "confusion_matrix": cm,
        "y_pred":           y_pred,
        "y_pred_proba":     y_pred_proba,
        "y_test":           y_test,
        "roc":              roc,
    }
