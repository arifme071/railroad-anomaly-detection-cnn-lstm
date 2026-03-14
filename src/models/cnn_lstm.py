"""
CNN-LSTM-SW Model for Railroad Anomaly Detection
Paper: Rahman et al., Green Energy and Intelligent Transportation, Elsevier 2024
DOI: 10.1016/j.geits.2024.100178
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional


def build_cnn_lstm(
    input_shape: Tuple[int, int],
    num_classes: int = 4,
    cnn_filters: list = [64, 128],
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build the CNN-LSTM architecture for DAS signal classification.

    Args:
        input_shape:   (timesteps, features) — shape of one input window
        num_classes:   Number of condition classes (NC, TP, AC1, AC2)
        cnn_filters:   Number of filters per CNN block
        lstm_units:    LSTM hidden units
        dropout_rate:  Dropout for regularization

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name="das_input")

    # --- CNN blocks: extract local spatial features ---
    x = inputs
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_cnn_{i+1}")(x)

    # --- LSTM: learn temporal dependencies ---
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        name="lstm"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm")(x)

    # --- Dense classifier ---
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(checkpoint_path: str = "results/best_model.keras") -> list:
    """Standard training callbacks."""
    return [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


if __name__ == "__main__":
    # Quick architecture check
    model = build_cnn_lstm(input_shape=(100, 18), num_classes=4)
    model.summary()
