"""
train.py — Main Training Script
Railroad Anomaly Detection: CNN-LSTM-SW

Usage:
    python train.py
    python train.py --epochs 100 --batch_size 64
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import json
import os

from src.preprocessing.signal_processor import generate_sample_signal
from src.preprocessing.feature_extractor import extract_features_from_windows
from src.models.cnn_lstm import build_cnn_lstm, get_callbacks
from src.models.sliding_window import sliding_window_correction, CONDITION_LABELS
from config import Config


def load_data(data_path: str = None) -> tuple:
    """Load DAS signal data. Falls back to synthetic sample if no path given."""
    if data_path and os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("No data path provided — generating synthetic sample data...")
        df = generate_sample_signal(n_points=10000, n_channels=8)

    channel_cols = [c for c in df.columns if c.startswith("channel_")]
    signal_matrix = df[channel_cols].values
    labels_raw = df["condition_label"].values

    return signal_matrix, labels_raw


def prepare_features(signal_matrix: np.ndarray, labels_raw: np.ndarray, cfg: Config) -> tuple:
    """Extract features and create windowed sequences."""
    print("Extracting features...")
    X = extract_features_from_windows(
        signal_matrix,
        window_size=cfg.WINDOW_SIZE,
        step_size=cfg.STEP_SIZE,
        fs=cfg.SAMPLING_RATE,
    )

    # Align labels to window centers
    centers = np.arange(
        cfg.WINDOW_SIZE // 2,
        len(labels_raw) - cfg.WINDOW_SIZE // 2,
        cfg.STEP_SIZE,
    )[:len(X)]
    y = labels_raw[centers]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for CNN-LSTM: (samples, timesteps=1, features)
    X = X[:, np.newaxis, :]

    return X, y, scaler


def train(args):
    cfg = Config()
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # Load and prepare data
    signal_matrix, labels_raw = load_data(args.data)
    X, y, scaler = prepare_features(signal_matrix, labels_raw, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=cfg.SEED
    )

    print(f"\nDataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]}")

    # Build and train model
    model = build_cnn_lstm(
        input_shape=X_train.shape[1:],
        num_classes=cfg.NUM_CLASSES,
        lstm_units=cfg.LSTM_UNITS,
        dropout_rate=cfg.DROPOUT_RATE,
    )

    os.makedirs("results/metrics", exist_ok=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Evaluate with sliding window correction
    y_pred_raw = np.argmax(model.predict(X_test), axis=1)
    y_pred_sw  = sliding_window_correction(y_pred_raw, window_size=cfg.SW_WINDOW)

    print("\n── CNN-LSTM Raw ──")
    print(classification_report(y_test, y_pred_raw,
          target_names=[CONDITION_LABELS[i] for i in range(cfg.NUM_CLASSES)]))

    print("\n── CNN-LSTM-SW (after sliding window) ──")
    print(classification_report(y_test, y_pred_sw,
          target_names=[CONDITION_LABELS[i] for i in range(cfg.NUM_CLASSES)]))

    # Save results
    results = {
        "raw_accuracy": float(np.mean(y_pred_raw == y_test)),
        "sw_accuracy":  float(np.mean(y_pred_sw  == y_test)),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    with open("results/metrics/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/metrics/training_results.json")
    print(f"Raw accuracy:  {results['raw_accuracy']:.4f}")
    print(f"SW  accuracy:  {results['sw_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    args = parser.parse_args()
    train(args)
