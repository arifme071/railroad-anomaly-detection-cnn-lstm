"""
Sample DAS Signal Generator
Generates synthetic data matching the statistical properties of the HTL loop
DAS dataset for demonstration and testing purposes.

Real dataset: AAR/TTCI test facility, Pueblo, CO (4.16 km fiber-optic track)
"""

import numpy as np
import pandas as pd
import os


# Condition class definitions (matching paper nomenclature)
CONDITIONS = {
    0: "NC",   # Normal Condition  — background rail/environment noise
    1: "TP",   # Train Position    — signal from passing train
    2: "AC1",  # Anomaly Class 1   — wheel flat / light defect
    3: "AC2",  # Anomaly Class 2   — rail joint / heavier anomaly
}

CONDITION_COLORS = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}


def generate_sample_signal(
    n_points: int = 5000,
    n_channels: int = 8,
    fs: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic DAS-like multi-channel signal with labeled conditions.

    Returns a DataFrame with columns:
        timestamp, channel_0..channel_N, condition_label, condition_name
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) / fs
    labels = np.zeros(n_points, dtype=int)

    # Define condition segments (index ranges)
    segments = [
        (0,    1200, 0),   # NC  — normal background
        (1200, 1800, 1),   # TP  — train passing
        (1800, 2400, 0),   # NC
        (2400, 2700, 2),   # AC1 — light anomaly
        (2700, 3500, 0),   # NC
        (3500, 3800, 3),   # AC2 — heavier anomaly
        (3800, 4200, 1),   # TP
        (4200, 5000, 0),   # NC
    ]
    for start, end, label in segments:
        labels[start:end] = label

    # Generate per-channel signal with condition-dependent amplitude
    channels = {}
    amplitude_map = {0: 0.3, 1: 2.5, 2: 1.2, 3: 1.8}

    for ch in range(n_channels):
        sig = np.zeros(n_points)
        for start, end, label in segments:
            amp = amplitude_map[label]
            freq = 50 + ch * 10 + rng.uniform(-5, 5)
            noise = rng.normal(0, 0.1, end - start)
            carrier = amp * np.sin(2 * np.pi * freq * t[start:end]) + noise
            sig[start:end] = carrier
        channels[f"channel_{ch}"] = sig

    df = pd.DataFrame({
        "timestamp": t,
        **channels,
        "condition_label": labels,
        "condition_name": [CONDITIONS[l] for l in labels],
    })

    return df


if __name__ == "__main__":
    print("Generating sample DAS signal...")
    df = generate_sample_signal(n_points=5000, n_channels=8)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "sample_das_signal.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} samples to {out_path}")
    print("\nCondition distribution:")
    print(df["condition_name"].value_counts())
