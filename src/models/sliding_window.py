"""
Sliding Window (SW) Post-Processing Module
Corrects CNN-LSTM misclassifications point-to-point along the railroad.

Paper: Rahman et al., Green Energy and Intelligent Transportation, Elsevier 2024
"""

import numpy as np
from collections import Counter
from typing import List


def sliding_window_correction(
    predictions: np.ndarray,
    window_size: int = 5,
    majority_threshold: float = 0.6,
) -> np.ndarray:
    """
    Apply sliding window majority vote to smooth CNN-LSTM predictions.

    The SW method addresses the key limitation of CNN-LSTM: point-to-point
    misclassifications near class boundaries (e.g., TP/AC2 confusion).
    A window votes on each prediction using its neighbors, improving
    spatial consistency along the track.

    Args:
        predictions:         Raw class predictions from CNN-LSTM (1D array)
        window_size:         Number of neighboring points to consider
        majority_threshold:  Minimum fraction to trigger correction (0.5–1.0)

    Returns:
        Corrected prediction array of the same length
    """
    n = len(predictions)
    corrected = predictions.copy()
    half_w = window_size // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window = predictions[start:end]

        counts = Counter(window)
        most_common_label, most_common_count = counts.most_common(1)[0]

        if most_common_count / len(window) >= majority_threshold:
            corrected[i] = most_common_label

    return corrected


def localize_anomalies(
    corrected_predictions: np.ndarray,
    timestamps: np.ndarray,
    anomaly_classes: List[int],
    sampling_rate_hz: float = 1.0,
) -> List[dict]:
    """
    Extract anomaly locations and durations from corrected predictions.

    Args:
        corrected_predictions:  Output of sliding_window_correction()
        timestamps:             Corresponding timestamps (seconds or index)
        anomaly_classes:        Class indices treated as anomalies (e.g. [2, 3])
        sampling_rate_hz:       Sensor sampling rate for duration calculation

    Returns:
        List of dicts: [{class, start_time, end_time, duration_s, n_points}]
    """
    anomalies = []
    in_anomaly = False
    start_idx = None
    current_class = None

    for i, pred in enumerate(corrected_predictions):
        if pred in anomaly_classes:
            if not in_anomaly:
                in_anomaly = True
                start_idx = i
                current_class = pred
        else:
            if in_anomaly:
                anomalies.append({
                    "class": int(current_class),
                    "start_time": float(timestamps[start_idx]),
                    "end_time": float(timestamps[i - 1]),
                    "duration_s": (i - start_idx) / sampling_rate_hz,
                    "n_points": i - start_idx,
                })
                in_anomaly = False

    # Close any open anomaly at end of signal
    if in_anomaly:
        anomalies.append({
            "class": int(current_class),
            "start_time": float(timestamps[start_idx]),
            "end_time": float(timestamps[-1]),
            "duration_s": (len(corrected_predictions) - start_idx) / sampling_rate_hz,
            "n_points": len(corrected_predictions) - start_idx,
        })

    return anomalies


# Class label mapping used in the paper
CONDITION_LABELS = {
    0: "NC",   # Normal Condition
    1: "TP",   # Train Position
    2: "AC1",  # Anomaly Class 1
    3: "AC2",  # Anomaly Class 2
}
