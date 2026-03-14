"""
Feature Extraction from DAS Fiber-Optic Signals
Time domain, frequency domain, and time-frequency domain features.

Paper: Rahman et al., Green Energy and Intelligent Transportation, Elsevier 2024
"""

import numpy as np
from scipy import signal, stats
from typing import Optional


# ── Time-domain features ──────────────────────────────────────────────────────

def time_domain_features(x: np.ndarray) -> np.ndarray:
    """
    Extract statistical time-domain features from a signal segment.
    Captures amplitude, variability, and temporal characteristics.

    Features: mean, std, rms, peak, crest_factor, skewness, kurtosis,
              zero_crossing_rate, peak_to_peak, variance
    """
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest = peak / (rms + 1e-10)
    zcr = np.mean(np.diff(np.sign(x)) != 0)

    return np.array([
        np.mean(x),
        np.std(x),
        rms,
        peak,
        crest,
        stats.skew(x),
        stats.kurtosis(x),
        zcr,
        np.ptp(x),           # peak-to-peak
        np.var(x),
    ])


# ── Frequency-domain features ─────────────────────────────────────────────────

def frequency_domain_features(
    x: np.ndarray,
    fs: float = 1000.0,
) -> np.ndarray:
    """
    Extract spectral features via FFT.

    Features: spectral_mean, spectral_std, spectral_centroid,
              dominant_frequency, spectral_entropy, band_power_ratio
    """
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = fft_vals ** 2
    total_power = np.sum(power) + 1e-10

    centroid = np.sum(freqs * power) / total_power
    dom_freq = freqs[np.argmax(power)]

    # Spectral entropy (normalized)
    psd_norm = power / total_power
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # Low/high band power ratio (split at Nyquist/4)
    mid = len(freqs) // 4
    band_ratio = np.sum(power[:mid]) / (np.sum(power[mid:]) + 1e-10)

    return np.array([
        np.mean(fft_vals),
        np.std(fft_vals),
        centroid,
        dom_freq,
        entropy,
        band_ratio,
    ])


# ── Time-frequency features ───────────────────────────────────────────────────

def timefreq_domain_features(
    x: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 64,
) -> np.ndarray:
    """
    Extract STFT-based time-frequency features.
    Captures how spectral content evolves over time.
    """
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Zxx)

    return np.array([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.median(magnitude),
    ])


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_all_features(
    x: np.ndarray,
    fs: float = 1000.0,
) -> np.ndarray:
    """
    Full feature extraction pipeline: time + frequency + time-frequency.
    Returns a flat feature vector of length 20.
    """
    td = time_domain_features(x)
    fd = frequency_domain_features(x, fs=fs)
    tfd = timefreq_domain_features(x, fs=fs)
    return np.concatenate([td, fd, tfd])


def extract_features_from_windows(
    signal_matrix: np.ndarray,
    window_size: int = 100,
    step_size: int = 50,
    fs: float = 1000.0,
) -> np.ndarray:
    """
    Slide a window over a multi-channel DAS signal matrix and extract features.

    Args:
        signal_matrix:  Shape (n_samples, n_channels)
        window_size:    Samples per window
        step_size:      Hop length between windows
        fs:             Sampling frequency (Hz)

    Returns:
        Feature matrix of shape (n_windows, n_features)
    """
    n_samples, n_channels = signal_matrix.shape
    feature_rows = []

    for start in range(0, n_samples - window_size + 1, step_size):
        window = signal_matrix[start: start + window_size, :]
        # Average features across channels (can also concatenate)
        window_features = np.mean(
            [extract_all_features(window[:, ch], fs=fs) for ch in range(n_channels)],
            axis=0,
        )
        feature_rows.append(window_features)

    return np.array(feature_rows)
