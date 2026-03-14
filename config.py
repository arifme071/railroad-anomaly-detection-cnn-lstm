"""
config.py — Hyperparameters and experiment configuration
"""


class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    SAMPLING_RATE   = 1000.0   # Hz — DAS fiber-optic sampling rate
    WINDOW_SIZE     = 100      # Samples per feature window
    STEP_SIZE       = 50       # Sliding window step (50% overlap)
    NUM_CLASSES     = 4        # NC, TP, AC1, AC2

    # ── Model ─────────────────────────────────────────────────────────────────
    CNN_FILTERS     = [64, 128]
    LSTM_UNITS      = 128
    DROPOUT_RATE    = 0.3
    DENSE_UNITS     = 64

    # ── Training ──────────────────────────────────────────────────────────────
    LEARNING_RATE   = 1e-3
    BATCH_SIZE      = 32
    EPOCHS          = 50
    SEED            = 42

    # ── Sliding Window Post-Processing ────────────────────────────────────────
    SW_WINDOW       = 5        # Neighborhood window size
    SW_THRESHOLD    = 0.6      # Majority vote threshold

    # ── Paths ─────────────────────────────────────────────────────────────────
    SAMPLE_DATA     = "data/sample/sample_das_signal.csv"
    CHECKPOINT_PATH = "results/best_model.keras"
    RESULTS_PATH    = "results/metrics/training_results.json"
