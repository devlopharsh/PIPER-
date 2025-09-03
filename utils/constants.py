
"""Project constants. Override via env or CLI where possible."""
from pathlib import Path
import os
from dotenv import load_dotenv

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Root data folder
DATA_DIR = Path(os.getenv("LEAK_DATA_DIR", "./LeakageData")).resolve()

# Model output directory
MODELS_DIR = Path(os.getenv("LEAK_MODELS_DIR", "./models")).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# Canonical column names
# --------------------
METER_ID_COL      = "Meter_ID"
TIMESTAMP_COL     = "Timestamp"
VOLUME_COL        = "Volume"
VOLUME_READING_COL= "Volume_Reading"
DELTA_VOLUME_COL  = "Delta_Volume"
LEAK_FLAG_COL     = "Leak_Flag"
LEAK_TYPE_COL     = "Leak_Type"
FORECAST_Y_COL    = "y"             # target used by forecasting builders

# Calendar feature set
CALENDAR_COLS = ["Hour", "DayOfWeek", "Month"]

# Baseline CLS features (used by tabular cls models)
CLS_FEATURE_COLS = [VOLUME_READING_COL, *CALENDAR_COLS, DELTA_VOLUME_COL]

# --------------------
# Default data knobs (shared)
# --------------------
RANDOM_SEED_DEFAULT          = 42
TEST_SIZE_DEFAULT            = 0.30
VAL_FRACTION_DEFAULT         = 0.10

LEAK_FRACTION_DEFAULT        = 0.15
SEGMENTS_RANGE_DEFAULT       = (1, 3)
MIN_LEN_FRAC_DEFAULT         = 0.05
TAIL_FRACTION_DEFAULT        = 0.30

# Forecasting defaults
LAGS_DEFAULT                 = 24
ROLLING_WINDOWS_DEFAULT      = (6, 24)
DROP_CALENDAR_DEFAULT        = False
RESIDUAL_THRESHOLD_PCT_DEFAULT = 85

# --------------------
# Model-specific sane defaults
# --------------------
# MLP
MLP_DEFAULTS = dict(
    hidden=[128, 64],
    dropout=0.1,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=20,
    early_patience=6,
)

# LSTM
LSTM_DEFAULTS = dict(
    window=24,
    stride=1,
    hidden=64,
    num_layers=1,
    bidirectional=False,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=20,
    early_patience=4,
)

# GRU (keras)
GRU_DEFAULTS = dict(
    seq_len=3*24,
    stride=1,
    batch_size=256,
    epochs=20,
    lr=1e-3,
    patience=6,
)

# SVM / MLflow (only used where relevant)
MLFLOW_URI_DEFAULT = "file:./mlruns"
MLFLOW_EXPERIMENT_DEFAULT = "LeakDetection-SVM"