
"""Project constants. Override via env or CLI where possible."""
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass
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
FORECAST_Y_COL    = "y"             # target used by forecasting builders

# Calendar feature set
CALENDAR_COLS = ["Hour", "DayOfWeek", "Month"]
ROLLING_COLS = [
    "roll6_mean", "roll24_mean", "roll168_mean", "roll720_mean",
    "roll6_std",  "roll24_std",  "roll168_std",  "roll720_std",
    # "rel_to_roll24"  # optional
]


# Baseline CLS features (used by tabular cls models)
CLS_FEATURE_COLS = [*CALENDAR_COLS, *ROLLING_COLS, DELTA_VOLUME_COL]

FORECAST_COLS = [*CALENDAR_COLS, *ROLLING_COLS,]
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
    weight_decay=1e-5,
    max_epochs=30,
    early_patience=8,
)

# LSTM
@dataclass(frozen=True)
class LSTMDefaults:
    window: int = 3*24
    stride: int = 1
    hidden: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    batch_size: int = 256
    max_epochs: int = 20
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_patience: int = 10

LSTM_DEFAULTS = LSTMDefaults()

# GRU 
@dataclass(frozen=True)
class GRUDefaults:
    window: int = 48
    stride: int = 1
    hidden: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    batch_size: int = 256
    max_epochs: int = 60
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_patience: int = 8

GRU_DEFAULTS = GRUDefaults()


# SVM / MLflow (only used where relevant)
MLFLOW_URI_DEFAULT = "file:./mlruns"
MLFLOW_EXPERIMENT_DEFAULT = "LeakDetection-SVM"