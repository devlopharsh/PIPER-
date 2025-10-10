
"""Project constants. Override via env or CLI where possible."""
from dataclasses import dataclass
import torch
from pathlib import Path

# Root data folder
DATA_DIR = Path("./LeakageData")
# Model output directory
MODELS_DIR = Path("./models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    "roll6_mean", "roll24_mean", "roll6_std",  "roll24_std",
    "rel_to_roll24"  # optional
]

# Baseline CLS features (used by tabular cls models)
CLS_FEATURE_COLS = [*CALENDAR_COLS, *ROLLING_COLS, DELTA_VOLUME_COL]

# --------------------
# Default data knobs (shared)
# --------------------
RANDOM_SEED_DEFAULT          = 42
TEST_SIZE_DEFAULT            = 0.30
VAL_FRACTION_DEFAULT         = 0.10

LEAK_FRACTION_DEFAULT        = 0.15
SEGMENTS_RANGE_DEFAULT       = (1, 3)
MIN_LEN_FRAC_DEFAULT         = 0.05

# --------------------
# Model-specific sane defaults
# --------------------
# MLP
MLP_DEFAULTS = dict(
    hidden=[128, 64],
    dropout=0.2,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=20,
    early_patience=10,
)

# LSTM
@dataclass(frozen=True)
class LSTMDefaults:
    window: int = 2*24
    stride: int = 1
    hidden: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    batch_size: int = 256
    max_epochs: int = 30
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_patience: int = 10

LSTM_DEFAULTS = LSTMDefaults()

# GRU 
@dataclass(frozen=True)
class GRUDefaults:
    window: int = 2*24
    stride: int = 1
    hidden: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    batch_size: int = 256
    max_epochs: int = 15
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_patience: int = 10

GRU_DEFAULTS = GRUDefaults()


# SVM / MLflow (only used where relevant)
MLFLOW_URI_DEFAULT = "file:./mlruns"
MLFLOW_EXPERIMENT_DEFAULT = "LeakDetection-SVM"