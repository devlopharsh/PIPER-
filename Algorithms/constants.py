
"""Project constants. Override via env or CLI where possible."""
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# Root data folder
DATA_DIR = Path(os.getenv("LEAK_DATA_DIR", "../LeakageData")).resolve()

# Model output directory
MODELS_DIR = Path(os.getenv("LEAK_MODELS_DIR", "../models")).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
