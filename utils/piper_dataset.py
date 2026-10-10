"""
PIPER Dataset Acquisition and Loading

This module handles:
1. Discovery and loading of raw datasets
2. Mapping source columns to canonical PIPER schema
3. Raw data preservation (immutable)
4. Dataset validation against PIPER requirements

The existing ApoErch dataset is loaded here for compatibility,
but it is flagged as MISSING mandatory PIPER telemetry.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import pandas as pd
import json
import shutil
from datetime import datetime

from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN, REQUIRED_COLUMNS,
    ColumnMapping, APOERCH_COLUMN_MAPPING, SCADA_COLUMN_MAPPING, validate_mapping, print_validation_report,
    TelemetryVariable, MetadataVariable, TargetVariable
)
from utils.constants import DATA_DIR as LEGACY_DATA_DIR


# =============================================================================
# PATHS
# =============================================================================

PIPER_RAW_DIR = Path("./data/raw")
PIPER_PROCESSED_DIR = Path("./data/processed")
PIPER_REPORTS_DIR = Path("./data/reports")

# Legacy data directory (existing ApoErch data)
LEGACY_DATA_DIR = Path("./LeakageData")


# =============================================================================
# DATASET CATALOG
# =============================================================================

@dataclass
class DatasetInfo:
    """Metadata about a discovered dataset."""
    name: str
    path: Path
    format: str  # csv, parquet, etc.
    n_files: int
    total_rows: int
    columns: List[str]
    date_range: Tuple[Optional[str], Optional[str]]
    sensors: List[str]
    mapping: ColumnMapping
    validation_report: Dict


class DatasetCatalog:
    """Catalog of available datasets with PIPER validation."""
    
    def __init__(self):
        self.datasets: Dict[str, DatasetInfo] = {}
    
    def register(self, info: DatasetInfo) -> None:
        self.datasets[info.name] = info
    
    def get(self, name: str) -> Optional[DatasetInfo]:
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())
    
    def get_piper_compatible(self) -> List[str]:
        """Return names of datasets that have all required PIPER telemetry."""
        return [
            name for name, info in self.datasets.items()
            if info.validation_report.get("all_required_mapped", False)
        ]
    
    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print("PIPER DATASET CATALOG")
        print(f"{'='*70}")
        for name, info in self.datasets.items():
            status = "PIPER-COMPATIBLE" if info.validation_report.get("all_required_mapped") else "INCOMPLETE"
            print(f"\n  {name} [{status}]")
            print(f"    Path: {info.path}")
            print(f"    Format: {info.format}")
            print(f"    Files: {info.n_files}, Rows: {info.total_rows:,}")
            print(f"    Columns: {info.columns}")
            print(f"    Date Range: {info.date_range[0]} to {info.date_range[1]}")
            print(f"    Sensors: {len(info.sensors)} ({info.sensors[:5]}{'...' if len(info.sensors) > 5 else ''})")
            if info.validation_report.get("blockers"):
                for b in info.validation_report["blockers"]:
                    print(f"    BLOCKER: {b}")
        print(f"{'='*70}\n")


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_apoerch_dataset(data_dir: Path = LEGACY_DATA_DIR) -> List[Path]:
    """Discover all CSV files in the legacy ApoErch data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Legacy data directory not found: {data_dir}")
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files


def load_apoerch_raw(data_dir: Path = LEGACY_DATA_DIR) -> pd.DataFrame:
    """
    Load raw ApoErch dataset (immutable raw data).
    
    Returns DataFrame with original columns: Meter_ID, Timestamp, Volume
    Does NOT apply any cleaning, feature engineering, or label generation.
    """
    csv_files = discover_apoerch_dataset(data_dir)
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Ensure consistent types
        df["Meter_ID"] = df["Meter_ID"].astype(str)
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"].astype(str).str.strip(),
            format="%d/%m/%Y %H:%M",
            dayfirst=True
        )
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["Meter_ID", "Timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    return combined


def apply_column_mapping(
    df: pd.DataFrame,
    mapping: ColumnMapping
) -> pd.DataFrame:
    """
    Apply column mapping to rename source columns to canonical PIPER names.
    
    Only renames columns that exist in the mapping.
    Does NOT create missing columns.
    """
    result = df.copy()
    rename_dict = {src: canon for src, canon in mapping.source_to_canonical.items() if src in result.columns}
    result = result.rename(columns=rename_dict)
    return result


def derive_flow_rate_from_volume(
    df: pd.DataFrame,
    volume_col: str = "Volume",
    sensor_col: str = "sensor_id",
    timestamp_col: str = "timestamp",
    output_col: str = "flow_rate"
) -> pd.DataFrame:
    """
    Derive flow rate from cumulative volume readings.
    
    This is a DERIVED feature, not a primary sensor reading.
    Computes first difference per sensor, ordered by timestamp.
    
    Args:
        df: DataFrame with cumulative volume, sensor_id, timestamp
        volume_col: Name of cumulative volume column
        sensor_col: Name of sensor identifier column
        timestamp_col: Name of timestamp column
        output_col: Name for derived flow rate column
    
    Returns:
        DataFrame with added flow_rate column
    """
    result = df.copy()
    result.sort_values([sensor_col, timestamp_col], inplace=True)
    
    # First difference per sensor
    result[output_col] = result.groupby(sensor_col)[volume_col].diff()
    
    # First row per sensor has NaN diff - fill with 0 (no prior reading)
    result[output_col] = result[output_col].fillna(0.0)
    
    # Negative values indicate meter reset or error - clip at 0
    # (cumulative volume should be non-decreasing)
    result[output_col] = result[output_col].clip(lower=0.0)
    
    return result


def create_apoerch_piper_dataset(
    data_dir: Path = LEGACY_DATA_DIR,
    save_raw_copy: bool = True
) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load ApoErch dataset and map to PIPER schema.
    
    This creates a PIPER-formatted dataset from the legacy data.
    IMPORTANT: This dataset is MISSING Pressure, Temperature, Vibration.
    Flow Rate is DERIVED from Volume.
    Labels are SYNTHETIC (handled in piper_labels.py).
    
    Returns:
        (mapped_dataframe, dataset_info)
    """
    # Load raw data
    raw_df = load_apoerch_raw(data_dir)
    
    # Save immutable raw copy to PIPER raw directory
    if save_raw_copy:
        PIPER_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_copy_path = PIPER_RAW_DIR / "apoerch_raw.parquet"
        raw_df.to_parquet(raw_copy_path, index=False)
        print(f"Saved raw copy to: {raw_copy_path}")
    
    # Apply column mapping
    mapped_df = apply_column_mapping(raw_df, APOERCH_COLUMN_MAPPING)
    
    # Derive flow_rate from Volume (cumulative)
    if "Volume" in mapped_df.columns and "sensor_id" in mapped_df.columns:
        mapped_df = derive_flow_rate_from_volume(
            mapped_df,
            volume_col="Volume",
            sensor_col="sensor_id",
            timestamp_col="timestamp",
            output_col=TelemetryVariable.FLOW_RATE.value
        )
        print(f"Derived {TelemetryVariable.FLOW_RATE.value} from Volume (cumulative)")
    
    # Validate
    validation = validate_mapping(APOERCH_COLUMN_MAPPING)
    
    # Build dataset info
    info = DatasetInfo(
        name="ApoErch_LeakageData",
        path=data_dir,
        format="csv",
        n_files=len(discover_apoerch_dataset(data_dir)),
        total_rows=len(raw_df),
        columns=raw_df.columns.tolist(),
        date_range=(
            str(raw_df["Timestamp"].min()) if "Timestamp" in raw_df.columns else None,
            str(raw_df["Timestamp"].max()) if "Timestamp" in raw_df.columns else None
        ),
        sensors=raw_df["Meter_ID"].unique().tolist() if "Meter_ID" in raw_df.columns else [],
        mapping=APOERCH_COLUMN_MAPPING,
        validation_report=validation
    )
    
    return mapped_df, info


def load_or_create_apoerch_piper_dataset(
    force_reload: bool = False
) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load cached PIPER-formatted ApoErch dataset or create it.
    
    Cache is stored in data/processed/apoerch_piper_mapped.parquet
    """
    cache_path = PIPER_PROCESSED_DIR / "apoerch_piper_mapped.parquet"
    
    if cache_path.exists() and not force_reload:
        df = pd.read_parquet(cache_path)
        # Rebuild info (could also cache this)
        _, info = create_apoerch_piper_dataset(save_raw_copy=False)
        return df, info
    
    df, info = create_apoerch_piper_dataset(save_raw_copy=True)
    
    # Save mapped version
    PIPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved mapped dataset to: {cache_path}")
    
    return df, info


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for dataset acquisition."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PIPER Dataset Acquisition")
    parser.add_argument("--force-reload", action="store_true", help="Force reload from raw CSV")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--catalog", action="store_true", help="Print dataset catalog")
    parser.add_argument("--dataset", choices=["apoerch", "scada", "both"], default="both", help="Which dataset to process")
    args = parser.parse_args()
    
    catalog = DatasetCatalog()
    
    if args.dataset in ["apoerch", "both"]:
        # Load/create ApoErch dataset
        df_apoerch, info_apoerch = load_or_create_apoerch_piper_dataset(force_reload=args.force_reload)
        catalog.register(info_apoerch)
        
        if args.validate_only:
            print_validation_report(info_apoerch.validation_report)
        
        print(f"\nLoaded ApoErch dataset: {len(df_apoerch):,} rows, {len(df_apoerch.columns)} columns")
        print(f"Columns: {df_apoerch.columns.tolist()}")
        
        # Check primary telemetry presence
        for pt in PRIMARY_TELEMETRY:
            present = pt in df_apoerch.columns
            print(f"  {pt}: {'PRESENT' if present else 'MISSING'}")
    
    if args.dataset in ["scada", "both"]:
        # Load/create SCADA dataset
        df_scada, info_scada = load_or_create_scada_piper_dataset(force_reload=args.force_reload)
        catalog.register(info_scada)
        
        if args.validate_only:
            print_validation_report(info_scada.validation_report)
        
        print(f"\nLoaded SCADA dataset: {len(df_scada):,} rows, {len(df_scada.columns)} columns")
        print(f"Columns: {df_scada.columns.tolist()}")
        
        # Check primary telemetry presence
        for pt in PRIMARY_TELEMETRY:
            present = pt in df_scada.columns
            print(f"  {pt}: {'PRESENT' if present else 'MISSING'}")
    
    if args.catalog:
        catalog.print_summary()


if __name__ == "__main__":
    main()


# =============================================================================
# SCADA DATASET LOADING
# =============================================================================

SCADA_RAW_PATH = Path("./data/raw/scada_pipeline.csv")


def load_scada_raw(data_path: Path = SCADA_RAW_PATH) -> pd.DataFrame:
    """
    Load raw SCADA dataset (immutable raw data).
    
    Returns DataFrame with original columns.
    Does NOT apply any cleaning, feature engineering, or label generation.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"SCADA data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure consistent types for known columns
    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype(int)
    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    if "valve_status" in df.columns:
        df["valve_status"] = df["valve_status"].astype(int)
    if "pump_state" in df.columns:
        df["pump_state"] = df["pump_state"].astype(int)
    if "compressor_state" in df.columns:
        df["compressor_state"] = df["compressor_state"].astype(int)
    if "alarm_triggered" in df.columns:
        df["alarm_triggered"] = df["alarm_triggered"].astype(int)
    
    # Parse timestamps (but keep original string column too)
    df["timestamp_parsed"] = pd.to_datetime(
        df["timestamp"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )
    
    # Sort by segment then timestamp
    df.sort_values(["segment_id", "timestamp_parsed"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def create_scada_piper_dataset(
    data_path: Path = SCADA_RAW_PATH,
    save_raw_copy: bool = True
) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load SCADA dataset and map to PIPER schema.
    
    This creates a PIPER-formatted dataset from the SCADA data.
    Primary telemetry present: pressure, flow_rate, temperature (3 of 4)
    Missing: vibration, pipeline_id
    Labels: target column is REAL ground truth (0=normal, 1=abnormal)
    
    Returns:
        (mapped_dataframe, dataset_info)
    """
    # Load raw data
    raw_df = load_scada_raw(data_path)
    
    # Save immutable raw copy to PIPER raw directory
    if save_raw_copy:
        PIPER_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_copy_path = PIPER_RAW_DIR / "scada_raw.parquet"
        raw_df.to_parquet(raw_copy_path, index=False)
        print(f"Saved raw copy to: {raw_copy_path}")
    
    # Apply column mapping
    mapped_df = apply_column_mapping(raw_df, SCADA_COLUMN_MAPPING)
    
    # Validate
    validation = validate_mapping(SCADA_COLUMN_MAPPING)
    
    # Build dataset info
    info = DatasetInfo(
        name="SCADA_Pipeline",
        path=data_path,
        format="csv",
        n_files=1,
        total_rows=len(raw_df),
        columns=raw_df.columns.tolist(),
        date_range=(
            str(raw_df["timestamp_parsed"].min()) if "timestamp_parsed" in raw_df.columns else None,
            str(raw_df["timestamp_parsed"].max()) if "timestamp_parsed" in raw_df.columns else None
        ),
        sensors=raw_df["segment_id"].unique().tolist() if "segment_id" in raw_df.columns else [],
        mapping=SCADA_COLUMN_MAPPING,
        validation_report=validation
    )
    
    return mapped_df, info


def load_or_create_scada_piper_dataset(
    force_reload: bool = False
) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load cached PIPER-formatted SCADA dataset or create it.
    
    Cache is stored in data/processed/scada_piper_mapped.parquet
    """
    cache_path = PIPER_PROCESSED_DIR / "scada_piper_mapped.parquet"
    
    if cache_path.exists() and not force_reload:
        df = pd.read_parquet(cache_path)
        # Rebuild info (could also cache this)
        _, info = create_scada_piper_dataset(save_raw_copy=False)
        return df, info
    
    df, info = create_scada_piper_dataset(save_raw_copy=True)
    
    # Save mapped version
    PIPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved mapped dataset to: {cache_path}")
    
    return df, info