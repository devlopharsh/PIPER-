"""
PIPER Telemetry Schema - Canonical column definitions and mappings.

This module defines the PIPER telemetry specification.
The three primary telemetry variables for the current tabular classification path:
    1. Pressure
    2. Flow Rate
    3. Temperature

Vibration is not required for the current adapted scope (SCADA dataset).
DO NOT modify these canonical names. Source datasets must map to these.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TelemetryVariable(str, Enum):
    """Canonical PIPER telemetry variable names. These are FIXED."""
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"  # Retained for future compatibility, not required for current scope


class MetadataVariable(str, Enum):
    """Canonical PIPER metadata variable names."""
    PIPELINE_ID = "pipeline_id"
    SENSOR_ID = "sensor_id"
    TIMESTAMP = "timestamp"


class TargetVariable(str, Enum):
    """Canonical PIPER target variable."""
    LABEL = "label"  # 0 = Normal, 1 = Abnormal/Leak


# =============================================================================
# CANONICAL PIPER SCHEMA
# =============================================================================

# Primary telemetry - three variables for current tabular classification scope
# Vibration is retained in TelemetryVariable enum for future compatibility
# but is NOT required for the current adapted PIPER scope (SCADA dataset)
PRIMARY_TELEMETRY: List[str] = [
    TelemetryVariable.PRESSURE.value,
    TelemetryVariable.FLOW_RATE.value,
    TelemetryVariable.TEMPERATURE.value,
]

# Metadata columns
METADATA_COLUMNS: List[str] = [
    MetadataVariable.PIPELINE_ID.value,
    MetadataVariable.SENSOR_ID.value,
    MetadataVariable.TIMESTAMP.value,
]

# Target column
TARGET_COLUMN: str = TargetVariable.LABEL.value

# All required columns for PIPER pipeline
REQUIRED_COLUMNS: List[str] = PRIMARY_TELEMETRY + METADATA_COLUMNS + [TARGET_COLUMN]

# Feature columns (primary telemetry only - derived features handled separately)
FEATURE_COLUMNS: List[str] = PRIMARY_TELEMETRY.copy()


# =============================================================================
# SOURCE DATASET COLUMN MAPPING
# =============================================================================
# Maps source dataset column names -> canonical PIPER column names.
# This mapping MUST be explicitly configured per dataset.
# DO NOT guess mappings automatically.

@dataclass(frozen=True)
class ColumnMapping:
    """Explicit mapping from source columns to canonical PIPER columns."""
    source_to_canonical: Dict[str, str]
    dataset_name: str
    description: str = ""

    def get_canonical(self, source_col: str) -> Optional[str]:
        """Get canonical name for a source column."""
        return self.source_to_canonical.get(source_col)

    def get_source(self, canonical_col: str) -> Optional[str]:
        """Get source column name for a canonical column."""
        for src, canon in self.source_to_canonical.items():
            if canon == canonical_col:
                return src
        return None

    def missing_canonical(self) -> List[str]:
        """Return list of canonical columns NOT mapped from source."""
        mapped = set(self.source_to_canonical.values())
        return [c for c in REQUIRED_COLUMNS if c not in mapped]

    def has_all_required(self) -> bool:
        """Check if all required PIPER columns are mapped."""
        return len(self.missing_canonical()) == 0


# -----------------------------------------------------------------------------
# ApoErch / Existing Repository Mapping
# -----------------------------------------------------------------------------
# The existing repository has ONLY:
#   - Meter_ID (sensor identifier)
#   - Timestamp
#   - Volume (cumulative meter reading)
#
# From Volume we can derive Flow Rate (first difference).
# Pressure, Temperature, Vibration are NOT available in this dataset.
# Leak_Flag is SYNTHETIC (injected by inject_random_leak_segments).

APOERCH_COLUMN_MAPPING = ColumnMapping(
    dataset_name="ApoErch_LeakageData",
    description="Mapping for existing ApoErch utility network leakage dataset. "
                "Only Volume and Timestamp are available. "
                "Pressure, Temperature, Vibration are MISSING. "
                "Flow Rate is DERIVED from Volume. "
                "Leak_Flag is SYNTHETIC.",
    source_to_canonical={
        "Meter_ID": MetadataVariable.SENSOR_ID.value,
        "Timestamp": MetadataVariable.TIMESTAMP.value,
        # Volume is cumulative - Flow Rate must be derived
        # "Volume": "volume_cumulative",  # Not a canonical PIPER column
        # Leak_Flag is synthetic - handled in label generation
    }
)

# Placeholder for future compatible dataset
# When a dataset with Pressure, Temperature, Vibration is available,
# create a new ColumnMapping with all required mappings.
# Example:
#
# COMPATIBLE_DATASET_MAPPING = ColumnMapping(
#     dataset_name="Piper_Compatible_Dataset",
#     description="Dataset with all required PIPER telemetry",
#     source_to_canonical={
#         "pressure_psi": TelemetryVariable.PRESSURE.value,
#         "flow_gpm": TelemetryVariable.FLOW_RATE.value,
#         "temp_c": TelemetryVariable.TEMPERATURE.value,
#         "vib_mm_s": TelemetryVariable.VIBRATION.value,
#         "pipeline_id": MetadataVariable.PIPELINE_ID.value,
#         "sensor_id": MetadataVariable.SENSOR_ID.value,
#         "ts": MetadataVariable.TIMESTAMP.value,
#         "leak_label": TargetVariable.LABEL.value,
#     }
# )

# -----------------------------------------------------------------------------
# SCADA Pipeline Dataset Mapping
# -----------------------------------------------------------------------------
# The SCADA dataset has:
#   - timestamp
#   - segment_id (maps to sensor_id)
#   - pressure, flow_rate, temperature (primary telemetry - 3 of 3 required for current scope)
#   - valve_status, pump_state, pump_speed, compressor_state, energy_consumption, alarm_triggered (auxiliary)
#   - event_type (event metadata)
#   - target (binary label: 0=normal, 1=abnormal)
#
# MISSING: pipeline_id (metadata), vibration (not required for current tabular scope)
# LABEL: target column is REAL ground truth (not synthetic)

SCADA_COLUMN_MAPPING = ColumnMapping(
    dataset_name="SCADA_Pipeline",
    description="SCADA pipeline dataset with pressure, flow_rate, temperature (3 primary telemetry). "
                "Vibration not required for current tabular classification scope. "
                "Pipeline_id is MISSING (single pipeline assumed). "
                "Target column is real ground truth. segment_id maps to sensor_id.",
    source_to_canonical={
        "timestamp": MetadataVariable.TIMESTAMP.value,
        "segment_id": MetadataVariable.SENSOR_ID.value,
        "pressure": TelemetryVariable.PRESSURE.value,
        "flow_rate": TelemetryVariable.FLOW_RATE.value,
        "temperature": TelemetryVariable.TEMPERATURE.value,
        # Auxiliary variables - kept as metadata, not primary telemetry
        "valve_status": "valve_status",
        "pump_state": "pump_state",
        "pump_speed": "pump_speed",
        "compressor_state": "compressor_state",
        "energy_consumption": "energy_consumption",
        "alarm_triggered": "alarm_triggered",
        "event_type": "event_type",
        "target": TargetVariable.LABEL.value,
    }
)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_mapping(mapping: ColumnMapping) -> Dict:
    """
    Validate a column mapping against PIPER requirements.
    Returns a validation report.
    """
    missing = mapping.missing_canonical()
    present = [c for c in REQUIRED_COLUMNS if c not in missing]

    # Primary telemetry status - these are REQUIRED for current scope
    primary_status = {}
    for pt in PRIMARY_TELEMETRY:
        primary_status[pt] = {
            "mapped": pt in present,
            "source_column": mapping.get_source(pt),
            "status": "MAPPED" if pt in present else "MISSING - BLOCKER"
        }

    # Metadata status - pipeline_id is optional for current scope
    metadata_status = {}
    for mc in METADATA_COLUMNS:
        is_required = mc != MetadataVariable.PIPELINE_ID.value  # pipeline_id optional
        metadata_status[mc] = {
            "mapped": mc in present,
            "source_column": mapping.get_source(mc),
            "status": "MAPPED" if mc in present else ("MISSING - OPTIONAL" if not is_required else "MISSING")
        }

    target_status = {
        TARGET_COLUMN: {
            "mapped": TARGET_COLUMN in present,
            "source_column": mapping.get_source(TARGET_COLUMN),
            "status": "MAPPED" if TARGET_COLUMN in present else "MISSING - BLOCKER"
        }
    }

    # Blockers only for required primary telemetry and target
    blockers = [f"Missing mandatory telemetry: {m}" for m in missing if m in PRIMARY_TELEMETRY]
    if TARGET_COLUMN in missing:
        blockers.append(f"Missing mandatory target: {TARGET_COLUMN}")

    return {
        "dataset": mapping.dataset_name,
        "description": mapping.description,
        "all_required_mapped": len(blockers) == 0,
        "primary_telemetry": primary_status,
        "metadata": metadata_status,
        "target": target_status,
        "missing_canonical_columns": missing,
        "blockers": blockers
    }


def print_validation_report(report: Dict) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"PIPER TELEMETRY VALIDATION: {report['dataset']}")
    print(f"{'='*60}")
    print(f"Description: {report['description']}")
    print(f"All Required Mapped: {report['all_required_mapped']}")
    print()
    print("NOTE: Current adapted scope requires 3 primary telemetry (pressure, flow_rate, temperature).")
    print("Vibration is retained for future compatibility but not required for tabular classification.")
    print()

    print("--- PRIMARY TELEMETRY (REQUIRED FOR CURRENT SCOPE) ---")
    for var, info in report['primary_telemetry'].items():
        status_marker = "[OK]" if info['mapped'] else "[MISSING]"
        src = info['source_column'] or "N/A"
        print(f"  {var:15s} : {status_marker} (source: {src})")

    print("\n--- METADATA ---")
    for var, info in report['metadata'].items():
        status_marker = "[OK]" if info['mapped'] else "[MISSING]"
        src = info['source_column'] or "N/A"
        print(f"  {var:15s} : {status_marker} (source: {src})")

    print("\n--- TARGET ---")
    for var, info in report['target'].items():
        status_marker = "[OK]" if info['mapped'] else "[MISSING]"
        src = info['source_column'] or "N/A"
        print(f"  {var:15s} : {status_marker} (source: {src})")

    if report['blockers']:
        print("\n!!! BLOCKERS !!!")
        for b in report['blockers']:
            print(f"  - {b}")
        print("\nThis dataset cannot be used for PIPER without resolving blockers.")

    print(f"{'='*60}\n")


# =============================================================================
# DERIVED FEATURE NAMES (for reference, not primary telemetry)
# =============================================================================
# These are feature engineering outputs, NOT primary telemetry.
# They must be clearly distinguished from PRIMARY_TELEMETRY.

DERIVED_FEATURE_PREFIXES = [
    "roll_",      # rolling statistics
    "diff_",      # differences / rate of change
    "rel_",       # relative / normalized
    "hour",       # time-of-day
    "dayofweek",  # day of week
    "month",      # month
]

def is_primary_telemetry(col: str) -> bool:
    """Check if a column is one of the four primary telemetry variables."""
    return col in PRIMARY_TELEMETRY

def is_derived_feature(col: str) -> bool:
    """Check if a column is a derived feature (not primary telemetry)."""
    return any(col.startswith(p) for p in DERIVED_FEATURE_PREFIXES) or col in [
        "Hour", "DayOfWeek", "Month",  # legacy names
        "roll6_mean", "roll24_mean", "roll6_std", "roll24_std", "rel_to_roll24"
    ]


if __name__ == "__main__":
    # Validate the existing ApoErch mapping
    report = validate_mapping(APOERCH_COLUMN_MAPPING)
    print_validation_report(report)