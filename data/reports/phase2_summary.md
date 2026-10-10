# PIPER Phase 2 Summary Report: SCADA_Pipeline

**Generated:** 2026-10-10T22:29:47.049251

## Dataset Overview

| Property | Value |
|----------|-------|
| Dataset Name | SCADA_Pipeline |
| Source Path | data\raw\scada_pipeline.csv |
| Format | csv |
| Total Rows | 1,000 |
| Total Columns (raw) | 14 |
| Date Range | 2024-01-01 00:00:00 to 2024-01-01 00:16:00 |
| Unique Sensors | 50 |

## PIPER Telemetry Validation

**All Required Mapped:** True

### Primary Telemetry (Mandatory)

- **pressure**: [OK] MAPPED (source: pressure)
- **flow_rate**: [OK] MAPPED (source: flow_rate)
- **temperature**: [OK] MAPPED (source: temperature)

### Metadata

- **pipeline_id**: [MISSING] MISSING (source: N/A)
- **sensor_id**: [OK] MAPPED (source: segment_id)
- **timestamp**: [OK] MAPPED (source: timestamp)

### Target

- **label**: [OK] MAPPED (source: target)


## Data Cleaning

| Metric | Value |
|--------|-------|
| Raw Rows | 1,000 |
| Clean Rows | 1,000 |
| Rows Removed | 0 |
| Cleaning Steps | 11 |

### Cleaning Steps Detail

#### validate_timestamps
- **Description:** Parse and validate timestamps (%m/%d/%Y %H:%M)
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - original_dtype: object
  - format_used: %m/%d/%Y %H:%M
  - parsing_failures: 0
  - rows_with_invalid_timestamp_removed: 0
  - timestamp_range: {'min': '2024-01-01 00:00:00', 'max': '2024-01-01 00:16:00'}

#### enforce_chronological_order
- **Description:** Sort by sensor_id, then timestamp
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - sort_columns: ['sensor_id', 'timestamp']
  - was_already_sorted: True

#### handle_duplicates
- **Description:** Handle duplicates on ['sensor_id', 'timestamp']: exact removed, conflicts preserved
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - subset_columns: ['sensor_id', 'timestamp']
  - keep_policy: first
  - exact_duplicate_rows_found: 0
  - exact_duplicate_rows_removed: 0
  - key_duplicate_groups_found: 339
  - conflicting_key_groups: 270
  - identical_key_groups: 0
  - conflicting_rows_preserved: 408
  - reading_index_added: True
  - action: Exact duplicates removed; conflicting key duplicates preserved with reading_index

#### detect_missing_values
- **Description:** Detect missing values in 5 columns
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - columns_checked: ['pressure', 'flow_rate', 'temperature', 'sensor_id', 'timestamp']
  - missing_per_column: {'pressure': 0, 'flow_rate': 0, 'temperature': 0, 'sensor_id': 0, 'timestamp': 0}
  - total_missing_values: 0
  - rows_with_any_missing: 0
  - action: REPORT_ONLY - no rows removed

#### validate_telemetry_numeric
- **Description:** Validate numeric type for telemetry columns
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - columns_checked: ['pressure', 'flow_rate', 'temperature']
  - non_numeric_per_column: {'pressure': 0, 'flow_rate': 0, 'temperature': 0}
  - total_non_numeric: 0
  - action: REPORT_ONLY - no conversion performed

#### coerce_telemetry_numeric
- **Description:** Coerce telemetry columns to numeric (coerce)
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - columns: ['pressure', 'flow_rate', 'temperature']
  - errors_policy: coerce
  - values_coerced_to_nan_per_column: {'pressure': 0, 'flow_rate': 0, 'temperature': 0}
  - total_coerced: 0
  - rows_removed_due_to_nan_telemetry: 0

#### detect_impossible_values
- **Description:** Check physical bounds on telemetry
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - rules_applied: {'pressure': {'min': 0, 'max': 2000}, 'flow_rate': {'min': 0, 'max': 50000}, 'temperature': {'min': -40, 'max': 200}, 'vibration': {'min': 0, 'max': 500}}
  - impossible_per_column: {'pressure': 0, 'flow_rate': 0, 'temperature': 0}
  - total_impossible_values: 0
  - rows_with_any_impossible: 0
  - action: REPORT_ONLY - abnormal values preserved for leak detection

#### inspect_outliers
- **Description:** Outlier inspection using iqr
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - method: iqr
  - columns_checked: ['pressure', 'flow_rate', 'temperature']
  - outliers_per_column: {'pressure': 100, 'flow_rate': 81, 'temperature': 6}
  - total_outliers: 187
  - outlier_details: {'pressure': {'method': 'iqr', 'q1': 71.1275, 'q3': 79.2825, 'iqr': 8.155000000000001, 'lower_bound': 58.894999999999996, 'upper_bound': 91.515, 'multiplier': 1.5}, 'flow_rate': {'method': 'iqr', 'q1': 4.04, 'q3': 4.88, 'iqr': 0.8399999999999999, 'lower_bound': 2.7800000000000002, 'upper_bound': 6.14, 'multiplier': 1.5}, 'temperature': {'method': 'iqr', 'q1': 30.69, 'q3': 33.394999999999996, 'iqr': 2.7049999999999947, 'lower_bound': 26.632500000000007, 'upper_bound': 37.452499999999986, 'multiplier': 1.5}}
  - action: REPORT_ONLY - outliers preserved (may indicate leaks)

#### check_sensor_consistency
- **Description:** Per-sensor consistency check
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - total_sensors: 50
  - sensor_statistics: [{'sensor_id': 1, 'n_readings': 17, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 17.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 2, 'n_readings': 19, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 3, 'n_readings': 24, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 24.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 4, 'n_readings': 13, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 13.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 5, 'n_readings': 23, 'date_range_start': '2024-01-01 00:02:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 23.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 6, 'n_readings': 20, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 7, 'n_readings': 15, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 8, 'n_readings': 20, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:14:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 9, 'n_readings': 21, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 21.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 10, 'n_readings': 20, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 11, 'n_readings': 18, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 12, 'n_readings': 22, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 22.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 13, 'n_readings': 14, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 14.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 14, 'n_readings': 12, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:14:00', 'time_span_hours': 0.21666666666666667, 'expected_readings_hourly': 1, 'completeness': 12.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 15, 'n_readings': 21, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 21.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 16, 'n_readings': 27, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 27.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 17, 'n_readings': 21, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 21.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 18, 'n_readings': 26, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 26.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 19, 'n_readings': 15, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 20, 'n_readings': 19, 'date_range_start': '2024-01-01 00:02:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 21, 'n_readings': 18, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:14:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 22, 'n_readings': 27, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 27.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 23, 'n_readings': 28, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 28.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 24, 'n_readings': 20, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:13:00', 'time_span_hours': 0.21666666666666667, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 25, 'n_readings': 22, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 22.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 26, 'n_readings': 20, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 27, 'n_readings': 17, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 17.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 28, 'n_readings': 22, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 22.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 29, 'n_readings': 23, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:14:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 23.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 30, 'n_readings': 20, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:14:00', 'time_span_hours': 0.21666666666666667, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 31, 'n_readings': 16, 'date_range_start': '2024-01-01 00:02:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.21666666666666667, 'expected_readings_hourly': 1, 'completeness': 16.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 32, 'n_readings': 21, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 21.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 33, 'n_readings': 20, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 20.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 34, 'n_readings': 18, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 35, 'n_readings': 19, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 36, 'n_readings': 28, 'date_range_start': '2024-01-01 00:02:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 28.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 37, 'n_readings': 15, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 38, 'n_readings': 15, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 39, 'n_readings': 18, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 40, 'n_readings': 19, 'date_range_start': '2024-01-01 00:02:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 41, 'n_readings': 18, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 42, 'n_readings': 18, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:12:00', 'time_span_hours': 0.2, 'expected_readings_hourly': 1, 'completeness': 18.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 43, 'n_readings': 33, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 33.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 44, 'n_readings': 15, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 45, 'n_readings': 23, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 23.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 46, 'n_readings': 19, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 47, 'n_readings': 19, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 19.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 48, 'n_readings': 15, 'date_range_start': '2024-01-01 00:01:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.23333333333333334, 'expected_readings_hourly': 1, 'completeness': 15.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 49, 'n_readings': 24, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:15:00', 'time_span_hours': 0.25, 'expected_readings_hourly': 1, 'completeness': 24.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}, {'sensor_id': 50, 'n_readings': 23, 'date_range_start': '2024-01-01 00:00:00', 'date_range_end': '2024-01-01 00:16:00', 'time_span_hours': 0.26666666666666666, 'expected_readings_hourly': 1, 'completeness': 23.0, 'n_gaps_gt_1h': 0, 'max_gap_hours': 0}]
  - sensors_with_gaps: 0
  - sensors_low_completeness: 0
  - single_reading_sensors: 0
  - action: REPORT_ONLY

#### check_sampling_interval
- **Description:** Sampling interval consistency (expected: 1.0h)
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - expected_interval_hours: 1.0
  - tolerance_hours: 0.1
  - total_intervals_analyzed: 950
  - overall_on_time_pct: 0.0
  - per_sensor_statistics: [{'sensor_id': 1, 'n_intervals': 16, 'mean_interval_hours': 0.016666666666666666, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01721325931647741, 'on_time_count': 0, 'early_count': 16, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 2, 'n_intervals': 18, 'mean_interval_hours': 0.013888888888888888, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013098420799889595, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 3, 'n_intervals': 23, 'mean_interval_hours': 0.011594202898550725, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013702081432144625, 'on_time_count': 0, 'early_count': 23, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 4, 'n_intervals': 12, 'mean_interval_hours': 0.022222222222222223, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.017884778999621993, 'on_time_count': 0, 'early_count': 12, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 5, 'n_intervals': 22, 'mean_interval_hours': 0.010606060606060607, 'median_interval_hours': 0.008333333333333333, 'std_interval_hours': 0.012112190004501526, 'on_time_count': 0, 'early_count': 22, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 6, 'n_intervals': 19, 'mean_interval_hours': 0.013157894736842105, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.011888041725295978, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 7, 'n_intervals': 14, 'mean_interval_hours': 0.019047619047619046, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.020524264207162735, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 8, 'n_intervals': 19, 'mean_interval_hours': 0.012280701754385965, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013428026319538053, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 9, 'n_intervals': 20, 'mean_interval_hours': 0.013333333333333332, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01159753956564099, 'on_time_count': 0, 'early_count': 20, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 10, 'n_intervals': 19, 'mean_interval_hours': 0.014035087719298248, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01690857578339294, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 11, 'n_intervals': 17, 'mean_interval_hours': 0.014705882352941176, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.014291548761875737, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 12, 'n_intervals': 21, 'mean_interval_hours': 0.011904761904761904, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.011952286093343936, 'on_time_count': 0, 'early_count': 21, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 13, 'n_intervals': 13, 'mean_interval_hours': 0.020512820512820513, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.020586140332494913, 'on_time_count': 0, 'early_count': 13, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 14, 'n_intervals': 11, 'mean_interval_hours': 0.0196969696969697, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.016360830286900715, 'on_time_count': 0, 'early_count': 11, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 15, 'n_intervals': 20, 'mean_interval_hours': 0.013333333333333332, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01585755303645848, 'on_time_count': 0, 'early_count': 20, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 16, 'n_intervals': 26, 'mean_interval_hours': 0.009615384615384616, 'median_interval_hours': 0.0, 'std_interval_hours': 0.013476792334377331, 'on_time_count': 0, 'early_count': 26, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 17, 'n_intervals': 20, 'mean_interval_hours': 0.0125, 'median_interval_hours': 0.008333333333333333, 'std_interval_hours': 0.01610909444930435, 'on_time_count': 0, 'early_count': 20, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 18, 'n_intervals': 25, 'mean_interval_hours': 0.01, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.010758287072798378, 'on_time_count': 0, 'early_count': 25, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 19, 'n_intervals': 14, 'mean_interval_hours': 0.016666666666666666, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.014617633655117155, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 20, 'n_intervals': 18, 'mean_interval_hours': 0.012962962962962963, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.016721044190151673, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 21, 'n_intervals': 17, 'mean_interval_hours': 0.013725490196078431, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.012126781251816649, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 22, 'n_intervals': 26, 'mean_interval_hours': 0.010256410256410256, 'median_interval_hours': 0.008333333333333333, 'std_interval_hours': 0.012540532575012805, 'on_time_count': 0, 'early_count': 26, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 23, 'n_intervals': 27, 'mean_interval_hours': 0.009876543209876545, 'median_interval_hours': 0.0, 'std_interval_hours': 0.016184868988416506, 'on_time_count': 0, 'early_count': 27, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 24, 'n_intervals': 19, 'mean_interval_hours': 0.011403508771929825, 'median_interval_hours': 0.0, 'std_interval_hours': 0.0167153285707093, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 25, 'n_intervals': 21, 'mean_interval_hours': 0.011904761904761908, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.011952286093343936, 'on_time_count': 0, 'early_count': 21, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 26, 'n_intervals': 19, 'mean_interval_hours': 0.014035087719298246, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013903501085343556, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 27, 'n_intervals': 16, 'mean_interval_hours': 0.016666666666666666, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013608276348795436, 'on_time_count': 0, 'early_count': 16, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 28, 'n_intervals': 21, 'mean_interval_hours': 0.011904761904761904, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.010727505072446485, 'on_time_count': 0, 'early_count': 21, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 29, 'n_intervals': 22, 'mean_interval_hours': 0.010606060606060607, 'median_interval_hours': 0.008333333333333333, 'std_interval_hours': 0.01315903389919538, 'on_time_count': 0, 'early_count': 22, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 30, 'n_intervals': 19, 'mean_interval_hours': 0.011403508771929825, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013667831453240051, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 31, 'n_intervals': 15, 'mean_interval_hours': 0.014444444444444444, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013896823130546533, 'on_time_count': 0, 'early_count': 15, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 32, 'n_intervals': 20, 'mean_interval_hours': 0.0125, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.010644439560975084, 'on_time_count': 0, 'early_count': 20, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 33, 'n_intervals': 19, 'mean_interval_hours': 0.013157894736842105, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013122108580302267, 'on_time_count': 0, 'early_count': 19, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 34, 'n_intervals': 17, 'mean_interval_hours': 0.014705882352941176, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.015458673560021056, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 35, 'n_intervals': 18, 'mean_interval_hours': 0.013888888888888888, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.018302112453523988, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 36, 'n_intervals': 27, 'mean_interval_hours': 0.008641975308641976, 'median_interval_hours': 0.0, 'std_interval_hours': 0.010712212826507533, 'on_time_count': 0, 'early_count': 27, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 37, 'n_intervals': 14, 'mean_interval_hours': 0.017857142857142856, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.015281246137553165, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 38, 'n_intervals': 14, 'mean_interval_hours': 0.017857142857142856, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013812365502745748, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 39, 'n_intervals': 17, 'mean_interval_hours': 0.01470588235294118, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013020220573817083, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 40, 'n_intervals': 18, 'mean_interval_hours': 0.012962962962962963, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01077936111272591, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 41, 'n_intervals': 17, 'mean_interval_hours': 0.01568627450980392, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.02462960915359445, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 42, 'n_intervals': 17, 'mean_interval_hours': 0.011764705882352941, 'median_interval_hours': 0.0, 'std_interval_hours': 0.016419739435729632, 'on_time_count': 0, 'early_count': 17, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 43, 'n_intervals': 32, 'mean_interval_hours': 0.008333333333333335, 'median_interval_hours': 0.0, 'std_interval_hours': 0.009466030570784413, 'on_time_count': 0, 'early_count': 32, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 44, 'n_intervals': 14, 'mean_interval_hours': 0.016666666666666666, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.019611613513818404, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 45, 'n_intervals': 22, 'mean_interval_hours': 0.012121212121212121, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01378387427377332, 'on_time_count': 0, 'early_count': 22, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 46, 'n_intervals': 18, 'mean_interval_hours': 0.014814814814814815, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.009716384802168958, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 47, 'n_intervals': 18, 'mean_interval_hours': 0.013888888888888892, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.013098420799889595, 'on_time_count': 0, 'early_count': 18, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 48, 'n_intervals': 14, 'mean_interval_hours': 0.016666666666666666, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.016012815380508715, 'on_time_count': 0, 'early_count': 14, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 49, 'n_intervals': 23, 'mean_interval_hours': 0.010869565217391304, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.012918446262719659, 'on_time_count': 0, 'early_count': 23, 'late_count': 0, 'on_time_pct': 0.0}, {'sensor_id': 50, 'n_intervals': 22, 'mean_interval_hours': 0.012121212121212121, 'median_interval_hours': 0.016666666666666666, 'std_interval_hours': 0.01170833622190348, 'on_time_count': 0, 'early_count': 22, 'late_count': 0, 'on_time_pct': 0.0}]
  - action: REPORT_ONLY

#### validate_data_integrity
- **Description:** Comprehensive data integrity validation
- **Rows Before:** 1,000
- **Rows After:** 1,000
- **Rows Affected:** 0
- **Details:**
  - checks: {'required_columns_present': True, 'missing_required_columns': [], 'no_duplicate_timestamps_per_sensor': False, 'duplicate_timestamp_count': 408, 'timestamps_monotonic_per_sensor': True, 'sensors_non_monotonic': 0, 'telemetry_coverage': {'pressure': 1.0, 'flow_rate': 1.0, 'temperature': 1.0}, 'target_column_present': True, 'target_unique_values': [0, 1], 'target_valid_binary': True}
  - all_checks_passed: True
  - action: VALIDATION_REPORT



## Feature Engineering

| Metric | Value |
|--------|-------|
| Input Columns | 15 |
| Output Columns | 31 |
| Primary Telemetry Present | 3/3 |
| Primary Telemetry Missing | 0/3 |
| Derived Features | 6 |
| Temporal Features | 13 |
| Metadata Features | 4 |

### Primary Telemetry Status

- [OK] pressure
- [OK] flow_rate
- [OK] temperature


### Feature Manifest Summary

| Category | Count |
|----------|-------|
| Total Features | 31 |
| Predictive Features | 28 |
| Excluded Features | 3 |

### Excluded Features (with reasons)

- **alarm_triggered** (auxiliary): Directly encodes abnormal condition - target leakage risk
- **event_type** (event_metadata): Post-event information - would cause target leakage
- **label** (target): Target variable - not a predictive feature


## Label Generation

| Metric | Value |
|--------|-------|
| Label Source | real_ground_truth |
| Label Method | existing_column |
| Normal (0) | 694 |
| Abnormal (1) | 306 |
| Total Labeled | 1,000 |
| Abnormal Rate | 0.3060 (30.60%) |
| Is Synthetic | False |



## Temporal Feasibility Audit

### Global Statistics

| Metric | Value |
|--------|-------|
| Total Observations | 1,000 |
| Total Segments | 50 |
| Total Unique Timestamps | 17 |
| Time Span (hours) | 0.27 |
| Duplicate (segment, timestamp) Keys | 408 |
| Exact Duplicate Rows | 0 |

### Per-Segment Summary

| Metric | Value |
|--------|-------|
| Segments with Duplicate Timestamps | 50 |
| Segments with Both Target Classes | 50 |
| Min Unique Timestamps/Segment | 8 |
| Max Unique Timestamps/Segment | 16 |
| Avg Unique Timestamps/Segment | 11.8 |

### Feasibility Assessment

| Capability | Supported |
|------------|-----------|
| Tabular Classification | [YES] |
| Sequence Construction | [NO] |
| Prediction Horizon Support | [NO] |

### Limitations

- 50/50 segments have multiple readings per timestamp (conflicting telemetry/target values). Preserved with reading_index; not a regular time series.
- Global time span is only 0.27 hours (< 1 hour). Insufficient for temporal sequence modeling with meaningful horizons.
- Minimum unique timestamps per segment is 8 (< 10). Insufficient for meaningful sequence windows.
- Metadata 'pipeline_id' is missing. Assuming single pipeline; segment_id used as sensor_id.
- Telemetry 'vibration' not present in dataset. Not required for current tabular classification scope (adapted PIPER scope uses 3 primary telemetry: pressure, flow_rate, temperature).


## Phase 2 Completion Status

**Status:** COMPLETE (adapted tabular-data scope)

### Artifacts Generated

- data/raw/scada_pipeline.csv (preserved unchanged)
- data/processed/scada_pipeline_cleaned.csv
- data/processed/scada_pipeline_features.csv
- data/processed/scada_feature_manifest.json
- data/reports/scada_dataset_profile.json
- data/reports/scada_cleaning_report.json
- data/reports/scada_label_report.json
- data/reports/scada_temporal_feasibility.json
- data/reports/phase2_summary.md


## Next Steps (Phase 3)

Phase 2 requirements for the adapted tabular classification scope are satisfied.

**Current scope:** Tabular abnormal-condition classification using 3 primary telemetry (pressure, flow_rate, temperature).

**Dataset limitations acknowledged:**
- Sequence construction (LSTM/GRU) NOT supported by current dataset (~16 min span, multiple readings per timestamp)
- Temporal forecasting NOT supported
- Vibration telemetry not required for current scope

**Phase 3 can begin** for tabular model development (Random Forest, XGBoost, MLP, SVM).
Sequence modeling track should be deferred until richer temporal data is available.
