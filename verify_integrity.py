import pandas as pd
import numpy as np

# Load original new dataset (updated)
original_path = r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv"
# Load corrected dataset
corrected_path = r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv"

orig = pd.read_csv(original_path)
corr = pd.read_csv(corrected_path)

print("="*70)
print("INTEGRITY VERIFICATION")
print("="*70)

# 1. Row count
print(f"\n1. ROW COUNT")
print(f"   Original: {len(orig)} rows")
print(f"   Corrected: {len(corr)} rows")
print(f"   Match: {len(orig) == len(corr)}")

# 2. Column count and order
print(f"\n2. COLUMNS")
print(f"   Original columns ({len(orig.columns)}): {list(orig.columns)}")
print(f"   Corrected columns ({len(corr.columns)}): {list(corr.columns)}")
print(f"   Match: {list(orig.columns) == list(corr.columns)}")

# 3. Column-by-column comparison (except valve_status)
print(f"\n3. COLUMN VALUES COMPARISON (excluding valve_status)")
for col in orig.columns:
    if col == 'valve_status':
        continue
    # For numeric columns, check exact equality
    if orig[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        match = orig[col].equals(corr[col])
        if not match:
            # Check if it's just floating point precision
            diff = (orig[col] - corr[col]).abs().max()
            print(f"   {col}: MISMATCH (max diff: {diff})")
        else:
            print(f"   {col}: OK")
    else:
        match = orig[col].equals(corr[col])
        if not match:
            print(f"   {col}: MISMATCH")
        else:
            print(f"   {col}: OK")

# 4. valve_status changes
print(f"\n4. VALVE_STATUS CHANGES")
print(f"   Original distribution:")
print(orig['valve_status'].value_counts().sort_index())
print(f"   Corrected distribution:")
print(corr['valve_status'].value_counts().sort_index())

changed = (orig['valve_status'] != corr['valve_status']).sum()
print(f"   Rows changed: {changed} ({changed/len(orig)*100:.2f}%)")
print(f"   Rows preserved: {len(orig) - changed}")

# 5. Check for new missing values or unexpected categories
print(f"\n5. DATA QUALITY CHECKS")
print(f"   Missing values in corrected:")
missing = corr.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   None")
print(f"   valve_status unique values: {sorted(corr['valve_status'].unique())}")

# 6. Row order check
print(f"\n6. ROW ORDER")
# Check if timestamp and segment_id order is preserved
orig_key = orig['timestamp'] + '_' + orig['segment_id'].astype(str)
corr_key = corr['timestamp'] + '_' + corr['segment_id'].astype(str)
order_match = (orig_key == corr_key).all()
print(f"   Row order preserved (by timestamp+segment_id): {order_match}")

# 7. Check that target, alarm_triggered, event_type unchanged
print(f"\n7. CRITICAL COLUMNS UNCHANGED")
for col in ['target', 'alarm_triggered', 'event_type']:
    match = orig[col].equals(corr[col])
    print(f"   {col}: {'OK' if match else 'MISMATCH'}")

print(f"\n{'='*70}")
print("INTEGRITY VERIFICATION COMPLETE")
print(f"{'='*70}")