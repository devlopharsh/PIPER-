import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the original data to get distribution shapes and categorical distributions
df = pd.read_csv('scada_pipeline.csv')

# Target medians
target_medians = {
    'pressure': 73.91,
    'flow_rate': 4.56,
    'temperature': 31.62,
    'pump_speed': 1233.10,
    'energy_consumption': 25.31
}

# Current medians
current_medians = {col: df[col].median() for col in target_medians.keys()}

# Calculate shift needed
shifts = {col: target_medians[col] - current_medians[col] for col in target_medians.keys()}

print("Shifts needed:", shifts)

# For categorical columns, get the distribution
categorical_cols = ['valve_status', 'pump_state', 'compressor_state', 'alarm_triggered', 'event_type', 'target']
cat_distributions = {}
for col in categorical_cols:
    cat_distributions[col] = df[col].value_counts(normalize=True)

# segment_id distribution
segment_dist = df['segment_id'].value_counts(normalize=True)

# Generate all 3600 rows from scratch using shifted empirical distribution
np.random.seed(42)
n_total_rows = 3600

new_rows = []
start_timestamp = pd.to_datetime('01/01/2024 00:00:01', format='%m/%d/%Y %H:%M:%S')

for i in range(n_total_rows):
    new_timestamp = start_timestamp + timedelta(seconds=i)
    
    # Sample from original data and shift to target median
    idx = np.random.randint(0, len(df))
    row = df.iloc[idx]
    
    pressure = row['pressure'] + shifts['pressure']
    flow_rate = row['flow_rate'] + shifts['flow_rate']
    temperature = row['temperature'] + shifts['temperature']
    pump_speed = row['pump_speed'] + shifts['pump_speed']
    energy_consumption = row['energy_consumption'] + shifts['energy_consumption']
    
    # Clip to reasonable ranges based on original data
    pressure = np.clip(pressure, df['pressure'].min(), df['pressure'].max())
    flow_rate = np.clip(flow_rate, df['flow_rate'].min(), df['flow_rate'].max())
    temperature = np.clip(temperature, df['temperature'].min(), df['temperature'].max())
    pump_speed = np.clip(pump_speed, df['pump_speed'].min(), df['pump_speed'].max())
    energy_consumption = np.clip(energy_consumption, df['energy_consumption'].min(), df['energy_consumption'].max())
    
    # Sample categorical values from existing distributions
    segment_id = np.random.choice(segment_dist.index, p=segment_dist.values)
    valve_status = np.random.choice(cat_distributions['valve_status'].index, p=cat_distributions['valve_status'].values)
    pump_state = np.random.choice(cat_distributions['pump_state'].index, p=cat_distributions['pump_state'].values)
    compressor_state = np.random.choice(cat_distributions['compressor_state'].index, p=cat_distributions['compressor_state'].values)
    alarm_triggered = np.random.choice(cat_distributions['alarm_triggered'].index, p=cat_distributions['alarm_triggered'].values)
    event_type = np.random.choice(cat_distributions['event_type'].index, p=cat_distributions['event_type'].values)
    target = np.random.choice(cat_distributions['target'].index, p=cat_distributions['target'].values)
    
    new_rows.append({
        'timestamp': new_timestamp.strftime('%m/%d/%Y %H:%M:%S'),
        'segment_id': int(segment_id),
        'pressure': round(pressure, 2),
        'flow_rate': round(flow_rate, 2),
        'temperature': round(temperature, 2),
        'valve_status': int(valve_status),
        'pump_state': int(pump_state),
        'pump_speed': round(pump_speed, 2),
        'compressor_state': int(compressor_state),
        'energy_consumption': round(energy_consumption, 2),
        'alarm_triggered': int(alarm_triggered),
        'event_type': event_type,
        'target': int(target)
    })

# Create dataframe
combined_df = pd.DataFrame(new_rows)

# Verify medians
print("\nVerification of medians in generated dataset:")
for col in target_medians.keys():
    median_val = combined_df[col].median()
    print(f"  {col}: {median_val:.2f} (target: {target_medians[col]})")

# Also verify categorical distributions
print("\nCategorical distributions:")
for col in categorical_cols:
    dist = combined_df[col].value_counts(normalize=True)
    print(f"  {col}: {dict(dist.round(3))}")

# Save
combined_df.to_csv('scada_pipeline_1hour.csv', index=False)
print(f"\nSaved dataset with {len(combined_df)} rows to scada_pipeline_1hour.csv")