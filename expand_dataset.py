import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the original data
df = pd.read_csv('scada_pipeline.csv')

# Target medians
target_medians = {
    'pressure': 73.91,
    'flow_rate': 4.56,
    'temperature': 31.62,
    'pump_speed': 1233.10,
    'energy_consumption': 25.31
}

# Analyze current distributions to match variance
current_stats = {}
for col in target_medians.keys():
    current_stats[col] = {
        'mean': df[col].mean(),
        'std': df[col].std(),
        'median': df[col].median(),
        'min': df[col].min(),
        'max': df[col].max()
    }

print("Current statistics:")
for col, stats in current_stats.items():
    print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, median={stats['median']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

# For categorical columns, get the distribution
categorical_cols = ['valve_status', 'pump_state', 'compressor_state', 'alarm_triggered', 'event_type', 'target']
cat_distributions = {}
for col in categorical_cols:
    cat_distributions[col] = df[col].value_counts(normalize=True)
    print(f"\n{col} distribution:")
    print(cat_distributions[col])

# segment_id distribution
segment_dist = df['segment_id'].value_counts(normalize=True)
print("\nsegment_id distribution:")
print(segment_dist.head(20))

# Generate new rows
np.random.seed(42)  # for reproducibility
n_new_rows = 3600 - len(df)  # 2600 more rows needed

new_rows = []
last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1], format='%m/%d/%Y %H:%M:%S')

for i in range(n_new_rows):
    new_timestamp = last_timestamp + timedelta(seconds=i+1)
    
    # Generate numerical values using normal distribution centered at target medians
    # with std matching current data
    pressure = np.random.normal(target_medians['pressure'], current_stats['pressure']['std'])
    flow_rate = np.random.normal(target_medians['flow_rate'], current_stats['flow_rate']['std'])
    temperature = np.random.normal(target_medians['temperature'], current_stats['temperature']['std'])
    pump_speed = np.random.normal(target_medians['pump_speed'], current_stats['pump_speed']['std'])
    energy_consumption = np.random.normal(target_medians['energy_consumption'], current_stats['energy_consumption']['std'])
    
    # Clip to reasonable ranges based on current data
    pressure = np.clip(pressure, current_stats['pressure']['min'], current_stats['pressure']['max'])
    flow_rate = np.clip(flow_rate, current_stats['flow_rate']['min'], current_stats['flow_rate']['max'])
    temperature = np.clip(temperature, current_stats['temperature']['min'], current_stats['temperature']['max'])
    pump_speed = np.clip(pump_speed, current_stats['pump_speed']['min'], current_stats['pump_speed']['max'])
    energy_consumption = np.clip(energy_consumption, current_stats['energy_consumption']['min'], current_stats['energy_consumption']['max'])
    
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

# Create new dataframe and combine
new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([df, new_df], ignore_index=True)

# Verify medians
print("\nVerification of medians in combined dataset:")
for col in target_medians.keys():
    median_val = combined_df[col].median()
    print(f"  {col}: {median_val:.2f} (target: {target_medians[col]})")

# Save
combined_df.to_csv('scada_pipeline_1hour.csv', index=False)
print(f"\nSaved expanded dataset with {len(combined_df)} rows to scada_pipeline_1hour.csv")