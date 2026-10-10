import pandas as pd
from pathlib import Path

data_dir = Path('./LeakageData')
csv_files = sorted(data_dir.glob('*.csv'))
print('Number of CSV files:', len(csv_files))

total_rows = 0
meters = set()
date_ranges = []

for f in csv_files:
    df = pd.read_csv(f)
    total_rows += len(df)
    meters.add(df['Meter_ID'].iloc[0])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', dayfirst=True)
    date_ranges.append((df['Timestamp'].min(), df['Timestamp'].max()))
    print(f'  {f.name}: {len(df)} rows, meter={df["Meter_ID"].iloc[0]}, date_range={df["Timestamp"].min()} to {df["Timestamp"].max()}')

print()
print('Total rows:', total_rows)
print('Unique meters:', len(meters))
print('Meters:', meters)

# Check columns
print('\nColumns in first file:')
df0 = pd.read_csv(csv_files[0])
print(df0.columns.tolist())
print(df0.dtypes)
print(df0.head())