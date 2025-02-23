import os
import pandas as pd


csv_folder = './Csv'
output_folder = 'FilteredData'
dataframes = []

for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder, filename)
        device_model = os.path.splitext(filename)[0]
        df = pd.read_csv(file_path, header=None, names=['Sensor_Model', 'Timestamp', 'Volume'])
        
        df['Device_Model'] = device_model
        
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Volume'])
        
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
aggregated_data = combined_df.groupby('Device_Model').agg({
    'Volume': ['mean', 'min', 'max'],
    'Timestamp': ['min', 'max']
}).reset_index()

aggregated_data.columns = [
    'Device_Model', 'Average_Volume', 'Min_Volume', 'Max_Volume',
    'Earliest_Timestamp', 'Latest_Timestamp'
]

output_path = os.path.join(output_folder, "aggregated_data.csv")
aggregated_data.to_csv(output_path, index=False)

print("Aggregated Data for Device Models:")
print(aggregated_data.head())
