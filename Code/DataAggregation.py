import pandas as pd
import os

csv_folder = './CSV'
dataframes = []

for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):  
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path)
        df['source_file'] = filename
        dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)

aggregated_data = combined_df.groupby('column_name').agg({
    'numerical_column': 'sum',  # Sum values
    'another_column': 'mean'   # Calculate the mean
})
aggregated_data = aggregated_data.reset_index()
output_path = "aggregated_data.csv"
aggregated_data.to_csv(output_path, index=False)

print("Aggregated Data:")
print(aggregated_data)