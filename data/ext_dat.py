import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup and authenticate
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")
api = KaggleApi()
api.authenticate()

# Download without unzipping
dataset = 'wordsforthewise/lending-club'
print("Downloading ZIP...")
api.dataset_download_files(dataset, path="1_data", unzip=False)

# Path to downloaded ZIP
zip_path = "1_data/lending-club.zip"
filtered_rows = []

# Stream process
print("Streaming and filtering rows from inside ZIP...")
with zipfile.ZipFile(zip_path) as z:
    with z.open('accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv') as f:
        chunks = pd.read_csv(f, chunksize=50000, low_memory=False)
        for chunk in chunks:
            if 'issue_d' in chunk.columns:
                # Extract year from issue date
                chunk['issue_year'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce').dt.year
                filtered = chunk[chunk['issue_year'] >= 2015]
                filtered_rows.append(filtered)

# Combine and save filtered data
df_filtered = pd.concat(filtered_rows)
output_path = "1_data/filtered_2015_onwards.csv"
df_filtered.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}. Shape: {df_filtered.shape}")

