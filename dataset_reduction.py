"""
dataset_reduction.py

First stage of the data pipeline.
This script safely samples massive log files (like 10M+ rows) in manageable chunks via Pandas.
It parses the baseline `auth` logs against the known `redteam` attacks to tag anomalies.
Then, it randomly samples normal activity proportional to the anomalies found
and concatenates everything into a reduced, memory-safe CSV file for preprocessing.
"""

import pandas as pd
import numpy as np
import os
import gc
from config import CHUNK_SIZE, NORMAL_ROWS_RATIO, NORMAL_ROWS_SAMPLING_RATE_FALLBACK

# --- File Paths ---
base_path = 'Data'
dataset_path = f'{base_path}/Dataset'
auth_file_path = f'{dataset_path}/auth.txt.gz' 
redteam_file_path = f'{dataset_path}/redteam.txt.gz'

reduced_dataset_path = f'{dataset_path}/dataset_reduced'
os.makedirs(reduced_dataset_path, exist_ok=True)
auth_reduced_path_csv_gz = f'{reduced_dataset_path}/auth_reduced_{NORMAL_ROWS_RATIO}.csv.gz'

# --- Columns Setup ---
auth_cols = [
    'time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 
    'auth_type', 'logon_type', 'auth_orientation', 'status'
]
redteam_cols = ['time_rt', 'src_user_rt', 'src_comp_rt', 'dst_comp_rt']

def main():
    print(f"Loading anomalous events from: {redteam_file_path}")
    redteam_df = pd.read_csv(redteam_file_path, header=None, names=redteam_cols, sep=',')
    print(f"Loaded redteam data shape: {redteam_df.shape}")

    # Create a fast lookup set of known anomalies to optimize intersection checks
    redteam_event_keys = set(tuple(x) for x in redteam_df.values)
    del redteam_df

    print(f"Created lookup hash map with {len(redteam_event_keys)} attack signatures.")
    print(f"Starting chunked stream processing on {auth_file_path} to prevent memory crashes...")

    chunk_iterator = pd.read_csv(auth_file_path, chunksize=CHUNK_SIZE, header=None, names=auth_cols, sep=',')
    reduced_dfs = []

    total_rows_processed = 0
    total_anomalies_found = 0

    for chunk_num, chunk_df in enumerate(chunk_iterator, 1):
        print(f"\\nProcessing chunk {chunk_num} ({len(chunk_df)} rows)")
        total_rows_processed += len(chunk_df)

        # Leverage Pandas vectorization to match rows against the redteam hash map
        chunk_keys_series = pd.Series([tuple(x) for x in chunk_df[['time', 'src_user', 'src_comp', 'dst_comp']].values])
        chunk_df['is_anomaly'] = chunk_keys_series.isin(redteam_event_keys).astype(int)

        anomalous_in_chunk = chunk_df[chunk_df['is_anomaly'] == 1]
        
        # Calculate exactly how many normal rows we need to sample relative to anomalies
        if not anomalous_in_chunk.empty:
            total_anomalies_found += len(anomalous_in_chunk)
            
            n_normal_to_sample = int(len(anomalous_in_chunk) * NORMAL_ROWS_RATIO)
            normal_in_chunk = chunk_df[chunk_df['is_anomaly'] == 0]
            
            # If we don't have enough normal rows in this chunk, take all of them
            if len(normal_in_chunk) > n_normal_to_sample:
                sampled_normal = normal_in_chunk.sample(n=n_normal_to_sample, random_state=42)
            else:
                sampled_normal = normal_in_chunk 
            
            reduced_chunk_df = pd.concat([anomalous_in_chunk, sampled_normal])
        else:
            # If the chunk has no anomalies at all, fallback to a tiny generic sample
            normal_in_chunk = chunk_df[chunk_df['is_anomaly'] == 0]
            sampled_normal = normal_in_chunk.sample(frac=NORMAL_ROWS_SAMPLING_RATE_FALLBACK, random_state=42)
            reduced_chunk_df = sampled_normal
            
        reduced_dfs.append(reduced_chunk_df)
        
        # Manually clear memory allocations to avoid Out-Of-Memory degradation over time
        del chunk_df, anomalous_in_chunk, normal_in_chunk, sampled_normal, reduced_chunk_df, chunk_keys_series
        gc.collect()

    print(f"\\nChunk processing complete. Total rows handled: {total_rows_processed}")

    print("Concatenating all processed chunks into the final dataframe...")
    auth_reduced_df = pd.concat(reduced_dfs, ignore_index=True)
    
    # Confirm that we didn't drop any anomalous behavior during reduction
    total_anomalies_in_reduced = auth_reduced_df['is_anomaly'].sum()
    print(f"Final reduced shape: {auth_reduced_df.shape}")
    print(f"Total anomalies kept: {total_anomalies_in_reduced}")

    print(f"Saving the compressed dataset to: {auth_reduced_path_csv_gz}")
    auth_reduced_df.drop(columns=['is_anomaly']).to_csv(auth_reduced_path_csv_gz, index=False, compression='gzip')
    print("Optimization phase completed successfully!")

if __name__ == "__main__":
    main()
