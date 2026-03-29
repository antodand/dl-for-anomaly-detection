"""
data_preprocessing.py

Second step of the data pipeline.
This script prepares the raw, condensed data for neural networks by standardizing
time characteristics and encoding categories (like Users, Computers, Protocols) into integers.
It groups events by user identity and generates chronological overlapping sequences
(often called 'sliding windows') mapping these events over time.
All structural hyperparameters (WINDOW, STRIDE, VERSION) are explicitly pulled from `config.py`.
"""

import pandas as pd
import numpy as np
import json
import gc
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import VERSION, WINDOW_SIZE, STRIDE, NORMAL_ROWS_RATIO

# --- Save Directories ---
base_path = 'Data'
dataset_path = f'{base_path}/Dataset'
embeddings_path = f'{base_path}/Embeddings'
sequences_path = f'{base_path}/Sequences'
auth_reduced_path = f'{dataset_path}/dataset_reduced/auth_reduced_{NORMAL_ROWS_RATIO}.csv.gz'
redteam_file_path = f'{dataset_path}/redteam.txt.gz'

os.makedirs(embeddings_path, exist_ok=True)
os.makedirs(sequences_path, exist_ok=True)

# --- Feature Columns ---
auth_cols = [
    'time', 'src_user', 'dst_user', 'src_comp', 'dst_comp', 
    'auth_type', 'logon_type', 'auth_orientation', 'status'
]
redteam_cols = ['time_rt', 'src_user_rt', 'src_comp_rt', 'dst_comp_rt']

def main():
    print(f"Loading the optimized dataset chunk: {auth_reduced_path}")
    auth_reduced_df = pd.read_csv(auth_reduced_path, na_values='?', compression='gzip')

    print(f"Loading the anomalous events target labels: {redteam_file_path}")
    redteam_df = pd.read_csv(redteam_file_path, header=None, names=redteam_cols, sep=',', compression='gzip', on_bad_lines='skip')

    # Convert the known anomalies list into a fast set to prevent slow matching
    redteam_event_keys = set(tuple(x) for x in redteam_df.values)
    del redteam_df 

    print("Iterating through the normal dataset to tag local anomalies...")
    reduced_df_keys = pd.Series([tuple(x) for x in auth_reduced_df[['time', 'src_user', 'src_comp', 'dst_comp']].values])
    auth_reduced_df['is_anomaly'] = reduced_df_keys.isin(redteam_event_keys).astype(int)
    del reduced_df_keys
    gc.collect()

    print(f"Anomaly occurrences inside the dataset:\\n{auth_reduced_df['is_anomaly'].value_counts(dropna=False)}")

    # --- Time Feature Standardization ---
    auth_reduced_df['time_scaled_per_user'] = np.nan
    for user_name, group_df in auth_reduced_df.groupby('src_user'):
        
        # Scale time instances per user to improve algorithmic tracking convergence
        if not group_df.empty:
            scaler = StandardScaler()
            time_values_reshaped = group_df['time'].values.reshape(-1, 1)
            scaled_time_values = scaler.fit_transform(time_values_reshaped)
            auth_reduced_df.loc[group_df.index, 'time_scaled_per_user'] = scaled_time_values.flatten()

    auth_reduced_df['time_scaled_per_user'] = auth_reduced_df['time_scaled_per_user'].fillna(0.0)
    print("Time values standardized using Zero-Mean limits.")

    # --- Text to Categorical Encodings ---
    categorical_cols_to_embed = [
        'src_user', 'dst_user', 'src_comp', 'dst_comp',
        'auth_type', 'logon_type', 'auth_orientation', 'status'
    ]
    feature_vocabs = {}
    embedding_data = {}
    missing_value_placeholder = "_MISSING_"

    for col in categorical_cols_to_embed:
        # Prevent unique protocol string overlaps from destroying vocabulary tracking
        if col == 'auth_type':
            condition = auth_reduced_df[col].astype(str).str.upper().str.startswith("MICROSOFT_AUTHENTICA")
            auth_reduced_df.loc[condition, col] = "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0"

        # Hardcode any empty values to avoid runtime breaking strings
        if auth_reduced_df[col].isnull().sum() > 0:
            auth_reduced_df[col] = auth_reduced_df[col].fillna(missing_value_placeholder)

        auth_reduced_df[col] = auth_reduced_df[col].astype('category')
        current_categories = auth_reduced_df[col].cat.categories

        vocab = {str(cat): i for i, cat in enumerate(current_categories)}
        feature_vocabs[col] = vocab
        
        vocab_size = len(current_categories)
        embedding_data[f'{col}_vocab_size'] = vocab_size
        auth_reduced_df[f'{col}_encoded'] = auth_reduced_df[col].cat.codes

    print("Missing values safely imputed and categoricals matched.")

    feature_vocabs_path = f'{embeddings_path}/feature_vocabs_{VERSION}.json'
    embedding_data_path = f'{embeddings_path}/embedding_data_{VERSION}.json'

    with open(feature_vocabs_path, 'w') as f: json.dump(feature_vocabs, f, indent=4)
    with open(embedding_data_path, 'w') as f: json.dump(embedding_data, f, indent=4)
    print("String translation arrays cached as dictionaries.")

    # --- Flattening Data Structure ---
    all_encoded_cols = [f'{col}_encoded' for col in categorical_cols_to_embed if f'{col}_encoded' in auth_reduced_df.columns]
    cols_for_final_df = ['time', 'time_scaled_per_user', 'src_user_encoded'] + [col for col in all_encoded_cols if col != 'src_user_encoded'] + ['is_anomaly']

    # Dump the cleaned dataframe sequentially before creating deep learning arrays
    encoded_events_df = auth_reduced_df[pd.Series(cols_for_final_df).drop_duplicates().tolist()].copy()
    encoded_events_filename = f"encoded_events_df_for_sequencing_{VERSION}.csv.gz"
    encoded_events_path = f'{sequences_path}/{encoded_events_filename}'

    encoded_events_df.to_csv(encoded_events_path, index=False, compression='gzip')
    print(f"Compressed numerical spreadsheet saved: {encoded_events_path}")

    encoded_events_df = pd.read_csv(encoded_events_path, compression='gzip')

    # --- Group Users & Calculate Overlapping Sequences ---
    feature_cols_for_sequence = [col for col in encoded_events_df.columns if col.endswith('_encoded')] + ['time_scaled_per_user']
    all_sequences_list = []
    all_micro_labels_list = []
    all_user_ids_for_sequences_list = []

    print("Extracting sequences using sliding windows bounded per discrete user entity...")
    grouped_by_user = encoded_events_df.groupby('src_user_encoded')
    for user_id, user_activity_df in grouped_by_user:
        # Guarantee sequential alignment prior to window segmentation
        user_activity_df_sorted = user_activity_df.sort_values(by='time')
        user_features_np = user_activity_df_sorted[feature_cols_for_sequence].values
        user_event_labels_np = user_activity_df_sorted['is_anomaly'].values

        for i in range(0, len(user_features_np) - WINDOW_SIZE + 1, STRIDE):
            sequence_features = user_features_np[i : i + WINDOW_SIZE]
            sequence_event_labels_for_window = user_event_labels_np[i : i + WINDOW_SIZE]
            
            # Label the entire window anomalous if ANY included event was malicious
            sequence_label = 1 if sequence_event_labels_for_window.any() else 0
            
            all_sequences_list.append(sequence_features.tolist())
            all_micro_labels_list.append(sequence_label)
            all_user_ids_for_sequences_list.append(user_id)

    print(f"Extraction created a total dataset size of N={len(all_sequences_list)} sliding properties.")

    # Convert the Python arrays natively into Numpy arrays required for neural networks
    X_final_sequences = np.array(all_sequences_list, dtype=np.float64)
    y_final_labels = np.array(all_micro_labels_list, dtype=np.int64)
    user_ids_final_array = np.array(all_user_ids_for_sequences_list)

    sequences_folder_name = f"sequences_{VERSION}_W{WINDOW_SIZE}_S{STRIDE}"
    sequences_output_path = f'{sequences_path}/{sequences_folder_name}'
    os.makedirs(sequences_output_path, exist_ok=True)

    np.save(f'{sequences_output_path}/X_sequences_{VERSION}.npy', X_final_sequences)
    np.save(f'{sequences_output_path}/y_labels_{VERSION}.npy', y_final_labels)
    np.save(f'{sequences_output_path}/user_ids_for_sequences_{VERSION}.npy', user_ids_final_array)

    # --- Build Dedicated Test / Validation Subsets ---
    # We restrict users geographically rather than rows, so models don't easily
    # track a specific unique user's behavior on the test and train set simultaneously.
    unique_users = np.unique(user_ids_final_array)
    users_train_ids, users_temp_pool = train_test_split(unique_users, test_size=0.40, random_state=42, shuffle=False)
    users_val_ids, users_test_ids = train_test_split(users_temp_pool, test_size=0.50, random_state=42, shuffle=False)

    train_user_mask = np.isin(user_ids_final_array, users_train_ids)
    X_from_train_users = X_final_sequences[train_user_mask]
    y_from_train_users = y_final_labels[train_user_mask]

    # Models only train exclusively on healthy sequences to learn normal behavior
    X_train = X_from_train_users[y_from_train_users == 0]
    y_train = y_from_train_users[y_from_train_users == 0]

    val_user_mask = np.isin(user_ids_final_array, users_val_ids)
    X_val = X_final_sequences[val_user_mask]
    y_val = y_final_labels[val_user_mask]

    test_user_mask = np.isin(user_ids_final_array, users_test_ids)
    X_test = X_final_sequences[test_user_mask]
    y_test = y_final_labels[test_user_mask]

    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test Evaluation shapes: X={X_test.shape}, y={y_test.shape}")

    user_split_output_dir = f'{sequences_path}/{sequences_folder_name}/user_split_W{WINDOW_SIZE}_S{STRIDE}'
    os.makedirs(user_split_output_dir, exist_ok=True)

    np.save(f'{user_split_output_dir}/X_train_normal_only.npy', X_train)
    np.save(f'{user_split_output_dir}/y_train_normal_only.npy', y_train)
    np.save(f'{user_split_output_dir}/X_val_user_split.npy', X_val)
    np.save(f'{user_split_output_dir}/y_val_user_split.npy', y_val)
    np.save(f'{user_split_output_dir}/X_test_user_split.npy', X_test)
    np.save(f'{user_split_output_dir}/y_test_user_split.npy', y_test)
    print("Pre-processed Numpy sequences successfully deployed to the filesystem.")

if __name__ == "__main__":
    main()
