"""
config.py

Global configuration file for the Anomaly Detection project.
This module stores the core parameters used across the entire pipeline.
Adjusting these values will automatically update the data reduction, 
preprocessing, and model training steps.
"""

# --- Sequence Hyperparameters ---
VERSION = "v2"
WINDOW_SIZE = 30       # Number of events per sliding window sequence
STRIDE = 5             # Number of events to slide forward to create the next window

# --- Dataset Reduction Limits ---
CHUNK_SIZE = 10_000_000 # Number of rows to process at once to avoid memory errors
NORMAL_ROWS_RATIO = 2.0  # How many normal rows to keep for every anomalous row
NORMAL_ROWS_SAMPLING_RATE_FALLBACK = 0.005 # Fallback sampling rate if no anomalies are found

# --- Model Training ---
BATCH_SIZE = 512       # Number of sequences processed together during training
