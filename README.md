# DL-Anomaly-Detection

A Deep Learning modular pipeline designed for sequential log anomaly detection. This repository utilizes PyTorch and Vectorized Pandas arrays natively executable via terminal (CLI). 

## Project Architecture

All interactive Jupyter notebooks have been refactored into robust `python` modules to solve memory constraints, random seeding problems, and Jupyter state synchronization bugs.

- **`config.py`**: The Central Nerve system. Controls sequence windows (`WINDOW_SIZE`, `STRIDE`), chunk sampling limits, and Deep Learning `BATCH_SIZE`. Any edit here automatically overrides the remaining scripts.
- **`dataset_reduction.py`**: Employs scalable, vectorized Pandas streams to chunk through massive multi-gigabyte files (e.g. `auth.txt.gz`) cutting them down against the `redteam` anomaly file without RAM exhaustion.
- **`data_preprocessing.py`**: Conducts Time Scaling (`StandardScaler`), builds categorical vocabularies, and slices temporal overlapping strings into sequential Numpy (`.npy`) data windows grouped by node/user identity.
- **`Models/`**:
  - `train_utils.py`: A unified Deep Learning engine containing shared DataLoaders, Early-Stopping evaluation logic, Loss plotting, and ASCII Confusion Matrix rendering.
  - `train_lstm.py`: The classic Long Short-Term Memory detection algorithm.
  - `train_bilstm.py`: A bidirectional LSTM featuring dynamic Time-Attention layers.
  - `train_transformer.py`: Top-tier Positional Encoded Self-Attention detection scheme.

## Quick Start

1. Create a `venv` and source it, downloading PyTorch & related components from `requirements.txt`:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Push your gigantic baseline files (`auth.txt.gz` and `redteam.txt.gz`) into the relative directory: `Data/Dataset/`. 

3. Process the entire sequence stream and cache the `.npy` deep-learning tensors:
```bash
python dataset_reduction.py
python data_preprocessing.py
```

4. Launch any of the Deep Learning evaluation trainers:
```bash
python Models/train_lstm.py
# or
python Models/train_bilstm.py
```

*The models will automatically detect and hook into your CUDA graphics engine. Optimal Loss weights (`.pt`) are continuously saved inside `Data/Models/` and Console Metric logs (Precision, F1) are printed at convergence threshold!*