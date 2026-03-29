"""
train_utils.py

Shared toolset for all deep learning anomaly detection models.
This script contains the core data loading functions and the unified
PyTorch training loop logic. It handles batching, calculating the Binary Cross 
Entropy Loss, Early Stopping, and evaluating final metrics like the F1-Score 
and Confusion Matrix directly onto the console.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from config import VERSION, WINDOW_SIZE, STRIDE, BATCH_SIZE

def load_data(base_path='../Data'):
    """Finds the Numpy sequences saved by preprocessing and wraps them into optimal PyTorch DataLoaders."""
    sequences_folder_name = f"sequences_{VERSION}_W{WINDOW_SIZE}_S{STRIDE}"
    user_split_strategy_folder_name = f"user_split_W{WINDOW_SIZE}_S{STRIDE}"
    sequences_dir = f'{base_path}/Sequences/{sequences_folder_name}/{user_split_strategy_folder_name}'

    print(f"Loading generated tensor arrays from: {sequences_dir}")
    X_train = np.load(f'{sequences_dir}/X_train_normal_only.npy')
    y_train = np.load(f'{sequences_dir}/y_train_normal_only.npy')
    X_val = np.load(f'{sequences_dir}/X_val_user_split.npy')
    y_val = np.load(f'{sequences_dir}/y_val_user_split.npy')
    X_test = np.load(f'{sequences_dir}/X_test_user_split.npy')
    y_test = np.load(f'{sequences_dir}/y_test_user_split.npy')

    print(f"Dataset Size Check:\\n- Training Array: X={X_train.shape}, y={y_train.shape}\\n- Validation Array: X={X_val.shape}, y={y_val.shape}\\n- Held-Out Test Arrays: X={X_test.shape}, y={y_test.shape}")

    # Convert standard arrays into floating point PyTorch tensors for neural layers
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train.shape[2] # Shape[2] provides the feature column limit

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, model_save_path):
    print("\\nStarting the Training Loop...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):    
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Sub-routine handling abstract tensor returns (e.g. for Tuple outputs in BiLSTM)
            if isinstance(outputs, tuple): outputs = outputs[0] 
            
            loss = criterion(outputs, labels)
            
            # Trigger backpropagation to learn parameter bounds
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Disable gradient calculations to test standard inference speeds during Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Training Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_val_loss:.4f}")
        
        # Test if the validation array is improving. If so, save the model configuration safely.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Model hasn't improved dynamically. Triggering Early Stopping sequence.")
                break

    # Re-equip the most optimal weights observed before performance plummeted
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

def get_predictions_and_labels(model, dataloader, device):
    """Loop through loaded batches and aggregate generic probabilities into Python lists."""
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            # Shrink arrays from nested tensors into basic 1D arrays
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    return np.array(all_labels).astype(int), np.array(all_probs)

def print_ascii_cm(cm, dataset_name):
    """Takes a standard Sklearn Confusion Matrix and renders it aesthetically into the terminal context."""
    print(f"\\n[{dataset_name}] Detailed Confusion Matrix Grid:")
    print("-" * 55)
    print(f"{'':>20} | Pred Normal (0) | Pred Anomaly (1)")
    print("-" * 55)
    print(f"True Normal (0)      | {cm[0,0]:<15} | {cm[0,1]:<15}")
    
    # Simple bounds check in case the matrix only tested arrays with normal behavior
    if cm.shape == (2,2):
        print(f"True Anomaly (1)     | {cm[1,0]:<15} | {cm[1,1]:<15}")
    else:
        print(f"True Anomaly (1)     | {'0':<15} | {'0':<15}")
    print("-" * 55)

def evaluate_and_plot(model, val_loader, test_loader, device):
    """Calculates optimal cutoff thresholds to balance precision vs recall efficiently."""
    print("\\nCompiling testing predictions against evaluated convergence bounds...")
    val_labels_final, val_probs_final = get_predictions_and_labels(model, val_loader, device)
    test_labels_final, test_probs_final = get_predictions_and_labels(model, test_loader, device)

    print("\\n[Baseline Analytics Outline]")
    val_probs_normal = val_probs_final[val_labels_final == 0]
    val_probs_anomalous = val_probs_final[val_labels_final == 1]
    
    print(f"Healthy Normal Logs Subsample > Size={len(val_probs_normal)}, Safest Minimum Error Score={np.min(val_probs_normal):.2e}")
    if len(val_probs_anomalous) > 0:
        print(f"Malicious Exploits Subsample > Size={len(val_probs_anomalous)}, Safest Minimum Error Score={np.min(val_probs_anomalous):.2e}")

    print("\\n[Testing Variable Thresholds to Find Best F1-Score Value]")
    percentiles_to_try = [85, 90, 95, 97, 98, 99, 99.5, 99.9]
    best_f1_val_percentile = -1
    final_optimal_threshold = 0.5
    num_anomalies_val = np.sum(val_labels_final == 1)
    
    if len(val_probs_final) > 0 and num_anomalies_val > 0:
        for p_val in percentiles_to_try:
            # Map simple boundaries looping through probabilities mathematically
            current_threshold = np.percentile(val_probs_final, p_val)
            val_preds_p = (val_probs_final >= current_threshold).astype(int)
            f1_p = f1_score(val_labels_final, val_preds_p, pos_label=1, zero_division=0)
            if f1_p > best_f1_val_percentile:
                best_f1_val_percentile = f1_p
                final_optimal_threshold = current_threshold
        print(f"Final Best Error Decision Boundary Found Check: {final_optimal_threshold:.2e}")
        
    print(f"\\n--- [VALIDATION SET EVALUATION] Final Algorithm Accuracy Test (Cutoff={final_optimal_threshold:.2e}) ---")
    val_predicted_labels_final = (val_probs_final >= final_optimal_threshold).astype(int)
    print(classification_report(val_labels_final, val_predicted_labels_final, target_names=['Normal Baseline (0)', 'Malicious Intrusion (1)'], zero_division=0))
    print_ascii_cm(confusion_matrix(val_labels_final, val_predicted_labels_final), "Validation Set")

    print(f"\\n--- [TEST SET EVALUATION] Generalization Profiling Matrix (Cutoff={final_optimal_threshold:.2e}) ---")
    test_predicted_labels_final = (test_probs_final >= final_optimal_threshold).astype(int)
    print(classification_report(test_labels_final, test_predicted_labels_final, target_names=['Normal Baseline (0)', 'Malicious Intrusion (1)'], zero_division=0))
    print_ascii_cm(confusion_matrix(test_labels_final, test_predicted_labels_final), "Test Set")
