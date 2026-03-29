"""
train_lstm.py

Training script for a classic Long Short-Term Memory (LSTM) network.
This model processes our sequential log sequences by remembering past 
states and learning normal behavior over time.
It's a foundational model for sequence anomaly detection.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Ensure config.py can be loaded from the parent directory
sys.path.append(os.path.abspath('..'))
from train_utils import load_data, train_model, evaluate_and_plot

# --- LSTM Model Definition ---
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTMAnomalyDetector, self).__init__()
        
        # Core LSTM processor analyzing the sequential relationships
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Dropout helps prevent the model from memorizing the data
        self.dropout = nn.Dropout(dropout_prob)
        
        # Final classification layer predicting the anomaly score
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Process the full tensor sequence: (Batch Size, Sequence Length, Features count)
        lstm_out, _ = self.lstm(x)
        
        # Since we only care about the sequence end-state, target the last time step
        last_time_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_time_step_out)
        
        # Compress the output naturally between a 0.0 and 1.0 probability score
        return torch.sigmoid(self.fc(out))

# --- Main Execution ---
def main():
    train_loader, val_loader, test_loader, input_size = load_data()

    # Core Training Hyperparameters
    hidden_size = 64
    num_layers = 2
    dropout_prob = 0.3
    learning_rate = 0.001
    num_epochs = 30
    patience = 5

    # Look for a GPU (CUDA) to speed up execution, otherwise run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing model on hardware device: {device}")
    
    model = LSTMAnomalyDetector(input_size, hidden_size, num_layers, dropout_prob).to(device)
    
    # Track the standard Binary Cross-Entropy Loss to measure accuracy
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model_save_path = '../Data/Models/lstm_best_model_weights.pt'

    # Shift heavy-lifting optimization code to the train_utils file
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, num_epochs, patience, model_save_path
    )

    evaluate_and_plot(model, val_loader, test_loader, device)

if __name__ == "__main__":
    main()
