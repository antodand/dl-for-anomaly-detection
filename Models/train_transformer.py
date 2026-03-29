"""
train_transformer.py

Training script for evaluating a Transformer Network for sequence anomaly detection.
Unlike recurrent models (LSTMs) that parse data step-by-step, the Transformer processes 
the entire sequence at once. It uses Positional Encoding to understand event order and 
Multi-Head Attention to compare every event against every other event instantly, 
making it extremely fast and effective at finding hidden patterns.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

# Ensure config.py can be loaded from the parent directory
sys.path.append(os.path.abspath('..'))
from train_utils import load_data, train_model, evaluate_and_plot
from config import WINDOW_SIZE

# --- Transformer Model Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Build a matrix to track the order of events
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate sine and cosine waves to embed the timeline distances
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        
        # Keep this matrix fixed (it doesn't learn, it's just positional reference data)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Simply add the absolute timeline location to the variable array
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_prob, max_seq_length):
        super(TransformerAnomalyDetector, self).__init__()
        
        # Expand simple starting features into a dense vector (d_model) for deeper learning
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Embed the mathematical timeline distances calculated above
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        
        # Create standard Self-Attention layers that correlate all sequences simultaneously
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Final fully connected layers compressing relationships into an initial score
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Feed the positional matrices into the parallel Multi-Head Attention blocks
        encoded = self.transformer_encoder(x)
        
        # Compress the sequence output by averaging it (Global Average Pooling)
        pooled = torch.mean(encoded, dim=1)
        
        # Pass the condensed array through standard activation nodes
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Output a normalized final probability anomaly score
        return torch.sigmoid(out)

# --- Main Execution ---
def main():
    train_loader, val_loader, test_loader, input_dim = load_data()

    # Core Training Hyperparameters
    d_model = 64
    nhead = 4
    num_encoder_layers = 1
    dim_feedforward = 128
    dropout_prob = 0.3
    max_seq_length = WINDOW_SIZE
    learning_rate = 0.0005
    num_epochs = 30
    patience = 5

    # Look for a GPU (CUDA) to speed up execution, otherwise run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing model on hardware device: {device}")
    
    model = TransformerAnomalyDetector(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_prob, max_seq_length).to(device)
    
    criterion = nn.BCELoss()
    # Trigger momentum-based Adam algorithm for fast backpropagation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model_save_path = '../Data/Models/transformer_best_model_weights.pt'

    # Shift heavy-lifting optimization code to the train_utils file
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, num_epochs, patience, model_save_path
    )

    evaluate_and_plot(model, val_loader, test_loader, device)

if __name__ == "__main__":
    main()
