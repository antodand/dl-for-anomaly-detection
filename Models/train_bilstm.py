"""
train_bilstm.py

Training script for evaluating a Bidirectional LSTM (BiLSTM) with an Attention mechanism.
Unlike a plain LSTM, this model reads the sequences both forwards and backwards simultaneously
to grasp full context. The injected Attention layer then highlights the most suspicious
and important events automatically, weighing them higher than normal noise.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Ensure config.py can be loaded from the parent directory
sys.path.append(os.path.abspath('..'))
from train_utils import load_data, train_model, evaluate_and_plot

# --- BiLSTM Model Definition ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        # Simple network to calculate importance scores for the timeline
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, lstm_output):
        # Calculate scores highlighting which part of the sequence matters most
        attn_weights = self.attention(lstm_output)
        
        # Combine the actual LSTM variables scaled tightly by the calculated attention weights
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector, attn_weights

class BiLSTMAttentionDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(BiLSTMAttentionDetector, self).__init__()
        
        # Dual-direction recurrent layer setting
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        # Link the custom attention calculator layer established above
        self.attention = Attention(hidden_size)
        
        # Dropout helps prevent the model from memorizing the entire array
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multiply linear size by 2 since the BiLSTM reads backward and forwards
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Scan through sequences from start-to-end and end-to-start
        lstm_out, _ = self.lstm(x)
        
        # Apply intelligent pooling isolating anomalous variables over time
        context_vector, attn_weights = self.attention(lstm_out)
        out = self.dropout(context_vector)
        
        # Condense model understanding into a simple scalar classification score
        return torch.sigmoid(self.fc(out))

# --- Main Execution ---
def main():
    train_loader, val_loader, test_loader, input_size = load_data()

    # Core Training Hyperparameters
    hidden_size = 64
    num_layers = 1
    dropout_prob = 0.4
    learning_rate = 0.001
    num_epochs = 30
    patience = 5

    # Look for a GPU (CUDA) to speed up execution, otherwise run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing model on hardware device: {device}")
    
    model = BiLSTMAttentionDetector(input_size, hidden_size, num_layers, dropout_prob).to(device)
    
    criterion = nn.BCELoss()
    # Trigger momentum-based Adam algorithm for fast backpropagation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model_save_path = '../Data/Models/bilstm_best_model_weights.pt'

    # Shift heavy-lifting optimization code to the train_utils file
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, num_epochs, patience, model_save_path
    )

    evaluate_and_plot(model, val_loader, test_loader, device)

if __name__ == "__main__":
    main()
