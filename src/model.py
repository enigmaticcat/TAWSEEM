"""
TAWSEEM MLP Model (PyTorch)
Multilayer Perceptron with 15 hidden layers and 7 dropout layers.
Architecture matches Table 5 of the paper.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_HIDDEN_LAYERS, HIDDEN_DIM, DROPOUT_RATE, NUM_CLASSES


class TAWSEEM_MLP(nn.Module):
    """
    Multilayer Perceptron for DNA contributor estimation.
    
    Architecture (from paper Table 5):
    - Input layer → 64 neurons
    - 15 hidden layers (64 neurons each), ReLU activation
    - 7 dropout layers (rate=0.2), placed every 2 hidden layers
    - Output layer: 5 classes (softmax)
    
    Structure:
        Input → [Linear→ReLU] → [Linear→ReLU, Linear→ReLU, Dropout] × 7 → Linear → Output
        = 1 + 14 hidden layers = 15 total, with 7 dropouts
    """
    
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super(TAWSEEM_MLP, self).__init__()
        
        layers = []
        
        # Input layer → first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 14 more hidden layers organized in 7 blocks of 2 layers + dropout
        for block in range(7):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Count layers for verification
        linear_layers = sum(1 for l in self.network if isinstance(l, nn.Linear))
        dropout_layers = sum(1 for l in self.network if isinstance(l, nn.Dropout))
        
        self._n_hidden = linear_layers - 1  # exclude output layer
        self._n_dropout = dropout_layers
        self._input_dim = input_dim
    
    def forward(self, x):
        return self.network(x)
    
    def summary(self):
        """Print model summary."""
        print(f"TAWSEEM MLP Model:")
        print(f"  Input dim:     {self._input_dim}")
        print(f"  Hidden layers: {self._n_hidden} ({HIDDEN_DIM} neurons each)")
        print(f"  Dropout layers: {self._n_dropout} (rate={DROPOUT_RATE})")
        print(f"  Output classes: {NUM_CLASSES}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:  {total_params:,}")
        print(f"  Trainable:     {trainable:,}")


if __name__ == "__main__":
    # Quick test
    model = TAWSEEM_MLP(input_dim=48)
    model.summary()
    
    # Test forward pass
    x = torch.randn(30, 48)  # batch_size=30, features=48
    output = model(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {torch.softmax(output[0], dim=0).detach().numpy()}")
