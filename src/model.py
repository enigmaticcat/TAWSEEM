"""
TAWSEEM MLP Model (PyTorch)
Multilayer Perceptron for DNA contributor number estimation.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_HIDDEN_LAYERS, HIDDEN_DIM, DROPOUT_RATE, NUM_CLASSES


class TAWSEEM_MLP(nn.Module):
    """
    Multilayer Perceptron for DNA contributor estimation.
    
    Profile-level architecture (for ~140 aggregated features):
    - Input -> 256 -> BN -> ReLU -> Dropout(0.4)
    - 256 -> 128 -> BN -> ReLU -> Dropout(0.4)
    - 128 -> 64 -> BN -> ReLU -> Dropout(0.3)
    - 64 -> 32 -> BN -> ReLU -> Dropout(0.3)
    - 32 -> 5 (output)
    
    Uses BatchNorm + higher dropout for regularization with small datasets.
    """
    
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super(TAWSEEM_MLP, self).__init__()
        
        self.network = nn.Sequential(
            # Block 1: Input -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Block 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Block 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Block 4: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output
            nn.Linear(32, num_classes),
        )
        
        self._input_dim = input_dim
        self._num_classes = num_classes
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def summary(self):
        """Print model summary."""
        linear_layers = [l for l in self.network if isinstance(l, nn.Linear)]
        dropout_layers = [l for l in self.network if isinstance(l, nn.Dropout)]
        
        print(f"TAWSEEM MLP Model:")
        print(f"  Input dim:     {self._input_dim}")
        print(f"  Architecture:  {' -> '.join([str(self._input_dim)] + [str(l.out_features) for l in linear_layers])}")
        print(f"  Hidden layers: {len(linear_layers) - 1}")
        print(f"  Dropout layers: {len(dropout_layers)}")
        print(f"  Output classes: {self._num_classes}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:  {total_params:,}")
        print(f"  Trainable:     {trainable:,}")


if __name__ == "__main__":
    model = TAWSEEM_MLP(input_dim=140)
    model.summary()
    
    x = torch.randn(30, 140)
    output = model(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {torch.softmax(output[0], dim=0).detach().numpy()}")
