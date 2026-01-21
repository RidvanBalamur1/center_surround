import torch
import torch.nn as nn


class LNModel(nn.Module):
    """Simple Linear-Nonlinear model for RGC cells."""
    
    def __init__(self, num_neurons, input_channels=1, kernel_size=31):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_neurons, kernel_size=kernel_size, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Softplus()
    
    def forward(self, x):
        x = self.conv(x)           # Linear filter
        x = self.pool(x)           # Spatial pooling
        x = x.flatten(1)           # (B, num_neurons)
        x = self.activation(x)     # Nonlinearity
        return x