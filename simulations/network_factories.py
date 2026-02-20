"""
Custom neural network architectures for semi-synthetic simulations.

Provides factory functions that return nn.Module instances compatible
with the deep_inference train_structural_net() API. Each module
maps X -> theta (d_theta-dimensional structural parameters).
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU encoder for sequential data.

    Treats 28x28 images as 28-step sequences of dimension 28.
    GRU -> final hidden state -> Linear head -> theta.
    """

    def __init__(self, input_dim: int, theta_dim: int,
                 seq_len: int = 28, seq_dim: int = 28,
                 hidden_size: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, theta_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 784) -> (batch, 28, 28)
        x = x.view(-1, self.seq_len, self.seq_dim)
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden)
        h = h_n.squeeze(0)    # (batch, hidden)
        return self.head(h)   # (batch, theta_dim)


class CNNEncoder(nn.Module):
    """CNN encoder for image data.

    Conv2d -> Pool -> Conv2d -> Pool -> Flatten -> Linear head -> theta.
    Designed for 28x28 grayscale images.
    """

    def __init__(self, input_dim: int, theta_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # -> (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),       # -> (32, 7, 7)
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, theta_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 784) -> (batch, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.head(features)


def gru_factory(input_dim: int, theta_dim: int) -> nn.Module:
    """Factory function for GRU encoder network."""
    return GRUEncoder(input_dim, theta_dim)


def cnn_factory(input_dim: int, theta_dim: int) -> nn.Module:
    """Factory function for CNN encoder network."""
    return CNNEncoder(input_dim, theta_dim)
