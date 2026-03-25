"""Convolutional autoencoder and latent-space RNN (matches original notebook architecture)."""

from __future__ import annotations

import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class ConvAE(nn.Module):
    """Encodes 3×64×64 RGB frames to ``dimz`` floats and decodes back to RGB."""

    def __init__(self, dimz: int = 50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1),
            nn.MaxPool2d(3, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 24, 5, 1),
            nn.ReLU(True),
            nn.Conv2d(24, 32, 5, 1, padding=2),
            nn.MaxPool2d(3, 2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4608, dimz),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimz, 4608),
            nn.ReLU(True),
            Reshape(-1, 32, 12, 12),
            nn.ConvTranspose2d(32, 24, 5, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 12, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 3, 3, stride=2, output_padding=1),
        )
        self.dimz = dimz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class SimpleRNN(nn.Module):
    """One-step latent predictor: tanh(W_x z + W_h h) -> next latent."""

    def __init__(self, latent_dim: int = 50, nhid: int = 200):
        super().__init__()
        self.nhid = nhid
        self.latent_dim = latent_dim
        self.ff = nn.Linear(latent_dim, nhid)
        self.rec = nn.Linear(nhid, nhid)
        self.out = nn.Linear(nhid, latent_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.ff(x) + self.rec(h)
        ht = torch.tanh(a)
        return self.out(ht), ht

    def init_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.nhid, device=device)
