"""Input embeddings for AeroGPT.

StateEmbedding: Linear projection R^8 -> R^hidden_dim per timestep.
TimeEncoding: Sinusoidal encoding using actual elapsed seconds for irregular ADS-B spacing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StateEmbedding(nn.Module):
    """Project 8-dim state vectors to hidden dimension.

    Per-timestep linear projection, not patch-based.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project state vectors to hidden space.

        Args:
            x: (batch, seq_len, input_dim) state vectors.

        Returns:
            (batch, seq_len, hidden_dim) embedded states.
        """
        return self.dropout(self.norm(self.projection(x)))


class TimeEncoding(nn.Module):
    """Sinusoidal encoding using actual elapsed seconds.

    Unlike standard positional encoding that assumes uniform spacing,
    this uses the actual time gap between observations. Critical for
    ADS-B data where observations are irregularly spaced (1s to 60s).

    Formula from ARCHITECTURE.md:
        time_enc(t) = [sin(t / 10^(2i/d)), cos(t / 10^(2i/d))] for i in 0..d/2
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Precompute frequency basis: 10^(2i/d) for i in 0..d/2
        half_dim = hidden_dim // 2
        freqs = torch.pow(10.0, 2.0 * torch.arange(half_dim).float() / hidden_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, elapsed_seconds: torch.Tensor) -> torch.Tensor:
        """Compute time encoding from elapsed seconds.

        Args:
            elapsed_seconds: (batch, seq_len) actual elapsed seconds since
                             start of sequence.

        Returns:
            (batch, seq_len, hidden_dim) time encoding.
        """
        # elapsed_seconds: (batch, seq_len) -> (batch, seq_len, 1)
        t = elapsed_seconds.unsqueeze(-1)
        # freqs: (half_dim,) -> (1, 1, half_dim)
        freqs = self.freqs.unsqueeze(0).unsqueeze(0)
        # Compute sin/cos: (batch, seq_len, half_dim)
        angles = t / freqs
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        # Interleave: (batch, seq_len, hidden_dim)
        return torch.cat([sin_enc, cos_enc], dim=-1)
