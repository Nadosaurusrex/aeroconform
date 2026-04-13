"""Patch-based tokenizer for continuous trajectory data.

Converts continuous delta-encoded state vectors into patch embeddings,
analogous to ViT patches for images or TimesFM patches for time series.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aeroconform.config import AeroConformConfig


class PatchTokenizer(nn.Module):
    """Convert continuous delta-encoded state vectors into patch embeddings.

    Groups P consecutive timesteps into a patch, then linearly projects
    the flattened patch into the model embedding dimension.

    Args:
        input_dim: Number of features per timestep (default: 6).
        patch_len: Number of timesteps per patch (default: 8).
        d_model: Embedding dimension (default: 256).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int = 6,
        patch_len: int = 8,
        d_model: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.d_model = d_model

        self.projection = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input sequence to patch embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
                seq_len must be divisible by patch_len.

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model).

        Raises:
            AssertionError: If seq_len is not divisible by patch_len.
        """
        b, t, d = x.shape
        assert t % self.patch_len == 0, (
            f"seq_len {t} must be divisible by patch_len {self.patch_len}"
        )
        num_patches = t // self.patch_len

        # Reshape: (B, num_patches, patch_len * D)
        x = x.reshape(b, num_patches, self.patch_len * d)
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    @classmethod
    def from_config(cls, config: AeroConformConfig) -> PatchTokenizer:
        """Create a PatchTokenizer from an AeroConformConfig.

        Args:
            config: AeroConform configuration.

        Returns:
            Configured PatchTokenizer instance.
        """
        return cls(
            input_dim=config.input_dim,
            patch_len=config.patch_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )
