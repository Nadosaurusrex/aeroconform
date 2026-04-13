"""Causal transformer foundation model for trajectory prediction.

Pre-trained on ADS-B state vectors via self-supervised next-patch
prediction with a Gaussian mixture output head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from aeroconform.config import AeroConformConfig
from aeroconform.models.gaussian_head import GaussianMixtureHead
from aeroconform.models.tokenizer import PatchTokenizer


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for length generalization.

    Uses fixed (not learned) sinusoidal patterns so the model can
    generalize to sequence lengths not seen during training.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encoding for the given sequence length.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Positional encoding of shape (1, seq_len, d_model).
        """
        return self.pe[:, : x.size(1), :]  # type: ignore[index]


class TrajectoryTransformer(nn.Module):
    """Causal transformer for next-patch prediction on ADS-B trajectories.

    Architecture: patch tokenization -> sinusoidal PE -> causal transformer
    decoder -> Gaussian mixture output head. Pre-norm transformer layers
    with GELU activation for training stability.

    Args:
        d_model: Model embedding dimension (default: 256).
        n_heads: Number of attention heads (default: 8).
        n_layers: Number of transformer layers (default: 6).
        d_ff: Feed-forward inner dimension (default: 1024).
        dropout: Dropout rate (default: 0.1).
        max_patches: Maximum number of patches (default: 64).
        input_dim: Features per timestep (default: 6).
        patch_len: Timesteps per patch (default: 8).
        n_components: Gaussian mixture components (default: 5).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_patches: int = 64,
        input_dim: int = 6,
        patch_len: int = 8,
        n_components: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.patch_len = patch_len

        self.tokenizer = PatchTokenizer(input_dim, patch_len, d_model, dropout)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_patches)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = GaussianMixtureHead(
            d_model, input_dim * patch_len, n_components
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the full model.

        Args:
            x: Raw delta-encoded states of shape (batch, seq_len, input_dim).
            mask: Optional validity mask of shape (batch, seq_len).

        Returns:
            Tuple of:
            - means: (batch, num_patches, n_components, output_dim)
            - log_vars: (batch, num_patches, n_components, output_dim)
            - log_weights: (batch, num_patches, n_components)
            - embeddings: (batch, num_patches, d_model) hidden states
        """
        patches = self.tokenizer(x)  # (B, P, d_model)
        patches = patches + self.pos_encoding(patches)

        # Causal mask: each patch can only attend to itself and previous patches
        seq_len = patches.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=patches.device
        )

        # Decoder-only: pass same tensor as both memory and target
        hidden = self.transformer(
            tgt=patches,
            memory=patches,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )  # (B, P, d_model)

        means, log_vars, log_weights = self.output_head(hidden)
        return means, log_vars, log_weights, hidden

    def get_trajectory_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the last hidden state as the trajectory embedding.

        Used as node features for the graph attention layer.

        Args:
            x: Raw delta-encoded states of shape (batch, seq_len, input_dim).

        Returns:
            Trajectory embeddings of shape (batch, d_model).
        """
        _, _, _, hidden = self.forward(x)
        return hidden[:, -1, :]  # (B, d_model) — last patch embedding

    @classmethod
    def from_config(cls, config: AeroConformConfig) -> TrajectoryTransformer:
        """Create a TrajectoryTransformer from an AeroConformConfig.

        Args:
            config: AeroConform configuration.

        Returns:
            Configured TrajectoryTransformer instance.
        """
        return cls(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_patches=config.max_patches,
            input_dim=config.input_dim,
            patch_len=config.patch_len,
            n_components=config.n_components,
        )
