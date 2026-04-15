"""AeroGPT: Causal decoder-only transformer for trajectory prediction.

Architecture per ARCHITECTURE.md:
- 6 layers, 8 heads, 256 hidden dim, 1024 FFN dim
- Per-timestep StateEmbedding + TimeEncoding
- GaussianHead output (mean + log_var per feature)
- ~8M parameters
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.embeddings import StateEmbedding, TimeEncoding
from src.models.heads import GaussianHead
from src.utils.config import ModelConfig


class AeroGPT(nn.Module):
    """Causal decoder-only transformer for trajectory foundation model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Input embedding
        self.state_embedding = StateEmbedding(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.time_encoding = TimeEncoding(hidden_dim=config.hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        # Output head
        self.head = GaussianHead(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )

        # Learnable mask token for masked pre-training
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_config(cls, config: ModelConfig) -> AeroGPT:
        """Create model from config."""
        return cls(config)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            (seq_len, seq_len) boolean mask where True means "do not attend".
        """
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: (batch, seq_len, input_dim) input state vectors (delta-encoded, normalized).
            time_gaps: (batch, seq_len) elapsed seconds since start of sequence.
            mask: (batch, seq_len) boolean mask, True for valid positions.

        Returns:
            means: (batch, seq_len, num_features) predicted delta means.
            log_vars: (batch, seq_len, num_features) predicted log-variances.
            hidden: (batch, seq_len, hidden_dim) last hidden states.
        """
        batch_size, seq_len, _ = x.shape

        # Embed states and add time encoding
        embedded = self.state_embedding(x)  # (batch, seq_len, hidden_dim)
        time_enc = self.time_encoding(time_gaps)  # (batch, seq_len, hidden_dim)
        hidden = embedded + time_enc

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Key padding mask: True means "ignore this position"
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = padded = ignore

        # Transformer forward
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        # Output head
        means, log_vars = self.head(hidden)

        return means, log_vars, hidden

    def get_embedding(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract trajectory embedding for graph layer input.

        Returns the last valid hidden state per sequence.

        Args:
            x: (batch, seq_len, input_dim) input state vectors.
            time_gaps: (batch, seq_len) elapsed seconds.
            mask: (batch, seq_len) validity mask.

        Returns:
            (batch, hidden_dim) per-sequence trajectory embedding.
        """
        _, _, hidden = self.forward(x, time_gaps, mask)

        if mask is not None:
            # Get last valid position per sequence
            lengths = mask.sum(dim=1).long() - 1  # (batch,)
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, lengths]

        return hidden[:, -1]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
