"""AirGraph: GATv2 graph attention network for multi-aircraft interaction.

Per ARCHITECTURE.md section 3:
- 2 GATv2Conv layers, 4 heads, edge features R^5
- Residual connections, LayerNorm
- Output: per-aircraft context embedding R^64
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from src.utils.config import GraphConfig


class AirGraph(nn.Module):
    """GATv2 network for multi-aircraft interaction modeling."""

    def __init__(self, config: GraphConfig, input_dim: int = 264) -> None:
        super().__init__()
        self.config = config

        # First GATv2 layer
        self.conv1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=config.hidden_dim // config.num_heads,
            heads=config.num_heads,
            edge_dim=config.edge_dim,
            dropout=config.dropout,
            concat=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)

        # Second GATv2 layer
        self.conv2 = GATv2Conv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim // config.num_heads,
            heads=config.num_heads,
            edge_dim=config.edge_dim,
            dropout=config.dropout,
            concat=True,
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Residual projection if dimensions differ
        if input_dim != config.hidden_dim:
            self.residual_proj = nn.Linear(input_dim, config.hidden_dim)
        else:
            self.residual_proj = nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim)

        # Prediction head for graph-enriched predictions
        self.pred_head = nn.Linear(config.output_dim, 16)  # 8 means + 8 log_vars

    @classmethod
    def from_config(cls, config: GraphConfig, input_dim: int = 264) -> AirGraph:
        """Create from config."""
        return cls(config, input_dim)

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through GATv2 layers.

        Args:
            data: PyG Data with x, edge_index, edge_attr.
            return_attention: If True, return attention weights from last layer.

        Returns:
            context: (num_nodes, output_dim) graph-enriched embeddings.
            attention: Optional (num_edges, num_heads) attention weights.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Handle empty graphs
        if x.size(0) == 0:
            return (
                torch.zeros(0, self.config.output_dim, device=x.device),
                None,
            )

        # Layer 1
        residual = self.residual_proj(x)
        h = self.conv1(x, edge_index, edge_attr)
        h = torch.nn.functional.elu(h)
        h = self.norm1(h + residual)

        # Layer 2
        residual = h
        if return_attention:
            h, (edge_idx, attention) = self.conv2(
                h, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            h = self.conv2(h, edge_index, edge_attr)
            attention = None
        h = torch.nn.functional.elu(h)
        h = self.norm2(h + residual)

        # Output projection
        context = self.output_norm(self.output_proj(h))

        return context, attention

    def predict(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from graph context.

        Args:
            context: (num_nodes, output_dim) graph-enriched embeddings.

        Returns:
            means: (num_nodes, 8) predicted means.
            log_vars: (num_nodes, 8) predicted log-variances.
        """
        out = self.pred_head(context)
        means = out[:, :8]
        log_vars = torch.clamp(out[:, 8:], -10.0, 10.0)
        return means, log_vars
