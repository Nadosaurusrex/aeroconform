"""GATv2 graph attention network for multi-aircraft interaction modeling.

Models dynamic airspace interactions using attention over aircraft pairs
connected by proximity and altitude overlap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as f_torch
from torch_geometric.nn import GATv2Conv

from aeroconform.config import AeroConformConfig


class AirspaceGATv2(nn.Module):
    """Graph attention network for multi-aircraft interaction modeling.

    Takes trajectory embeddings from the foundation model as node features
    and learns which aircraft interactions are operationally significant
    via attention over edge-connected aircraft pairs.

    Architecture: multi-layer GATv2 with residual connections and LayerNorm.
    First layer: concatenated multi-head output.
    Last layer: single-head mean aggregation to match d_model.

    Args:
        in_channels: Input node feature dimension (d_model from foundation model).
        hidden_channels: Hidden dimension per attention head.
        out_channels: Output dimension (same as d_model for residual).
        edge_dim: Edge feature dimension (distance, closing speed, alt diff, bearing).
        heads: Number of attention heads.
        n_layers: Number of GATv2 layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 256,
        edge_dim: int = 4,
        heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if n_layers == 1:
            # Single layer: direct in -> out
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    out_channels,
                    heads=1,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.norms.append(nn.LayerNorm(out_channels))
        else:
            # First layer: in -> hidden * heads (concatenated)
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

            # Hidden layers
            for _ in range(n_layers - 2):
                self.convs.append(
                    GATv2Conv(
                        hidden_channels * heads,
                        hidden_channels,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.norms.append(nn.LayerNorm(hidden_channels * heads))

            # Final layer: hidden * heads -> out (mean over heads)
            self.convs.append(
                GATv2Conv(
                    hidden_channels * heads,
                    out_channels,
                    heads=1,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.norms.append(nn.LayerNorm(out_channels))

        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection when dimensions don't match
        self.residual_proj: nn.Linear | None = None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run forward pass through the GATv2 layers.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Graph connectivity in COO format, shape (2, E).
            edge_attr: Edge features of shape (E, edge_dim).

        Returns:
            Tuple of:
            - x: Context-enriched node embeddings of shape (N, out_channels).
            - attention_weights: List of (edge_index, attention) tuples per layer.
        """
        x_input = x
        attention_weights: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=False)):
            x_out, alpha = conv(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(alpha)
            x_out = norm(x_out)

            if i < len(self.convs) - 1:
                x_out = f_torch.elu(x_out)
                x_out = self.dropout(x_out)

            x = x_out

        # Residual connection from input to output
        if self.residual_proj is not None:
            x = x + self.residual_proj(x_input)
        elif x_input.shape[-1] == x.shape[-1]:
            x = x + x_input

        return x, attention_weights

    @classmethod
    def from_config(cls, config: AeroConformConfig) -> AirspaceGATv2:
        """Create an AirspaceGATv2 from an AeroConformConfig.

        Args:
            config: AeroConform configuration.

        Returns:
            Configured AirspaceGATv2 instance.
        """
        return cls(
            in_channels=config.d_model,
            hidden_channels=config.graph_hidden,
            out_channels=config.d_model,
            edge_dim=config.edge_dim,
            heads=config.graph_heads,
            n_layers=config.graph_layers,
            dropout=config.dropout,
        )
