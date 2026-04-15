"""Combined AeroConform model: AeroGPT + AirGraph + prediction.

Chains foundation model -> graph attention -> prediction head.
Joint loss = foundation NLL + 0.1 * graph NLL.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from src.models.aerogpt import AeroGPT
from src.models.airgraph import AirGraph
from src.models.graph_builder import AirspaceGraphBuilder
from src.models.losses import gaussian_nll_loss
from src.utils.config import GraphConfig, ModelConfig


class AeroConformModel(nn.Module):
    """Combined model chaining AeroGPT -> AirGraph -> predictions."""

    def __init__(
        self,
        model_config: ModelConfig,
        graph_config: GraphConfig,
    ) -> None:
        super().__init__()
        self.aerogpt = AeroGPT(model_config)
        self.airgraph = AirGraph(
            graph_config,
            input_dim=model_config.hidden_dim + model_config.input_dim,
        )
        self.graph_builder = AirspaceGraphBuilder()
        self.joint_loss_weight = graph_config.joint_loss_weight

    def forward_foundation(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run foundation model only.

        Returns:
            means, log_vars, hidden from AeroGPT.
        """
        return self.aerogpt(x, time_gaps, mask)

    def forward_graph(
        self,
        states: npt.NDArray[np.float32],
        embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build graph and run AirGraph.

        Args:
            states: (num_aircraft, 8) current kinematic states.
            embeddings: (num_aircraft, hidden_dim) trajectory embeddings.
            return_attention: Return attention weights.

        Returns:
            context: (num_aircraft, output_dim) graph-enriched embeddings.
            attention: Optional attention weights.
        """
        graph = self.graph_builder.build_graph(states, embeddings)
        graph = graph.to(embeddings.device)
        return self.airgraph(graph, return_attention)

    def compute_joint_loss(
        self,
        foundation_means: torch.Tensor,
        foundation_log_vars: torch.Tensor,
        graph_means: torch.Tensor,
        graph_log_vars: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint loss = foundation NLL + weight * graph NLL.

        Args:
            foundation_means, foundation_log_vars: From AeroGPT.
            graph_means, graph_log_vars: From AirGraph prediction head.
            targets: Ground truth deltas.
            mask: Validity mask.

        Returns:
            total_loss, foundation_loss, graph_loss.
        """
        foundation_loss = gaussian_nll_loss(
            foundation_means, foundation_log_vars, targets, mask
        )
        graph_loss = gaussian_nll_loss(
            graph_means, graph_log_vars, targets, mask
        )
        total_loss = foundation_loss + self.joint_loss_weight * graph_loss
        return total_loss, foundation_loss, graph_loss
