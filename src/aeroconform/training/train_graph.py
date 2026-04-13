"""Graph attention layer training.

Trains the GATv2 layer to improve next-state prediction using
multi-aircraft context while keeping the foundation model frozen.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader

from aeroconform.config import AeroConformConfig
from aeroconform.models.gaussian_head import GaussianMixtureHead
from aeroconform.models.graph_attention import AirspaceGATv2
from aeroconform.models.trajectory_model import TrajectoryTransformer
from aeroconform.training.utils import (
    clip_gradients,
    log_metrics,
    save_checkpoint,
)

logger = structlog.get_logger(__name__)


def train_graph(
    foundation_model: TrajectoryTransformer,
    graph_model: AirspaceGATv2,
    dataset: DataLoader,
    config: AeroConformConfig,
    device: str = "cpu",
) -> AirspaceGATv2:
    """Train the graph attention layer.

    Freezes the foundation model and trains the GATv2 layer using
    the same NLL loss but with graph-enhanced embeddings.

    Args:
        foundation_model: Pre-trained foundation model (frozen).
        graph_model: GATv2 model to train.
        dataset: DataLoader yielding airspace snapshot batches.
        config: Training configuration.
        device: Device to train on.

    Returns:
        The trained graph model.
    """
    foundation_model = foundation_model.to(device)
    graph_model = graph_model.to(device)

    # Freeze foundation model
    foundation_model.eval()
    if config.freeze_foundation:
        for param in foundation_model.parameters():
            param.requires_grad = False

    graph_model.train()

    # Create a projection head for graph-enhanced predictions
    projection_head = GaussianMixtureHead(
        d_model=config.d_model,
        output_dim=config.output_dim,
        n_components=config.n_components,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(graph_model.parameters()) + list(projection_head.parameters()),
        lr=config.graph_lr,
        weight_decay=config.graph_weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir)

    for epoch in range(config.graph_epochs):
        graph_model.train()
        projection_head.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataset:
            node_features = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            targets = batch["targets"].to(device)

            # Run graph model
            enhanced, _ = graph_model(node_features, edge_index, edge_attr)

            # Predict using enhanced embeddings
            enhanced_3d = enhanced.unsqueeze(0)  # (1, N, d_model)
            means, log_vars, log_weights = projection_head(enhanced_3d)

            # Compute loss
            targets_3d = targets.unsqueeze(0)  # (1, N, output_dim)
            loss = projection_head.nll_loss(means, log_vars, log_weights, targets_3d)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(graph_model, config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        log_metrics("graph", epoch, {"train_loss": avg_loss})

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                graph_model, optimizer, epoch,
                checkpoint_dir / f"graph_epoch_{epoch}.pt",
                train_loss=avg_loss,
            )

    save_checkpoint(
        graph_model, optimizer, config.graph_epochs - 1,
        checkpoint_dir / "best_graph.pt",
    )

    return graph_model
