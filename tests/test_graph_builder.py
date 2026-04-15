"""Tests for graph construction."""

from __future__ import annotations

import numpy as np
import torch
from src.models.graph_builder import AirspaceGraphBuilder


def test_empty_graph() -> None:
    """Empty input produces empty graph."""
    builder = AirspaceGraphBuilder()
    states = np.zeros((0, 8), dtype=np.float32)
    embeddings = torch.zeros(0, 256)

    graph = builder.build_graph(states, embeddings)
    assert graph.x.shape[0] == 0
    assert graph.edge_index.shape[1] == 0


def test_single_aircraft() -> None:
    """Single aircraft produces graph with no edges."""
    builder = AirspaceGraphBuilder()
    states = np.array([[45.0, 9.0, 10000, 250, 0.7, 0.7, 0, 0]], dtype=np.float32)
    embeddings = torch.randn(1, 256)

    graph = builder.build_graph(states, embeddings)
    assert graph.x.shape == (1, 264)
    assert graph.edge_index.shape[1] == 0


def test_close_aircraft_connected() -> None:
    """Aircraft within proximity threshold are connected."""
    builder = AirspaceGraphBuilder()
    # Two aircraft very close together
    states = np.array(
        [
            [45.0, 9.0, 10000, 250, 0.7, 0.7, 0, 0],
            [45.01, 9.01, 10000, 250, 0.7, 0.7, 0, 0],  # ~1.5 km away
        ],
        dtype=np.float32,
    )
    embeddings = torch.randn(2, 256)

    graph = builder.build_graph(states, embeddings)
    assert graph.edge_index.shape[1] > 0
    assert graph.edge_attr.shape[1] == 5


def test_far_aircraft_not_connected() -> None:
    """Aircraft far apart are not connected."""
    builder = AirspaceGraphBuilder()
    # Two aircraft very far apart (>50 NM)
    states = np.array(
        [
            [45.0, 9.0, 10000, 250, 0.7, 0.7, 0, 0],
            [47.0, 12.0, 5000, 200, 0.7, 0.7, 0, 0],  # ~300 km away, different altitude
        ],
        dtype=np.float32,
    )
    embeddings = torch.randn(2, 256)

    graph = builder.build_graph(states, embeddings)
    assert graph.edge_index.shape[1] == 0


def test_edge_features_dimensions() -> None:
    """Edge features have 5 dimensions."""
    builder = AirspaceGraphBuilder()
    states = np.array(
        [
            [45.0, 9.0, 10000, 250, 0.7, 0.7, 0, 0],
            [45.05, 9.05, 10000, 260, 0.5, 0.86, 0, 0],
            [45.1, 9.0, 10100, 240, 0.7, 0.7, 0, 0],
        ],
        dtype=np.float32,
    )
    embeddings = torch.randn(3, 256)

    graph = builder.build_graph(states, embeddings)
    if graph.edge_index.shape[1] > 0:
        assert graph.edge_attr.shape[1] == 5
