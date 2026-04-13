"""Tests for the GATv2 airspace graph attention model."""

from __future__ import annotations

import torch
import pytest

from aeroconform.config import AeroConformConfig
from aeroconform.models.graph_attention import AirspaceGATv2


class TestAirspaceGATv2:
    """Tests for AirspaceGATv2."""

    @pytest.fixture
    def model(self) -> AirspaceGATv2:
        """Create a small GATv2 for testing."""
        return AirspaceGATv2(
            in_channels=64,
            hidden_channels=32,
            out_channels=64,
            edge_dim=4,
            heads=2,
            n_layers=2,
            dropout=0.0,
        )

    def test_output_shape(self, model: AirspaceGATv2) -> None:
        """Output should have shape (N, out_channels)."""
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        out, attn = model(x, edge_index, edge_attr)
        assert out.shape == (10, 64)

    def test_variable_size_graphs(self, model: AirspaceGATv2) -> None:
        """Should handle graphs of different sizes."""
        for n_nodes in [3, 10, 50, 100]:
            x = torch.randn(n_nodes, 64)
            # Create a chain graph
            src = list(range(n_nodes - 1))
            dst = list(range(1, n_nodes))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            edge_attr = torch.randn(edge_index.shape[1], 4)
            out, _ = model(x, edge_index, edge_attr)
            assert out.shape == (n_nodes, 64)

    def test_single_node_no_edges(self, model: AirspaceGATv2) -> None:
        """Should handle a single-node graph with no edges."""
        x = torch.randn(1, 64)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4))
        out, _ = model(x, edge_index, edge_attr)
        assert out.shape == (1, 64)

    def test_attention_weights_returned(self, model: AirspaceGATv2) -> None:
        """Should return attention weights for each layer."""
        x = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, 4)
        _, attn_weights = model(x, edge_index, edge_attr)
        assert len(attn_weights) == 2  # 2 layers

    def test_gradient_flow(self, model: AirspaceGATv2) -> None:
        """Gradients should flow through the model."""
        x = torch.randn(5, 64, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        out, _ = model(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_edge_features_used(self, model: AirspaceGATv2) -> None:
        """Different edge features should produce different outputs."""
        model.eval()
        x = torch.randn(3, 64)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        edge_attr1 = torch.ones(2, 4)
        edge_attr2 = torch.ones(2, 4) * 10

        with torch.no_grad():
            out1, _ = model(x, edge_index, edge_attr1)
            out2, _ = model(x, edge_index, edge_attr2)

        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_residual_connection(self) -> None:
        """Residual connection should preserve input information."""
        model = AirspaceGATv2(
            in_channels=64, hidden_channels=32, out_channels=64,
            edge_dim=4, heads=2, n_layers=2, dropout=0.0,
        )
        model.eval()
        x = torch.randn(5, 64)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4))
        with torch.no_grad():
            out, _ = model(x, edge_index, edge_attr)
        # With no edges, output should still be meaningful (not zero)
        assert out.abs().mean() > 0.01

    def test_from_config(self) -> None:
        """Should create correctly from config."""
        config = AeroConformConfig(d_model=64, graph_hidden=32, graph_heads=2, graph_layers=2)
        model = AirspaceGATv2.from_config(config)
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.randn(2, config.edge_dim)
        out, _ = model(x, edge_index, edge_attr)
        assert out.shape == (10, 64)

    def test_fully_connected_graph(self, model: AirspaceGATv2) -> None:
        """Should handle a fully connected graph."""
        n = 8
        x = torch.randn(n, 64)
        # All pairs connected
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.randn(len(src), 4)
        out, _ = model(x, edge_index, edge_attr)
        assert out.shape == (n, 64)
