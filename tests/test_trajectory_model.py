"""Tests for the trajectory transformer foundation model."""

from __future__ import annotations

import torch
import pytest

from aeroconform.config import AeroConformConfig
from aeroconform.models.trajectory_model import (
    SinusoidalPositionalEncoding,
    TrajectoryTransformer,
)


class TestSinusoidalPE:
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self) -> None:
        """PE should have correct shape."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=64)
        x = torch.randn(4, 16, 256)
        out = pe(x)
        assert out.shape == (1, 16, 256)

    def test_deterministic(self) -> None:
        """PE should be deterministic (not learned)."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=64)
        x = torch.randn(4, 16, 256)
        out1 = pe(x)
        out2 = pe(x)
        torch.testing.assert_close(out1, out2)

    def test_different_positions_different_encodings(self) -> None:
        """Different positions should have different encodings."""
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=64)
        x = torch.randn(1, 64, 256)
        out = pe(x)
        # Positions 0 and 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestTrajectoryTransformer:
    """Tests for the TrajectoryTransformer model."""

    @pytest.fixture
    def model(self) -> TrajectoryTransformer:
        """Create a small model for testing."""
        return TrajectoryTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            max_patches=64,
            input_dim=6,
            patch_len=8,
            n_components=3,
        )

    def test_forward_shapes(self, model: TrajectoryTransformer) -> None:
        """Forward pass should produce correct output shapes."""
        x = torch.randn(4, 128, 6)
        means, log_vars, log_weights, hidden = model(x)
        assert means.shape == (4, 16, 3, 48)
        assert log_vars.shape == (4, 16, 3, 48)
        assert log_weights.shape == (4, 16, 3)
        assert hidden.shape == (4, 16, 64)

    def test_full_size_model_shapes(self) -> None:
        """Full-size model with spec dimensions should work."""
        model = TrajectoryTransformer(
            d_model=256, n_heads=8, n_layers=6, d_ff=1024,
            input_dim=6, patch_len=8, n_components=5,
        )
        x = torch.randn(32, 128, 6)
        means, log_vars, log_weights, hidden = model(x)
        assert means.shape == (32, 16, 5, 48)
        assert log_vars.shape == (32, 16, 5, 48)
        assert log_weights.shape == (32, 16, 5)
        assert hidden.shape == (32, 16, 256)

    def test_trajectory_embedding(self, model: TrajectoryTransformer) -> None:
        """get_trajectory_embedding should return (B, d_model)."""
        x = torch.randn(8, 128, 6)
        emb = model.get_trajectory_embedding(x)
        assert emb.shape == (8, 64)

    def test_gradient_flow(self, model: TrajectoryTransformer) -> None:
        """Gradients should flow through the entire model."""
        x = torch.randn(2, 128, 6, requires_grad=True)
        means, log_vars, log_weights, hidden = model(x)
        targets = torch.randn(2, 16, 48)
        loss = model.output_head.nll_loss(means, log_vars, log_weights, targets)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_causal_masking(self, model: TrajectoryTransformer) -> None:
        """Causal masking: changing future patches should not affect past predictions.

        Modifying patch k+1 should not change the output at patch k.
        """
        model.eval()
        x1 = torch.randn(1, 128, 6)
        x2 = x1.clone()
        # Modify the last patch (timesteps 120-127)
        x2[0, 120:128, :] = torch.randn(8, 6)

        with torch.no_grad():
            _, _, _, h1 = model(x1)
            _, _, _, h2 = model(x2)

        # Outputs for patches before the last should be identical
        # (patches 0 through 13 cover timesteps 0-111, unaffected)
        torch.testing.assert_close(h1[:, :14, :], h2[:, :14, :], atol=1e-5, rtol=1e-5)

    def test_from_config(self) -> None:
        """Should create correctly from config."""
        config = AeroConformConfig()
        model = TrajectoryTransformer.from_config(config)
        x = torch.randn(2, config.seq_len, config.input_dim)
        means, log_vars, log_weights, hidden = model(x)
        assert means.shape[0] == 2
        assert hidden.shape[-1] == config.d_model

    def test_parameter_count(self) -> None:
        """Full model should have approximately 5M parameters."""
        model = TrajectoryTransformer(
            d_model=256, n_heads=8, n_layers=6, d_ff=1024,
            input_dim=6, patch_len=8, n_components=5,
        )
        param_count = sum(p.numel() for p in model.parameters())
        # Should be roughly 5M (allow range 3M-8M)
        assert 3_000_000 < param_count < 8_000_000

    def test_batch_size_one(self, model: TrajectoryTransformer) -> None:
        """Model should work with batch size 1."""
        x = torch.randn(1, 128, 6)
        means, log_vars, log_weights, hidden = model(x)
        assert means.shape[0] == 1

    def test_deterministic_eval(self, model: TrajectoryTransformer) -> None:
        """In eval mode, outputs should be deterministic."""
        model.eval()
        x = torch.randn(2, 128, 6)
        with torch.no_grad():
            _, _, _, h1 = model(x)
            _, _, _, h2 = model(x)
        torch.testing.assert_close(h1, h2)
