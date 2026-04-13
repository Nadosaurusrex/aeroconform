"""Tests for the Gaussian mixture output head."""

from __future__ import annotations

import math

import torch
import pytest

from aeroconform.config import AeroConformConfig
from aeroconform.models.gaussian_head import GaussianMixtureHead


class TestGaussianMixtureHead:
    """Tests for GaussianMixtureHead."""

    @pytest.fixture
    def head(self) -> GaussianMixtureHead:
        """Create a default Gaussian mixture head."""
        return GaussianMixtureHead(d_model=256, output_dim=48, n_components=5)

    def test_output_shapes(self, head: GaussianMixtureHead) -> None:
        """Output shapes should be correct."""
        hidden = torch.randn(32, 16, 256)
        means, log_vars, log_weights = head(hidden)
        assert means.shape == (32, 16, 5, 48)
        assert log_vars.shape == (32, 16, 5, 48)
        assert log_weights.shape == (32, 16, 5)

    def test_log_weights_sum_to_one(self, head: GaussianMixtureHead) -> None:
        """Mixture weights should sum to 1 (in probability space)."""
        hidden = torch.randn(4, 16, 256)
        _, _, log_weights = head(hidden)
        weights = torch.exp(log_weights)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_log_vars_clamped(self, head: GaussianMixtureHead) -> None:
        """Log-variances should be clamped to [-10, 2]."""
        hidden = torch.randn(4, 16, 256) * 100  # Large inputs
        _, log_vars, _ = head(hidden)
        assert log_vars.min() >= -10
        assert log_vars.max() <= 2

    def test_nll_loss_finite(self, head: GaussianMixtureHead) -> None:
        """NLL loss should be finite and positive."""
        hidden = torch.randn(4, 16, 256)
        means, log_vars, log_weights = head(hidden)
        targets = torch.randn(4, 16, 48)
        loss = head.nll_loss(means, log_vars, log_weights, targets)
        assert torch.isfinite(loss)

    def test_nll_loss_gradient(self, head: GaussianMixtureHead) -> None:
        """Gradients should flow through the NLL loss."""
        hidden = torch.randn(4, 16, 256, requires_grad=True)
        means, log_vars, log_weights = head(hidden)
        targets = torch.randn(4, 16, 48)
        loss = head.nll_loss(means, log_vars, log_weights, targets)
        loss.backward()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()

    def test_nll_loss_decreases_for_better_predictions(self, head: GaussianMixtureHead) -> None:
        """NLL loss should be lower when predictions are closer to targets."""
        hidden = torch.randn(4, 16, 256)
        means, log_vars, log_weights = head(hidden)
        # Loss with random targets
        random_targets = torch.randn(4, 16, 48) * 10
        loss_random = head.nll_loss(means, log_vars, log_weights, random_targets)
        # Loss with targets close to mean of first component
        close_targets = means[:, :, 0, :].detach() + torch.randn(4, 16, 48) * 0.01
        loss_close = head.nll_loss(means, log_vars, log_weights, close_targets)
        assert loss_close < loss_random

    def test_sample_shape(self, head: GaussianMixtureHead) -> None:
        """Samples should have correct shape."""
        hidden = torch.randn(4, 16, 256)
        means, log_vars, log_weights = head(hidden)
        samples = head.sample(means, log_vars, log_weights)
        assert samples.shape == (4, 16, 48)

    def test_from_config(self) -> None:
        """Should create correctly from config."""
        config = AeroConformConfig()
        head = GaussianMixtureHead.from_config(config)
        hidden = torch.randn(4, config.num_patches, config.d_model)
        means, log_vars, log_weights = head(hidden)
        assert means.shape == (4, config.num_patches, config.n_components, config.output_dim)
