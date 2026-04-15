"""Tests for loss functions."""

from __future__ import annotations

import torch
from src.models.losses import gaussian_nll_loss


def test_nll_loss_positive() -> None:
    """NLL loss is positive for random inputs."""
    means = torch.randn(4, 16, 8)
    log_vars = torch.zeros(4, 16, 8)
    targets = torch.randn(4, 16, 8)

    loss = gaussian_nll_loss(means, log_vars, targets)
    assert loss.item() > 0


def test_nll_loss_zero_variance() -> None:
    """With zero log_var and perfect prediction, loss is just the constant term."""
    targets = torch.randn(4, 16, 8)
    means = targets.clone()
    log_vars = torch.zeros(4, 16, 8)

    loss = gaussian_nll_loss(means, log_vars, targets)
    # Should be 0 (all terms cancel: log_var=0, residual=0)
    assert loss.item() < 0.01


def test_nll_loss_with_mask() -> None:
    """Masked positions don't contribute to loss."""
    means = torch.randn(2, 8, 8)
    log_vars = torch.zeros(2, 8, 8)
    targets = torch.randn(2, 8, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)

    # All masked out -> loss should be 0
    loss = gaussian_nll_loss(means, log_vars, targets, mask)
    assert loss.item() == 0.0


def test_nll_loss_higher_with_worse_predictions() -> None:
    """Loss increases when predictions are worse."""
    targets = torch.randn(4, 16, 8)
    log_vars = torch.zeros(4, 16, 8)

    # Good predictions
    good_means = targets + torch.randn_like(targets) * 0.1
    good_loss = gaussian_nll_loss(good_means, log_vars, targets)

    # Bad predictions
    bad_means = targets + torch.randn_like(targets) * 10.0
    bad_loss = gaussian_nll_loss(bad_means, log_vars, targets)

    assert bad_loss.item() > good_loss.item()


def test_nll_loss_gradient_flow() -> None:
    """Loss gradients flow to means and log_vars."""
    means = torch.randn(2, 8, 8, requires_grad=True)
    log_vars = torch.randn(2, 8, 8, requires_grad=True)
    targets = torch.randn(2, 8, 8)

    loss = gaussian_nll_loss(means, log_vars, targets)
    loss.backward()

    assert means.grad is not None
    assert log_vars.grad is not None
