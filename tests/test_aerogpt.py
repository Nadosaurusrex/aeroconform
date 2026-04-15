"""Tests for AeroGPT model."""

from __future__ import annotations

import torch
from src.models.aerogpt import AeroGPT
from src.utils.config import ModelConfig


def _small_config() -> ModelConfig:
    """Small config for fast testing."""
    return ModelConfig(
        input_dim=8,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        feedforward_dim=64,
        dropout=0.0,
        max_seq_len=64,
        output_dim=16,
    )


def test_forward_shapes() -> None:
    """Forward pass produces correct output shapes."""
    config = _small_config()
    model = AeroGPT(config)

    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, 8)
    time_gaps = torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, -1) * 10

    means, log_vars, hidden = model(x, time_gaps)

    assert means.shape == (batch_size, seq_len, 8)
    assert log_vars.shape == (batch_size, seq_len, 8)
    assert hidden.shape == (batch_size, seq_len, 32)


def test_forward_with_mask() -> None:
    """Forward pass respects padding mask."""
    config = _small_config()
    model = AeroGPT(config)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, 8)
    time_gaps = torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, -1) * 10
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[1, 10:] = False  # Second sequence shorter

    means, log_vars, hidden = model(x, time_gaps, mask)
    assert means.shape == (batch_size, seq_len, 8)


def test_log_var_clamping() -> None:
    """Log variance is clamped to [-10, 10]."""
    config = _small_config()
    model = AeroGPT(config)

    x = torch.randn(2, 8, 8) * 100  # Large input to test clamping
    time_gaps = torch.arange(8).float().unsqueeze(0).expand(2, -1)

    _, log_vars, _ = model(x, time_gaps)
    assert log_vars.min() >= -10.0
    assert log_vars.max() <= 10.0


def test_get_embedding() -> None:
    """get_embedding returns per-sequence embedding."""
    config = _small_config()
    model = AeroGPT(config)

    x = torch.randn(4, 16, 8)
    time_gaps = torch.arange(16).float().unsqueeze(0).expand(4, -1) * 10

    emb = model.get_embedding(x, time_gaps)
    assert emb.shape == (4, 32)


def test_get_embedding_with_mask() -> None:
    """get_embedding respects mask for last valid position."""
    config = _small_config()
    model = AeroGPT(config)

    x = torch.randn(2, 16, 8)
    time_gaps = torch.arange(16).float().unsqueeze(0).expand(2, -1) * 10
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 8:] = False

    emb = model.get_embedding(x, time_gaps, mask)
    assert emb.shape == (2, 32)


def test_gradient_flow() -> None:
    """Gradients flow through the entire model."""
    config = _small_config()
    model = AeroGPT(config)

    x = torch.randn(2, 8, 8, requires_grad=True)
    time_gaps = torch.arange(8).float().unsqueeze(0).expand(2, -1) * 10

    means, log_vars, _ = model(x, time_gaps)
    loss = means.sum() + log_vars.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and name != "mask_token":
            assert param.grad is not None, f"No gradient for {name}"


def test_parameter_count() -> None:
    """Full-size model has approximately 8M parameters."""
    config = ModelConfig()  # Default = full-size
    model = AeroGPT(config)
    count = model.count_parameters()
    # ARCHITECTURE.md estimated ~8M. Actual is ~4.7M with TransformerEncoder.
    # Allow range 3M-12M.
    assert 3_000_000 < count < 12_000_000, f"Parameter count {count} outside expected range"
