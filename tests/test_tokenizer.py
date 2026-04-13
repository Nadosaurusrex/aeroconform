"""Tests for the patch tokenizer."""

from __future__ import annotations

import torch
import pytest

from aeroconform.config import AeroConformConfig
from aeroconform.models.tokenizer import PatchTokenizer


class TestPatchTokenizer:
    """Tests for PatchTokenizer."""

    def test_output_shape(self) -> None:
        """Output should have correct shape (B, num_patches, d_model)."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256)
        x = torch.randn(32, 128, 6)
        out = tokenizer(x)
        assert out.shape == (32, 16, 256)

    def test_single_sample(self) -> None:
        """Should work with batch size 1."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256)
        x = torch.randn(1, 128, 6)
        out = tokenizer(x)
        assert out.shape == (1, 16, 256)

    def test_different_seq_lengths(self) -> None:
        """Should work with different sequence lengths divisible by patch_len."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256)
        for seq_len in [8, 16, 64, 128, 256]:
            x = torch.randn(4, seq_len, 6)
            out = tokenizer(x)
            assert out.shape == (4, seq_len // 8, 256)

    def test_non_divisible_seq_len_raises(self) -> None:
        """Should raise AssertionError for seq_len not divisible by patch_len."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256)
        x = torch.randn(4, 100, 6)
        with pytest.raises(AssertionError):
            tokenizer(x)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the tokenizer."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256)
        x = torch.randn(4, 128, 6, requires_grad=True)
        out = tokenizer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_from_config(self) -> None:
        """Should create correctly from config."""
        config = AeroConformConfig()
        tokenizer = PatchTokenizer.from_config(config)
        x = torch.randn(4, config.seq_len, config.input_dim)
        out = tokenizer(x)
        assert out.shape == (4, config.num_patches, config.d_model)

    def test_eval_mode_no_dropout(self) -> None:
        """In eval mode, outputs should be deterministic."""
        tokenizer = PatchTokenizer(input_dim=6, patch_len=8, d_model=256, dropout=0.5)
        tokenizer.eval()
        x = torch.randn(4, 128, 6)
        out1 = tokenizer(x)
        out2 = tokenizer(x)
        torch.testing.assert_close(out1, out2)
