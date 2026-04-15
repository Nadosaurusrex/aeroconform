"""Loss functions for AeroGPT.

Gaussian NLL loss as specified in ARCHITECTURE.md:
    NLL = 0.5 * sum_i [log_var_i + (delta_i - mu_i)^2 / exp(log_var_i)]
"""

from __future__ import annotations

import torch


def gaussian_nll_loss(
    means: torch.Tensor,
    log_vars: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Gaussian negative log-likelihood loss.

    Args:
        means: (batch, seq_len, features) predicted means.
        log_vars: (batch, seq_len, features) predicted log-variances.
        targets: (batch, seq_len, features) ground truth deltas.
        mask: (batch, seq_len) boolean mask, True for valid positions.

    Returns:
        Scalar loss averaged over valid positions and features.
    """
    # Per-feature NLL: 0.5 * [log_var + (target - mean)^2 / exp(log_var)]
    precision = torch.exp(-log_vars)
    nll = 0.5 * (log_vars + (targets - means) ** 2 * precision)

    # Sum over features
    nll = nll.sum(dim=-1)  # (batch, seq_len)

    if mask is not None:
        # Zero out padded positions
        nll = nll * mask.float()
        num_valid = mask.float().sum()
        if num_valid > 0:
            return nll.sum() / num_valid
        return nll.sum()

    return nll.mean()
