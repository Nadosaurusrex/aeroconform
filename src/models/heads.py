"""Output heads for AeroGPT.

GaussianHead: predicts mean + log_variance per feature for probabilistic output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianHead(nn.Module):
    """Single Gaussian output per feature.

    Predicts 8 means and 8 log-variances (16 total).
    Log-variance is clamped to [-10, 10] for numerical stability.
    """

    LOG_VAR_MIN = -10.0
    LOG_VAR_MAX = 10.0

    def __init__(self, hidden_dim: int = 256, output_dim: int = 16) -> None:
        super().__init__()
        self.num_features = output_dim // 2
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict Gaussian parameters.

        Args:
            hidden: (batch, seq_len, hidden_dim) transformer output.

        Returns:
            means: (batch, seq_len, num_features) predicted delta means.
            log_vars: (batch, seq_len, num_features) clamped log-variances.
        """
        output = self.linear(hidden)
        means = output[..., : self.num_features]
        log_vars = output[..., self.num_features :]
        log_vars = torch.clamp(log_vars, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        return means, log_vars
