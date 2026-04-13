"""Gaussian mixture density network output head.

Predicts a mixture of Gaussians for next-patch prediction,
providing both point predictions and uncertainty estimates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as f_torch

from aeroconform.config import AeroConformConfig


class GaussianMixtureHead(nn.Module):
    """Mixture density network output head for the trajectory transformer.

    Predicts parameters of a Gaussian mixture model:
    - means: center of each component
    - log_vars: log-variance of each component (diagonal covariance)
    - log_weights: log mixing weights (log-softmax)

    Args:
        d_model: Input hidden dimension (default: 256).
        output_dim: Output dimension per component (patch_len * input_dim, default: 48).
        n_components: Number of mixture components (default: 5).
    """

    def __init__(
        self,
        d_model: int = 256,
        output_dim: int = 48,
        n_components: int = 5,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        self.d_model = d_model

        self.mean_head = nn.Linear(d_model, n_components * output_dim)
        self.logvar_head = nn.Linear(d_model, n_components * output_dim)
        self.weight_head = nn.Linear(d_model, n_components)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict mixture of Gaussian parameters.

        Args:
            hidden: Transformer output of shape (batch, num_patches, d_model).

        Returns:
            Tuple of:
            - means: (batch, num_patches, n_components, output_dim)
            - log_vars: (batch, num_patches, n_components, output_dim), clamped to [-10, 2]
            - log_weights: (batch, num_patches, n_components), log-softmax normalized
        """
        b, p, d = hidden.shape

        means = self.mean_head(hidden).reshape(b, p, self.n_components, self.output_dim)
        log_vars = self.logvar_head(hidden).reshape(b, p, self.n_components, self.output_dim)
        log_vars = torch.clamp(log_vars, min=-10, max=2)
        log_weights = f_torch.log_softmax(self.weight_head(hidden), dim=-1)

        return means, log_vars, log_weights

    def nll_loss(
        self,
        means: torch.Tensor,
        log_vars: torch.Tensor,
        log_weights: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss for the mixture model.

        Args:
            means: (B, P, K, D) mixture component means.
            log_vars: (B, P, K, D) mixture component log-variances.
            log_weights: (B, P, K) mixture log-weights.
            targets: (B, P, D) actual next patches.

        Returns:
            Scalar NLL loss averaged over batch and patches.
        """
        targets_expanded = targets.unsqueeze(2)  # (B, P, 1, D)

        var = torch.exp(log_vars)
        log_probs = -0.5 * (
            log_vars + (targets_expanded - means) ** 2 / (var + 1e-8) + math.log(2 * math.pi)
        )  # (B, P, K, D)

        log_probs = log_probs.sum(dim=-1)  # (B, P, K) — sum over features
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)  # (B, P)

        return -log_mixture.mean()

    def sample(
        self,
        means: torch.Tensor,
        log_vars: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the mixture distribution.

        Args:
            means: (B, P, K, D) mixture means.
            log_vars: (B, P, K, D) mixture log-variances.
            log_weights: (B, P, K) mixture log-weights.

        Returns:
            Samples of shape (B, P, D).
        """
        # Select component
        weights = torch.exp(log_weights)  # (B, P, K)
        component_idx = torch.multinomial(
            weights.reshape(-1, self.n_components), 1
        ).reshape(means.shape[0], means.shape[1])  # (B, P)

        # Gather selected component parameters
        b, p, k, d = means.shape
        idx = component_idx.unsqueeze(-1).unsqueeze(-1).expand(b, p, 1, d)
        selected_means = means.gather(2, idx).squeeze(2)  # (B, P, D)
        selected_log_vars = log_vars.gather(2, idx).squeeze(2)  # (B, P, D)

        # Sample from selected Gaussian
        std = torch.exp(0.5 * selected_log_vars)
        samples = selected_means + std * torch.randn_like(std)
        return samples

    @classmethod
    def from_config(cls, config: AeroConformConfig) -> GaussianMixtureHead:
        """Create a GaussianMixtureHead from an AeroConformConfig.

        Args:
            config: AeroConform configuration.

        Returns:
            Configured GaussianMixtureHead instance.
        """
        return cls(
            d_model=config.d_model,
            output_dim=config.output_dim,
            n_components=config.n_components,
        )
