"""Adaptive conformal prediction for anomaly detection.

Per ARCHITECTURE.md section 4:
- Mahalanobis non-conformity score
- Sliding calibration buffer (N=2000, decay=0.995)
- Weighted quantile for adaptive threshold
- P-value computation with coverage guarantees
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class AnomalyScore:
    """Result of conformal anomaly scoring."""

    score: float
    p_value: float
    threshold: float
    is_anomaly: bool


def mahalanobis_score(
    observed: npt.NDArray[np.float32],
    means: npt.NDArray[np.float32],
    log_vars: npt.NDArray[np.float32],
) -> float:
    """Compute Mahalanobis non-conformity score.

    Per ARCHITECTURE.md:
        s = sqrt(sum_i (delta_i - mu_i)^2 / exp(log_var_i))

    Args:
        observed: (features,) observed delta values.
        means: (features,) predicted means.
        log_vars: (features,) predicted log-variances.

    Returns:
        Non-conformity score.
    """
    precision = np.exp(-log_vars)
    squared_diff = (observed - means) ** 2 * precision
    return float(np.sqrt(np.sum(squared_diff)))


def weighted_quantile(
    values: list[float],
    quantile: float,
    decay: float,
) -> float:
    """Compute exponentially weighted quantile.

    More recent scores get higher weight, allowing adaptation
    to distribution shift.

    Args:
        values: List of scores (oldest first).
        quantile: Target quantile (e.g., 0.99 for alpha=0.01).
        decay: Exponential decay factor per step.

    Returns:
        Weighted quantile value.
    """
    n = len(values)
    if n == 0:
        return float("inf")

    # Compute weights: newest has highest weight
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    # Sort values and corresponding weights
    sorted_idx = np.argsort(values)
    sorted_values = np.array(values)[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Find weighted quantile
    cumsum = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumsum, quantile)
    idx = min(idx, n - 1)

    return float(sorted_values[idx])


class AdaptiveConformal:
    """Adaptive conformal prediction with sliding calibration.

    Maintains a buffer of non-conformity scores from "normal" traffic
    and uses exponentially weighted quantiles to handle distribution shift.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        buffer_size: int = 2000,
        decay: float = 0.995,
    ) -> None:
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.decay = decay
        self.buffer: deque[float] = deque(maxlen=buffer_size)

    def calibrate(self, scores: list[float] | npt.NDArray[np.float64]) -> None:
        """Initialize calibration buffer with clean scores.

        Args:
            scores: Non-conformity scores from known-normal traffic.
        """
        self.buffer.clear()
        self.buffer.extend(float(s) for s in scores)

    def score(
        self,
        observed: npt.NDArray[np.float32],
        means: npt.NDArray[np.float32],
        log_vars: npt.NDArray[np.float32],
    ) -> AnomalyScore:
        """Score a single observation against the calibration buffer.

        Args:
            observed: (features,) observed delta.
            means: (features,) predicted means.
            log_vars: (features,) predicted log-variances.

        Returns:
            AnomalyScore with score, p-value, threshold, and anomaly flag.
        """
        s = mahalanobis_score(observed, means, log_vars)

        if len(self.buffer) == 0:
            return AnomalyScore(score=s, p_value=1.0, threshold=float("inf"), is_anomaly=False)

        # Compute threshold as weighted quantile
        threshold = weighted_quantile(
            list(self.buffer), 1 - self.alpha, self.decay
        )

        # Compute p-value
        num_geq = sum(1 for c in self.buffer if c >= s)
        p_value = (num_geq + 1) / (len(self.buffer) + 1)

        is_anomaly = s > threshold

        # Update buffer: only add non-anomalous scores
        if not is_anomaly:
            self.buffer.append(s)

        return AnomalyScore(
            score=s,
            p_value=p_value,
            threshold=threshold,
            is_anomaly=is_anomaly,
        )

    @property
    def buffer_size_current(self) -> int:
        """Current number of scores in calibration buffer."""
        return len(self.buffer)
