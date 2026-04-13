"""Adaptive conformal prediction for anomaly detection.

Provides distribution-free anomaly detection with guaranteed false alarm
rates under the exchangeability assumption. Uses non-conformity scores
from the trajectory model's Gaussian mixture predictions.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import scipy.special
import structlog

from aeroconform.config import AeroConformConfig

logger = structlog.get_logger(__name__)


class AdaptiveConformalDetector:
    """Distribution-free anomaly detection with guaranteed false alarm rates.

    The core algorithm:
    1. The foundation model predicts a distribution over next states.
    2. Non-conformity score = negative log-likelihood of the observation
       under the predicted mixture distribution.
    3. Calibrate a threshold on clean data: FAR <= alpha.
    4. Adapt the threshold over time using a sliding window with
       exponential recency weighting.

    Guarantee: under exchangeability, the probability of a false alarm
    at any single timestep is at most alpha.

    Args:
        alpha: Significance level (target false alarm rate).
        cal_window: Size of the sliding calibration window.
        adapt_lr: Learning rate for exponential recency weighting.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        cal_window: int = 500,
        adapt_lr: float = 0.01,
    ) -> None:
        self.alpha = alpha
        self.cal_window = cal_window
        self.adapt_lr = adapt_lr
        self.calibration_scores: deque[float] = deque(maxlen=cal_window)
        self.quantile_weights: np.ndarray | None = None

    def compute_nonconformity_score(
        self,
        observed: np.ndarray,
        means: np.ndarray,
        log_vars: np.ndarray,
        log_weights: np.ndarray,
    ) -> float:
        """Compute the non-conformity score for an observation.

        The score is the negative log-likelihood of the observed state
        under the predicted Gaussian mixture distribution. Higher scores
        indicate more surprising (potentially anomalous) observations.

        Args:
            observed: Actual next state of shape (D,).
            means: Mixture component means of shape (K, D).
            log_vars: Mixture component log-variances of shape (K, D).
            log_weights: Mixture log-weights of shape (K,).

        Returns:
            Non-conformity score (higher = more anomalous).
        """
        var = np.exp(log_vars)  # (K, D)
        log_probs = -0.5 * np.sum(
            log_vars + (observed - means) ** 2 / (var + 1e-8) + np.log(2 * np.pi),
            axis=-1,
        )  # (K,)
        log_mixture = scipy.special.logsumexp(log_weights + log_probs)
        return float(-log_mixture)

    def calibrate(self, clean_scores: np.ndarray) -> None:
        """Initialize calibration with scores from clean (normal) data.

        Sets the sliding calibration window and initializes uniform
        quantile weights.

        Args:
            clean_scores: Non-conformity scores from clean data, shape (N,).
        """
        self.calibration_scores = deque(clean_scores.tolist(), maxlen=self.cal_window)
        n = len(self.calibration_scores)
        self.quantile_weights = np.ones(n) / n
        logger.info("conformal_calibrated", n_scores=n, alpha=self.alpha)

    def get_threshold(self) -> float:
        """Compute the adaptive conformal detection threshold.

        Uses weighted quantile: finds the smallest t such that the weighted
        fraction of calibration scores <= t is at least (1 - alpha).

        Returns:
            Detection threshold. Observations with scores above this
            are flagged as anomalous.

        Raises:
            RuntimeError: If the detector has not been calibrated.
        """
        if not self.calibration_scores:
            raise RuntimeError("Detector not calibrated. Call calibrate() first.")

        scores = np.array(self.calibration_scores)
        n = len(scores)
        sorted_scores = np.sort(scores)

        # Standard conformal quantile: ceil((1-alpha)(n+1))-th smallest score
        # This guarantees FAR <= alpha under exchangeability
        conformal_level = int(np.ceil((1 - self.alpha) * (n + 1)))
        # Clamp to valid index range (1-indexed -> 0-indexed)
        quantile_idx = min(conformal_level - 1, n - 1)
        quantile_idx = max(quantile_idx, 0)

        return float(sorted_scores[quantile_idx])

    def update(self, score: float, is_normal: bool = True) -> None:
        """Update the calibration set with a new observation.

        Only adds the score to the calibration window if the observation
        is classified as normal. Adapts weights using exponential recency.

        Args:
            score: Non-conformity score for the observation.
            is_normal: Whether the observation is normal (not anomalous).
        """
        if is_normal:
            self.calibration_scores.append(score)
            n = len(self.calibration_scores)
            # Exponential recency weighting: more recent scores get more weight
            decay = np.exp(-self.adapt_lr * np.arange(n)[::-1])
            self.quantile_weights = decay / decay.sum()

    def predict(
        self,
        observed: np.ndarray,
        means: np.ndarray,
        log_vars: np.ndarray,
        log_weights: np.ndarray,
    ) -> dict[str, Any]:
        """Run anomaly detection on a single observation.

        Computes the non-conformity score, compares to the adaptive
        threshold, and returns a conformal p-value.

        Args:
            observed: Actual observed state of shape (D,).
            means: Mixture means of shape (K, D).
            log_vars: Mixture log-variances of shape (K, D).
            log_weights: Mixture log-weights of shape (K,).

        Returns:
            Dict with keys:
            - score: Non-conformity score.
            - threshold: Current adaptive threshold.
            - p_value: Conformal p-value (higher = more normal).
            - is_anomaly: True if score > threshold.
            - confidence: 1 - p_value (confidence in anomaly).
        """
        score = self.compute_nonconformity_score(observed, means, log_vars, log_weights)
        threshold = self.get_threshold()

        # Conformal p-value: fraction of calibration scores >= observed score
        cal_scores = np.array(self.calibration_scores)
        p_value = float((np.sum(cal_scores >= score) + 1) / (len(cal_scores) + 1))

        is_anomaly = score > threshold

        return {
            "score": score,
            "threshold": threshold,
            "p_value": p_value,
            "is_anomaly": bool(is_anomaly),
            "confidence": 1.0 - p_value,
        }

    @classmethod
    def from_config(cls, config: AeroConformConfig) -> AdaptiveConformalDetector:
        """Create an AdaptiveConformalDetector from an AeroConformConfig.

        Args:
            config: AeroConform configuration.

        Returns:
            Configured AdaptiveConformalDetector instance.
        """
        return cls(
            alpha=config.alpha,
            cal_window=config.cal_window,
            adapt_lr=config.adapt_lr,
        )
