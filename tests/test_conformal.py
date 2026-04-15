"""Tests for conformal prediction."""

from __future__ import annotations

import numpy as np
from src.models.conformal import AdaptiveConformal, mahalanobis_score, weighted_quantile


def test_mahalanobis_zero_residual() -> None:
    """Perfect prediction gives score 0."""
    observed = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    means = observed.copy()
    log_vars = np.zeros(3, dtype=np.float32)

    score = mahalanobis_score(observed, means, log_vars)
    assert score == 0.0


def test_mahalanobis_positive() -> None:
    """Non-zero residual gives positive score."""
    observed = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    log_vars = np.zeros(3, dtype=np.float32)

    score = mahalanobis_score(observed, means, log_vars)
    assert score > 0


def test_mahalanobis_variance_scaling() -> None:
    """Higher variance reduces score for same residual."""
    observed = np.array([2.0, 2.0], dtype=np.float32)
    means = np.zeros(2, dtype=np.float32)

    low_var = np.zeros(2, dtype=np.float32)  # var = 1
    high_var = np.full(2, 2.0, dtype=np.float32)  # var = e^2

    score_low = mahalanobis_score(observed, means, low_var)
    score_high = mahalanobis_score(observed, means, high_var)

    assert score_high < score_low


def test_weighted_quantile_uniform() -> None:
    """Weighted quantile on uniform data."""
    values = list(range(100))
    q = weighted_quantile(values, 0.99, 1.0)  # No decay
    assert q >= 98


def test_weighted_quantile_empty() -> None:
    """Empty buffer returns inf."""
    q = weighted_quantile([], 0.99, 0.995)
    assert q == float("inf")


def test_adaptive_conformal_calibration() -> None:
    """Calibration fills buffer."""
    ac = AdaptiveConformal(alpha=0.05, buffer_size=100)
    scores = np.random.default_rng(42).normal(0, 1, size=100).tolist()
    ac.calibrate(scores)
    assert ac.buffer_size_current == 100


def test_adaptive_conformal_scoring() -> None:
    """Score produces valid AnomalyScore."""
    ac = AdaptiveConformal(alpha=0.05, buffer_size=100)
    scores = np.random.default_rng(42).exponential(1, size=100).tolist()
    ac.calibrate(scores)

    observed = np.array([0.5] * 8, dtype=np.float32)
    means = np.zeros(8, dtype=np.float32)
    log_vars = np.zeros(8, dtype=np.float32)

    result = ac.score(observed, means, log_vars)
    assert 0 <= result.p_value <= 1
    assert result.score >= 0
    assert result.threshold >= 0


def test_adaptive_conformal_anomaly_not_added() -> None:
    """Anomalous scores should not contaminate the buffer."""
    ac = AdaptiveConformal(alpha=0.05, buffer_size=50)
    # Fill with small scores
    ac.calibrate([0.1] * 50)
    initial_size = ac.buffer_size_current

    # Score something very anomalous
    observed = np.full(8, 100.0, dtype=np.float32)
    means = np.zeros(8, dtype=np.float32)
    log_vars = np.zeros(8, dtype=np.float32)

    result = ac.score(observed, means, log_vars)
    assert result.is_anomaly
    # Buffer should NOT have grown
    assert ac.buffer_size_current == initial_size
