"""Combined scoring and multi-level alerting.

Per ARCHITECTURE.md section 4:
- Combined score: alpha * s_individual + (1 - alpha) * s_contextual
- Alert levels: RED (p<0.01), AMBER (p<0.05), YELLOW (p<0.10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    """Multi-level alerting thresholds."""

    RED = "red"  # p < 0.01: highly anomalous
    AMBER = "amber"  # p < 0.05: unusual
    YELLOW = "yellow"  # p < 0.10: worth monitoring
    NORMAL = "normal"  # p >= 0.10: normal traffic


def classify_alert(p_value: float) -> AlertLevel:
    """Classify p-value into alert level.

    Args:
        p_value: Conformal p-value.

    Returns:
        Corresponding AlertLevel.
    """
    if p_value < 0.01:
        return AlertLevel.RED
    if p_value < 0.05:
        return AlertLevel.AMBER
    if p_value < 0.10:
        return AlertLevel.YELLOW
    return AlertLevel.NORMAL


def combined_score(
    s_individual: float,
    s_contextual: float,
    alpha_weight: float = 0.5,
) -> float:
    """Compute combined anomaly score.

    Args:
        s_individual: Foundation model non-conformity score.
        s_contextual: Graph-enriched non-conformity score.
        alpha_weight: Weight for individual score (1-alpha for contextual).

    Returns:
        Combined score.
    """
    return alpha_weight * s_individual + (1 - alpha_weight) * s_contextual


@dataclass
class Alert:
    """Anomaly alert for an aircraft."""

    icao24: str
    timestamp: int
    alert_level: AlertLevel
    p_value: float
    score: float
    latitude: float
    longitude: float
    altitude: float
    explanation: str = ""
    attention_pairs: list[tuple[str, float]] = field(default_factory=list)
