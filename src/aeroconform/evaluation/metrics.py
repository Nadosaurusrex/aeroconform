"""Evaluation metrics for anomaly detection.

Implements FAR, detection rate, detection delay, AUC-ROC, F1,
and per-anomaly-type breakdown.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.metrics import f1_score, roc_auc_score

logger = structlog.get_logger(__name__)


def false_alarm_rate(
    predictions: np.ndarray, labels: np.ndarray
) -> float:
    """Compute the false alarm rate.

    FAR = number of clean timesteps flagged as anomalous / total clean timesteps.

    Args:
        predictions: Binary predictions (1=anomaly) of shape (N,).
        labels: Ground truth labels (1=anomaly) of shape (N,).

    Returns:
        False alarm rate in [0, 1].
    """
    clean_mask = labels == 0
    n_clean = clean_mask.sum()
    if n_clean == 0:
        return 0.0
    false_alarms = (predictions[clean_mask] == 1).sum()
    return float(false_alarms / n_clean)


def detection_rate(
    predictions: np.ndarray, labels: np.ndarray
) -> float:
    """Compute the detection rate (recall / true positive rate).

    DR = number of anomalous timesteps correctly flagged / total anomalous timesteps.

    Args:
        predictions: Binary predictions (1=anomaly) of shape (N,).
        labels: Ground truth labels (1=anomaly) of shape (N,).

    Returns:
        Detection rate in [0, 1].
    """
    anomaly_mask = labels == 1
    n_anomalous = anomaly_mask.sum()
    if n_anomalous == 0:
        return 1.0
    detections = (predictions[anomaly_mask] == 1).sum()
    return float(detections / n_anomalous)


def detection_delay(
    predictions: np.ndarray, labels: np.ndarray
) -> float | None:
    """Compute the average detection delay.

    Detection delay = number of timesteps between anomaly onset and first detection.

    Args:
        predictions: Binary predictions of shape (N,).
        labels: Ground truth labels of shape (N,).

    Returns:
        Average detection delay in timesteps, or None if no anomalies detected.
    """
    # Find anomaly onset points (0 -> 1 transitions in labels)
    label_diff = np.diff(labels.astype(int))
    onset_indices = np.where(label_diff == 1)[0] + 1

    if len(onset_indices) == 0:
        return None

    delays: list[int] = []
    for onset in onset_indices:
        # Find first detection after onset
        detected = np.where(predictions[onset:] == 1)[0]
        if len(detected) > 0:
            delays.append(int(detected[0]))

    if not delays:
        return None
    return float(np.mean(delays))


def compute_auc_roc(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Compute AUC-ROC using continuous anomaly scores.

    Args:
        scores: Continuous anomaly scores (higher = more anomalous).
        labels: Binary ground truth labels.

    Returns:
        AUC-ROC score in [0, 1].
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_f1(
    predictions: np.ndarray, labels: np.ndarray
) -> float:
    """Compute F1 score.

    Args:
        predictions: Binary predictions of shape (N,).
        labels: Ground truth labels of shape (N,).

    Returns:
        F1 score in [0, 1].
    """
    return float(f1_score(labels, predictions, zero_division=0.0))


def per_type_detection_rates(
    predictions: list[np.ndarray],
    labels: list[np.ndarray],
    types: list[str],
) -> dict[str, float]:
    """Compute detection rates per anomaly type.

    Args:
        predictions: List of binary prediction arrays.
        labels: List of ground truth label arrays.
        types: Anomaly type for each trajectory.

    Returns:
        Dict mapping anomaly type to detection rate.
    """
    type_results: dict[str, list[float]] = {}

    for pred, label, atype in zip(predictions, labels, types, strict=False):
        if atype == "clean":
            continue
        dr = detection_rate(pred, label)
        if atype not in type_results:
            type_results[atype] = []
        type_results[atype].append(dr)

    return {
        atype: float(np.mean(rates))
        for atype, rates in type_results.items()
    }


def compute_all_metrics(
    predictions: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute all standard metrics.

    Args:
        predictions: Binary predictions of shape (N,).
        scores: Continuous anomaly scores of shape (N,).
        labels: Ground truth labels of shape (N,).

    Returns:
        Dict with all metric values.
    """
    metrics = {
        "far": false_alarm_rate(predictions, labels),
        "detection_rate": detection_rate(predictions, labels),
        "f1": compute_f1(predictions, labels),
        "auc_roc": compute_auc_roc(scores, labels),
    }

    delay = detection_delay(predictions, labels)
    if delay is not None:
        metrics["detection_delay"] = delay

    return metrics
