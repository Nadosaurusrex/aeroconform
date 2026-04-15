"""Evaluation metrics for anomaly detection.

Detection rate, false alarm rate, detection latency, ADE/FDE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DetectionResult:
    """Result of anomaly detection evaluation."""

    detection_rate: float
    false_alarm_rate: float
    detection_latency: float  # Average timesteps to first alert
    num_true_positives: int
    num_false_positives: int
    num_true_negatives: int
    num_false_negatives: int


def compute_detection_rate(
    predictions: npt.NDArray[np.bool_],
    labels: npt.NDArray[np.bool_],
) -> DetectionResult:
    """Compute detection metrics from binary predictions and labels.

    Args:
        predictions: (N,) predicted anomaly flags.
        labels: (N,) ground truth anomaly flags.

    Returns:
        DetectionResult with rates and counts.
    """
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))

    detection_rate = tp / max(1, tp + fn)
    false_alarm_rate = fp / max(1, fp + tn)

    # Detection latency: average index of first True prediction in anomalous regions
    latencies = []
    in_anomaly = False
    anomaly_start = 0
    for i in range(len(labels)):
        if labels[i] and not in_anomaly:
            in_anomaly = True
            anomaly_start = i
        elif not labels[i] and in_anomaly:
            in_anomaly = False

        if in_anomaly and predictions[i]:
            latencies.append(i - anomaly_start)
            in_anomaly = False

    avg_latency = float(np.mean(latencies)) if latencies else float("inf")

    return DetectionResult(
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        detection_latency=avg_latency,
        num_true_positives=tp,
        num_false_positives=fp,
        num_true_negatives=tn,
        num_false_negatives=fn,
    )


def compute_calibration_error(
    p_values: npt.NDArray[np.float64],
    labels: npt.NDArray[np.bool_],
    alpha: float = 0.01,
) -> float:
    """Compute calibration error: |empirical FAR - target alpha|.

    Args:
        p_values: (N,) conformal p-values for normal traffic.
        labels: (N,) True = anomalous (excluded from calibration check).
        alpha: Target false alarm rate.

    Returns:
        Absolute calibration error.
    """
    normal_mask = ~labels
    if normal_mask.sum() == 0:
        return 0.0

    normal_p = p_values[normal_mask]
    empirical_far = float(np.mean(normal_p < alpha))

    return abs(empirical_far - alpha)
