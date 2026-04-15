"""Evaluation harness for AeroConform.

Runs all anomaly types, computes metrics, generates report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from src.data.preprocessing import delta_encode, normalize
from src.data.schemas import Flight, NormStats
from src.data.synthetic import (
    inject_ghost,
    inject_gps_drift,
    inject_impossible_maneuver,
    inject_position_spoofing,
)
from src.evaluation.metrics import DetectionResult, compute_detection_rate
from src.models.conformal import AdaptiveConformal

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from running the full evaluation benchmark."""

    spoofing: DetectionResult
    ghost: DetectionResult
    gps_drift: DetectionResult
    impossible_maneuver: DetectionResult
    calibration_error: float
    mean_ade: float
    mean_fde: float


def evaluate_anomaly_type(
    conformal: AdaptiveConformal,
    normal_flights: list[Flight],
    anomalous_flights: list[tuple[Flight, np.ndarray]],
    norm_stats: NormStats,
    predict_fn,  # noqa: ANN001
) -> DetectionResult:
    """Evaluate detection on a specific anomaly type.

    Args:
        conformal: Calibrated conformal detector.
        normal_flights: Clean flights for false alarm measurement.
        anomalous_flights: List of (flight, labels) tuples.
        norm_stats: Normalization statistics.
        predict_fn: Function(features) -> (means, log_vars).

    Returns:
        DetectionResult for this anomaly type.
    """
    all_predictions = []
    all_labels = []

    # Score anomalous flights
    for flight, labels in anomalous_flights:
        deltas = delta_encode(flight.features)
        normalized = normalize(deltas, norm_stats)

        predictions = np.zeros(flight.num_steps, dtype=np.bool_)
        for t in range(1, flight.num_steps):
            means, log_vars = predict_fn(normalized[:t])
            result = conformal.score(normalized[t], means, log_vars)
            predictions[t] = result.is_anomaly

        all_predictions.append(predictions)
        all_labels.append(labels)

    # Score normal flights for false alarm rate
    for flight in normal_flights[:10]:
        deltas = delta_encode(flight.features)
        normalized = normalize(deltas, norm_stats)

        predictions = np.zeros(flight.num_steps, dtype=np.bool_)
        for t in range(1, flight.num_steps):
            means, log_vars = predict_fn(normalized[:t])
            result = conformal.score(normalized[t], means, log_vars)
            predictions[t] = result.is_anomaly

        all_predictions.append(predictions)
        all_labels.append(np.zeros(flight.num_steps, dtype=np.bool_))

    predictions_cat = np.concatenate(all_predictions)
    labels_cat = np.concatenate(all_labels)

    return compute_detection_rate(predictions_cat, labels_cat)


def run_benchmark(
    normal_flights: list[Flight],
    norm_stats: NormStats,
    predict_fn,  # noqa: ANN001
    conformal: AdaptiveConformal,
    *,
    num_anomalies: int = 20,
    seed: int = 42,
) -> BenchmarkResult:
    """Run full evaluation benchmark.

    Args:
        normal_flights: Clean flights for evaluation.
        norm_stats: Normalization statistics.
        predict_fn: Prediction function.
        conformal: Calibrated conformal detector.
        num_anomalies: Number of anomalous samples per type.
        seed: Random seed.

    Returns:
        BenchmarkResult with all metrics.
    """
    # Generate anomalous samples
    source_flights = normal_flights[:num_anomalies]

    spoofing_data = []
    ghost_data = []
    drift_data = []
    maneuver_data = []

    for i, flight in enumerate(source_flights):
        # Spoofing
        lf = inject_position_spoofing(flight)
        labels = np.zeros(flight.num_steps, dtype=np.bool_)
        labels[lf.labels[0].start_idx : lf.labels[0].end_idx] = True
        spoofing_data.append((lf.flight, labels))

        # Ghost
        lf = inject_ghost(seed=seed + i)
        labels = np.ones(lf.flight.num_steps, dtype=np.bool_)
        ghost_data.append((lf.flight, labels))

        # GPS drift
        lf = inject_gps_drift(flight)
        labels = np.zeros(flight.num_steps, dtype=np.bool_)
        labels[lf.labels[0].start_idx : lf.labels[0].end_idx] = True
        drift_data.append((lf.flight, labels))

        # Impossible maneuver
        lf = inject_impossible_maneuver(flight)
        labels = np.zeros(flight.num_steps, dtype=np.bool_)
        labels[lf.labels[0].start_idx : lf.labels[0].end_idx] = True
        maneuver_data.append((lf.flight, labels))

    logger.info("running_benchmark", anomalies_per_type=num_anomalies)

    spoofing_result = evaluate_anomaly_type(conformal, normal_flights, spoofing_data, norm_stats, predict_fn)
    ghost_result = evaluate_anomaly_type(conformal, normal_flights, ghost_data, norm_stats, predict_fn)
    drift_result = evaluate_anomaly_type(conformal, normal_flights, drift_data, norm_stats, predict_fn)
    maneuver_result = evaluate_anomaly_type(conformal, normal_flights, maneuver_data, norm_stats, predict_fn)

    logger.info(
        "benchmark_complete",
        spoofing_recall=f"{spoofing_result.detection_rate:.3f}",
        ghost_recall=f"{ghost_result.detection_rate:.3f}",
        drift_recall=f"{drift_result.detection_rate:.3f}",
        maneuver_recall=f"{maneuver_result.detection_rate:.3f}",
    )

    return BenchmarkResult(
        spoofing=spoofing_result,
        ghost=ghost_result,
        gps_drift=drift_result,
        impossible_maneuver=maneuver_result,
        calibration_error=0.0,  # Computed separately with p-values
        mean_ade=0.0,  # Computed from model predictions
        mean_fde=0.0,
    )
