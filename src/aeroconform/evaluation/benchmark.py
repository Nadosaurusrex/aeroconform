"""Baseline comparisons for AeroConform evaluation.

Implements simple baselines for fair comparison: Isolation Forest,
LSTM Autoencoder, One-Class SVM, and threshold-based rules.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from aeroconform.evaluation.metrics import compute_all_metrics

logger = structlog.get_logger(__name__)


class IsolationForestBaseline:
    """Isolation Forest anomaly detector on raw state features.

    Args:
        contamination: Expected fraction of outliers.
        random_state: Random seed.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    def fit(self, trajectories: list[np.ndarray]) -> None:
        """Fit on clean training trajectories.

        Args:
            trajectories: List of clean trajectory arrays.
        """
        features = np.concatenate(trajectories, axis=0)
        self.model.fit(features)

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """Predict anomaly labels for a trajectory.

        Args:
            trajectory: State vectors of shape (T, 6).

        Returns:
            Binary labels of shape (T,) where 1 = anomaly.
        """
        preds: np.ndarray = self.model.predict(trajectory)
        result: np.ndarray = (preds == -1).astype(int)
        return result

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous).

        Args:
            trajectory: State vectors of shape (T, 6).

        Returns:
            Anomaly scores of shape (T,).
        """
        result: np.ndarray = -self.model.score_samples(trajectory)
        return result


class OneClassSVMBaseline:
    """One-Class SVM anomaly detector on trajectory features.

    Args:
        kernel: SVM kernel type.
        nu: Upper bound on fraction of outliers.
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.05) -> None:
        self.model = OneClassSVM(kernel=kernel, nu=nu)

    def fit(self, trajectories: list[np.ndarray]) -> None:
        """Fit on clean training trajectories.

        Args:
            trajectories: List of clean trajectory arrays.
        """
        features = np.concatenate(trajectories, axis=0)
        # Subsample if too large for SVM
        if len(features) > 10000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(features), 10000, replace=False)
            features = features[idx]
        self.model.fit(features)

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            trajectory: State vectors of shape (T, 6).

        Returns:
            Binary labels where 1 = anomaly.
        """
        preds: np.ndarray = self.model.predict(trajectory)
        result: np.ndarray = (preds == -1).astype(int)
        return result

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """Get anomaly scores.

        Args:
            trajectory: State vectors of shape (T, 6).

        Returns:
            Anomaly scores (higher = more anomalous).
        """
        result: np.ndarray = -self.model.score_samples(trajectory)
        return result


class ThresholdBaseline:
    """Simple physics-based threshold anomaly detector.

    Flags timesteps where state changes exceed physical limits.

    Args:
        max_speed_kts: Maximum plausible speed.
        max_climb_rate_fpm: Maximum plausible climb rate.
        max_turn_rate_dps: Maximum plausible turn rate (degrees/sec).
        max_jump_nm: Maximum plausible position change per timestep.
    """

    def __init__(
        self,
        max_speed_kts: float = 600.0,
        max_climb_rate_fpm: float = 6000.0,
        max_turn_rate_dps: float = 5.0,
        max_jump_nm: float = 0.5,
    ) -> None:
        self.max_speed = max_speed_kts
        self.max_climb = max_climb_rate_fpm
        self.max_turn = max_turn_rate_dps
        self.max_jump = max_jump_nm

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """Predict anomaly labels based on physics thresholds.

        Args:
            trajectory: State vectors (T, 6) [lat, lon, alt, vel, hdg, vrate].

        Returns:
            Binary labels where 1 = anomaly.
        """
        labels = np.zeros(len(trajectory), dtype=int)

        for t in range(1, len(trajectory)):
            # Check speed
            if trajectory[t, 3] > self.max_speed:
                labels[t] = 1

            # Check climb rate
            if abs(trajectory[t, 5]) > self.max_climb:
                labels[t] = 1

            # Check turn rate
            hdg_diff = ((trajectory[t, 4] - trajectory[t - 1, 4] + 180) % 360) - 180
            if abs(hdg_diff) > self.max_turn:
                labels[t] = 1

            # Check position jump (approximate)
            dlat = trajectory[t, 0] - trajectory[t - 1, 0]
            dlon = trajectory[t, 1] - trajectory[t - 1, 1]
            jump_deg = np.sqrt(dlat**2 + dlon**2)
            jump_nm = jump_deg * 60  # Approximate
            if jump_nm > self.max_jump:
                labels[t] = 1

        return labels

    def score(self, trajectory: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on deviation magnitude.

        Args:
            trajectory: State vectors of shape (T, 6).

        Returns:
            Anomaly scores of shape (T,).
        """
        scores = np.zeros(len(trajectory))

        for t in range(1, len(trajectory)):
            speed_dev = max(0, trajectory[t, 3] - self.max_speed) / self.max_speed
            climb_dev = max(0, abs(trajectory[t, 5]) - self.max_climb) / self.max_climb
            hdg_diff = abs(((trajectory[t, 4] - trajectory[t - 1, 4] + 180) % 360) - 180)
            turn_dev = max(0, hdg_diff - self.max_turn) / self.max_turn
            scores[t] = speed_dev + climb_dev + turn_dev

        return scores


def run_benchmark(
    train_trajectories: list[np.ndarray],
    test_trajectories: list[np.ndarray],
    test_labels: list[np.ndarray],
    test_types: list[str],
) -> dict[str, dict[str, float]]:
    """Run all baselines and compute metrics.

    Args:
        train_trajectories: Clean training trajectories.
        test_trajectories: Mixed test trajectories.
        test_labels: Per-timestep labels for test trajectories.
        test_types: Anomaly type for each test trajectory.

    Returns:
        Dict mapping baseline name to metrics dict.
    """
    results: dict[str, dict[str, float]] = {}

    # Isolation Forest
    iso = IsolationForestBaseline()
    iso.fit(train_trajectories)
    all_preds = np.concatenate([iso.predict(t) for t in test_trajectories])
    all_scores = np.concatenate([iso.score(t) for t in test_trajectories])
    all_labels = np.concatenate(test_labels)
    results["isolation_forest"] = compute_all_metrics(all_preds, all_scores, all_labels)

    # One-Class SVM
    svm = OneClassSVMBaseline()
    svm.fit(train_trajectories)
    all_preds = np.concatenate([svm.predict(t) for t in test_trajectories])
    all_scores = np.concatenate([svm.score(t) for t in test_trajectories])
    results["one_class_svm"] = compute_all_metrics(all_preds, all_scores, all_labels)

    # Threshold
    thresh = ThresholdBaseline()
    all_preds = np.concatenate([thresh.predict(t) for t in test_trajectories])
    all_scores = np.concatenate([thresh.score(t) for t in test_trajectories])
    results["threshold"] = compute_all_metrics(all_preds, all_scores, all_labels)

    logger.info("benchmark_complete", baselines=list(results.keys()))
    return results
