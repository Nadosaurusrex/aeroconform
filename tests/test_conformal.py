"""Tests for the adaptive conformal prediction module.

Critical tests: empirically verify the coverage guarantee that
FAR <= alpha + tolerance in 95% of runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from aeroconform.models.conformal import AdaptiveConformalDetector


class TestNonconformityScore:
    """Tests for non-conformity score computation."""

    def test_perfect_prediction_low_score(self) -> None:
        """When observed matches the mean, score should be relatively low."""
        detector = AdaptiveConformalDetector()
        observed = np.zeros(6)
        means = np.zeros((3, 6))
        log_vars = np.zeros((3, 6))  # variance = 1
        log_weights = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))

        score = detector.compute_nonconformity_score(observed, means, log_vars, log_weights)
        assert np.isfinite(score)

    def test_outlier_high_score(self) -> None:
        """When observed is far from all means, score should be high."""
        detector = AdaptiveConformalDetector()
        observed = np.ones(6) * 100
        means = np.zeros((3, 6))
        log_vars = np.zeros((3, 6))
        log_weights = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))

        score = detector.compute_nonconformity_score(observed, means, log_vars, log_weights)
        assert score > 100

    def test_score_increases_with_distance(self) -> None:
        """Score should increase as observation moves away from prediction."""
        detector = AdaptiveConformalDetector()
        means = np.zeros((2, 6))
        log_vars = np.zeros((2, 6))
        log_weights = np.log(np.array([0.5, 0.5]))

        scores = []
        for dist in [0.0, 1.0, 5.0, 10.0]:
            observed = np.ones(6) * dist
            score = detector.compute_nonconformity_score(observed, means, log_vars, log_weights)
            scores.append(score)

        # Scores should be monotonically increasing
        for i in range(len(scores) - 1):
            assert scores[i + 1] > scores[i]


class TestCalibration:
    """Tests for calibration procedure."""

    def test_calibrate_sets_scores(self) -> None:
        """Calibration should set the calibration scores."""
        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=100)
        scores = np.random.randn(100)
        detector.calibrate(scores)
        assert len(detector.calibration_scores) == 100

    def test_threshold_exists_after_calibration(self) -> None:
        """Should be able to get threshold after calibration."""
        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=100)
        scores = np.random.randn(100)
        detector.calibrate(scores)
        threshold = detector.get_threshold()
        assert np.isfinite(threshold)

    def test_threshold_increases_with_lower_alpha(self) -> None:
        """Lower alpha (stricter) should produce higher threshold."""
        scores = np.random.randn(1000)

        d1 = AdaptiveConformalDetector(alpha=0.10, cal_window=1000)
        d1.calibrate(scores)

        d2 = AdaptiveConformalDetector(alpha=0.01, cal_window=1000)
        d2.calibrate(scores)

        assert d2.get_threshold() > d1.get_threshold()

    def test_uncalibrated_raises(self) -> None:
        """Getting threshold before calibration should raise."""
        detector = AdaptiveConformalDetector()
        with pytest.raises(RuntimeError):
            detector.get_threshold()


class TestUpdate:
    """Tests for online updates."""

    def test_update_adds_score(self) -> None:
        """Normal update should add to calibration window."""
        detector = AdaptiveConformalDetector(cal_window=100)
        detector.calibrate(np.random.randn(50))
        initial_len = len(detector.calibration_scores)
        detector.update(1.0, is_normal=True)
        assert len(detector.calibration_scores) == initial_len + 1

    def test_anomalous_update_skipped(self) -> None:
        """Anomalous observations should not be added to calibration."""
        detector = AdaptiveConformalDetector(cal_window=100)
        detector.calibrate(np.random.randn(50))
        initial_len = len(detector.calibration_scores)
        detector.update(1.0, is_normal=False)
        assert len(detector.calibration_scores) == initial_len

    def test_window_bounded(self) -> None:
        """Calibration window should not exceed max size."""
        detector = AdaptiveConformalDetector(cal_window=100)
        detector.calibrate(np.random.randn(100))
        for i in range(50):
            detector.update(float(i), is_normal=True)
        assert len(detector.calibration_scores) == 100


class TestPredict:
    """Tests for prediction output."""

    def test_predict_output_keys(self) -> None:
        """Predict should return all expected keys."""
        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=100)
        detector.calibrate(np.random.randn(100))

        result = detector.predict(
            observed=np.zeros(6),
            means=np.zeros((3, 6)),
            log_vars=np.zeros((3, 6)),
            log_weights=np.log(np.array([1 / 3, 1 / 3, 1 / 3])),
        )
        assert "score" in result
        assert "threshold" in result
        assert "p_value" in result
        assert "is_anomaly" in result
        assert "confidence" in result

    def test_p_value_range(self) -> None:
        """P-value should be in (0, 1]."""
        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=100)
        detector.calibrate(np.random.randn(100))

        result = detector.predict(
            observed=np.zeros(6),
            means=np.zeros((3, 6)),
            log_vars=np.zeros((3, 6)),
            log_weights=np.log(np.array([1 / 3, 1 / 3, 1 / 3])),
        )
        assert 0 < result["p_value"] <= 1.0

    def test_confidence_complement_of_p_value(self) -> None:
        """Confidence should equal 1 - p_value."""
        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=100)
        detector.calibrate(np.random.randn(100))

        result = detector.predict(
            observed=np.zeros(6),
            means=np.zeros((3, 6)),
            log_vars=np.zeros((3, 6)),
            log_weights=np.log(np.array([1 / 3, 1 / 3, 1 / 3])),
        )
        assert result["confidence"] == pytest.approx(1.0 - result["p_value"])


class TestCoverageGuarantee:
    """Critical tests: empirically verify the conformal coverage guarantee.

    The guarantee states that under exchangeable data, the false alarm
    rate should be at most alpha (plus a small tolerance for finite samples).
    """

    def _run_coverage_trial(self, alpha: float, n_cal: int, n_test: int, seed: int) -> float:
        """Run a single coverage trial and return the empirical FAR.

        Generates data from a known Gaussian distribution, calibrates
        on n_cal samples, and computes FAR on n_test clean samples.
        """
        rng = np.random.default_rng(seed)

        # Generate data from a known 1-component Gaussian
        d = 6
        true_mean = np.zeros(d)
        true_var = np.ones(d)

        # Simulate model predictions: perfect knowledge of the distribution
        means = true_mean.reshape(1, d)
        log_vars = np.log(true_var).reshape(1, d)
        log_weights = np.array([0.0])  # Single component

        detector = AdaptiveConformalDetector(alpha=alpha, cal_window=n_cal)

        # Calibrate on clean data
        cal_scores = []
        for _ in range(n_cal):
            obs = rng.normal(true_mean, np.sqrt(true_var))
            score = detector.compute_nonconformity_score(obs, means, log_vars, log_weights)
            cal_scores.append(score)
        detector.calibrate(np.array(cal_scores))

        # Test on clean data
        false_alarms = 0
        for _ in range(n_test):
            obs = rng.normal(true_mean, np.sqrt(true_var))
            result = detector.predict(obs, means, log_vars, log_weights)
            if result["is_anomaly"]:
                false_alarms += 1

        return false_alarms / n_test

    def test_coverage_single_run(self) -> None:
        """Single run: FAR should be approximately <= alpha."""
        far = self._run_coverage_trial(alpha=0.05, n_cal=500, n_test=1000, seed=42)
        # Allow generous tolerance for single run
        assert far <= 0.10, f"FAR {far:.4f} exceeds tolerance"

    def test_coverage_guarantee_95_percent(self) -> None:
        """Critical test: FAR <= alpha + 0.01 in at least 95% of 100 runs.

        This is the core conformal guarantee. With alpha=0.05:
        - In at least 95 out of 100 runs, FAR should be <= 0.06.
        Uses a large calibration set (2000) to reduce threshold variance.
        """
        alpha = 0.05
        tolerance = 0.01
        n_runs = 100
        n_cal = 5000
        n_test = 5000

        passes = 0
        for i in range(n_runs):
            far = self._run_coverage_trial(
                alpha=alpha, n_cal=n_cal, n_test=n_test, seed=i
            )
            if far <= alpha + tolerance:
                passes += 1

        pass_rate = passes / n_runs
        assert pass_rate >= 0.95, (
            f"Coverage guarantee failed: only {passes}/{n_runs} runs "
            f"({pass_rate:.2%}) satisfied FAR <= {alpha + tolerance}"
        )

    def test_coverage_at_different_alphas(self) -> None:
        """Coverage should hold at different significance levels."""
        for alpha in [0.01, 0.05, 0.10]:
            passes = 0
            for i in range(50):
                far = self._run_coverage_trial(
                    alpha=alpha, n_cal=2000, n_test=1000, seed=i * 100 + int(alpha * 100)
                )
                if far <= alpha + 0.02:
                    passes += 1

            pass_rate = passes / 50
            assert pass_rate >= 0.90, (
                f"Coverage failed at alpha={alpha}: {passes}/50 "
                f"({pass_rate:.2%}) passed"
            )

    def test_anomalies_detected(self) -> None:
        """Injected anomalies (far from mean) should be detected."""
        rng = np.random.default_rng(42)
        d = 6
        true_mean = np.zeros(d)
        true_var = np.ones(d)

        means = true_mean.reshape(1, d)
        log_vars = np.log(true_var).reshape(1, d)
        log_weights = np.array([0.0])

        detector = AdaptiveConformalDetector(alpha=0.05, cal_window=500)

        # Calibrate
        cal_scores = []
        for _ in range(500):
            obs = rng.normal(true_mean, np.sqrt(true_var))
            score = detector.compute_nonconformity_score(obs, means, log_vars, log_weights)
            cal_scores.append(score)
        detector.calibrate(np.array(cal_scores))

        # Test with anomalies (shifted by 10 std devs)
        detections = 0
        for _ in range(100):
            anomalous = rng.normal(true_mean + 10.0, np.sqrt(true_var))
            result = detector.predict(anomalous, means, log_vars, log_weights)
            if result["is_anomaly"]:
                detections += 1

        # Should detect nearly all strong anomalies
        assert detections >= 95, f"Only detected {detections}/100 anomalies"
