"""Tests for data preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeroconform.config import AeroConformConfig
from aeroconform.data.preprocessing import (
    compute_norm_stats,
    delta_encode,
    denormalize,
    extract_trajectories,
    normalize,
    preprocess_pipeline,
    window_trajectory,
)


class TestDeltaEncode:
    """Tests for delta encoding of trajectories."""

    def test_basic_delta(self) -> None:
        """Delta encoding should compute differences between consecutive steps."""
        traj = np.array([
            [45.0, 9.0, 35000.0, 450.0, 90.0, 0.0],
            [45.1, 9.1, 35100.0, 451.0, 91.0, 100.0],
            [45.2, 9.2, 35200.0, 452.0, 92.0, 100.0],
        ])
        deltas = delta_encode(traj)
        assert deltas.shape == (2, 6)
        assert deltas[0, 0] == pytest.approx(0.1, abs=1e-10)
        assert deltas[0, 1] == pytest.approx(0.1, abs=1e-10)
        assert deltas[0, 2] == pytest.approx(100.0, abs=1e-10)

    def test_output_shape(self, synthetic_trajectory: np.ndarray) -> None:
        """Delta encoding reduces length by 1."""
        deltas = delta_encode(synthetic_trajectory)
        assert deltas.shape == (synthetic_trajectory.shape[0] - 1, 6)

    def test_heading_wraparound(self) -> None:
        """Heading deltas should handle 360->0 wraparound correctly."""
        traj = np.array([
            [45.0, 9.0, 35000.0, 450.0, 350.0, 0.0],
            [45.0, 9.0, 35000.0, 450.0, 10.0, 0.0],
        ])
        deltas = delta_encode(traj)
        # Should be +20, not -340
        assert deltas[0, 4] == pytest.approx(20.0, abs=0.1)

    def test_heading_wraparound_negative(self) -> None:
        """Heading deltas should handle 0->350 wraparound."""
        traj = np.array([
            [45.0, 9.0, 35000.0, 450.0, 10.0, 0.0],
            [45.0, 9.0, 35000.0, 450.0, 350.0, 0.0],
        ])
        deltas = delta_encode(traj)
        # Should be -20, not +340
        assert deltas[0, 4] == pytest.approx(-20.0, abs=0.1)

    def test_heading_180_difference(self) -> None:
        """180-degree heading change should be exactly +-180."""
        traj = np.array([
            [45.0, 9.0, 35000.0, 450.0, 0.0, 0.0],
            [45.0, 9.0, 35000.0, 450.0, 180.0, 0.0],
        ])
        deltas = delta_encode(traj)
        assert abs(deltas[0, 4]) == pytest.approx(180.0, abs=0.1)


class TestNormalization:
    """Tests for robust normalization."""

    def test_compute_stats(self) -> None:
        """Norm stats should compute median and IQR."""
        data = [np.random.randn(100, 6)]
        stats = compute_norm_stats(data)
        assert len(stats["median"]) == 6
        assert len(stats["iqr"]) == 6
        assert all(iqr >= 0 for iqr in stats["iqr"])

    def test_normalize_denormalize_roundtrip(self) -> None:
        """Normalizing then denormalizing should recover original data."""
        data = np.random.randn(50, 6)
        stats = compute_norm_stats([data])
        normed = normalize(data, stats)
        recovered = denormalize(normed, stats)
        np.testing.assert_allclose(recovered, data, atol=1e-6)

    def test_normalized_centered(self) -> None:
        """Normalized data should be approximately centered at zero."""
        data = np.random.randn(1000, 6) * 10 + 5
        stats = compute_norm_stats([data])
        normed = normalize(data, stats)
        assert abs(np.median(normed, axis=0)).max() < 0.5


class TestWindowing:
    """Tests for trajectory windowing."""

    def test_window_shape(self) -> None:
        """Windows should have the correct shape."""
        config = AeroConformConfig(seq_len=128)
        data = np.random.randn(200, 6)
        windows = window_trajectory(data, "test", 0, config)
        assert len(windows) > 0
        for w in windows:
            assert w["data"].shape == (128, 6)
            assert w["mask"].shape == (128,)

    def test_short_trajectory_padding(self) -> None:
        """Short trajectories should be zero-padded with mask."""
        config = AeroConformConfig(seq_len=128)
        data = np.random.randn(50, 6)
        windows = window_trajectory(data, "test", 0, config)
        assert len(windows) == 1
        w = windows[0]
        assert w["mask"][:50].all()
        assert not w["mask"][50:].any()
        np.testing.assert_array_equal(w["data"][50:], 0.0)

    def test_overlap(self) -> None:
        """Windows should overlap by stride amount."""
        config = AeroConformConfig(seq_len=128, window_stride_divisor=2)
        data = np.random.randn(256, 6)
        windows = window_trajectory(data, "test", 0, config)
        assert len(windows) >= 2

    def test_metadata(self) -> None:
        """Windows should carry icao24 and start_time metadata."""
        config = AeroConformConfig(seq_len=128)
        data = np.random.randn(128, 6)
        windows = window_trajectory(data, "ABC123", 1000, config)
        assert windows[0]["icao24"] == "ABC123"
        assert windows[0]["start_time"] == 1000


class TestExtractTrajectories:
    """Tests for trajectory extraction from state vector DataFrames."""

    def test_basic_extraction(self) -> None:
        """Should extract trajectories grouped by icao24."""
        n = 50
        df = pd.DataFrame({
            "icao24": ["a1b2c3"] * n,
            "timestamp": list(range(n)),
            "latitude": np.linspace(45, 46, n),
            "longitude": np.linspace(9, 10, n),
            "baro_altitude": np.full(n, 35000.0),
            "velocity": np.full(n, 450.0),
            "true_track": np.full(n, 90.0),
            "vertical_rate": np.zeros(n),
            "on_ground": [False] * n,
        })
        trajs = extract_trajectories(df)
        assert "a1b2c3" in trajs
        assert trajs["a1b2c3"].shape == (n, 6)

    def test_filters_short_trajectories(self) -> None:
        """Should filter out trajectories shorter than min_trajectory_len."""
        config = AeroConformConfig(min_trajectory_len=30)
        df = pd.DataFrame({
            "icao24": ["short"] * 10,
            "timestamp": list(range(10)),
            "latitude": np.linspace(45, 46, 10),
            "longitude": np.linspace(9, 10, 10),
            "baro_altitude": np.full(10, 35000.0),
            "velocity": np.full(10, 450.0),
            "true_track": np.full(10, 90.0),
            "vertical_rate": np.zeros(10),
            "on_ground": [False] * 10,
        })
        trajs = extract_trajectories(df, config)
        assert len(trajs) == 0

    def test_filters_ground_only(self) -> None:
        """Should filter out aircraft that are on ground for entire trajectory."""
        n = 50
        df = pd.DataFrame({
            "icao24": ["ground"] * n,
            "timestamp": list(range(n)),
            "latitude": np.linspace(45, 46, n),
            "longitude": np.linspace(9, 10, n),
            "baro_altitude": np.full(n, 0.0),
            "velocity": np.full(n, 20.0),
            "true_track": np.full(n, 90.0),
            "vertical_rate": np.zeros(n),
            "on_ground": [True] * n,
        })
        trajs = extract_trajectories(df)
        assert len(trajs) == 0

    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        trajs = extract_trajectories(df)
        assert len(trajs) == 0


class TestPreprocessPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_end_to_end(self) -> None:
        """Pipeline should produce windowed trajectories from raw data."""
        n = 200
        df = pd.DataFrame({
            "icao24": ["test1"] * n,
            "timestamp": list(range(n)),
            "latitude": np.linspace(45, 46, n),
            "longitude": np.linspace(9, 10, n),
            "baro_altitude": np.linspace(30000, 35000, n),
            "velocity": np.full(n, 450.0) + np.random.randn(n) * 5,
            "true_track": np.full(n, 90.0) + np.random.randn(n) * 2,
            "vertical_rate": np.random.randn(n) * 50,
            "on_ground": [False] * n,
        })
        windows, stats = preprocess_pipeline(df)
        assert len(windows) > 0
        assert len(stats["median"]) == 6
        assert len(stats["iqr"]) == 6
        assert windows[0]["data"].shape[1] == 6
