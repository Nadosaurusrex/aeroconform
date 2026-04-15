"""Tests for preprocessing: encoding, delta, normalization."""

from __future__ import annotations

import numpy as np
import polars as pl
from src.data.preprocessing import (
    compute_norm_stats,
    delta_decode,
    delta_encode,
    denormalize,
    encode_state_vectors,
    extract_features,
    normalize,
)
from src.data.schemas import NormStats


def test_encode_state_vectors_shape(synthetic_state_vectors: pl.DataFrame) -> None:
    """Encoded DataFrame has correct columns."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    expected_cols = {
        "icao24",
        "time",
        "callsign",
        "latitude",
        "longitude",
        "baro_altitude",
        "velocity",
        "sin_track",
        "cos_track",
        "vertical_rate",
        "on_ground",
    }
    assert set(encoded.columns) == expected_cols


def test_encode_sin_cos_range(synthetic_state_vectors: pl.DataFrame) -> None:
    """Sin/cos heading values are in [-1, 1]."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    sin_vals = encoded["sin_track"].to_numpy()
    cos_vals = encoded["cos_track"].to_numpy()
    assert np.all(sin_vals >= -1.0 - 1e-6)
    assert np.all(sin_vals <= 1.0 + 1e-6)
    assert np.all(cos_vals >= -1.0 - 1e-6)
    assert np.all(cos_vals <= 1.0 + 1e-6)


def test_encode_sin_cos_identity(synthetic_state_vectors: pl.DataFrame) -> None:
    """sin^2 + cos^2 == 1 for heading encoding."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    sin_vals = encoded["sin_track"].to_numpy()
    cos_vals = encoded["cos_track"].to_numpy()
    identity = sin_vals**2 + cos_vals**2
    np.testing.assert_allclose(identity, 1.0, atol=1e-5)


def test_encode_on_ground_float(synthetic_state_vectors: pl.DataFrame) -> None:
    """on_ground is encoded as float 0/1."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    on_ground = encoded["on_ground"].to_numpy()
    assert set(np.unique(on_ground)).issubset({0.0, 1.0})


def test_extract_features_shape(synthetic_state_vectors: pl.DataFrame) -> None:
    """Feature extraction produces (n, 8) array."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    features = extract_features(encoded)
    assert features.shape == (len(encoded), 8)
    assert features.dtype == np.float32


def test_delta_encode_roundtrip() -> None:
    """Delta encoding is invertible via delta_decode."""
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, size=(50, 8)).astype(np.float32)

    deltas = delta_encode(features)
    reconstructed = delta_decode(deltas, features[0])
    np.testing.assert_allclose(reconstructed, features, atol=1e-5)


def test_delta_encode_first_row_zero() -> None:
    """First row of deltas is zero (no predecessor)."""
    features = np.ones((10, 8), dtype=np.float32)
    deltas = delta_encode(features)
    np.testing.assert_array_equal(deltas[0], np.zeros(8))


def test_normalization_roundtrip() -> None:
    """Normalize then denormalize recovers original."""
    rng = np.random.default_rng(42)
    deltas = rng.normal(0, 1, size=(100, 8)).astype(np.float32)

    stats = compute_norm_stats([deltas])
    normalized = normalize(deltas, stats)
    recovered = denormalize(normalized, stats)
    np.testing.assert_allclose(recovered, deltas, atol=1e-5)


def test_norm_stats_save_load(tmp_path) -> None:
    """NormStats save and load roundtrip."""
    stats = NormStats(
        mean=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        std=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    )
    path = str(tmp_path / "stats.npz")
    stats.save(path)

    loaded = NormStats.load(path)
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)


def test_compute_norm_stats_dimensions() -> None:
    """Norm stats have correct shape (8,)."""
    rng = np.random.default_rng(42)
    trajectories = [rng.normal(0, 1, size=(50, 8)).astype(np.float32) for _ in range(5)]
    stats = compute_norm_stats(trajectories)
    assert stats.mean.shape == (8,)
    assert stats.std.shape == (8,)
    assert np.all(stats.std > 0)
