"""State vector preprocessing: encoding, delta computation, normalization.

All operations use polars for DataFrame processing and numpy for tensor operations.
Follows ARCHITECTURE.md: 8-dim model input with sin/cos heading encoding.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import polars as pl

from src.data.schemas import NormStats


def encode_state_vectors(df: pl.DataFrame) -> pl.DataFrame:
    """Encode raw state vectors into 8-dim model input features.

    Input columns expected: lat, lon, baroaltitude, velocity, heading, vertrate, onground
    Output columns: latitude, longitude, baro_altitude, velocity, sin_track, cos_track,
                     vertical_rate, on_ground

    Args:
        df: Raw state vector DataFrame from OpenSky (Trino or REST).

    Returns:
        DataFrame with 8 encoded feature columns plus time and icao24.
    """
    return df.with_columns(
        # Keep identifiers
        pl.col("icao24"),
        pl.col("time"),
        pl.col("callsign"),
        # Lat/lon pass through
        pl.col("lat").alias("latitude"),
        pl.col("lon").alias("longitude"),
        # Altitude: fill nulls with 0
        pl.col("baroaltitude").fill_null(0.0).alias("baro_altitude"),
        # Velocity: fill nulls with 0
        pl.col("velocity").fill_null(0.0).alias("velocity_encoded"),
        # Heading: sin/cos encoding to handle wraparound
        (pl.col("heading").fill_null(0.0) * math.pi / 180.0).sin().alias("sin_track"),
        (pl.col("heading").fill_null(0.0) * math.pi / 180.0).cos().alias("cos_track"),
        # Vertical rate: None when on ground -> 0.0
        pl.col("vertrate").fill_null(0.0).alias("vertical_rate"),
        # On ground: boolean -> float 0/1
        pl.col("onground").cast(pl.Float64).fill_null(0.0).alias("on_ground"),
    ).select(
        "icao24",
        "time",
        "callsign",
        "latitude",
        "longitude",
        "baro_altitude",
        pl.col("velocity_encoded").alias("velocity"),
        "sin_track",
        "cos_track",
        "vertical_rate",
        "on_ground",
    )


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "baro_altitude",
    "velocity",
    "sin_track",
    "cos_track",
    "vertical_rate",
    "on_ground",
]


def extract_features(df: pl.DataFrame) -> npt.NDArray[np.float32]:
    """Extract 8-dim feature array from encoded DataFrame.

    Args:
        df: Encoded DataFrame with feature columns.

    Returns:
        Array of shape (num_steps, 8).
    """
    return df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)


def compute_time_gaps(timestamps: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Compute time gaps in seconds between consecutive observations.

    First gap is 0 (no predecessor).

    Args:
        timestamps: Unix timestamps array.

    Returns:
        Array of time gaps in seconds, shape (num_steps,).
    """
    gaps = np.zeros(len(timestamps), dtype=np.float32)
    gaps[1:] = np.diff(timestamps).astype(np.float32)
    return gaps


def compute_elapsed_seconds(timestamps: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Compute elapsed seconds since start of sequence.

    Used for TimeEncoding: actual elapsed seconds, not position indices.

    Args:
        timestamps: Unix timestamps array.

    Returns:
        Array of elapsed seconds from first observation, shape (num_steps,).
    """
    return (timestamps - timestamps[0]).astype(np.float32)


def delta_encode(features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute delta encoding: differences between consecutive timesteps.

    The model predicts change-in-state, not absolute state.
    First timestep delta is zero (no predecessor).

    Args:
        features: Absolute features of shape (num_steps, 8).

    Returns:
        Delta-encoded features of shape (num_steps, 8).
    """
    deltas = np.zeros_like(features)
    deltas[1:] = np.diff(features, axis=0)
    return deltas


def delta_decode(deltas: npt.NDArray[np.float32], initial_state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Reconstruct absolute features from deltas and initial state.

    Args:
        deltas: Delta-encoded features of shape (num_steps, 8).
        initial_state: Initial absolute state, shape (8,).

    Returns:
        Absolute features of shape (num_steps, 8).
    """
    features = np.zeros_like(deltas)
    features[0] = initial_state
    for t in range(1, len(deltas)):
        features[t] = features[t - 1] + deltas[t]
    return features


def compute_norm_stats(all_deltas: list[npt.NDArray[np.float32]]) -> NormStats:
    """Compute per-feature mean and std from a collection of delta-encoded trajectories.

    Args:
        all_deltas: List of delta arrays, each of shape (num_steps, 8).

    Returns:
        NormStats with mean and std arrays of shape (8,).
    """
    concatenated = np.concatenate(all_deltas, axis=0)
    mean = concatenated.mean(axis=0).astype(np.float32)
    std = concatenated.std(axis=0).astype(np.float32)
    # Avoid division by zero: clamp std to a minimum value
    std = np.maximum(std, 1e-8)
    return NormStats(mean=mean, std=std)


def normalize(
    deltas: npt.NDArray[np.float32],
    stats: NormStats,
) -> npt.NDArray[np.float32]:
    """Normalize delta features using mean/std.

    Args:
        deltas: Delta features of shape (num_steps, 8).
        stats: Normalization statistics.

    Returns:
        Normalized deltas of shape (num_steps, 8).
    """
    return ((deltas - stats.mean) / stats.std).astype(np.float32)


def denormalize(
    normalized: npt.NDArray[np.float32],
    stats: NormStats,
) -> npt.NDArray[np.float32]:
    """Denormalize features back to original scale.

    Args:
        normalized: Normalized features of shape (..., 8).
        stats: Normalization statistics.

    Returns:
        Denormalized features in original scale.
    """
    return (normalized * stats.std + stats.mean).astype(np.float32)
