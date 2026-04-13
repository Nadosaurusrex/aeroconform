"""Preprocessing pipeline for ADS-B trajectory data.

Handles trajectory extraction, delta encoding, robust normalization,
and windowing for model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import structlog

from aeroconform.config import AeroConformConfig

logger = structlog.get_logger(__name__)

FEATURE_COLUMNS = ["latitude", "longitude", "baro_altitude", "velocity", "true_track", "vertical_rate"]


class NormStats(TypedDict):
    """Normalization statistics for each feature."""

    median: list[float]
    iqr: list[float]


class WindowedTrajectory(TypedDict):
    """A windowed trajectory sample ready for the model."""

    data: np.ndarray  # (seq_len, 6)
    mask: np.ndarray  # (seq_len,)
    icao24: str
    start_time: int


def extract_trajectories(
    states_df: pd.DataFrame,
    config: AeroConformConfig | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-aircraft trajectories from state vector snapshots.

    Groups state vectors by icao24, sorts by timestamp, and filters
    out aircraft with insufficient data or those on the ground.

    Args:
        states_df: DataFrame with state vector columns including 'icao24' and 'timestamp'.
        config: Configuration for minimum trajectory length.

    Returns:
        Dict mapping icao24 to trajectory arrays of shape (T, 6).
    """
    cfg = config or AeroConformConfig()
    trajectories: dict[str, np.ndarray] = {}

    if states_df.empty:
        return trajectories

    for icao24, group in states_df.groupby("icao24"):
        group = group.sort_values("timestamp")

        # Filter out entries with null position
        valid = group.dropna(subset=["latitude", "longitude", "baro_altitude"])
        if len(valid) < cfg.min_trajectory_len:
            continue

        # Skip if aircraft is on ground for the entire trajectory
        if valid["on_ground"].all():
            continue

        # Extract feature columns
        features = valid[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

        # Fill any remaining NaN values with forward fill then backward fill
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                # Forward fill
                for i in range(1, len(col)):
                    if mask[i]:
                        col[i] = col[i - 1]
                # Backward fill for leading NaNs
                for i in range(len(col) - 2, -1, -1):
                    if mask[i]:
                        col[i] = col[i + 1]

        # Skip if still has NaN
        if np.isnan(features).any():
            continue

        trajectories[str(icao24)] = features

    logger.info("trajectories_extracted", count=len(trajectories))
    return trajectories


def delta_encode(trajectory: np.ndarray) -> np.ndarray:
    """Convert absolute state vectors to deltas.

    Computes the difference between consecutive timesteps.
    Special handling for heading (true_track) to handle wrap-around.

    Args:
        trajectory: Absolute state vectors of shape (T, 6) where features
            are [lat, lon, alt, vel, hdg, vrate].

    Returns:
        Delta-encoded state vectors of shape (T-1, 6) where features
        are [dlat, dlon, dalt, dvel, dhdg, dvrate].
    """
    deltas = np.diff(trajectory, axis=0)

    # Fix heading wrap-around (column index 4)
    # Use shortest angular difference: ((delta + 180) % 360) - 180
    deltas[:, 4] = ((deltas[:, 4] + 180) % 360) - 180

    return deltas


def compute_norm_stats(trajectories: list[np.ndarray]) -> NormStats:
    """Compute robust normalization statistics from training trajectories.

    Uses median and IQR (interquartile range) instead of mean/std
    to handle outliers in ADS-B data.

    Args:
        trajectories: List of delta-encoded trajectory arrays, each (T, 6).

    Returns:
        NormStats dict with median and IQR per feature.
    """
    all_data = np.concatenate(trajectories, axis=0)

    median = np.median(all_data, axis=0).tolist()
    q25 = np.percentile(all_data, 25, axis=0)
    q75 = np.percentile(all_data, 75, axis=0)
    iqr = (q75 - q25).tolist()

    return NormStats(median=median, iqr=iqr)


def normalize(data: np.ndarray, stats: NormStats, eps: float = 1e-8) -> np.ndarray:
    """Apply robust normalization to data.

    Normalizes using: x_norm = (x - median) / (IQR + eps)

    Args:
        data: Array of shape (..., 6) to normalize.
        stats: Normalization statistics (median and IQR).
        eps: Small constant for numerical stability.

    Returns:
        Normalized array of the same shape.
    """
    median = np.array(stats["median"])
    iqr = np.array(stats["iqr"])
    result: np.ndarray = (data - median) / (iqr + eps)
    return result


def denormalize(data: np.ndarray, stats: NormStats, eps: float = 1e-8) -> np.ndarray:
    """Reverse the robust normalization.

    Args:
        data: Normalized array of shape (..., 6).
        stats: Normalization statistics used for normalization.
        eps: Same eps used in normalize.

    Returns:
        Denormalized array of the same shape.
    """
    median = np.array(stats["median"])
    iqr = np.array(stats["iqr"])
    result: np.ndarray = data * (iqr + eps) + median
    return result


def window_trajectory(
    trajectory: np.ndarray,
    icao24: str,
    start_time: int,
    config: AeroConformConfig | None = None,
) -> list[WindowedTrajectory]:
    """Slice a trajectory into fixed-length windows with overlap.

    Windows shorter than seq_len are zero-padded with a mask.
    Stride is seq_len // window_stride_divisor for overlap.

    Args:
        trajectory: Delta-encoded, normalized trajectory of shape (T, 6).
        icao24: ICAO24 address for metadata.
        start_time: Unix timestamp of the first state.
        config: Configuration for seq_len and stride.

    Returns:
        List of windowed trajectory dicts.
    """
    cfg = config or AeroConformConfig()
    seq_len = cfg.seq_len
    stride = cfg.window_stride
    t_len = trajectory.shape[0]

    windows: list[WindowedTrajectory] = []

    start_indices = list(range(0, max(1, t_len - seq_len + 1), stride))
    # Always include a window starting at the end if trajectory is long enough
    if t_len > seq_len and start_indices[-1] + seq_len < t_len:
        start_indices.append(t_len - seq_len)

    for start_idx in start_indices:
        end_idx = min(start_idx + seq_len, t_len)
        chunk = trajectory[start_idx:end_idx]

        # Pad if necessary
        if chunk.shape[0] < seq_len:
            padded = np.zeros((seq_len, trajectory.shape[1]), dtype=trajectory.dtype)
            mask = np.zeros(seq_len, dtype=bool)
            padded[: chunk.shape[0]] = chunk
            mask[: chunk.shape[0]] = True
        else:
            padded = chunk
            mask = np.ones(seq_len, dtype=bool)

        windows.append(
            WindowedTrajectory(
                data=padded,
                mask=mask,
                icao24=icao24,
                start_time=start_time + start_idx,
            )
        )

    return windows


def save_norm_stats(stats: NormStats, filepath: Path) -> None:
    """Save normalization statistics to a JSON file.

    Args:
        stats: Normalization statistics to save.
        filepath: Path to the output JSON file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(dict(stats), f, indent=2)
    logger.info("norm_stats_saved", filepath=str(filepath))


def load_norm_stats(filepath: Path) -> NormStats:
    """Load normalization statistics from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        NormStats dict with median and IQR per feature.

    Raises:
        FileNotFoundError: If the stats file does not exist.
    """
    with open(filepath) as f:
        data = json.load(f)
    return NormStats(median=data["median"], iqr=data["iqr"])


def preprocess_pipeline(
    states_df: pd.DataFrame,
    config: AeroConformConfig | None = None,
    norm_stats: NormStats | None = None,
) -> tuple[list[WindowedTrajectory], NormStats]:
    """Run the full preprocessing pipeline on raw state vectors.

    Extracts trajectories, delta-encodes, normalizes, and windows.

    Args:
        states_df: Raw state vectors with icao24 and timestamp columns.
        config: Configuration.
        norm_stats: Pre-computed normalization stats (computed from data if None).

    Returns:
        Tuple of (list of windowed trajectories, normalization stats).
    """
    cfg = config or AeroConformConfig()

    # Step 1: Extract trajectories
    trajectories = extract_trajectories(states_df, cfg)

    # Step 2: Delta encode
    delta_trajs = {
        icao: delta_encode(traj)
        for icao, traj in trajectories.items()
        if len(traj) > 1
    }

    # Step 3: Compute or use provided norm stats
    if norm_stats is None:
        if not delta_trajs:
            norm_stats = NormStats(median=[0.0] * 6, iqr=[1.0] * 6)
        else:
            norm_stats = compute_norm_stats(list(delta_trajs.values()))

    # Step 4: Normalize and window
    all_windows: list[WindowedTrajectory] = []
    for icao, delta_traj in delta_trajs.items():
        normed = normalize(delta_traj, norm_stats)
        windows = window_trajectory(normed, icao, start_time=0, config=cfg)
        all_windows.extend(windows)

    logger.info(
        "preprocessing_complete",
        trajectories=len(delta_trajs),
        windows=len(all_windows),
    )
    return all_windows, norm_stats
