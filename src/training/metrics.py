"""Evaluation metrics for trajectory prediction.

ADE (Average Displacement Error) and FDE (Final Displacement Error)
computed in geographic coordinates after denormalization.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_ade(
    predictions: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32],
) -> float:
    """Compute Average Displacement Error in meters.

    Uses Euclidean distance on lat/lon (first 2 features) converted to
    approximate meters using the lat/lon degree-to-meter conversion.

    Args:
        predictions: (num_steps, 8) predicted absolute states.
        targets: (num_steps, 8) ground truth absolute states.

    Returns:
        ADE in meters.
    """
    lat_diff = predictions[:, 0] - targets[:, 0]
    lon_diff = predictions[:, 1] - targets[:, 1]

    # Approximate degree-to-meter conversion
    lat_m = lat_diff * 111_320.0
    avg_lat = np.mean(targets[:, 0])
    lon_m = lon_diff * 111_320.0 * np.cos(np.radians(avg_lat))

    distances = np.sqrt(lat_m**2 + lon_m**2)
    return float(np.mean(distances))


def compute_fde(
    predictions: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32],
) -> float:
    """Compute Final Displacement Error in meters.

    Distance between last predicted and last target position.

    Args:
        predictions: (num_steps, 8) predicted absolute states.
        targets: (num_steps, 8) ground truth absolute states.

    Returns:
        FDE in meters.
    """
    lat_diff = predictions[-1, 0] - targets[-1, 0]
    lon_diff = predictions[-1, 1] - targets[-1, 1]

    lat_m = lat_diff * 111_320.0
    lon_m = lon_diff * 111_320.0 * np.cos(np.radians(targets[-1, 0]))

    return float(np.sqrt(lat_m**2 + lon_m**2))
