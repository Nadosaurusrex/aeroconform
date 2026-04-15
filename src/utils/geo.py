"""Geodesic utility functions for graph construction.

All functions use metric units (km, m/s, radians) per ARCHITECTURE.md.
Vectorized with numpy for performance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.utils.constants import EARTH_RADIUS_KM


def haversine_km(
    lat1: npt.NDArray[np.float64],
    lon1: npt.NDArray[np.float64],
    lat2: npt.NDArray[np.float64],
    lon2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute great-circle distance in kilometers.

    Args:
        lat1, lon1: First point(s) in decimal degrees.
        lat2, lon2: Second point(s) in decimal degrees.

    Returns:
        Distance(s) in kilometers.
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return EARTH_RADIUS_KM * c


def bearing_rad(
    lat1: npt.NDArray[np.float64],
    lon1: npt.NDArray[np.float64],
    lat2: npt.NDArray[np.float64],
    lon2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute bearing from point 1 to point 2 in radians.

    Args:
        lat1, lon1: Origin point(s) in decimal degrees.
        lat2, lon2: Destination point(s) in decimal degrees.

    Returns:
        Bearing(s) in radians [0, 2*pi).
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlon = lon2_r - lon1_r

    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)

    return np.arctan2(x, y) % (2 * np.pi)


def closing_speed_mps(
    vel1: npt.NDArray[np.float64],
    heading1_rad: npt.NDArray[np.float64],
    vel2: npt.NDArray[np.float64],
    heading2_rad: npt.NDArray[np.float64],
    relative_bearing_rad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute closing speed between two aircraft in m/s.

    Positive = converging, negative = diverging.

    Args:
        vel1, vel2: Ground speeds in m/s.
        heading1_rad, heading2_rad: Headings in radians.
        relative_bearing_rad: Bearing from aircraft 1 to aircraft 2.

    Returns:
        Closing speed in m/s.
    """
    # Velocity components toward each other
    v1_toward = vel1 * np.cos(heading1_rad - relative_bearing_rad)
    v2_toward = vel2 * np.cos(heading2_rad - (relative_bearing_rad + np.pi))

    return v1_toward + v2_toward


def time_to_cpa_seconds(
    dist_km: npt.NDArray[np.float64],
    close_speed_mps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Estimate time to closest point of approach in seconds.

    Simple linear estimate: distance / closing_speed.

    Args:
        dist_km: Current distance in km.
        close_speed_mps: Closing speed in m/s (positive = converging).

    Returns:
        Estimated time to CPA in seconds. Inf if diverging.
    """
    dist_m = dist_km * 1000.0
    # Avoid division by zero: if closing speed <= 0, set to inf
    result = np.full_like(dist_m, np.inf)
    converging = close_speed_mps > 0.1  # Small threshold
    result[converging] = dist_m[converging] / close_speed_mps[converging]
    return np.clip(result, 0, 3600)  # Cap at 1 hour
