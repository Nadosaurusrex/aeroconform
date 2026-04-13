"""Geodesic utility functions for aviation calculations.

All functions operate in standard aviation units:
- Distances in nautical miles (nm)
- Bearings in degrees (0-360, clockwise from true north)
- Speeds in knots (kts)
- Altitudes in feet (ft)
"""

from __future__ import annotations

import math

# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in nautical miles.

    Uses the haversine formula for numerical stability.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in nautical miles.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_NM * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the initial bearing from point 1 to point 2 in degrees.

    Returns a value in [0, 360).

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Initial bearing in degrees (0-360, clockwise from north).
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)

    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def closing_speed_kts(
    lat1: float,
    lon1: float,
    vel1: float,
    hdg1: float,
    lat2: float,
    lon2: float,
    vel2: float,
    hdg2: float,
) -> float:
    """Compute the rate of change of distance between two aircraft in knots.

    Negative values indicate converging (closing) aircraft.
    Positive values indicate diverging aircraft.

    Uses velocity vector projections along the line connecting the two aircraft.

    Args:
        lat1: Latitude of aircraft 1 in degrees.
        lon1: Longitude of aircraft 1 in degrees.
        vel1: Ground speed of aircraft 1 in knots.
        hdg1: True track (heading) of aircraft 1 in degrees.
        lat2: Latitude of aircraft 2 in degrees.
        lon2: Longitude of aircraft 2 in degrees.
        vel2: Ground speed of aircraft 2 in knots.
        hdg2: True track (heading) of aircraft 2 in degrees.

    Returns:
        Closing speed in knots (negative = converging, positive = diverging).
    """
    bearing_1_to_2 = bearing_deg(lat1, lon1, lat2, lon2)
    bearing_2_to_1 = bearing_deg(lat2, lon2, lat1, lon1)

    # Project velocity of aircraft 1 along the line from 1 to 2
    # Positive component means moving toward aircraft 2
    v1_toward = vel1 * math.cos(math.radians(hdg1 - bearing_1_to_2))

    # Project velocity of aircraft 2 along the line from 2 to 1
    # Positive component means moving toward aircraft 1
    v2_toward = vel2 * math.cos(math.radians(hdg2 - bearing_2_to_1))

    # Closing speed: rate at which distance decreases
    # Both aircraft moving toward each other increases closing speed
    # Return negative for converging (convention: closing = negative)
    return -(v1_toward + v2_toward)


def point_in_bbox(
    lat: float, lon: float, bbox: tuple[float, float, float, float]
) -> bool:
    """Check if a point is within a bounding box.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        bbox: Bounding box as (min_lat, max_lat, min_lon, max_lon).

    Returns:
        True if the point is within the bounding box.
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def angular_difference(angle1: float, angle2: float) -> float:
    """Compute the shortest angular difference between two angles in degrees.

    Returns a value in [-180, 180].

    Args:
        angle1: First angle in degrees.
        angle2: Second angle in degrees.

    Returns:
        Shortest angular difference in degrees (positive = clockwise).
    """
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff


def ft_to_nm(ft: float) -> float:
    """Convert feet to nautical miles.

    Args:
        ft: Distance in feet.

    Returns:
        Distance in nautical miles.
    """
    return ft / 6076.12


def nm_to_ft(nm: float) -> float:
    """Convert nautical miles to feet.

    Args:
        nm: Distance in nautical miles.

    Returns:
        Distance in feet.
    """
    return nm * 6076.12
