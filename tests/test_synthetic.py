"""Tests for synthetic anomaly injection."""

from __future__ import annotations

import numpy as np
from src.data.schemas import Flight
from src.data.synthetic import (
    inject_ghost,
    inject_gps_drift,
    inject_impossible_maneuver,
    inject_position_spoofing,
)


def test_spoofing_offset(synthetic_flight: Flight) -> None:
    """Spoofing shifts position by expected offset."""
    result = inject_position_spoofing(synthetic_flight, offset_lat=1.0, offset_lon=0.5)
    assert result.is_anomalous

    original_lat = synthetic_flight.features[:, 0]
    spoofed_lat = result.flight.features[:, 0]
    np.testing.assert_allclose(spoofed_lat - original_lat, 1.0, atol=1e-5)

    assert len(result.labels) == 1
    assert result.labels[0].anomaly_type == "spoofing"


def test_spoofing_preserves_icao(synthetic_flight: Flight) -> None:
    """Spoofed flight has same icao24 as original."""
    result = inject_position_spoofing(synthetic_flight)
    assert result.flight.icao24 == synthetic_flight.icao24


def test_ghost_shape() -> None:
    """Ghost injection produces valid flight."""
    result = inject_ghost(num_steps=80, seed=42)
    assert result.is_anomalous
    assert result.flight.num_steps == 80
    assert result.flight.features.shape == (80, 8)
    assert result.labels[0].anomaly_type == "ghost"
    assert "ghost_" in result.flight.icao24


def test_ghost_in_bbox() -> None:
    """Ghost trajectory stays near bounding box."""
    bbox = (6.5, 44.0, 13.5, 47.0)
    result = inject_ghost(bbox=bbox, num_steps=50, seed=42)
    lats = result.flight.features[:, 0]
    lons = result.flight.features[:, 1]
    # Should at least start within bbox
    assert bbox[1] <= lats[0] <= bbox[3]
    assert bbox[0] <= lons[0] <= bbox[2]


def test_gps_drift_gradual(synthetic_flight: Flight) -> None:
    """GPS drift gradually increases position offset."""
    result = inject_gps_drift(synthetic_flight, start_fraction=0.3)
    assert result.is_anomalous

    start_idx = result.labels[0].start_idx
    # Before drift: should be identical
    np.testing.assert_array_equal(result.flight.features[:start_idx], synthetic_flight.features[:start_idx])
    # After drift: position should be shifted
    lat_diff = result.flight.features[-1, 0] - synthetic_flight.features[-1, 0]
    assert abs(lat_diff) > 0.001  # Drift should be noticeable


def test_impossible_maneuver_detectable(synthetic_flight: Flight) -> None:
    """Impossible maneuver creates detectable change."""
    result = inject_impossible_maneuver(synthetic_flight, turn_degrees=90.0)
    assert result.is_anomalous

    label = result.labels[0]
    assert label.anomaly_type == "impossible_maneuver"

    # Heading components should change at maneuver point
    idx = label.start_idx
    original_heading = np.arctan2(synthetic_flight.features[idx, 4], synthetic_flight.features[idx, 5])
    modified_heading = np.arctan2(result.flight.features[idx, 4], result.flight.features[idx, 5])
    heading_diff = abs(modified_heading - original_heading)
    # Allow for wrapping
    heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
    assert heading_diff > np.radians(45)  # Should rotate significantly

    # Vertical rate should be extreme
    assert abs(result.flight.features[idx, 6]) > 20.0
