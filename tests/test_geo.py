"""Tests for geodesic utility functions."""

from __future__ import annotations

import math

import pytest

from aeroconform.utils.geo import (
    angular_difference,
    bearing_deg,
    closing_speed_kts,
    ft_to_nm,
    haversine_nm,
    nm_to_ft,
    point_in_bbox,
)


class TestHaversine:
    """Tests for haversine distance calculation."""

    def test_same_point_is_zero(self) -> None:
        """Distance from a point to itself should be zero."""
        assert haversine_nm(45.0, 9.0, 45.0, 9.0) == 0.0

    def test_london_to_paris(self) -> None:
        """London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is ~187 nm."""
        dist = haversine_nm(51.5074, -0.1278, 48.8566, 2.3522)
        assert 185 < dist < 190

    def test_new_york_to_london(self) -> None:
        """JFK (40.6413, -73.7781) to LHR (51.4700, -0.4543) is ~2999 nm."""
        dist = haversine_nm(40.6413, -73.7781, 51.4700, -0.4543)
        assert 2990 < dist < 3010

    def test_symmetry(self) -> None:
        """Distance should be symmetric: d(A,B) == d(B,A)."""
        d1 = haversine_nm(45.0, 9.0, 48.0, 12.0)
        d2 = haversine_nm(48.0, 12.0, 45.0, 9.0)
        assert d1 == pytest.approx(d2, abs=1e-10)

    def test_equator_one_degree(self) -> None:
        """One degree of longitude at the equator is ~60 nm."""
        dist = haversine_nm(0.0, 0.0, 0.0, 1.0)
        assert 59.5 < dist < 60.5

    def test_poles(self) -> None:
        """Distance from North Pole to South Pole is half the circumference (~10800 nm)."""
        dist = haversine_nm(90.0, 0.0, -90.0, 0.0)
        assert 10790 < dist < 10810

    def test_antimeridian(self) -> None:
        """Distance across the antimeridian should work correctly."""
        dist = haversine_nm(0.0, 179.0, 0.0, -179.0)
        # 2 degrees of longitude at equator ≈ 120 nm
        assert 119 < dist < 121


class TestBearing:
    """Tests for bearing calculation."""

    def test_due_north(self) -> None:
        """Bearing from a point to a point due north should be ~0."""
        b = bearing_deg(45.0, 9.0, 46.0, 9.0)
        assert b == pytest.approx(0.0, abs=0.1)

    def test_due_east(self) -> None:
        """Bearing from a point to a point due east should be ~90."""
        b = bearing_deg(0.0, 0.0, 0.0, 1.0)
        assert b == pytest.approx(90.0, abs=0.1)

    def test_due_south(self) -> None:
        """Bearing from a point to a point due south should be ~180."""
        b = bearing_deg(46.0, 9.0, 45.0, 9.0)
        assert b == pytest.approx(180.0, abs=0.1)

    def test_due_west(self) -> None:
        """Bearing from a point to a point due west should be ~270."""
        b = bearing_deg(0.0, 1.0, 0.0, 0.0)
        assert b == pytest.approx(270.0, abs=0.1)

    def test_bearing_range(self) -> None:
        """Bearing should always be in [0, 360)."""
        b = bearing_deg(51.5, -0.1, 48.9, 2.4)
        assert 0 <= b < 360

    def test_reverse_bearings_differ_by_180(self) -> None:
        """Forward and reverse bearings should differ by approximately 180 degrees."""
        b_forward = bearing_deg(45.0, 9.0, 46.0, 10.0)
        b_reverse = bearing_deg(46.0, 10.0, 45.0, 9.0)
        diff = abs(angular_difference(b_forward, b_reverse))
        assert diff == pytest.approx(180.0, abs=1.0)


class TestClosingSpeed:
    """Tests for closing speed calculation."""

    def test_head_on(self) -> None:
        """Two aircraft flying toward each other should have negative closing speed."""
        cs = closing_speed_kts(
            45.0, 9.0, 450.0, 90.0,   # aircraft 1: flying east
            45.0, 10.0, 450.0, 270.0,  # aircraft 2: flying west, east of aircraft 1
        )
        assert cs < -800  # Combined ~900 kts closing speed

    def test_diverging(self) -> None:
        """Two aircraft flying away from each other should have positive closing speed."""
        cs = closing_speed_kts(
            45.0, 9.0, 450.0, 270.0,  # aircraft 1: flying west
            45.0, 10.0, 450.0, 90.0,  # aircraft 2: flying east, east of aircraft 1
        )
        assert cs > 800

    def test_parallel_same_direction(self) -> None:
        """Parallel aircraft at same speed should have near-zero closing speed."""
        cs = closing_speed_kts(
            45.0, 9.0, 450.0, 90.0,
            45.1, 9.0, 450.0, 90.0,
        )
        assert abs(cs) < 50  # Small due to geometry


class TestPointInBbox:
    """Tests for point-in-bounding-box check."""

    def test_inside(self) -> None:
        """Point inside the LIMM FIR bbox."""
        assert point_in_bbox(45.0, 10.0, (43.5, 47.0, 6.5, 14.0))

    def test_outside(self) -> None:
        """Point outside the LIMM FIR bbox."""
        assert not point_in_bbox(42.0, 10.0, (43.5, 47.0, 6.5, 14.0))

    def test_on_boundary(self) -> None:
        """Point on the boundary should be inside."""
        assert point_in_bbox(43.5, 6.5, (43.5, 47.0, 6.5, 14.0))

    def test_outside_lon(self) -> None:
        """Point outside longitude range."""
        assert not point_in_bbox(45.0, 15.0, (43.5, 47.0, 6.5, 14.0))


class TestAngularDifference:
    """Tests for angular difference calculation."""

    def test_zero(self) -> None:
        """Same angle should give zero difference."""
        assert angular_difference(90.0, 90.0) == pytest.approx(0.0)

    def test_positive(self) -> None:
        """Clockwise difference should be positive."""
        assert angular_difference(0.0, 90.0) == pytest.approx(90.0)

    def test_negative(self) -> None:
        """Counter-clockwise difference should be negative."""
        assert angular_difference(90.0, 0.0) == pytest.approx(-90.0)

    def test_wraparound_positive(self) -> None:
        """Wrap-around should take the shortest path."""
        assert angular_difference(350.0, 10.0) == pytest.approx(20.0)

    def test_wraparound_negative(self) -> None:
        """Wrap-around in the other direction."""
        assert angular_difference(10.0, 350.0) == pytest.approx(-20.0)

    def test_opposite(self) -> None:
        """180 degrees should be exactly 180 (or -180)."""
        diff = angular_difference(0.0, 180.0)
        assert abs(diff) == pytest.approx(180.0)


class TestConversions:
    """Tests for unit conversion functions."""

    def test_ft_to_nm(self) -> None:
        """6076.12 feet should be approximately 1 nm."""
        assert ft_to_nm(6076.12) == pytest.approx(1.0, abs=0.01)

    def test_nm_to_ft(self) -> None:
        """1 nm should be approximately 6076.12 feet."""
        assert nm_to_ft(1.0) == pytest.approx(6076.12, abs=0.01)

    def test_roundtrip(self) -> None:
        """Converting back and forth should preserve the value."""
        original = 10000.0
        assert nm_to_ft(ft_to_nm(original)) == pytest.approx(original, abs=0.01)
