"""Shared test fixtures for AeroConform."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from src.data.schemas import Flight, NormStats


@pytest.fixture()
def synthetic_state_vectors() -> pl.DataFrame:
    """Generate a synthetic state vectors DataFrame mimicking Trino output.

    Creates 3 aircraft with 50 observations each in the LIMM FIR.
    """
    rng = np.random.default_rng(42)
    rows = []

    for i, icao in enumerate(["aabbcc", "ddeeff", "112233"]):
        base_lat = 45.0 + i * 0.5
        base_lon = 9.0 + i * 0.5
        heading = rng.uniform(0, 360)
        heading_rad = math.radians(heading)

        for t in range(50):
            lat = base_lat + t * 0.01 * math.cos(heading_rad)
            lon = base_lon + t * 0.01 * math.sin(heading_rad)
            rows.append(
                {
                    "icao24": icao,
                    "time": 1700000000 + t * 15,
                    "callsign": f"TST{i:03d}",
                    "lat": lat,
                    "lon": lon,
                    "baroaltitude": 10000.0 + rng.normal(0, 10),
                    "velocity": 250.0 + rng.normal(0, 5),
                    "heading": heading + rng.normal(0, 2),
                    "vertrate": rng.normal(0, 1),
                    "onground": False,
                    "geoaltitude": 10050.0,
                    "squawk": "7000",
                }
            )

    return pl.DataFrame(rows)


@pytest.fixture()
def synthetic_flight() -> Flight:
    """Generate a single synthetic flight with 100 observations."""
    rng = np.random.default_rng(42)
    num_steps = 100

    heading_rad = math.radians(45)
    features = np.zeros((num_steps, 8), dtype=np.float32)
    timestamps = np.arange(num_steps, dtype=np.int64) * 10 + 1700000000

    lat, lon = 45.5, 9.0
    alt, vel = 10000.0, 250.0
    vrate = 0.0

    for t in range(num_steps):
        heading_rad += rng.normal(0, 0.01)
        vel += rng.normal(0, 0.5)
        alt += vrate * 10
        vrate = rng.normal(0, 0.3)

        lat += vel * math.cos(heading_rad) * 10 / 111320.0
        lon += vel * math.sin(heading_rad) * 10 / (111320.0 * math.cos(math.radians(lat)))

        features[t] = [
            lat,
            lon,
            alt,
            vel,
            math.sin(heading_rad),
            math.cos(heading_rad),
            vrate,
            0.0,
        ]

    return Flight(
        icao24="aabbcc",
        callsign="TST001",
        timestamps=timestamps,
        features=features,
    )


@pytest.fixture()
def sample_norm_stats() -> NormStats:
    """Sample normalization stats for testing."""
    return NormStats(
        mean=np.zeros(8, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
    )
