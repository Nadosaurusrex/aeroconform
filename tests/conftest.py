"""Shared test fixtures for AeroConform tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeroconform.config import AeroConformConfig


@pytest.fixture
def config() -> AeroConformConfig:
    """Provide a default configuration for tests."""
    return AeroConformConfig()


@pytest.fixture
def synthetic_trajectory() -> np.ndarray:
    """Generate a synthetic but realistic aircraft trajectory.

    Simulates straight-and-level flight with gentle turns and altitude changes.
    Returns absolute state vectors: (T, 6) with [lat, lon, alt, vel, hdg, vrate].

    The trajectory starts near Milan (LIMM FIR) and flies northeast.
    """
    rng = np.random.default_rng(42)
    t_steps = 200

    # Starting position near Milan
    lat = 45.5
    lon = 9.0
    alt = 35000.0  # feet
    vel = 450.0  # knots
    hdg = 45.0  # northeast
    vrate = 0.0  # level flight

    trajectory = np.zeros((t_steps, 6))
    for t in range(t_steps):
        # Small random perturbations for realism
        hdg += rng.normal(0, 0.5)  # slight heading changes
        hdg = hdg % 360
        vel += rng.normal(0, 1.0)
        vel = np.clip(vel, 400, 500)
        vrate = rng.normal(0, 50)  # small vertical rate fluctuations
        alt += vrate / 60  # vrate is ft/min, timestep is ~1s so /60
        alt = np.clip(alt, 30000, 40000)

        # Approximate position update (simplified, ~1 second timestep)
        # 1 knot = 1 nm/hr, so in 1 second = vel/3600 nm
        dist_nm = vel / 3600
        lat += dist_nm * np.cos(np.radians(hdg)) / 60  # 1 nm ≈ 1/60 degree lat
        lon += dist_nm * np.sin(np.radians(hdg)) / (60 * np.cos(np.radians(lat)))

        trajectory[t] = [lat, lon, alt, vel, hdg, vrate]

    return trajectory


@pytest.fixture
def synthetic_airspace_snapshot() -> pd.DataFrame:
    """Generate a snapshot with 20 aircraft at various positions/altitudes.

    Returns a DataFrame matching the OpenSky state vector format.
    """
    rng = np.random.default_rng(123)
    n_aircraft = 20

    data = {
        "icao24": [f"{i:06x}" for i in range(n_aircraft)],
        "callsign": [f"TST{i:03d}" for i in range(n_aircraft)],
        "origin_country": ["Italy"] * n_aircraft,
        "timestamp": [1700000000] * n_aircraft,
        "latitude": rng.uniform(44.0, 47.0, n_aircraft),
        "longitude": rng.uniform(7.0, 13.0, n_aircraft),
        "baro_altitude": rng.uniform(5000, 40000, n_aircraft),
        "on_ground": [False] * n_aircraft,
        "velocity": rng.uniform(200, 500, n_aircraft),
        "true_track": rng.uniform(0, 360, n_aircraft),
        "vertical_rate": rng.normal(0, 200, n_aircraft),
        "geo_altitude": rng.uniform(5000, 40000, n_aircraft),
    }

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_trajectories(synthetic_trajectory: np.ndarray) -> list[np.ndarray]:
    """Generate multiple synthetic trajectories with slight variations."""
    rng = np.random.default_rng(99)
    trajectories = [synthetic_trajectory]
    for i in range(9):
        # Offset the base trajectory slightly
        offset = rng.normal(0, 0.01, size=synthetic_trajectory.shape)
        offset[:, 2] *= 100  # altitude offsets in feet
        offset[:, 3] *= 5  # velocity offsets in knots
        offset[:, 4] *= 2  # heading offsets in degrees
        offset[:, 5] *= 10  # vrate offsets
        trajectories.append(synthetic_trajectory + offset)
    return trajectories
