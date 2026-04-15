"""Synthetic anomaly injection for evaluation.

Generates 4 attack types per ARCHITECTURE.md section 6:
1. Position spoofing: duplicate icao24 at different position
2. Ghost injection: plausible but fabricated trajectory
3. GPS drift: gradual position shift
4. Impossible maneuver: sudden extreme maneuver
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.data.schemas import AnomalyLabel, Flight, LabeledFlight


def inject_position_spoofing(
    flight: Flight,
    *,
    offset_lat: float = 0.5,
    offset_lon: float = 0.5,
) -> LabeledFlight:
    """Create a spoofed copy of a flight at a different position.

    Clones the trajectory but shifts lat/lon. The graph layer should
    detect spatial inconsistency between the real and spoofed aircraft.

    Args:
        flight: Original flight to spoof.
        offset_lat: Latitude offset in degrees.
        offset_lon: Longitude offset in degrees.

    Returns:
        LabeledFlight with spoofed trajectory and labels.
    """
    spoofed_features = flight.features.copy()
    spoofed_features[:, 0] += offset_lat  # latitude
    spoofed_features[:, 1] += offset_lon  # longitude

    spoofed = Flight(
        icao24=flight.icao24,  # Same icao24 is the spoofing signature
        callsign=flight.callsign,
        timestamps=flight.timestamps.copy(),
        features=spoofed_features,
    )

    label = AnomalyLabel(
        anomaly_type="spoofing",
        start_idx=0,
        end_idx=flight.num_steps,
        metadata={"offset_lat": offset_lat, "offset_lon": offset_lon},
    )

    return LabeledFlight(flight=spoofed, labels=[label], is_anomalous=True)


def inject_ghost(
    *,
    bbox: tuple[float, float, float, float] = (6.5, 44.0, 13.5, 47.0),
    num_steps: int = 100,
    dt: int = 10,
    seed: int | None = None,
) -> LabeledFlight:
    """Generate a ghost aircraft with plausible but fabricated trajectory.

    Creates a realistic-looking trajectory with smooth dynamics but no
    corresponding real aircraft. The foundation model should detect
    subtle kinematic inconsistencies.

    Args:
        bbox: (west, south, east, north) geographic bounds.
        num_steps: Number of observations to generate.
        dt: Time between observations in seconds.
        seed: Random seed for reproducibility.

    Returns:
        LabeledFlight with ghost trajectory.
    """
    rng = np.random.default_rng(seed)

    west, south, east, north = bbox
    lat = rng.uniform(south + 0.5, north - 0.5)
    lon = rng.uniform(west + 0.5, east - 0.5)
    alt = rng.uniform(5000, 12000)  # meters
    vel = rng.uniform(150, 280)  # m/s
    heading_rad = rng.uniform(0, 2 * np.pi)
    vrate = 0.0

    features = np.zeros((num_steps, 8), dtype=np.float32)
    timestamps = np.arange(num_steps, dtype=np.int64) * dt + int(1e9)

    for t in range(num_steps):
        # Add small perturbations that may not follow real flight dynamics
        heading_rad += rng.normal(0, 0.02)
        vel += rng.normal(0, 1.0)
        alt += rng.normal(0, 5.0)
        vrate = rng.normal(0, 2.0)

        lat += vel * np.cos(heading_rad) * dt / 111320.0
        lon += vel * np.sin(heading_rad) * dt / (111320.0 * np.cos(np.radians(lat)))

        features[t] = [
            lat, lon, alt, vel,
            np.sin(heading_rad), np.cos(heading_rad),
            vrate, 0.0,
        ]

    icao24 = f"ghost_{rng.integers(0, 0xFFFFFF):06x}"
    flight = Flight(
        icao24=icao24,
        callsign="GHOST",
        timestamps=timestamps,
        features=features,
    )

    label = AnomalyLabel(
        anomaly_type="ghost",
        start_idx=0,
        end_idx=num_steps,
    )

    return LabeledFlight(flight=flight, labels=[label], is_anomalous=True)


def inject_gps_drift(
    flight: Flight,
    *,
    start_fraction: float = 0.3,
    drift_rate_deg_per_min: float = 0.01,
) -> LabeledFlight:
    """Gradually shift an aircraft's reported position.

    Applies a linear drift to lat/lon starting at a fraction of the flight.
    The model should detect the trajectory deviating from learned dynamics.

    Args:
        flight: Original flight to modify.
        start_fraction: Fraction of flight at which drift begins.
        drift_rate_deg_per_min: Position drift rate in degrees per minute.

    Returns:
        LabeledFlight with drifted trajectory and labels.
    """
    drifted_features = flight.features.copy()
    start_idx = int(flight.num_steps * start_fraction)

    for t in range(start_idx, flight.num_steps):
        elapsed_minutes = (flight.timestamps[t] - flight.timestamps[start_idx]) / 60.0
        drift = drift_rate_deg_per_min * elapsed_minutes
        drifted_features[t, 0] += drift  # latitude drift
        drifted_features[t, 1] += drift * 0.7  # longitude drift (slightly less)

    drifted = Flight(
        icao24=flight.icao24,
        callsign=flight.callsign,
        timestamps=flight.timestamps.copy(),
        features=drifted_features,
    )

    label = AnomalyLabel(
        anomaly_type="gps_drift",
        start_idx=start_idx,
        end_idx=flight.num_steps,
        metadata={
            "drift_rate_deg_per_min": drift_rate_deg_per_min,
            "start_fraction": start_fraction,
        },
    )

    return LabeledFlight(flight=drifted, labels=[label], is_anomalous=True)


def inject_impossible_maneuver(
    flight: Flight,
    *,
    maneuver_idx: int | None = None,
    turn_degrees: float = 90.0,
    climb_rate_mps: float = 25.4,  # ~5000 ft/min
) -> LabeledFlight:
    """Inject a sudden impossible maneuver into a trajectory.

    Applies a sudden 90-degree turn and/or extreme climb rate.
    Should trigger immediate high non-conformity score.

    Args:
        flight: Original flight to modify.
        maneuver_idx: Timestep at which to inject maneuver. Defaults to middle.
        turn_degrees: Heading change in degrees.
        climb_rate_mps: Vertical rate in m/s (25.4 = ~5000 ft/min).

    Returns:
        LabeledFlight with impossible maneuver injected.
    """
    if maneuver_idx is None:
        maneuver_idx = flight.num_steps // 2

    modified_features = flight.features.copy()
    turn_rad = np.radians(turn_degrees)

    # Apply sudden heading change via sin/cos
    for t in range(maneuver_idx, min(maneuver_idx + 5, flight.num_steps)):
        old_sin = modified_features[t, 4]
        old_cos = modified_features[t, 5]
        # Rotate heading
        new_sin = old_sin * np.cos(turn_rad) + old_cos * np.sin(turn_rad)
        new_cos = old_cos * np.cos(turn_rad) - old_sin * np.sin(turn_rad)
        modified_features[t, 4] = new_sin
        modified_features[t, 5] = new_cos
        # Extreme climb rate
        modified_features[t, 6] = climb_rate_mps

    modified = Flight(
        icao24=flight.icao24,
        callsign=flight.callsign,
        timestamps=flight.timestamps.copy(),
        features=modified_features,
    )

    label = AnomalyLabel(
        anomaly_type="impossible_maneuver",
        start_idx=maneuver_idx,
        end_idx=min(maneuver_idx + 5, flight.num_steps),
        metadata={"turn_degrees": turn_degrees, "climb_rate_mps": climb_rate_mps},
    )

    return LabeledFlight(flight=modified, labels=[label], is_anomalous=True)
