"""Synthetic anomaly injection for evaluation of AeroConform.

Implements 5 realistic attack scenarios for evaluating anomaly detection:
GPS spoofing, position jumps, ghost aircraft, replay attacks, and
altitude manipulation.
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class AnomalyInjector:
    """Inject realistic anomalies into clean trajectories for evaluation.

    All methods operate on absolute state vectors of shape (T, 6)
    with features [lat, lon, alt, vel, hdg, vrate]. They return
    the modified trajectory and a binary label array indicating
    which timesteps are anomalous.

    Args:
        rng: NumPy random number generator for reproducibility.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def inject_gps_spoofing(
        self,
        traj: np.ndarray,
        start_idx: int | None = None,
        offset_nm: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradually shift position by offset_nm starting at start_idx.

        Simulates GPS spoofing that slowly pulls aircraft off course.
        The offset increases linearly from start_idx to the end.

        Args:
            traj: Clean trajectory of shape (T, 6).
            start_idx: Timestep at which spoofing begins (random if None).
            offset_nm: Maximum position offset in nautical miles.

        Returns:
            Tuple of (modified trajectory, binary labels).
        """
        t_len = traj.shape[0]
        if start_idx is None:
            start_idx = self.rng.integers(t_len // 4, t_len // 2)

        modified = traj.copy()
        labels = np.zeros(t_len, dtype=int)

        # Random direction for the spoofing drift
        drift_bearing = self.rng.uniform(0, 360)
        drift_lat = np.cos(np.radians(drift_bearing))
        drift_lon = np.sin(np.radians(drift_bearing))

        for t in range(start_idx, t_len):
            progress = (t - start_idx) / max(1, t_len - start_idx - 1)
            offset = offset_nm * progress

            # Convert nm offset to degrees (approximate)
            # 1 nm ≈ 1/60 degree latitude
            lat_offset = offset * drift_lat / 60.0
            lon_offset = offset * drift_lon / (60.0 * np.cos(np.radians(modified[t, 0])))

            modified[t, 0] += lat_offset
            modified[t, 1] += lon_offset
            labels[t] = 1

        logger.debug(
            "gps_spoofing_injected",
            start_idx=start_idx,
            offset_nm=offset_nm,
        )
        return modified, labels

    def inject_position_jump(
        self,
        traj: np.ndarray,
        idx: int | None = None,
        jump_nm: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Instantaneous position teleportation at a single timestep.

        Simulates crude spoofing with an abrupt position change.

        Args:
            traj: Clean trajectory of shape (T, 6).
            idx: Timestep of the jump (random if None).
            jump_nm: Jump distance in nautical miles.

        Returns:
            Tuple of (modified trajectory, binary labels).
        """
        t_len = traj.shape[0]
        if idx is None:
            idx = self.rng.integers(t_len // 4, 3 * t_len // 4)

        modified = traj.copy()
        labels = np.zeros(t_len, dtype=int)

        # Random direction for the jump
        bearing = self.rng.uniform(0, 360)
        lat_offset = jump_nm * np.cos(np.radians(bearing)) / 60.0
        lon_offset = jump_nm * np.sin(np.radians(bearing)) / (
            60.0 * np.cos(np.radians(modified[idx, 0]))
        )

        # Apply jump to this timestep and all subsequent
        modified[idx:, 0] += lat_offset
        modified[idx:, 1] += lon_offset
        labels[idx:] = 1

        logger.debug("position_jump_injected", idx=idx, jump_nm=jump_nm)
        return modified, labels

    def inject_ghost_aircraft(
        self,
        clean_traj: np.ndarray,
        noise_scale: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic trajectory that looks plausible but fake.

        Creates a ghost aircraft by interpolating between random waypoints
        with small noise added for realism.

        Args:
            clean_traj: Reference trajectory for shape and altitude range.
            noise_scale: Scale of random noise added to the ghost trajectory.

        Returns:
            Tuple of (ghost trajectory, all-ones labels).
        """
        t_len = clean_traj.shape[0]

        # Generate random waypoints
        n_waypoints = max(3, t_len // 50)
        waypoint_indices = np.linspace(0, t_len - 1, n_waypoints, dtype=int)

        # Random positions near the reference trajectory
        waypoints = np.zeros((n_waypoints, 6))
        for i, wp_idx in enumerate(waypoint_indices):
            ref = clean_traj[wp_idx]
            waypoints[i, 0] = ref[0] + self.rng.normal(0, 0.5)  # lat
            waypoints[i, 1] = ref[1] + self.rng.normal(0, 0.5)  # lon
            waypoints[i, 2] = self.rng.uniform(10000, 40000)     # alt
            waypoints[i, 3] = self.rng.uniform(200, 500)         # vel
            waypoints[i, 4] = self.rng.uniform(0, 360)           # hdg
            waypoints[i, 5] = self.rng.normal(0, 100)            # vrate

        # Interpolate between waypoints
        ghost = np.zeros_like(clean_traj)
        for col in range(6):
            ghost[:, col] = np.interp(
                np.arange(t_len),
                waypoint_indices,
                waypoints[:, col],
            )

        # Add noise
        ghost += self.rng.normal(0, noise_scale, ghost.shape)

        # Ensure heading wraps properly
        ghost[:, 4] = ghost[:, 4] % 360

        labels = np.ones(t_len, dtype=int)

        logger.debug("ghost_aircraft_injected", t_len=t_len)
        return ghost, labels

    def inject_replay_attack(
        self,
        traj: np.ndarray,
        replay_traj: np.ndarray,
        start_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replace part of a trajectory with a previously recorded one.

        Simulates a replay attack by splicing in a different aircraft's
        trajectory, shifted to start from the current position.

        Args:
            traj: Target trajectory of shape (T, 6).
            replay_traj: Source trajectory to replay from, shape (T2, 6).
            start_idx: Timestep at which replay begins (random if None).

        Returns:
            Tuple of (modified trajectory, binary labels).
        """
        t_len = traj.shape[0]
        if start_idx is None:
            start_idx = self.rng.integers(t_len // 4, t_len // 2)

        modified = traj.copy()
        labels = np.zeros(t_len, dtype=int)

        # Compute position offset to align replay at start point
        replay_start = self.rng.integers(0, max(1, len(replay_traj) - (t_len - start_idx)))
        lat_offset = traj[start_idx, 0] - replay_traj[replay_start, 0]
        lon_offset = traj[start_idx, 1] - replay_traj[replay_start, 1]

        # Splice in the replay trajectory
        replay_len = min(t_len - start_idx, len(replay_traj) - replay_start)
        replay_segment = replay_traj[replay_start : replay_start + replay_len].copy()
        replay_segment[:, 0] += lat_offset
        replay_segment[:, 1] += lon_offset

        modified[start_idx : start_idx + replay_len] = replay_segment
        labels[start_idx : start_idx + replay_len] = 1

        logger.debug(
            "replay_attack_injected",
            start_idx=start_idx,
            replay_len=replay_len,
        )
        return modified, labels

    def inject_altitude_manipulation(
        self,
        traj: np.ndarray,
        start_idx: int | None = None,
        alt_offset_ft: float = 2000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradually manipulate reported altitude.

        Dangerous because it could cause loss of separation in
        altitude-based conflict detection.

        Args:
            traj: Clean trajectory of shape (T, 6).
            start_idx: Timestep at which manipulation begins (random if None).
            alt_offset_ft: Maximum altitude offset in feet.

        Returns:
            Tuple of (modified trajectory, binary labels).
        """
        t_len = traj.shape[0]
        if start_idx is None:
            start_idx = self.rng.integers(t_len // 4, t_len // 2)

        modified = traj.copy()
        labels = np.zeros(t_len, dtype=int)

        direction = self.rng.choice([-1.0, 1.0])

        for t in range(start_idx, t_len):
            progress = (t - start_idx) / max(1, t_len - start_idx - 1)
            offset = direction * alt_offset_ft * progress
            modified[t, 2] += offset
            labels[t] = 1

        logger.debug(
            "altitude_manipulation_injected",
            start_idx=start_idx,
            alt_offset_ft=alt_offset_ft,
        )
        return modified, labels


def generate_evaluation_set(
    clean_trajectories: list[np.ndarray],
    anomalies_per_type: int = 200,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Generate a mixed evaluation set with clean and anomalous trajectories.

    Creates a balanced set with clean trajectories and injected anomalies
    of each type for evaluation.

    Args:
        clean_trajectories: List of clean trajectory arrays.
        anomalies_per_type: Number of anomalous trajectories per type.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trajectories, per-timestep labels, anomaly type strings).
    """
    rng = np.random.default_rng(seed)
    injector = AnomalyInjector(rng=rng)

    all_trajs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_types: list[str] = []

    # Add clean trajectories
    for traj in clean_trajectories:
        all_trajs.append(traj)
        all_labels.append(np.zeros(len(traj), dtype=int))
        all_types.append("clean")

    n_clean = len(clean_trajectories)
    anomaly_methods = [
        ("gps_spoofing", injector.inject_gps_spoofing),
        ("position_jump", injector.inject_position_jump),
        ("altitude_manipulation", injector.inject_altitude_manipulation),
    ]

    for anomaly_name, method in anomaly_methods:
        for _i in range(min(anomalies_per_type, n_clean)):
            idx = rng.integers(0, n_clean)
            modified, labels = method(clean_trajectories[idx])
            all_trajs.append(modified)
            all_labels.append(labels)
            all_types.append(anomaly_name)

    # Ghost aircraft
    for _i in range(min(anomalies_per_type, n_clean)):
        idx = rng.integers(0, n_clean)
        ghost, labels = injector.inject_ghost_aircraft(clean_trajectories[idx])
        all_trajs.append(ghost)
        all_labels.append(labels)
        all_types.append("ghost_aircraft")

    # Replay attack (needs two trajectories)
    for _i in range(min(anomalies_per_type, n_clean)):
        idx1 = rng.integers(0, n_clean)
        idx2 = rng.integers(0, n_clean)
        modified, labels = injector.inject_replay_attack(
            clean_trajectories[idx1], clean_trajectories[idx2]
        )
        all_trajs.append(modified)
        all_labels.append(labels)
        all_types.append("replay_attack")

    logger.info(
        "evaluation_set_generated",
        clean=n_clean,
        anomalous=len(all_trajs) - n_clean,
    )
    return all_trajs, all_labels, all_types
