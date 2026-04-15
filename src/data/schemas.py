"""Data schemas for AeroConform pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class StateVector:
    """Raw ADS-B state vector from OpenSky."""

    icao24: str
    callsign: str | None
    time_position: int
    latitude: float
    longitude: float
    baro_altitude: float
    velocity: float
    true_track: float
    vertical_rate: float
    on_ground: bool
    geo_altitude: float | None = None
    squawk: str | None = None
    origin_country: str | None = None
    last_contact: int | None = None


@dataclass
class Flight:
    """Segmented flight trajectory."""

    icao24: str
    callsign: str | None
    timestamps: npt.NDArray[np.int64]
    features: npt.NDArray[np.float32]  # Shape: (num_steps, 8)
    origin_country: str | None = None

    @property
    def num_steps(self) -> int:
        """Number of observation timesteps."""
        return len(self.timestamps)

    @property
    def duration_seconds(self) -> int:
        """Total flight duration in seconds."""
        return int(self.timestamps[-1] - self.timestamps[0])


@dataclass
class TrajectoryWindow:
    """Fixed-length window for model input."""

    data: npt.NDArray[np.float32]  # (context_length, 8) - delta-encoded, normalized
    time_gaps: npt.NDArray[np.float32]  # (context_length,) - seconds between observations
    mask: npt.NDArray[np.bool_]  # (context_length,) - True for valid positions
    icao24: str
    seq_len: int  # Actual length before padding

    @property
    def context_length(self) -> int:
        """Maximum window length."""
        return self.data.shape[0]


@dataclass
class NormStats:
    """Per-feature normalization statistics (mean/std)."""

    mean: npt.NDArray[np.float32]  # (8,)
    std: npt.NDArray[np.float32]  # (8,)

    def save(self, path: str) -> None:
        """Save normalization stats to .npz file."""
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> NormStats:
        """Load normalization stats from .npz file."""
        data = np.load(path)
        return cls(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))


@dataclass
class AnomalyLabel:
    """Label for synthetic anomaly injection."""

    anomaly_type: str  # spoofing, ghost, gps_drift, impossible_maneuver
    start_idx: int
    end_idx: int
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass
class LabeledFlight:
    """Flight with optional anomaly labels for evaluation."""

    flight: Flight
    labels: list[AnomalyLabel] = field(default_factory=list)
    is_anomalous: bool = False
