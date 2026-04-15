"""Per-aircraft state buffer management for real-time inference.

Maintains sliding windows per icao24, handles aircraft entering/leaving
coverage, garbage collects stale entries.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import structlog

from src.utils.constants import CONTEXT_LENGTH

logger = structlog.get_logger(__name__)

# Stale entry timeout in seconds (5 minutes)
STALE_TIMEOUT = 300
# Minimum observations before aircraft is "ready" for inference
MIN_READY_OBSERVATIONS = 5


@dataclass
class AircraftBuffer:
    """Rolling buffer for a single aircraft's observations."""

    icao24: str
    features: deque[npt.NDArray[np.float32]] = field(
        default_factory=lambda: deque(maxlen=CONTEXT_LENGTH + 10)
    )
    timestamps: deque[int] = field(
        default_factory=lambda: deque(maxlen=CONTEXT_LENGTH + 10)
    )
    last_update: float = 0.0
    callsign: str | None = None

    @property
    def num_observations(self) -> int:
        """Number of observations in buffer."""
        return len(self.features)

    @property
    def is_ready(self) -> bool:
        """Whether aircraft has enough observations for inference."""
        return self.num_observations >= MIN_READY_OBSERVATIONS

    def add_observation(
        self,
        features: npt.NDArray[np.float32],
        timestamp: int,
        callsign: str | None = None,
    ) -> None:
        """Add a new observation to the buffer.

        Args:
            features: (8,) encoded state vector.
            timestamp: Unix timestamp.
            callsign: Optional callsign update.
        """
        self.features.append(features)
        self.timestamps.append(timestamp)
        self.last_update = time.time()
        if callsign:
            self.callsign = callsign

    def get_window(
        self, context_length: int = CONTEXT_LENGTH
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Get the most recent window of observations.

        Args:
            context_length: Maximum window length.

        Returns:
            features: (seq_len, 8) array.
            timestamps: (seq_len,) array.
        """
        n = min(len(self.features), context_length)
        features = np.array(list(self.features)[-n:], dtype=np.float32)
        timestamps = np.array(list(self.timestamps)[-n:], dtype=np.int64)
        return features, timestamps

    def get_current_state(self) -> npt.NDArray[np.float32]:
        """Get most recent state vector."""
        return self.features[-1]


class BufferManager:
    """Manages per-aircraft observation buffers."""

    def __init__(
        self,
        context_length: int = CONTEXT_LENGTH,
        stale_timeout: float = STALE_TIMEOUT,
    ) -> None:
        self.context_length = context_length
        self.stale_timeout = stale_timeout
        self.buffers: dict[str, AircraftBuffer] = {}

    def update(
        self,
        icao24: str,
        features: npt.NDArray[np.float32],
        timestamp: int,
        callsign: str | None = None,
    ) -> None:
        """Add an observation for an aircraft.

        Args:
            icao24: Aircraft identifier.
            features: (8,) encoded state vector.
            timestamp: Unix timestamp.
            callsign: Optional callsign.
        """
        if icao24 not in self.buffers:
            self.buffers[icao24] = AircraftBuffer(icao24=icao24)

        self.buffers[icao24].add_observation(features, timestamp, callsign)

    def get_ready_aircraft(self) -> list[str]:
        """Get list of aircraft with enough observations for inference.

        Returns:
            List of icao24 addresses.
        """
        return [
            icao24
            for icao24, buf in self.buffers.items()
            if buf.is_ready
        ]

    def get_buffer(self, icao24: str) -> AircraftBuffer | None:
        """Get buffer for a specific aircraft."""
        return self.buffers.get(icao24)

    def garbage_collect(self) -> int:
        """Remove stale aircraft entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        stale = [
            icao24
            for icao24, buf in self.buffers.items()
            if (now - buf.last_update) > self.stale_timeout
        ]

        for icao24 in stale:
            del self.buffers[icao24]

        if stale:
            logger.info("buffer_gc", removed=len(stale), remaining=len(self.buffers))

        return len(stale)

    @property
    def num_tracked(self) -> int:
        """Number of currently tracked aircraft."""
        return len(self.buffers)

    @property
    def num_ready(self) -> int:
        """Number of aircraft ready for inference."""
        return len(self.get_ready_aircraft())
