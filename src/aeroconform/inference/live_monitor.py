"""Real-time airspace monitoring for anomaly detection.

Polls the OpenSky API, maintains per-aircraft trajectory buffers,
runs inference through the AeroConform pipeline, and emits alerts.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from aeroconform.config import AeroConformConfig
from aeroconform.data.opensky_client import OpenSkyClient
from aeroconform.models.pipeline import AeroConformPipeline

logger = structlog.get_logger(__name__)


class LiveAirspaceMonitor:
    """Real-time anomaly detection on live OpenSky data.

    Polls the API at regular intervals, maintains per-aircraft trajectory
    buffers, and runs the full pipeline to produce anomaly alerts.

    Args:
        pipeline: Calibrated AeroConform inference pipeline.
        config: Configuration.
        client: OpenSky API client (created if not provided).
    """

    def __init__(
        self,
        pipeline: AeroConformPipeline,
        config: AeroConformConfig | None = None,
        client: OpenSkyClient | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or AeroConformConfig()
        self.client = client or OpenSkyClient(config=self.config)
        self.trajectory_buffers: dict[str, deque[np.ndarray]] = {}
        self.buffer_maxlen = self.config.seq_len + 50

    async def run(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        poll_interval: float | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Main monitoring loop.

        Args:
            bbox: Bounding box to monitor (default from config).
            poll_interval: Seconds between polls (default from config).
            max_iterations: Stop after this many iterations (None = forever).
        """
        bbox = bbox or self.config.bbox
        interval = poll_interval or float(self.config.opensky_poll_interval)
        iteration = 0

        logger.info("monitor_started", bbox=bbox, interval=interval)

        while max_iterations is None or iteration < max_iterations:
            try:
                states = await self.client.get_states(bbox=bbox)
                alerts = self.process_snapshot(states)
                for alert in alerts:
                    self._emit_alert(alert)
            except Exception as exc:
                logger.error("monitor_error", error=str(exc))

            iteration += 1
            await asyncio.sleep(interval)

        logger.info("monitor_stopped", iterations=iteration)

    def process_snapshot(self, states: pd.DataFrame) -> list[dict[str, Any]]:
        """Process one airspace snapshot and return anomaly alerts.

        Args:
            states: DataFrame with current aircraft state vectors.

        Returns:
            List of anomaly alert dicts.
        """
        if states.empty:
            return []

        alerts: list[dict[str, Any]] = []

        for _, row in states.iterrows():
            icao24 = row["icao24"]

            # Skip aircraft on ground
            if row.get("on_ground", False):
                continue

            # Build state vector
            state = np.array([
                row.get("latitude", np.nan),
                row.get("longitude", np.nan),
                row.get("baro_altitude", np.nan),
                row.get("velocity", np.nan),
                row.get("true_track", np.nan),
                row.get("vertical_rate", np.nan),
            ])

            # Skip if position is missing
            if np.isnan(state[:3]).any():
                continue

            # Fill remaining NaN with zeros
            state = np.nan_to_num(state, nan=0.0)

            # Update trajectory buffer
            if icao24 not in self.trajectory_buffers:
                self.trajectory_buffers[icao24] = deque(maxlen=self.buffer_maxlen)
            self.trajectory_buffers[icao24].append(state)

            # Check if we have enough history
            buffer = self.trajectory_buffers[icao24]
            if len(buffer) < self.config.min_trajectory_len:
                continue

            # Run detection
            trajectory = np.array(list(buffer))
            result = self.pipeline.detect_anomalies(trajectory)

            if result["is_anomalous"]:
                alert = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "icao24": icao24,
                    "callsign": row.get("callsign", ""),
                    "position": {
                        "lat": float(state[0]),
                        "lon": float(state[1]),
                        "alt_ft": float(state[2]),
                    },
                    "anomaly_type": "trajectory_deviation",
                    "confidence": result["results"][-1]["confidence"] if result["results"] else 0.0,
                    "p_value": result["results"][-1]["p_value"] if result["results"] else 1.0,
                    "max_score": result["max_score"],
                    "anomalous_patches": result["anomalous_patches"],
                }
                alerts.append(alert)

        # Clean up stale buffers
        active_icao24s = set(states["icao24"]) if not states.empty else set()
        stale = [k for k in self.trajectory_buffers if k not in active_icao24s]
        for k in stale:
            del self.trajectory_buffers[k]

        return alerts

    def _emit_alert(self, alert: dict[str, Any]) -> None:
        """Log an anomaly alert.

        Args:
            alert: Alert dict with detection details.
        """
        logger.warning(
            "anomaly_detected",
            icao24=alert["icao24"],
            callsign=alert["callsign"],
            confidence=alert["confidence"],
            position=alert["position"],
        )

    def get_status(self) -> dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Dict with tracked aircraft count and buffer sizes.
        """
        return {
            "tracked_aircraft": len(self.trajectory_buffers),
            "buffer_sizes": {
                k: len(v) for k, v in self.trajectory_buffers.items()
            },
        }
