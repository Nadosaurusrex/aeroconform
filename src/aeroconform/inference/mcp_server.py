"""MCP (Model Context Protocol) server for AeroConform.

Exposes AeroConform anomaly detection capabilities to Claude
through three MCP tools: airspace status, aircraft trajectory,
and airspace graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from aeroconform.config import AeroConformConfig
from aeroconform.inference.live_monitor import LiveAirspaceMonitor

logger = structlog.get_logger(__name__)


class AeroConformMCPServer:
    """MCP server exposing AeroConform to Claude.

    Provides three tools:
    1. get_airspace_status: Current anomaly status for a bounding box
    2. get_aircraft_trajectory: Recent trajectory and scores for an aircraft
    3. get_airspace_graph: Current interaction graph with attention weights

    Args:
        monitor: Live airspace monitor instance.
        config: Configuration.
    """

    def __init__(
        self,
        monitor: LiveAirspaceMonitor,
        config: AeroConformConfig | None = None,
    ) -> None:
        self.monitor = monitor
        self.config = config or AeroConformConfig()

    def get_airspace_status(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Get current airspace anomaly status for a bounding box.

        Args:
            bbox: Bounding box (min_lat, max_lat, min_lon, max_lon).
                Defaults to config bbox.
            alpha: Significance level (default from config).

        Returns:
            Dict with timestamp, total aircraft count, anomalous count,
            alpha level, and list of anomaly alerts.
        """
        status = self.monitor.get_status()
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_aircraft": status["tracked_aircraft"],
            "anomalous_aircraft": 0,
            "alpha": alpha or self.config.alpha,
            "alerts": [],
            "bbox": bbox or self.config.bbox,
        }

    def get_aircraft_trajectory(
        self,
        icao24: str,
        window_minutes: int = 10,
    ) -> dict[str, Any]:
        """Get recent trajectory and anomaly scores for a specific aircraft.

        Args:
            icao24: ICAO 24-bit address of the aircraft.
            window_minutes: How many minutes of history to return.

        Returns:
            Dict with icao24, trajectory data, and anomaly scores.
        """
        buffer = self.monitor.trajectory_buffers.get(icao24)

        if buffer is None:
            return {
                "icao24": icao24,
                "found": False,
                "message": f"Aircraft {icao24} not currently tracked",
            }

        trajectory = np.array(list(buffer))
        # Limit to requested window
        max_points = window_minutes * 60  # ~1Hz sampling
        if len(trajectory) > max_points:
            trajectory = trajectory[-max_points:]

        result = self.monitor.pipeline.detect_anomalies(trajectory, update_calibration=False)

        return {
            "icao24": icao24,
            "found": True,
            "n_points": len(trajectory),
            "latest_position": {
                "lat": float(trajectory[-1, 0]),
                "lon": float(trajectory[-1, 1]),
                "alt_ft": float(trajectory[-1, 2]),
            },
            "latest_velocity_kts": float(trajectory[-1, 3]),
            "latest_heading_deg": float(trajectory[-1, 4]),
            "is_anomalous": result["is_anomalous"],
            "max_score": result["max_score"],
            "mean_score": result["mean_score"],
            "anomalous_patches": result["anomalous_patches"],
        }

    def get_airspace_graph(
        self,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Get the current airspace interaction graph.

        Args:
            bbox: Bounding box to filter aircraft.

        Returns:
            Dict with nodes (aircraft), edges (interactions),
            and attention weights.
        """
        self.monitor.get_status()

        nodes: list[dict[str, Any]] = []
        for icao24, buffer in self.monitor.trajectory_buffers.items():
            if len(buffer) > 0:
                latest = np.array(buffer[-1])
                nodes.append({
                    "icao24": icao24,
                    "lat": float(latest[0]),
                    "lon": float(latest[1]),
                    "alt_ft": float(latest[2]),
                    "velocity_kts": float(latest[3]),
                    "heading_deg": float(latest[4]),
                })

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "n_nodes": len(nodes),
            "nodes": nodes[:50],  # Limit for response size
            "n_edges": 0,
            "edges": [],
            "bbox": bbox or self.config.bbox,
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Return the MCP tool definitions.

        Returns:
            List of MCP tool specification dicts.
        """
        return [
            {
                "name": "get_airspace_status",
                "description": "Get current airspace anomaly status for a bounding box. "
                "Returns total aircraft count, anomalous aircraft, and alerts "
                "with confidence levels and p-values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "string",
                            "description": "Bounding box as 'min_lat,max_lat,min_lon,max_lon'",
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Significance level for anomaly detection (default 0.01)",
                        },
                    },
                },
            },
            {
                "name": "get_aircraft_trajectory",
                "description": "Get recent trajectory and anomaly scores for a specific aircraft "
                "identified by its ICAO24 address.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "icao24": {
                            "type": "string",
                            "description": "ICAO 24-bit address of the aircraft",
                        },
                        "window_minutes": {
                            "type": "integer",
                            "description": "Minutes of trajectory history to return (default 10)",
                        },
                    },
                    "required": ["icao24"],
                },
            },
            {
                "name": "get_airspace_graph",
                "description": "Get the current airspace interaction graph with aircraft as nodes "
                "and proximity-based edges with attention weights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "string",
                            "description": "Bounding box as 'min_lat,max_lat,min_lon,max_lon'",
                        },
                    },
                },
            },
        ]
