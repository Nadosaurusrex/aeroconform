"""Pydantic response models for the AeroConform API."""

from __future__ import annotations

from pydantic import BaseModel


class AircraftState(BaseModel):
    """Current state of a tracked aircraft."""

    icao24: str
    callsign: str | None = None
    latitude: float
    longitude: float
    altitude: float
    velocity: float
    heading: float
    vertical_rate: float
    on_ground: bool
    anomaly_score: float = 0.0
    p_value: float = 1.0
    alert_level: str = "normal"


class AnomalyAlert(BaseModel):
    """Anomaly alert details."""

    icao24: str
    timestamp: int
    alert_level: str
    p_value: float
    score: float
    latitude: float
    longitude: float
    altitude: float
    explanation: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    tracked_aircraft: int = 0
    ready_aircraft: int = 0
    buffer_size: int = 0
    uptime_seconds: float = 0.0


class AlertsResponse(BaseModel):
    """List of recent alerts."""

    alerts: list[AnomalyAlert] = []
    count: int = 0
