"""FastAPI application for AeroConform.

Endpoints:
- GET /health: Health check
- GET /aircraft: Current state + scores for all tracked aircraft
- GET /alerts: Recent anomaly alerts
- WebSocket /ws/alerts: Live alert stream
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    AircraftState,
    AlertsResponse,
    AnomalyAlert,
    HealthResponse,
)
from src.models.scoring import Alert

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="AeroConform",
    description="Conformalized trajectory anomaly detection API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
_start_time = time.time()
_recent_alerts: deque[Alert] = deque(maxlen=100)
_connected_websockets: set[WebSocket] = set()
_pipeline = None


def set_pipeline(pipeline) -> None:  # noqa: ANN001
    """Set the inference pipeline instance."""
    global _pipeline  # noqa: PLW0603
    _pipeline = pipeline


def on_alert(alert: Alert) -> None:
    """Callback for new alerts from the pipeline."""
    _recent_alerts.append(alert)
    # Broadcast to WebSocket clients
    alert_data = AnomalyAlert(
        icao24=alert.icao24,
        timestamp=alert.timestamp,
        alert_level=alert.alert_level.value,
        p_value=alert.p_value,
        score=alert.score,
        latitude=alert.latitude,
        longitude=alert.longitude,
        altitude=alert.altitude,
        explanation=alert.explanation,
    )
    asyncio.create_task(_broadcast_alert(alert_data))


async def _broadcast_alert(alert: AnomalyAlert) -> None:
    """Send alert to all connected WebSocket clients."""
    global _connected_websockets  # noqa: PLW0603
    message = alert.model_dump_json()
    disconnected: set[WebSocket] = set()
    for ws in _connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    _connected_websockets -= disconnected


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        tracked_aircraft=_pipeline.buffer_manager.num_tracked if _pipeline else 0,
        ready_aircraft=_pipeline.buffer_manager.num_ready if _pipeline else 0,
        buffer_size=_pipeline.conformal.buffer_size_current if _pipeline else 0,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/aircraft", response_model=list[AircraftState])
async def get_aircraft() -> list[AircraftState]:
    """Get current state and scores for all tracked aircraft."""
    if not _pipeline:
        return []

    aircraft = []
    for icao24 in _pipeline.buffer_manager.get_ready_aircraft():
        buf = _pipeline.buffer_manager.get_buffer(icao24)
        if buf is None:
            continue

        state = buf.get_current_state()
        import math

        heading_rad = math.atan2(state[4], state[5])
        heading_deg = math.degrees(heading_rad) % 360

        aircraft.append(
            AircraftState(
                icao24=icao24,
                callsign=buf.callsign,
                latitude=float(state[0]),
                longitude=float(state[1]),
                altitude=float(state[2]),
                velocity=float(state[3]),
                heading=heading_deg,
                vertical_rate=float(state[6]),
                on_ground=bool(state[7] > 0.5),
            )
        )

    return aircraft


@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts() -> AlertsResponse:
    """Get recent anomaly alerts."""
    alerts = [
        AnomalyAlert(
            icao24=a.icao24,
            timestamp=a.timestamp,
            alert_level=a.alert_level.value,
            p_value=a.p_value,
            score=a.score,
            latitude=a.latitude,
            longitude=a.longitude,
            altitude=a.altitude,
            explanation=a.explanation,
        )
        for a in _recent_alerts
    ]
    return AlertsResponse(alerts=alerts, count=len(alerts))


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint for live alert streaming."""
    await websocket.accept()
    _connected_websockets.add(websocket)
    logger.info("websocket_connected", total=len(_connected_websockets))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connected_websockets.discard(websocket)
        logger.info("websocket_disconnected", total=len(_connected_websockets))
