"""End-to-end real-time inference pipeline.

Polls OpenSky REST API, maintains per-aircraft buffers, runs
AeroGPT + AirGraph + AeroConformal, and emits alerts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog
import torch

from src.data.opensky_client import RESTClient
from src.data.preprocessing import (
    compute_elapsed_seconds,
    delta_encode,
    normalize,
)
from src.data.schemas import NormStats
from src.inference.buffer_manager import BufferManager
from src.models.aerogpt import AeroGPT
from src.models.airgraph import AirGraph
from src.models.conformal import AdaptiveConformal
from src.models.graph_builder import AirspaceGraphBuilder
from src.models.scoring import Alert, AlertLevel, classify_alert
from src.utils.constants import LIMM_BBOX

logger = structlog.get_logger(__name__)


class InferencePipeline:
    """Real-time anomaly detection pipeline."""

    def __init__(
        self,
        aerogpt: AeroGPT,
        airgraph: AirGraph | None,
        conformal: AdaptiveConformal,
        norm_stats: NormStats,
        *,
        poll_interval: float = 10.0,
        alert_callback: Callable[[Alert], None] | None = None,
        device: str = "cpu",
    ) -> None:
        self.aerogpt = aerogpt
        self.airgraph = airgraph
        self.conformal = conformal
        self.norm_stats = norm_stats
        self.poll_interval = poll_interval
        self.alert_callback = alert_callback
        self.device = torch.device(device)

        self.buffer_manager = BufferManager()
        self.graph_builder = AirspaceGraphBuilder()
        self.rest_client = RESTClient()
        self._running = False

        self.aerogpt.to(self.device)
        self.aerogpt.eval()
        if self.airgraph:
            self.airgraph.to(self.device)
            self.airgraph.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        norm_stats_path: Path,
        **kwargs,
    ) -> InferencePipeline:
        """Load pipeline from saved checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            norm_stats_path: Path to normalization stats.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Initialized InferencePipeline.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]

        model = AeroGPT(config)
        model.load_state_dict(ckpt["model_state_dict"])

        norm_stats = NormStats.load(str(norm_stats_path))
        conformal = AdaptiveConformal()

        return cls(
            aerogpt=model,
            airgraph=None,
            conformal=conformal,
            norm_stats=norm_stats,
            **kwargs,
        )

    async def poll_and_process(self) -> list[Alert]:
        """Single poll-and-process cycle.

        Returns:
            List of generated alerts.
        """
        start_time = time.time()

        # Fetch current state vectors
        try:
            states_df = await self.rest_client.get_states(LIMM_BBOX)
        except Exception:
            logger.exception("rest_api_error")
            return []

        if len(states_df) == 0:
            return []

        # Update buffers with new observations
        import math

        for row in states_df.iter_rows(named=True):
            lat = row.get("lat")
            lon = row.get("lon")
            if lat is None or lon is None:
                continue

            heading_rad = math.radians(row.get("heading", 0) or 0)
            features = np.array(
                [
                    lat,
                    lon,
                    row.get("baroaltitude", 0) or 0,
                    row.get("velocity", 0) or 0,
                    math.sin(heading_rad),
                    math.cos(heading_rad),
                    row.get("vertrate", 0) or 0,
                    float(row.get("onground", False) or False),
                ],
                dtype=np.float32,
            )
            self.buffer_manager.update(
                icao24=row["icao24"],
                features=features,
                timestamp=row.get("time", 0),
                callsign=row.get("callsign"),
            )

        # Garbage collect stale entries
        self.buffer_manager.garbage_collect()

        # Run inference on ready aircraft
        ready = self.buffer_manager.get_ready_aircraft()
        alerts = self._run_inference(ready)

        elapsed = time.time() - start_time
        logger.info(
            "poll_cycle_complete",
            aircraft_count=len(states_df),
            tracked=self.buffer_manager.num_tracked,
            ready=len(ready),
            alerts=len(alerts),
            elapsed_ms=f"{elapsed * 1000:.0f}",
        )

        return alerts

    @torch.no_grad()
    def _run_inference(self, aircraft_ids: list[str]) -> list[Alert]:
        """Run model inference on ready aircraft.

        Args:
            aircraft_ids: List of icao24 addresses to process.

        Returns:
            List of alerts for anomalous aircraft.
        """
        if not aircraft_ids:
            return []

        alerts: list[Alert] = []

        # Batch inference: prepare all sequences
        batch_data = []
        for icao24 in aircraft_ids:
            buf = self.buffer_manager.get_buffer(icao24)
            if buf is None:
                continue

            features, timestamps = buf.get_window()
            deltas = delta_encode(features)
            normalized = normalize(deltas, self.norm_stats)
            elapsed = compute_elapsed_seconds(timestamps)

            batch_data.append({
                "icao24": icao24,
                "input": torch.from_numpy(normalized[:-1]).unsqueeze(0),
                "time_gaps": torch.from_numpy(elapsed[:-1]).unsqueeze(0),
                "target_delta": normalized[-1],
                "current_state": features[-1],
                "callsign": buf.callsign,
            })

        if not batch_data:
            return []

        # Stack into batch tensors
        inputs = torch.cat([d["input"] for d in batch_data]).to(self.device)
        time_gaps = torch.cat([d["time_gaps"] for d in batch_data]).to(self.device)

        # AeroGPT forward
        means, log_vars, hidden = self.aerogpt(inputs, time_gaps)

        # Score each aircraft
        for i, data in enumerate(batch_data):
            pred_means = means[i, -1].cpu().numpy()
            pred_log_vars = log_vars[i, -1].cpu().numpy()
            observed = data["target_delta"]

            result = self.conformal.score(observed, pred_means, pred_log_vars)
            alert_level = classify_alert(result.p_value)

            if alert_level != AlertLevel.NORMAL:
                state = data["current_state"]
                alert = Alert(
                    icao24=data["icao24"],
                    timestamp=int(time.time()),
                    alert_level=alert_level,
                    p_value=result.p_value,
                    score=result.score,
                    latitude=float(state[0]),
                    longitude=float(state[1]),
                    altitude=float(state[2]),
                    explanation=f"{alert_level.value} alert: p={result.p_value:.4f}, score={result.score:.2f}",
                )
                alerts.append(alert)

                if self.alert_callback:
                    self.alert_callback(alert)

        return alerts

    async def run(self) -> None:
        """Run continuous inference loop."""
        self._running = True
        logger.info("pipeline_started", poll_interval=self.poll_interval)

        while self._running:
            await self.poll_and_process()
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the inference loop."""
        self._running = False
        logger.info("pipeline_stopped")
