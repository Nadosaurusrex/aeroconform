"""OpenSky Network data clients: Trino (historical) and REST (live).

Trino is the primary data source via pyopensky. REST is used for
live state vector polling and as a fallback for historical data.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime

import httpx
import polars as pl
import structlog

from src.utils.constants import (
    LIMM_BBOX,
    OPENSKY_API_BASE,
    OPENSKY_RATE_LIMIT_SECONDS,
    OPENSKY_TOKEN_URL,
)

logger = structlog.get_logger(__name__)


class TrinoUnavailableError(Exception):
    """Raised when Trino connection is unavailable."""


class TrinoClient:
    """Historical data access via pyopensky Trino interface.

    Wraps pyopensky.trino.Trino for querying state_vectors_data4.
    pyopensky handles OAuth2 token refresh and automatic hour-partition
    splitting internally.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = LIMM_BBOX,
    ) -> None:
        self.bounds = bounds
        self._trino = None

    def _get_trino(self):  # noqa: ANN202
        """Lazy-initialize pyopensky Trino connection."""
        if self._trino is None:
            try:
                from pyopensky.trino import Trino

                self._trino = Trino()
            except Exception as e:
                raise TrinoUnavailableError(f"Cannot connect to Trino: {e}") from e
        return self._trino

    def query_state_vectors(
        self,
        start: str | datetime,
        end: str | datetime,
        *,
        icao24: str | list[str] | None = None,
        columns: tuple[str, ...] | None = None,
    ) -> pl.DataFrame:
        """Query historical state vectors from Trino.

        pyopensky automatically splits queries by hour partition and handles
        OAuth2 authentication.

        Args:
            start: Start time (ISO string or datetime).
            end: End time (ISO string or datetime).
            icao24: Filter by specific aircraft ICAO24 address(es).
            columns: Specific columns to select. Defaults to all needed columns.

        Returns:
            polars DataFrame with state vectors.

        Raises:
            TrinoUnavailableError: If Trino connection fails.
        """
        trino = self._get_trino()

        kwargs: dict = {
            "start": start,
            "stop": end,
            "bounds": self.bounds,
        }
        if icao24 is not None:
            kwargs["icao24"] = icao24
        if columns is not None:
            kwargs["selected_columns"] = columns

        logger.info("trino_query_start", start=str(start), end=str(end), bounds=self.bounds)

        try:
            result = trino.history(**kwargs)
        except Exception as e:
            raise TrinoUnavailableError(f"Trino query failed: {e}") from e

        if result is None or (hasattr(result, '__len__') and len(result) == 0):
            logger.warning("trino_query_empty", start=str(start), end=str(end))
            return pl.DataFrame()

        # pyopensky returns a pandas DataFrame directly
        pdf = result.data if hasattr(result, 'data') else result
        df = pl.from_pandas(pdf)

        # Rename columns to match our schema (handle both pyopensky output formats)
        rename_map = {
            "timestamp": "time",
            "latitude": "lat",
            "longitude": "lon",
            "altitude": "baroaltitude",
            "groundspeed": "velocity",
            "track": "heading",
            "vertical_rate": "vertrate",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename({old: new})

        # Convert timestamp to unix int if needed
        if "time" in df.columns:
            if df["time"].dtype == pl.Datetime or str(df["time"].dtype).startswith("Datetime"):
                df = df.with_columns(pl.col("time").dt.epoch("s").alias("time"))

        logger.info("trino_query_complete", rows=len(df))
        return df


class RESTClient:
    """Live state vector access via OpenSky REST API.

    Uses OAuth2 client credentials flow. Rate-limited to 1 request per 5 seconds.
    """

    def __init__(self) -> None:
        self._client_id = os.environ.get("OPENSKY_CLIENT_ID", "")
        self._client_secret = os.environ.get("OPENSKY_CLIENT_SECRET", "")
        self._token: str | None = None
        self._token_expires_at: float = 0
        self._last_request_time: float = 0
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _ensure_token(self) -> str:
        """Get valid OAuth2 access token, refreshing if expired."""
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        client = await self._get_client()
        resp = await client.post(
            OPENSKY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 1800)
        logger.info("oauth2_token_refreshed", expires_in=data.get("expires_in"))
        return self._token  # type: ignore[return-value]

    async def _rate_limit(self) -> None:
        """Enforce rate limiting: 1 request per 5 seconds."""
        elapsed = time.time() - self._last_request_time
        if elapsed < OPENSKY_RATE_LIMIT_SECONDS:
            await asyncio.sleep(OPENSKY_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    async def get_states(
        self,
        bbox: tuple[float, float, float, float] = LIMM_BBOX,
    ) -> pl.DataFrame:
        """Fetch current state vectors within bounding box.

        Args:
            bbox: (west, south, east, north) bounding box.

        Returns:
            polars DataFrame with current state vectors.
        """
        await self._rate_limit()
        token = await self._ensure_token()
        client = await self._get_client()

        west, south, east, north = bbox
        resp = await client.get(
            f"{OPENSKY_API_BASE}/states/all",
            params={
                "lamin": south,
                "lamax": north,
                "lomin": west,
                "lomax": east,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()

        states = data.get("states", [])
        if not states:
            return pl.DataFrame()

        columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "lon", "lat", "baroaltitude", "onground",
            "velocity", "heading", "vertrate", "sensors", "geoaltitude",
            "squawk", "spi", "position_source",
        ]

        rows = []
        for s in states:
            row = dict(zip(columns, s, strict=False))
            row["time"] = data.get("time", row.get("time_position", 0))
            rows.append(row)

        df = pl.DataFrame(rows)
        logger.info("rest_states_fetched", count=len(df))
        return df

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
