"""OpenSky REST API client for live ADS-B state vectors.

Provides async access to the OpenSky Network API with rate limiting,
exponential backoff, and response caching.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

import httpx
import pandas as pd
import structlog

from aeroconform.config import AeroConformConfig

logger = structlog.get_logger(__name__)

# OpenSky state vector column names
STATE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
]

# Columns we actually use
USED_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "latitude",
    "longitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "geo_altitude",
]


class OpenSkyClient:
    """Wrapper around OpenSky REST API for live state vectors.

    Supports authenticated and unauthenticated access with automatic
    rate limiting and exponential backoff on 429 responses.

    Args:
        config: AeroConform configuration.
        username: OpenSky username for authenticated access (optional).
        password: OpenSky password for authenticated access (optional).
    """

    def __init__(
        self,
        config: AeroConformConfig | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.config = config or AeroConformConfig()
        self.base_url = self.config.opensky_base_url
        self._auth = (username, password) if username and password else None
        self._min_interval = 5.0 if self._auth else 10.0
        self._last_request_time: float = 0.0
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 10.0

    async def get_states(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        icao24: list[str] | None = None,
        time_secs: int = 0,
    ) -> pd.DataFrame:
        """Fetch current state vectors from the OpenSky API.

        Args:
            bbox: Bounding box as (min_lat, max_lat, min_lon, max_lon).
            icao24: List of ICAO24 addresses to filter by.
            time_secs: Unix timestamp to request (0 = most recent).

        Returns:
            DataFrame with state vector columns for airborne aircraft.
        """
        cache_key = f"{bbox}_{icao24}_{time_secs}"
        now = time.monotonic()
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return cached_df

        # Rate limiting
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)

        params: dict[str, str | int | float] = {}
        if time_secs > 0:
            params["time"] = time_secs
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            params["lamin"] = min_lat
            params["lamax"] = max_lat
            params["lomin"] = min_lon
            params["lomax"] = max_lon
        if icao24 is not None:
            params["icao24"] = ",".join(icao24)

        df = await self._request_with_backoff(params)
        self._cache[cache_key] = (time.monotonic(), df)
        return df

    async def _request_with_backoff(
        self, params: dict[str, str | int | float], max_retries: int = 5
    ) -> pd.DataFrame:
        """Make an API request with exponential backoff on rate limiting.

        Args:
            params: Query parameters for the API call.
            max_retries: Maximum number of retry attempts.

        Returns:
            DataFrame with parsed state vectors.

        Raises:
            httpx.HTTPStatusError: If all retries are exhausted.
        """
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                auth = self._auth
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/states/all",
                        params=params,
                        auth=auth,
                        timeout=30.0,
                    )
                self._last_request_time = time.monotonic()

                if response.status_code == 429:
                    logger.warning(
                        "rate_limited",
                        attempt=attempt,
                        backoff=backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "request_failed",
                    attempt=attempt,
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        return pd.DataFrame(columns=USED_COLUMNS)

    def _parse_response(self, data: dict) -> pd.DataFrame:
        """Parse the OpenSky JSON response into a DataFrame.

        Args:
            data: Raw JSON response from the API.

        Returns:
            Cleaned DataFrame with used columns only.
        """
        if not data or "states" not in data or data["states"] is None:
            return pd.DataFrame(columns=USED_COLUMNS)

        df = pd.DataFrame(data["states"], columns=STATE_COLUMNS)
        df = df[USED_COLUMNS].copy()

        # Strip whitespace from callsigns
        df["callsign"] = df["callsign"].str.strip()

        # Convert types
        for col in ["latitude", "longitude", "baro_altitude", "velocity", "true_track", "vertical_rate", "geo_altitude"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["on_ground"] = df["on_ground"].astype(bool)

        logger.info("states_fetched", aircraft_count=len(df))
        return df

    async def stream_states(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        poll_interval: float | None = None,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Continuously poll the API and yield state vector snapshots.

        Args:
            bbox: Bounding box to filter by.
            poll_interval: Seconds between polls (default from config).

        Yields:
            DataFrame with state vectors at each polling interval.
        """
        interval = poll_interval or float(self.config.opensky_poll_interval)
        while True:
            try:
                states = await self.get_states(bbox=bbox)
                if len(states) > 0:
                    yield states
            except Exception as exc:
                logger.error("stream_error", error=str(exc))
            await asyncio.sleep(interval)
