"""OpenSky Network data clients: Trino (historical) and REST (live).

Trino is the primary data source via pyopensky. REST is used for
live state vector polling and as a fallback for historical data.

pyopensky's Trino class has no timeout configuration, which causes
hangs on cloud environments (Colab, etc.). We monkey-patch it to add
explicit timeouts at both the auth and query levels.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta

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

# Timeout settings for Trino (seconds)
TRINO_AUTH_TIMEOUT = 30.0
TRINO_REQUEST_TIMEOUT = 300


class TrinoUnavailableError(Exception):
    """Raised when Trino connection is unavailable."""


def _patch_pyopensky_timeouts() -> None:
    """Monkey-patch pyopensky's Trino class to add timeout configuration.

    pyopensky's Trino.token() uses httpx.post() with no timeout,
    and Trino.engine() creates a SQLAlchemy engine with no request_timeout.
    Both hang on cloud environments like Colab.

    This patches both methods to add explicit timeouts.
    """
    try:
        import pyopensky.trino as _trino_mod
    except ImportError:
        return

    _original_token = _trino_mod.Trino.token
    _original_engine = _trino_mod.Trino.engine

    def _patched_token(self, **kwargs):  # noqa: ANN001, ANN003, ANN202
        """Token method with explicit httpx timeout."""
        kwargs.setdefault("timeout", TRINO_AUTH_TIMEOUT)
        return _original_token(self, **kwargs)

    def _patched_engine(self):  # noqa: ANN001, ANN202
        """Engine method with request_timeout in connect_args."""
        from sqlalchemy import create_engine
        from trino.auth import JWTAuthentication, OAuth2Authentication
        from trino.sqlalchemy import URL

        token = self.token()

        trino_username = getattr(_trino_mod, "trino_username", None)
        engine = create_engine(
            URL(
                "trino.opensky-network.org",
                port=443,
                user=trino_username,
                catalog="minio",
                schema="osky",
                source="pyopensky",
            ),
            connect_args={
                "auth": JWTAuthentication(token) if token is not None else OAuth2Authentication(),
                "http_scheme": "https",
                "legacy_prepared_statements": True,
                "request_timeout": TRINO_REQUEST_TIMEOUT,
            },
        )
        return engine

    _trino_mod.Trino.token = _patched_token
    _trino_mod.Trino.engine = _patched_engine
    logger.info(
        "pyopensky_patched",
        auth_timeout=TRINO_AUTH_TIMEOUT,
        request_timeout=TRINO_REQUEST_TIMEOUT,
    )


class TrinoClient:
    """Historical data access via pyopensky Trino interface.

    Wraps pyopensky.trino.Trino for querying state_vectors_data4.
    Patches pyopensky to add timeouts that prevent hangs on cloud.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = LIMM_BBOX,
    ) -> None:
        self.bounds = bounds
        self._trino = None

    def _get_trino(self):  # noqa: ANN202
        """Lazy-initialize pyopensky Trino connection with timeout patches."""
        if self._trino is None:
            try:
                _patch_pyopensky_timeouts()
                from pyopensky.trino import Trino

                self._trino = Trino()
                logger.info("trino_client_initialized")
            except Exception as e:
                raise TrinoUnavailableError(f"Cannot initialize Trino: {e}") from e
        return self._trino

    def test_connectivity(self) -> dict[str, bool | str]:
        """Test connectivity to Trino auth and query servers.

        Returns dict with 'auth_ok', 'query_ok', and any error messages.
        Useful for diagnosing Colab connectivity issues.
        """
        result: dict[str, bool | str] = {"auth_ok": False, "query_ok": False}

        # Test 1: Can we reach the auth server?
        try:
            resp = httpx.get(
                "https://auth.opensky-network.org/auth/realms/opensky-network",
                timeout=10.0,
            )
            result["auth_ok"] = resp.status_code == 200
            result["auth_status"] = str(resp.status_code)
        except Exception as e:
            result["auth_error"] = str(e)

        # Test 2: Can we reach the Trino server?
        try:
            resp = httpx.get(
                "https://trino.opensky-network.org/ui/",
                timeout=10.0,
            )
            result["query_ok"] = resp.status_code in (200, 301, 302, 303, 307, 401)
            result["query_status"] = str(resp.status_code)
        except Exception as e:
            result["query_error"] = str(e)

        logger.info("trino_connectivity_test", **result)
        return result

    def query_state_vectors(
        self,
        start: str | datetime,
        end: str | datetime,
        *,
        icao24: str | list[str] | None = None,
        columns: tuple[str, ...] | None = None,
        chunk_hours: int = 2,
        max_retries: int = 3,
    ) -> pl.DataFrame:
        """Query historical state vectors from Trino.

        Splits queries into smaller time chunks to avoid timeouts.
        Retries failed chunks with exponential backoff.

        Args:
            start: Start time (ISO string or datetime).
            end: End time (ISO string or datetime).
            icao24: Filter by specific aircraft ICAO24 address(es).
            columns: Specific columns to select.
            chunk_hours: Hours per query chunk (default 2h).
            max_retries: Number of retries per chunk on failure.

        Returns:
            polars DataFrame with state vectors.
        """
        trino = self._get_trino()

        start_dt = start if isinstance(start, datetime) else datetime.fromisoformat(str(start))
        end_dt = end if isinstance(end, datetime) else datetime.fromisoformat(str(end))

        all_dfs: list[pl.DataFrame] = []
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(hours=chunk_hours), end_dt)

            kwargs: dict = {
                "start": current.strftime("%Y-%m-%d %H:%M"),
                "stop": chunk_end.strftime("%Y-%m-%d %H:%M"),
                "bounds": self.bounds,
            }
            if icao24 is not None:
                kwargs["icao24"] = icao24
            if columns is not None:
                kwargs["selected_columns"] = columns

            logger.info(
                "trino_chunk_query",
                start=current.strftime("%Y-%m-%d %H:%M"),
                end=chunk_end.strftime("%Y-%m-%d %H:%M"),
            )

            chunk_result = None
            for attempt in range(max_retries):
                try:
                    chunk_result = trino.history(**kwargs)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 15 * (attempt + 1)
                        logger.warning(
                            "trino_retry",
                            attempt=attempt + 1,
                            wait=wait,
                            error=f"{type(e).__name__}: {e}",
                        )
                        time.sleep(wait)
                        # Re-initialize connection on retry
                        self._trino = None
                        trino = self._get_trino()
                    else:
                        raise TrinoUnavailableError(
                            f"Trino query failed after {max_retries} retries: {type(e).__name__}: {e}"
                        ) from e

            if chunk_result is not None and (not hasattr(chunk_result, "__len__") or len(chunk_result) > 0):
                pdf = chunk_result.data if hasattr(chunk_result, "data") else chunk_result
                chunk_df = pl.from_pandas(pdf)
                all_dfs.append(chunk_df)
                logger.info("trino_chunk_complete", rows=len(chunk_df))
            else:
                logger.info("trino_chunk_empty", start=current.strftime("%Y-%m-%d %H:%M"))

            current = chunk_end

        if not all_dfs:
            logger.warning("trino_query_empty", start=str(start), end=str(end))
            return pl.DataFrame()

        df = pl.concat(all_dfs)

        # Rename columns to match our schema
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
        if "time" in df.columns and (df["time"].dtype == pl.Datetime or str(df["time"].dtype).startswith("Datetime")):
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
            "icao24",
            "callsign",
            "origin_country",
            "time_position",
            "last_contact",
            "lon",
            "lat",
            "baroaltitude",
            "onground",
            "velocity",
            "heading",
            "vertrate",
            "sensors",
            "geoaltitude",
            "squawk",
            "spi",
            "position_source",
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
