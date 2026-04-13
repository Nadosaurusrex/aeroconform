"""Historical data management for AeroConform.

Handles Parquet I/O for historical ADS-B state vectors collected
from the OpenSky Network, with hourly file rotation and append mode.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from aeroconform.config import AeroConformConfig

logger = structlog.get_logger(__name__)

# Parquet schema for stored state vectors
PARQUET_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("icao24", pa.string()),
    ("callsign", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("baro_altitude", pa.float64()),
    ("on_ground", pa.bool_()),
    ("velocity", pa.float64()),
    ("true_track", pa.float64()),
    ("vertical_rate", pa.float64()),
    ("geo_altitude", pa.float64()),
    ("origin_country", pa.string()),
])


class HistoricalDataManager:
    """Manages historical ADS-B data in Parquet format.

    Supports hourly file rotation for data collection and efficient
    reading of time-range queries for training.

    Args:
        config: AeroConform configuration.
    """

    def __init__(self, config: AeroConformConfig | None = None) -> None:
        self.config = config or AeroConformConfig()
        self.data_dir = Path(self.config.data_dir) / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_hourly_path(self, timestamp: int | None = None) -> Path:
        """Get the Parquet file path for a given hour.

        Args:
            timestamp: Unix timestamp (default: current time).

        Returns:
            Path to the hourly Parquet file.
        """
        ts = timestamp or int(time.time())
        hour_ts = ts - (ts % 3600)
        return self.data_dir / f"states_{hour_ts}.parquet"

    def append_states(self, states: pd.DataFrame, timestamp: int | None = None) -> Path:
        """Append state vectors to the current hourly Parquet file.

        Args:
            states: DataFrame with state vector columns.
            timestamp: Unix timestamp for the snapshot.

        Returns:
            Path to the file that was written.
        """
        ts = timestamp or int(time.time())
        filepath = self.get_hourly_path(ts)

        # Add timestamp column
        df = states.copy()
        df["timestamp"] = ts

        # Ensure columns match schema
        for col in PARQUET_SCHEMA.names:
            if col not in df.columns:
                df[col] = None

        df = df[PARQUET_SCHEMA.names]

        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)

        if filepath.exists():
            existing_table = pq.read_table(filepath)
            combined = pa.concat_tables([existing_table, table])
            pq.write_table(combined, filepath)
        else:
            pq.write_table(table, filepath)

        logger.info(
            "states_appended",
            filepath=str(filepath),
            rows=len(df),
        )
        return filepath

    def read_time_range(
        self,
        start_ts: int,
        end_ts: int,
    ) -> pd.DataFrame:
        """Read state vectors within a time range.

        Args:
            start_ts: Start Unix timestamp (inclusive).
            end_ts: End Unix timestamp (inclusive).

        Returns:
            DataFrame with state vectors in the time range.
        """
        frames: list[pd.DataFrame] = []

        # Iterate over hourly files that could contain data in the range
        current_hour = start_ts - (start_ts % 3600)
        while current_hour <= end_ts:
            filepath = self.data_dir / f"states_{current_hour}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
                if len(df) > 0:
                    frames.append(df)
            current_hour += 3600

        if not frames:
            return pd.DataFrame(columns=PARQUET_SCHEMA.names)

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            "time_range_read",
            start=start_ts,
            end=end_ts,
            rows=len(result),
        )
        return result

    def list_files(self) -> list[Path]:
        """List all available Parquet data files sorted by timestamp.

        Returns:
            Sorted list of Parquet file paths.
        """
        files = sorted(self.data_dir.glob("states_*.parquet"))
        return files

    def get_statistics(self) -> dict[str, int | float]:
        """Get collection statistics across all files.

        Returns:
            Dict with total_files, total_rows, earliest_ts, latest_ts.
        """
        files = self.list_files()
        if not files:
            return {"total_files": 0, "total_rows": 0}

        total_rows = 0
        earliest_ts = float("inf")
        latest_ts = float("-inf")

        for filepath in files:
            df = pd.read_parquet(filepath, columns=["timestamp"])
            total_rows += len(df)
            if len(df) > 0:
                earliest_ts = min(earliest_ts, df["timestamp"].min())
                latest_ts = max(latest_ts, df["timestamp"].max())

        return {
            "total_files": len(files),
            "total_rows": total_rows,
            "earliest_ts": int(earliest_ts) if earliest_ts != float("inf") else 0,
            "latest_ts": int(latest_ts) if latest_ts != float("-inf") else 0,
        }
