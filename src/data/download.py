"""Data download script: historical state vectors from Trino or REST API.

Primary path: pyopensky Trino for bulk historical data.
Fallback: REST API with rate limiting.

Usage:
    python -m src.data.download --start 2024-06-01 --end 2024-12-01 --output data/raw
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

from src.data.flight_segmentation import segment_flights
from src.data.opensky_client import TrinoClient, TrinoUnavailableError
from src.data.preprocessing import (
    compute_norm_stats,
    delta_encode,
    encode_state_vectors,
    extract_features,
)
from src.data.schemas import NormStats
from src.utils.constants import LIMM_BBOX
from src.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def download_trino(
    start: str,
    end: str,
    output_dir: Path,
    *,
    chunk_days: int = 1,
) -> list[Path]:
    """Download historical data via Trino in daily chunks.

    Args:
        start: Start date (ISO format).
        end: End date (ISO format).
        output_dir: Directory to save Parquet files.
        chunk_days: Number of days per query chunk.

    Returns:
        List of saved Parquet file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = TrinoClient(bounds=LIMM_BBOX)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    current = start_dt

    saved_paths: list[Path] = []

    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        date_str = current.strftime("%Y-%m-%d")
        output_path = output_dir / f"state_vectors_{date_str}.parquet"

        if output_path.exists():
            logger.info("chunk_already_exists", date=date_str)
            saved_paths.append(output_path)
            current = chunk_end
            continue

        logger.info("downloading_chunk", start=date_str, end=chunk_end.strftime("%Y-%m-%d"))

        try:
            df = client.query_state_vectors(current, chunk_end)
            if len(df) > 0:
                # Encode state vectors
                encoded = encode_state_vectors(df)
                encoded.write_parquet(output_path)
                saved_paths.append(output_path)
                logger.info("chunk_saved", date=date_str, rows=len(encoded))
            else:
                logger.warning("chunk_empty", date=date_str)
        except TrinoUnavailableError:
            logger.error("trino_unavailable", date=date_str)
            raise

        current = chunk_end

    return saved_paths


def compute_and_save_norm_stats(
    parquet_paths: list[Path],
    output_path: Path,
) -> NormStats:
    """Compute normalization statistics from downloaded data.

    Args:
        parquet_paths: List of Parquet files to process.
        output_path: Path to save norm stats.

    Returns:
        Computed NormStats.
    """
    logger.info("computing_norm_stats", num_files=len(parquet_paths))

    all_deltas = []
    total_flights = 0

    for path in parquet_paths:
        df = pl.read_parquet(path)
        flights = segment_flights(df)
        total_flights += len(flights)

        for flight in flights:
            deltas = delta_encode(flight.features)
            all_deltas.append(deltas)

    stats = compute_norm_stats(all_deltas)
    stats.save(str(output_path))

    logger.info(
        "norm_stats_computed",
        total_flights=total_flights,
        mean=stats.mean.tolist(),
        std=stats.std.tolist(),
    )

    return stats


def main() -> None:
    """CLI entry point for data download."""
    parser = argparse.ArgumentParser(description="Download OpenSky historical data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--chunk-days", type=int, default=1, help="Days per query chunk")
    args = parser.parse_args()

    setup_logging(level="INFO")

    output_dir = Path(args.output)
    paths = download_trino(args.start, args.end, output_dir, chunk_days=args.chunk_days)

    if paths:
        stats_path = output_dir.parent / "norm_stats.npz"
        compute_and_save_norm_stats(paths, stats_path)
        logger.info("download_complete", num_files=len(paths))
    else:
        logger.warning("no_data_downloaded")


if __name__ == "__main__":
    main()
