"""CLI script to download historical ADS-B data from OpenSky Network.

Polls the OpenSky API at regular intervals and saves state vectors
to hourly Parquet files for training data collection.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import structlog

from aeroconform.config import AeroConformConfig
from aeroconform.data.historical import HistoricalDataManager
from aeroconform.data.opensky_client import OpenSkyClient
from aeroconform.utils.airspace import get_fir_bbox
from aeroconform.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


async def collect_data(args: argparse.Namespace) -> None:
    """Run the data collection loop.

    Args:
        args: Parsed command-line arguments.
    """
    config = AeroConformConfig(
        target_fir=args.fir,
        data_dir=args.output_dir,
    )

    bbox = get_fir_bbox(args.fir)
    client = OpenSkyClient(config=config)
    manager = HistoricalDataManager(config=config)

    logger.info(
        "collection_started",
        fir=args.fir,
        bbox=bbox,
        duration_hours=args.duration,
        interval=args.interval,
    )

    end_time = time.time() + args.duration * 3600
    snapshots = 0

    async for states in client.stream_states(bbox=bbox, poll_interval=args.interval):
        if time.time() >= end_time:
            break

        ts = int(time.time())
        manager.append_states(states, timestamp=ts)
        snapshots += 1

        if snapshots % 100 == 0:
            stats = manager.get_statistics()
            logger.info(
                "collection_progress",
                snapshots=snapshots,
                **stats,
            )

    stats = manager.get_statistics()
    logger.info("collection_complete", snapshots=snapshots, **stats)


def main() -> None:
    """Entry point for the data collection CLI."""
    parser = argparse.ArgumentParser(
        description="Download historical ADS-B data from OpenSky Network"
    )
    parser.add_argument(
        "--fir", type=str, default="LIMM",
        help="FIR code to collect data for (default: LIMM)",
    )
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="Collection duration in hours (default: 1.0)",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Polling interval in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    asyncio.run(collect_data(args))


if __name__ == "__main__":
    main()
