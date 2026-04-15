"""Flight segmentation: group state vectors by aircraft and segment into individual flights.

Logic per ARCHITECTURE.md:
- Group by icao24
- Sort by time within each group
- Split into flights on gaps > 30 minutes
- Discard flights with fewer than 20 observations
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

from src.data.preprocessing import FEATURE_COLUMNS, extract_features
from src.data.schemas import Flight
from src.utils.constants import GAP_THRESHOLD_SECONDS, MIN_FLIGHT_OBSERVATIONS

logger = structlog.get_logger(__name__)


def segment_flights(
    df: pl.DataFrame,
    gap_threshold_seconds: int = GAP_THRESHOLD_SECONDS,
    min_observations: int = MIN_FLIGHT_OBSERVATIONS,
) -> list[Flight]:
    """Segment encoded state vectors into individual flights.

    Groups by icao24, sorts by time, splits on time gaps > threshold,
    and filters out flights with too few observations.

    Args:
        df: Encoded DataFrame with columns: icao24, time, callsign, + FEATURE_COLUMNS.
        gap_threshold_seconds: Maximum gap between consecutive observations
            before splitting into a new flight. Default 30 minutes.
        min_observations: Minimum number of observations for a valid flight.
            Default 20.

    Returns:
        List of Flight objects, each representing one continuous trajectory.
    """
    required_cols = {"icao24", "time", *FEATURE_COLUMNS}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    flights: list[Flight] = []
    total_segments = 0
    total_discarded = 0

    # Group by aircraft
    for icao24, group_df in df.group_by("icao24"):
        icao = icao24[0] if isinstance(icao24, tuple) else icao24
        group_sorted = group_df.sort("time")

        # Compute time gaps and identify segment boundaries
        times = group_sorted["time"].to_numpy()
        gaps = np.diff(times)
        split_indices = np.where(gaps > gap_threshold_seconds)[0] + 1

        # Split into segments
        segments = np.split(np.arange(len(group_sorted)), split_indices)

        for segment_indices in segments:
            total_segments += 1

            if len(segment_indices) < min_observations:
                total_discarded += 1
                continue

            segment_df = group_sorted[segment_indices.tolist()]
            timestamps = segment_df["time"].to_numpy().astype(np.int64)
            features = extract_features(segment_df)

            callsign = None
            cs_col = segment_df["callsign"].drop_nulls()
            if len(cs_col) > 0:
                callsign = cs_col[0]

            flights.append(
                Flight(
                    icao24=str(icao),
                    callsign=callsign,
                    timestamps=timestamps,
                    features=features,
                )
            )

    logger.info(
        "flight_segmentation_complete",
        total_segments=total_segments,
        valid_flights=len(flights),
        discarded=total_discarded,
        unique_aircraft=df["icao24"].n_unique(),
    )

    return flights
