"""Tests for flight segmentation."""

from __future__ import annotations

import numpy as np
import polars as pl
from src.data.flight_segmentation import segment_flights
from src.data.preprocessing import encode_state_vectors


def test_segment_basic(synthetic_state_vectors: pl.DataFrame) -> None:
    """Basic segmentation produces flights from valid data."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    flights = segment_flights(encoded)
    assert len(flights) > 0
    for flight in flights:
        assert flight.num_steps >= 20
        assert flight.features.shape[1] == 8


def test_segment_splits_on_gap(synthetic_state_vectors: pl.DataFrame) -> None:
    """Flights are split when time gap exceeds threshold."""
    encoded = encode_state_vectors(synthetic_state_vectors)

    # Insert a 2-hour gap in the middle of one aircraft's data
    mask = encoded["icao24"] == "aabbcc"
    aabbcc = encoded.filter(mask)
    others = encoded.filter(~mask)

    mid = len(aabbcc) // 2
    modified = (
        aabbcc.with_row_index("_idx")
        .with_columns(
            pl.when(pl.col("_idx") >= mid)
            .then(pl.col("time") + 7200)  # +2 hours
            .otherwise(pl.col("time"))
            .alias("time")
        )
        .drop("_idx")
    )

    combined = pl.concat([modified, others])
    flights = segment_flights(combined)

    # Should have more flights since aabbcc was split
    aabbcc_flights = [f for f in flights if f.icao24 == "aabbcc"]
    # With 50 obs split in half (25 each), and min_observations=20, both halves should pass
    assert len(aabbcc_flights) == 2


def test_segment_discards_short() -> None:
    """Flights with fewer than min_observations are discarded."""
    # Create a very short trajectory
    rows = [
        {
            "icao24": "short1",
            "time": 1700000000 + t * 10,
            "callsign": "SH1",
            "latitude": 45.0,
            "longitude": 9.0,
            "baro_altitude": 10000.0,
            "velocity": 250.0,
            "sin_track": 0.0,
            "cos_track": 1.0,
            "vertical_rate": 0.0,
            "on_ground": 0.0,
        }
        for t in range(10)  # Only 10 observations
    ]
    df = pl.DataFrame(rows)
    flights = segment_flights(df, min_observations=20)
    assert len(flights) == 0


def test_segment_preserves_icao24(synthetic_state_vectors: pl.DataFrame) -> None:
    """Each flight has a valid icao24."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    flights = segment_flights(encoded)
    for flight in flights:
        assert isinstance(flight.icao24, str)
        assert len(flight.icao24) > 0


def test_segment_timestamps_sorted(synthetic_state_vectors: pl.DataFrame) -> None:
    """Flight timestamps are monotonically increasing."""
    encoded = encode_state_vectors(synthetic_state_vectors)
    flights = segment_flights(encoded)
    for flight in flights:
        assert np.all(np.diff(flight.timestamps) > 0)
