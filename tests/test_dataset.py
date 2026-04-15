"""Tests for trajectory dataset."""

from __future__ import annotations

import numpy as np
from src.data.dataset import (
    TrajectoryMapDataset,
    collate_trajectories,
    flight_to_windows,
)
from src.data.schemas import Flight, NormStats


def test_flight_to_windows_shape(synthetic_flight: Flight) -> None:
    """Windows have correct shape."""
    windows = flight_to_windows(synthetic_flight, context_length=32)
    assert len(windows) > 0
    for w in windows:
        assert w.data.shape == (32, 8)
        assert w.time_gaps.shape == (32,)
        assert w.mask.shape == (32,)


def test_flight_to_windows_mask(synthetic_flight: Flight) -> None:
    """Mask correctly indicates valid positions."""
    windows = flight_to_windows(synthetic_flight, context_length=128)
    for w in windows:
        # Valid positions should be True
        assert w.mask[: w.seq_len].all()
        # Padded positions should be False (if any)
        if w.seq_len < 128:
            assert not w.mask[w.seq_len :].any()


def test_flight_to_windows_with_normalization(synthetic_flight: Flight, sample_norm_stats: NormStats) -> None:
    """Windows are normalized when stats provided."""
    windows = flight_to_windows(synthetic_flight, context_length=32, norm_stats=sample_norm_stats)
    assert len(windows) > 0


def test_map_dataset(synthetic_flight: Flight) -> None:
    """MapDataset yields correct tensor shapes."""
    dataset = TrajectoryMapDataset([synthetic_flight], context_length=32)
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["input"].shape == (31, 8)
    assert sample["target"].shape == (31, 8)
    assert sample["time_gaps"].shape == (31,)
    assert sample["mask"].shape == (31,)


def test_collate_trajectories(synthetic_flight: Flight) -> None:
    """Collate function produces batched tensors."""
    dataset = TrajectoryMapDataset([synthetic_flight], context_length=32)
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    collated = collate_trajectories(batch)

    batch_size = len(batch)
    assert collated["input"].shape == (batch_size, 31, 8)
    assert collated["target"].shape == (batch_size, 31, 8)
    assert collated["time_gaps"].shape == (batch_size, 31)
    assert collated["mask"].shape == (batch_size, 31)


def test_short_flight_no_windows() -> None:
    """Flight with < 2 steps produces no windows."""
    flight = Flight(
        icao24="test",
        callsign="T",
        timestamps=np.array([0], dtype=np.int64),
        features=np.zeros((1, 8), dtype=np.float32),
    )
    windows = flight_to_windows(flight, context_length=128)
    assert len(windows) == 0
