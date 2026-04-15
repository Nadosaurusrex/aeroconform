"""PyTorch datasets for trajectory model training.

Provides IterableDataset for streaming from Parquet files with random
window sampling, and a MapDataset for pre-loaded smaller datasets.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from torch.utils.data import Dataset, IterableDataset

from src.data.flight_segmentation import segment_flights
from src.data.preprocessing import (
    FEATURE_COLUMNS,
    compute_elapsed_seconds,
    compute_time_gaps,
    delta_encode,
    encode_state_vectors,
    extract_features,
    normalize,
)
from src.data.schemas import Flight, NormStats, TrajectoryWindow
from src.utils.constants import CONTEXT_LENGTH


def flight_to_windows(
    flight: Flight,
    context_length: int = CONTEXT_LENGTH,
    norm_stats: NormStats | None = None,
    stride: int | None = None,
) -> list[TrajectoryWindow]:
    """Convert a flight into fixed-length trajectory windows.

    Args:
        flight: Flight object with features and timestamps.
        context_length: Window length in timesteps.
        norm_stats: If provided, normalize deltas.
        stride: Step size for sliding window. Defaults to context_length (no overlap).

    Returns:
        List of TrajectoryWindow objects.
    """
    if stride is None:
        stride = context_length

    if flight.num_steps < 2:
        return []

    deltas = delta_encode(flight.features)
    if norm_stats is not None:
        deltas = normalize(deltas, norm_stats)

    elapsed = compute_elapsed_seconds(flight.timestamps)
    time_gaps = compute_time_gaps(flight.timestamps)

    windows: list[TrajectoryWindow] = []

    for start in range(0, max(1, flight.num_steps - context_length + 1), stride):
        end = min(start + context_length, flight.num_steps)
        seq_len = end - start

        # Pad if needed
        data = np.zeros((context_length, 8), dtype=np.float32)
        tg = np.zeros(context_length, dtype=np.float32)
        mask = np.zeros(context_length, dtype=np.bool_)

        data[:seq_len] = deltas[start:end]
        tg[:seq_len] = elapsed[start:end] - elapsed[start]  # Relative elapsed time
        mask[:seq_len] = True

        windows.append(
            TrajectoryWindow(
                data=data,
                time_gaps=tg,
                mask=mask,
                icao24=flight.icao24,
                seq_len=seq_len,
            )
        )

    return windows


class TrajectoryMapDataset(Dataset):
    """Pre-loaded trajectory dataset for smaller datasets.

    Loads all flights into memory and pre-computes windows.
    """

    def __init__(
        self,
        flights: list[Flight],
        context_length: int = CONTEXT_LENGTH,
        norm_stats: NormStats | None = None,
    ) -> None:
        self.windows: list[TrajectoryWindow] = []
        for flight in flights:
            self.windows.extend(
                flight_to_windows(flight, context_length, norm_stats)
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = self.windows[idx]
        return {
            "input": torch.from_numpy(w.data[:-1]),   # (context_length-1, 8)
            "target": torch.from_numpy(w.data[1:]),    # (context_length-1, 8)
            "time_gaps": torch.from_numpy(w.time_gaps[:-1]),  # (context_length-1,)
            "mask": torch.from_numpy(w.mask[:-1]),     # (context_length-1,)
        }


class TrajectoryIterableDataset(IterableDataset):
    """Streaming trajectory dataset from Parquet files.

    Reads Parquet files, segments flights, and yields random windows.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        context_length: int = CONTEXT_LENGTH,
        norm_stats: NormStats | None = None,
        shuffle: bool = True,
    ) -> None:
        self.parquet_paths = parquet_paths
        self.context_length = context_length
        self.norm_stats = norm_stats
        self.shuffle = shuffle

    def __iter__(self):  # noqa: ANN204
        paths = list(self.parquet_paths)
        if self.shuffle:
            random.shuffle(paths)

        for path in paths:
            df = pl.read_parquet(path)
            # If raw data, encode first
            if "baroaltitude" in df.columns and "latitude" not in df.columns:
                df = encode_state_vectors(df)

            flights = segment_flights(df)
            if self.shuffle:
                random.shuffle(flights)

            for flight in flights:
                windows = flight_to_windows(
                    flight, self.context_length, self.norm_stats
                )
                if self.shuffle:
                    random.shuffle(windows)

                for w in windows:
                    yield {
                        "input": torch.from_numpy(w.data[:-1]),
                        "target": torch.from_numpy(w.data[1:]),
                        "time_gaps": torch.from_numpy(w.time_gaps[:-1]),
                        "mask": torch.from_numpy(w.mask[:-1]),
                    }


def collate_trajectories(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Stacks tensors from individual samples into batches.

    Args:
        batch: List of sample dicts from dataset.

    Returns:
        Dict with batched tensors.
    """
    return {
        "input": torch.stack([s["input"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
        "time_gaps": torch.stack([s["time_gaps"] for s in batch]),
        "mask": torch.stack([s["mask"] for s in batch]),
    }
