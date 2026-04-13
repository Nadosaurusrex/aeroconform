"""PyTorch Dataset classes for AeroConform training.

Provides TrajectoryDataset for foundation model pre-training and
AirspaceSnapshotDataset for graph layer training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aeroconform.config import AeroConformConfig
from aeroconform.data.preprocessing import WindowedTrajectory


class TrajectoryDataset(Dataset):
    """PyTorch dataset for pre-training the trajectory foundation model.

    Each sample consists of:
    - input: delta-encoded, normalized state vectors, shape (seq_len, 6)
    - target: reshaped into patches for next-patch prediction, shape (num_patches, patch_len * 6)
    - mask: boolean mask for valid timesteps, shape (seq_len,)
    - metadata: dict with icao24, start_time

    Args:
        windows: List of windowed trajectory dicts from preprocessing.
        config: AeroConform configuration.
    """

    def __init__(
        self,
        windows: list[WindowedTrajectory],
        config: AeroConformConfig | None = None,
    ) -> None:
        self.config = config or AeroConformConfig()
        self.windows = windows

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'input', 'target', 'mask', 'metadata' keys.
        """
        window = self.windows[idx]
        data = window["data"]  # (seq_len, 6)
        mask = window["mask"]  # (seq_len,)

        input_tensor = torch.from_numpy(data).float()
        mask_tensor = torch.from_numpy(mask).bool()

        # Create target: reshape into patches, shifted by 1 patch
        # Each patch covers patch_len timesteps
        patch_len = self.config.patch_len
        input_dim = self.config.input_dim
        num_patches = self.config.num_patches

        # Reshape data into patches: (num_patches, patch_len * input_dim)
        patched = data.reshape(num_patches, patch_len * input_dim)
        # Target for next-patch prediction: patches shifted by 1
        # For patch i, the target is patch i+1
        target = np.zeros_like(patched)
        target[:-1] = patched[1:]
        target_tensor = torch.from_numpy(target).float()

        return {
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "metadata": {
                "icao24": window["icao24"],
                "start_time": window["start_time"],
            },
        }


class AirspaceSnapshotDataset(Dataset):
    """Dataset for graph layer training on airspace snapshots.

    Each sample represents a full airspace snapshot at time t with
    aircraft as nodes and proximity-based edges.

    Args:
        snapshots: List of snapshot dicts, each containing:
            - node_features: (N, d_model) trajectory embeddings
            - edge_index: (2, E) connectivity
            - edge_attr: (E, 4) edge features
            - targets: (N, output_dim) next-state targets
    """

    def __init__(self, snapshots: list[dict[str, Any]]) -> None:
        self.snapshots = snapshots

    def __len__(self) -> int:
        """Return the number of snapshots."""
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single airspace snapshot.

        Args:
            idx: Snapshot index.

        Returns:
            Dict with 'node_features', 'edge_index', 'edge_attr', 'targets'.
        """
        snap = self.snapshots[idx]

        node_features = snap["node_features"]
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).float()

        edge_index = snap["edge_index"]
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()

        edge_attr = snap["edge_attr"]
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.from_numpy(edge_attr).float()

        targets = snap["targets"]
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "targets": targets,
        }
