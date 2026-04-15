"""Dynamic airspace graph construction for AirGraph.

Builds PyTorch Geometric Data objects from concurrent aircraft states.
Per ARCHITECTURE.md section 3:
- Nodes: aircraft with >=5 recent observations
- Node features: R^264 (trajectory embedding R^256 + kinematic state R^8)
- Edges: proximity <50 NM or altitude sep <1000 ft when lateral <10 NM
- Edge features: R^5 (distance_km, closing_speed_mps, altitude_diff_m,
                       relative_bearing_rad, time_to_cpa_s)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data

from src.utils.constants import FT_TO_M, NM_TO_KM
from src.utils.geo import bearing_rad, closing_speed_mps, haversine_km, time_to_cpa_seconds

# Edge criteria thresholds
MAX_DISTANCE_NM = 50.0
MAX_DISTANCE_KM = MAX_DISTANCE_NM * NM_TO_KM
ALTITUDE_SEP_FT = 1000.0
ALTITUDE_SEP_M = ALTITUDE_SEP_FT * FT_TO_M
LATERAL_DISTANCE_NM = 10.0
LATERAL_DISTANCE_KM = LATERAL_DISTANCE_NM * NM_TO_KM


class AirspaceGraphBuilder:
    """Build dynamic graphs from concurrent aircraft states."""

    def __init__(
        self,
        max_distance_km: float = MAX_DISTANCE_KM,
        altitude_sep_m: float = ALTITUDE_SEP_M,
        lateral_distance_km: float = LATERAL_DISTANCE_KM,
    ) -> None:
        self.max_distance_km = max_distance_km
        self.altitude_sep_m = altitude_sep_m
        self.lateral_distance_km = lateral_distance_km

    def build_graph(
        self,
        states: npt.NDArray[np.float32],
        embeddings: torch.Tensor,
    ) -> Data:
        """Construct a PyG Data object from concurrent aircraft states.

        Args:
            states: (num_aircraft, 8) current kinematic states
                    [lat, lon, alt, vel, sin_track, cos_track, vrate, on_ground].
            embeddings: (num_aircraft, embed_dim) trajectory embeddings from AeroGPT.

        Returns:
            PyG Data with node features, edge index, and edge attributes.
        """
        num_aircraft = len(states)

        if num_aircraft == 0:
            return Data(
                x=torch.zeros(0, embeddings.shape[1] + 8),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 5),
            )

        # Node features: concat embedding + state
        states_tensor = torch.from_numpy(states).float()
        node_features = torch.cat([embeddings, states_tensor], dim=-1)

        # Compute pairwise edges
        edge_index, edge_attr = self._compute_edges(states)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def _compute_edges(
        self,
        states: npt.NDArray[np.float32],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge connectivity and features.

        Uses vectorized computation for all pairs.

        Args:
            states: (N, 8) aircraft states.

        Returns:
            edge_index: (2, num_edges) source-target pairs.
            edge_attr: (num_edges, 5) edge features.
        """
        n = len(states)
        if n < 2:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 5, dtype=torch.float32),
            )

        lats = states[:, 0].astype(np.float64)
        lons = states[:, 1].astype(np.float64)
        alts = states[:, 2].astype(np.float64)
        vels = states[:, 3].astype(np.float64)
        sin_tracks = states[:, 4].astype(np.float64)
        cos_tracks = states[:, 5].astype(np.float64)
        headings = np.arctan2(sin_tracks, cos_tracks)

        # Compute all pairwise distances (vectorized)
        src_indices = []
        dst_indices = []
        edge_features = []

        for i in range(n):
            # Compute distance from aircraft i to all others
            others = np.arange(n)
            others = others[others != i]

            dists = haversine_km(
                np.full(len(others), lats[i]),
                np.full(len(others), lons[i]),
                lats[others],
                lons[others],
            )

            alt_diffs = np.abs(alts[i] - alts[others])

            # Edge criteria: distance < 50 NM OR (alt_sep < 1000 ft AND lateral < 10 NM)
            criterion1 = dists < self.max_distance_km
            criterion2 = (alt_diffs < self.altitude_sep_m) & (dists < self.lateral_distance_km)
            connected = criterion1 | criterion2

            if not connected.any():
                continue

            connected_idx = others[connected]
            connected_dists = dists[connected]
            connected_alt_diffs = alt_diffs[connected]

            # Compute edge features for connected pairs
            bearings = bearing_rad(
                np.full(len(connected_idx), lats[i]),
                np.full(len(connected_idx), lons[i]),
                lats[connected_idx],
                lons[connected_idx],
            )

            close_speeds = closing_speed_mps(
                np.full(len(connected_idx), vels[i]),
                np.full(len(connected_idx), headings[i]),
                vels[connected_idx],
                headings[connected_idx],
                bearings,
            )

            tcpa = time_to_cpa_seconds(connected_dists, close_speeds)

            for j_local, j_global in enumerate(connected_idx):
                src_indices.append(i)
                dst_indices.append(j_global)
                edge_features.append([
                    connected_dists[j_local],
                    close_speeds[j_local],
                    connected_alt_diffs[j_local],
                    bearings[j_local],
                    tcpa[j_local],
                ])

        if not src_indices:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 5, dtype=torch.float32),
            )

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return edge_index, edge_attr
