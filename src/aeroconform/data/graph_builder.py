"""Build dynamic airspace graphs from aircraft state vectors.

Constructs PyTorch Geometric graph objects where aircraft are nodes
(with trajectory embeddings as features) and edges connect aircraft
within proximity and altitude thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
import torch

from aeroconform.config import AeroConformConfig
from aeroconform.utils.geo import bearing_deg, closing_speed_kts, haversine_nm

logger = structlog.get_logger(__name__)


class AirspaceGraphBuilder:
    """Build a PyTorch Geometric graph from a set of aircraft states.

    Connects aircraft pairs that are within proximity_threshold_nm and
    have overlapping altitude bands. Caps edges per node for efficiency.

    Args:
        config: AeroConform configuration.
    """

    def __init__(self, config: AeroConformConfig | None = None) -> None:
        self.config = config or AeroConformConfig()

    def build_graph(
        self,
        states: pd.DataFrame,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build an airspace graph from aircraft states and embeddings.

        Args:
            states: DataFrame with columns: latitude, longitude, baro_altitude,
                velocity, true_track. One row per aircraft.
            embeddings: (N, d_model) trajectory embeddings from the foundation model.

        Returns:
            Dict with:
            - x: (N, d_model) node features (trajectory embeddings)
            - edge_index: (2, E) COO format connectivity
            - edge_attr: (E, 4) edge features [distance_nm, closing_speed_kts,
              alt_diff_ft, bearing_deg]
        """
        n_aircraft = len(states)

        if n_aircraft <= 1:
            return {
                "x": embeddings,
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, self.config.edge_dim), dtype=torch.float32),
            }

        lats = states["latitude"].values
        lons = states["longitude"].values
        alts = states["baro_altitude"].values
        vels = states["velocity"].values
        hdgs = states["true_track"].values

        src_list: list[int] = []
        dst_list: list[int] = []
        attr_list: list[list[float]] = []

        # Track edge count per node
        edge_counts = np.zeros(n_aircraft, dtype=int)

        for i in range(n_aircraft):
            if edge_counts[i] >= self.config.max_edges_per_node:
                continue

            candidates: list[tuple[float, int, list[float]]] = []

            for j in range(i + 1, n_aircraft):
                if edge_counts[j] >= self.config.max_edges_per_node:
                    continue

                # Check altitude band overlap
                alt_diff = abs(alts[i] - alts[j])
                if alt_diff > self.config.altitude_band_ft:
                    continue

                # Check proximity
                dist = haversine_nm(lats[i], lons[i], lats[j], lons[j])
                if dist > self.config.proximity_threshold_nm:
                    continue

                # Compute edge features
                brg = bearing_deg(lats[i], lons[i], lats[j], lons[j])
                cs = closing_speed_kts(
                    lats[i], lons[i], vels[i], hdgs[i],
                    lats[j], lons[j], vels[j], hdgs[j],
                )

                edge_features = [dist, cs, alt_diff, brg]
                candidates.append((dist, j, edge_features))

            # Sort by distance and take closest up to max_edges_per_node
            candidates.sort(key=lambda x: x[0])
            remaining_i = self.config.max_edges_per_node - edge_counts[i]

            for _, j, feats in candidates[:remaining_i]:
                if edge_counts[j] >= self.config.max_edges_per_node:
                    continue

                # Add bidirectional edges
                src_list.extend([i, j])
                dst_list.extend([j, i])
                attr_list.extend([feats, feats])

                edge_counts[i] += 1
                edge_counts[j] += 1

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.config.edge_dim), dtype=torch.float32)

        logger.info(
            "graph_built",
            nodes=n_aircraft,
            edges=edge_index.shape[1],
        )

        return {
            "x": embeddings,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }
