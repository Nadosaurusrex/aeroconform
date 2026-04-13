"""Full AeroConform inference pipeline.

Connects the trajectory foundation model, graph attention network,
and conformal prediction layer into a single inference pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch

from aeroconform.config import AeroConformConfig
from aeroconform.data.preprocessing import NormStats, delta_encode, normalize
from aeroconform.models.conformal import AdaptiveConformalDetector
from aeroconform.models.graph_attention import AirspaceGATv2
from aeroconform.models.trajectory_model import TrajectoryTransformer

logger = structlog.get_logger(__name__)


class AeroConformPipeline:
    """Full AeroConform inference pipeline.

    Processes raw state vectors through the foundation model, optional
    graph layer, and conformal detector to produce anomaly alerts.

    Args:
        model: Trained trajectory transformer.
        detector: Calibrated conformal detector.
        config: Configuration.
        graph_model: Optional trained graph attention model.
        norm_stats: Normalization statistics for preprocessing.
        device: Inference device.
    """

    def __init__(
        self,
        model: TrajectoryTransformer,
        detector: AdaptiveConformalDetector,
        config: AeroConformConfig | None = None,
        graph_model: AirspaceGATv2 | None = None,
        norm_stats: NormStats | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or AeroConformConfig()
        self.model = model.to(device)
        self.model.eval()
        self.detector = detector
        self.graph_model = graph_model
        if self.graph_model is not None:
            self.graph_model = self.graph_model.to(device)
            self.graph_model.eval()
        self.norm_stats = norm_stats or NormStats(median=[0.0] * 6, iqr=[1.0] * 6)
        self.device = device

    def process_trajectory(
        self, trajectory: np.ndarray
    ) -> list[dict[str, Any]]:
        """Process a single trajectory and return per-timestep anomaly results.

        Takes raw absolute state vectors and runs through the full pipeline:
        delta encoding -> normalization -> model -> conformal detection.

        Args:
            trajectory: Raw state vectors of shape (T, 6) with features
                [lat, lon, alt, vel, hdg, vrate].

        Returns:
            List of anomaly detection results, one per predicted patch.
        """
        if len(trajectory) < self.config.patch_len + 1:
            return []

        # Delta encode
        deltas = delta_encode(trajectory)

        # Normalize
        normed = normalize(deltas, self.norm_stats)

        # Truncate or pad to seq_len
        seq_len = self.config.seq_len
        if len(normed) >= seq_len:
            normed = normed[:seq_len]
        else:
            padded = np.zeros((seq_len, normed.shape[1]))
            padded[: len(normed)] = normed
            normed = padded

        # Run model
        x = torch.from_numpy(normed).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            means, log_vars, log_weights, embeddings = self.model(x)

        # Run conformal prediction for each patch
        results: list[dict[str, Any]] = []
        n_patches = means.shape[1]

        for p in range(n_patches - 1):
            # Use the model's prediction for patch p+1
            patch_means = means[0, p].cpu().numpy()
            patch_log_vars = log_vars[0, p].cpu().numpy()
            patch_log_weights = log_weights[0, p].cpu().numpy()

            # Target is the actual next patch (if we have it)
            target_start = (p + 1) * self.config.patch_len
            target_end = target_start + self.config.patch_len
            if target_end > len(deltas):
                break

            target_patch = normed[target_start:target_end].flatten()

            result = self.detector.predict(
                target_patch,
                patch_means,
                patch_log_vars,
                patch_log_weights,
            )
            result["patch_idx"] = p
            results.append(result)

        return results

    def detect_anomalies(
        self, trajectory: np.ndarray, update_calibration: bool = True
    ) -> dict[str, Any]:
        """Run anomaly detection on a trajectory and return a summary.

        Args:
            trajectory: Raw state vectors of shape (T, 6).
            update_calibration: Whether to update the calibration set with
                normal observations.

        Returns:
            Dict with:
            - is_anomalous: bool
            - max_score: float
            - mean_score: float
            - anomalous_patches: list of patch indices
            - results: full per-patch results
        """
        results = self.process_trajectory(trajectory)

        if not results:
            return {
                "is_anomalous": False,
                "max_score": 0.0,
                "mean_score": 0.0,
                "anomalous_patches": [],
                "results": [],
            }

        scores = [r["score"] for r in results]
        anomalous_patches = [r["patch_idx"] for r in results if r["is_anomaly"]]

        # Update calibration with normal observations
        if update_calibration:
            for r in results:
                if not r["is_anomaly"]:
                    self.detector.update(r["score"], is_normal=True)

        return {
            "is_anomalous": len(anomalous_patches) > 0,
            "max_score": max(scores),
            "mean_score": sum(scores) / len(scores),
            "anomalous_patches": anomalous_patches,
            "results": results,
        }
