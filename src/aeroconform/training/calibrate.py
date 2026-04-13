"""Conformal calibration procedure.

Runs the trained model on clean calibration data to compute
non-conformity scores and initialize the conformal detector.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader

from aeroconform.config import AeroConformConfig
from aeroconform.models.conformal import AdaptiveConformalDetector
from aeroconform.models.graph_attention import AirspaceGATv2
from aeroconform.models.trajectory_model import TrajectoryTransformer

logger = structlog.get_logger(__name__)


def calibrate_conformal(
    model: TrajectoryTransformer,
    cal_loader: DataLoader,
    config: AeroConformConfig,
    graph_model: AirspaceGATv2 | None = None,
    device: str = "cpu",
) -> AdaptiveConformalDetector:
    """Run conformal calibration on clean data.

    Computes non-conformity scores for each prediction in the calibration
    set and initializes the adaptive conformal detector.

    Args:
        model: Trained trajectory foundation model.
        cal_loader: DataLoader with clean calibration trajectories.
        config: Configuration.
        graph_model: Optional trained graph model.
        device: Device to run inference on.

    Returns:
        Calibrated AdaptiveConformalDetector.
    """
    model = model.to(device)
    model.eval()
    if graph_model is not None:
        graph_model = graph_model.to(device)
        graph_model.eval()

    detector = AdaptiveConformalDetector.from_config(config)
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in cal_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)

            means, log_vars, log_weights, _ = model(x)

            # Compute non-conformity scores for each sample and patch
            batch_size = x.shape[0]
            n_patches = means.shape[1]

            for i in range(batch_size):
                for p in range(n_patches - 1):  # Skip last patch (no target)
                    score = detector.compute_nonconformity_score(
                        target[i, p].cpu().numpy(),
                        means[i, p].cpu().numpy(),
                        log_vars[i, p].cpu().numpy(),
                        log_weights[i, p].cpu().numpy(),
                    )
                    if np.isfinite(score):
                        all_scores.append(score)

    if not all_scores:
        logger.warning("no_valid_calibration_scores")
        all_scores = [0.0]

    detector.calibrate(np.array(all_scores))
    logger.info(
        "conformal_calibration_complete",
        n_scores=len(all_scores),
        threshold=detector.get_threshold(),
    )
    return detector
