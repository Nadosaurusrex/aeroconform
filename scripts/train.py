"""CLI entry point for AeroConform training.

Supports three training phases:
- pretrain: Foundation model pre-training
- graph: Graph attention layer training
- calibrate: Conformal calibration
"""

from __future__ import annotations

import argparse

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader

from aeroconform.config import AeroConformConfig
from aeroconform.data.dataset import TrajectoryDataset
from aeroconform.data.preprocessing import (
    NormStats,
    delta_encode,
    compute_norm_stats,
    normalize,
    window_trajectory,
)
from aeroconform.models.graph_attention import AirspaceGATv2
from aeroconform.models.trajectory_model import TrajectoryTransformer
from aeroconform.training.calibrate import calibrate_conformal
from aeroconform.training.pretrain import pretrain
from aeroconform.training.utils import load_checkpoint, save_checkpoint
from aeroconform.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def generate_synthetic_training_data(
    config: AeroConformConfig, n_trajectories: int = 100
) -> tuple[list[dict], NormStats]:
    """Generate synthetic training data for testing the training loop.

    Args:
        config: Configuration.
        n_trajectories: Number of synthetic trajectories.

    Returns:
        Tuple of (windowed trajectories, normalization stats).
    """
    rng = np.random.default_rng(42)
    trajectories = []

    for _ in range(n_trajectories):
        t_len = rng.integers(config.seq_len + 10, config.seq_len + 100)
        traj = np.zeros((t_len, 6))
        traj[:, 0] = np.linspace(45, 46, t_len) + rng.normal(0, 0.001, t_len)
        traj[:, 1] = np.linspace(9, 10, t_len) + rng.normal(0, 0.001, t_len)
        traj[:, 2] = 35000 + rng.normal(0, 100, t_len)
        traj[:, 3] = 450 + rng.normal(0, 5, t_len)
        traj[:, 4] = 90 + rng.normal(0, 1, t_len)
        traj[:, 5] = rng.normal(0, 50, t_len)
        trajectories.append(traj)

    delta_trajs = [delta_encode(t) for t in trajectories]
    norm_stats = compute_norm_stats(delta_trajs)

    windows = []
    for i, dt in enumerate(delta_trajs):
        normed = normalize(dt, norm_stats)
        wins = window_trajectory(normed, f"synth_{i:04d}", 0, config)
        windows.extend(wins)

    return windows, norm_stats


def main() -> None:
    """Entry point for the training CLI."""
    parser = argparse.ArgumentParser(description="AeroConform Training")
    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["pretrain", "graph", "calibrate"],
        help="Training phase to run",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to train on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)
    config = AeroConformConfig()

    if args.epochs is not None:
        if args.phase == "pretrain":
            config.pretrain_epochs = args.epochs
        elif args.phase == "graph":
            config.graph_epochs = args.epochs

    device = args.device

    if args.phase == "pretrain":
        logger.info("starting_pretrain", device=device)
        model = TrajectoryTransformer.from_config(config)

        if args.synthetic:
            windows, norm_stats = generate_synthetic_training_data(config)
        else:
            logger.error("real_data_not_implemented")
            return

        dataset = TrajectoryDataset(windows, config)
        model = pretrain(model, dataset, config, device=device)
        logger.info("pretrain_complete")

    elif args.phase == "graph":
        logger.info("starting_graph_training", device=device)
        logger.info("graph_training_placeholder")

    elif args.phase == "calibrate":
        logger.info("starting_calibration", device=device)

        model = TrajectoryTransformer.from_config(config)

        if args.synthetic:
            windows, _ = generate_synthetic_training_data(config, n_trajectories=50)
        else:
            logger.error("real_data_not_implemented")
            return

        dataset = TrajectoryDataset(windows, config)
        cal_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        detector = calibrate_conformal(model, cal_loader, config, device=device)
        logger.info("calibration_complete", threshold=detector.get_threshold())


if __name__ == "__main__":
    main()
