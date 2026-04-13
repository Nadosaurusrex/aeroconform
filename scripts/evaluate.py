"""CLI entry point for AeroConform evaluation and benchmarking."""

from __future__ import annotations

import argparse

import numpy as np
import structlog

from aeroconform.config import AeroConformConfig
from aeroconform.data.synthetic_anomalies import generate_evaluation_set
from aeroconform.evaluation.benchmark import run_benchmark
from aeroconform.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def main() -> None:
    """Entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(description="AeroConform Evaluation")
    parser.add_argument(
        "--n-clean", type=int, default=100,
        help="Number of clean test trajectories",
    )
    parser.add_argument(
        "--anomalies-per-type", type=int, default=20,
        help="Number of anomalous trajectories per type",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    # Generate synthetic evaluation data
    rng = np.random.default_rng(args.seed)
    clean_trajs = []
    for _ in range(args.n_clean):
        t_len = 200
        traj = np.zeros((t_len, 6))
        traj[:, 0] = np.linspace(45, 46, t_len) + rng.normal(0, 0.001, t_len)
        traj[:, 1] = np.linspace(9, 10, t_len) + rng.normal(0, 0.001, t_len)
        traj[:, 2] = 35000 + rng.normal(0, 100, t_len)
        traj[:, 3] = 450 + rng.normal(0, 5, t_len)
        traj[:, 4] = 90 + rng.normal(0, 1, t_len)
        traj[:, 5] = rng.normal(0, 50, t_len)
        clean_trajs.append(traj)

    test_trajs, test_labels, test_types = generate_evaluation_set(
        clean_trajs, anomalies_per_type=args.anomalies_per_type, seed=args.seed,
    )

    # Run baselines
    results = run_benchmark(
        train_trajectories=clean_trajs[:50],
        test_trajectories=test_trajs,
        test_labels=test_labels,
        test_types=test_types,
    )

    # Print results
    for name, metrics in results.items():
        logger.info("baseline_results", name=name, **metrics)


if __name__ == "__main__":
    main()
