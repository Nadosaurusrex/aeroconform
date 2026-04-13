"""Visualization utilities for AeroConform evaluation.

Generates ROC curves, coverage plots, detection delay histograms,
attention heatmaps, trajectory overlays, and score distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import structlog
from sklearn.metrics import roc_curve

logger = structlog.get_logger(__name__)


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "ROC Curves",
) -> None:
    """Plot ROC curves for multiple methods.

    Args:
        results: Dict mapping method name to (scores, labels) tuples.
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (scores, labels) in results.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_conformal_coverage(
    alpha_values: np.ndarray,
    empirical_fars: np.ndarray,
    output_path: Path,
) -> None:
    """Plot empirical FAR vs nominal alpha for conformal coverage.

    Args:
        alpha_values: Array of alpha values tested.
        empirical_fars: Corresponding empirical false alarm rates.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(alpha_values, empirical_fars, "o-", label="Empirical FAR")
    ax.plot([0, max(alpha_values)], [0, max(alpha_values)], "k--", alpha=0.3, label="Ideal (FAR = alpha)")
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Empirical FAR")
    ax.set_title("Conformal Coverage Guarantee")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_detection_delay_histogram(
    delays: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Plot histogram of detection delays per anomaly type.

    Args:
        delays: Dict mapping anomaly type to list of delays.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for atype, d in delays.items():
        if d:
            ax.hist(d, bins=20, alpha=0.5, label=atype)

    ax.set_xlabel("Detection Delay (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title("Detection Delay Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_score_distribution(
    clean_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot non-conformity score distribution for clean vs anomalous.

    Args:
        clean_scores: Scores from clean trajectories.
        anomaly_scores: Scores from anomalous trajectories.
        threshold: Detection threshold.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(clean_scores, bins=50, alpha=0.5, label="Clean", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.5, label="Anomalous", density=True)
    ax.axvline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.2f})")

    ax.set_xlabel("Non-conformity Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Clean vs Anomalous")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_trajectory_anomaly_overlay(
    trajectory: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = "Trajectory Anomaly Overlay",
) -> None:
    """Plot a trajectory colored by anomaly score.

    Args:
        trajectory: State vectors (T, 6) with lat/lon in columns 0/1.
        scores: Anomaly scores of shape (T,).
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        trajectory[:, 1], trajectory[:, 0],
        c=scores, cmap="RdYlGn_r", s=10, alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="Anomaly Score")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "GATv2 Attention Weights",
) -> None:
    """Plot attention weight heatmap for an airspace graph.

    Args:
        attention_weights: Attention matrix of shape (N, N).
        labels: Node labels (e.g., callsigns).
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    n = len(labels)
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))
