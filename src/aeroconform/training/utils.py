"""Training utilities for AeroConform.

Provides learning rate scheduling, gradient clipping, checkpoint
management, and structured logging for training loops.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = structlog.get_logger(__name__)


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a cosine annealing scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps with linear LR increase.
        total_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients by global norm.

    Args:
        model: The model whose gradients to clip.
        max_norm: Maximum gradient norm.

    Returns:
        The total gradient norm before clipping.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    filepath: Path,
    **extra: Any,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        filepath: Path to save the checkpoint.
        **extra: Additional items to include in the checkpoint.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        **extra,
    }
    torch.save(checkpoint, filepath)
    logger.info("checkpoint_saved", filepath=str(filepath), epoch=epoch)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Dict with checkpoint metadata (epoch, etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint: dict[str, Any] = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        "checkpoint_loaded",
        filepath=str(filepath),
        epoch=checkpoint.get("epoch", -1),
    )
    return checkpoint


def log_metrics(
    phase: str,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Log training metrics using structured logging.

    Args:
        phase: Training phase name (e.g., 'pretrain', 'graph').
        epoch: Current epoch number.
        metrics: Dict of metric name -> value.
    """
    logger.info(
        "training_metrics",
        phase=phase,
        epoch=epoch,
        **metrics,
    )
