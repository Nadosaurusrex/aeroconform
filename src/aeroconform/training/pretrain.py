"""Foundation model pre-training loop.

Self-supervised next-patch prediction on ADS-B trajectory data
using the Gaussian mixture NLL loss with mixed-precision training.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader, random_split

from aeroconform.config import AeroConformConfig
from aeroconform.data.dataset import TrajectoryDataset
from aeroconform.models.trajectory_model import TrajectoryTransformer
from aeroconform.training.utils import (
    clip_gradients,
    get_cosine_warmup_scheduler,
    log_metrics,
    save_checkpoint,
)

logger = structlog.get_logger(__name__)


def pretrain(
    model: TrajectoryTransformer,
    dataset: TrajectoryDataset,
    config: AeroConformConfig,
    device: str = "cpu",
) -> TrajectoryTransformer:
    """Pre-train the foundation model on trajectory data.

    Uses AdamW optimizer with cosine warmup LR schedule, mixed precision
    training, gradient accumulation, and early stopping.

    Args:
        model: TrajectoryTransformer to train.
        dataset: TrajectoryDataset with windowed trajectories.
        config: Training configuration.
        device: Device to train on ('cpu', 'cuda', 'mps').

    Returns:
        The trained model.
    """
    model = model.to(device)
    model.train()

    # Train/val split
    val_size = max(1, int(len(dataset) * config.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.pretrain_batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * config.pretrain_epochs // config.gradient_accumulation_steps
    scheduler = get_cosine_warmup_scheduler(
        optimizer, config.pretrain_warmup_steps, total_steps
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(config.checkpoint_dir)

    for epoch in range(config.pretrain_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            x = batch["input"].to(device)
            target = batch["target"].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                means, log_vars, log_weights, _ = model(x)
                loss = model.output_head.nll_loss(means, log_vars, log_weights, target)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                clip_gradients(model, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)

        # Validation
        val_loss = _evaluate(model, val_loader, device, use_amp)
        log_metrics("pretrain", epoch, {
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Checkpointing and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                checkpoint_dir / "best_pretrain.pt",
                val_loss=val_loss,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info("early_stopping", epoch=epoch, best_val_loss=best_val_loss)
                break

        if (epoch + 1) % config.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch,
                checkpoint_dir / f"pretrain_epoch_{epoch}.pt",
                val_loss=val_loss,
            )

    return model


def _evaluate(
    model: TrajectoryTransformer,
    val_loader: DataLoader,
    device: str,
    use_amp: bool,
) -> float:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate.
        val_loader: Validation DataLoader.
        device: Device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                means, log_vars, log_weights, _ = model(x)
                loss = model.output_head.nll_loss(means, log_vars, log_weights, target)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)
