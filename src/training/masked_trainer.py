"""Masked pre-training for AeroGPT.

After initial next-step pre-training, fine-tune with masked prediction:
randomly mask 15% of states, predict masked states from context.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.aerogpt import AeroGPT
from src.models.losses import gaussian_nll_loss
from src.training.trainer import CosineWarmupScheduler
from src.utils.config import TrainingConfig

logger = structlog.get_logger(__name__)


class MaskedTrainer:
    """Trainer for masked prediction fine-tuning."""

    def __init__(
        self,
        model: AeroGPT,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        checkpoint_dir: Path | None = None,
        mask_ratio: float = 0.15,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.mask_ratio = mask_ratio

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.global_step = 0

    def _apply_mask(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to input sequences.

        Args:
            x: (batch, seq_len, features) input tensor.
            mask: (batch, seq_len) validity mask.

        Returns:
            masked_x: Input with masked positions replaced by mask token.
            mask_positions: (batch, seq_len) boolean, True where masked.
        """
        batch_size, seq_len, _ = x.shape

        # Random mask: mask_ratio of valid positions
        rand = torch.rand(batch_size, seq_len, device=x.device)
        mask_positions = (rand < self.mask_ratio) & mask

        # Zero out masked positions in input space
        masked_x = x.clone()
        masked_x[mask_positions] = 0.0

        return masked_x, mask_positions

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute one masked training step."""
        self.model.train()

        x = batch["input"].to(self.device)
        target = batch["target"].to(self.device)
        time_gaps = batch["time_gaps"].to(self.device)
        mask = batch["mask"].to(self.device)

        # Apply masking
        masked_x, mask_positions = self._apply_mask(x, mask)

        self.optimizer.zero_grad()

        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            means, log_vars, _ = self.model(masked_x, time_gaps, mask)
            # Loss only on masked positions
            loss = gaussian_nll_loss(means, log_vars, target, mask_positions)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def train(self) -> dict[str, list[float]]:
        """Run masked pre-training loop."""
        history: dict[str, list[float]] = {"train_loss": [], "lr": []}
        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss = self.train_step(batch)
            lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(loss)
            history["lr"].append(lr)

            if self.global_step % 100 == 0:
                logger.info(
                    "masked_training_step",
                    step=self.global_step,
                    loss=f"{loss:.4f}",
                    lr=f"{lr:.2e}",
                )

            if self.global_step % self.config.checkpoint_every == 0 and self.checkpoint_dir:
                path = self.checkpoint_dir / f"masked_step_{self.global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "global_step": self.global_step,
                        "config": self.model.config,
                    },
                    path,
                )

        return history
