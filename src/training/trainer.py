"""Step-based training loop for AeroGPT.

Per ARCHITECTURE.md: 100K steps, AdamW, cosine warmup, bf16, gradient clipping.
"""

from __future__ import annotations

import math
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.aerogpt import AeroGPT
from src.models.losses import gaussian_nll_loss
from src.utils.config import TrainingConfig

logger = structlog.get_logger(__name__)


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Learning rate scheduler: linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


class Trainer:
    """Step-based trainer for AeroGPT foundation model."""

    def __init__(
        self,
        model: AeroGPT,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        checkpoint_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir

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

        # Mixed precision
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute one training step.

        Args:
            batch: Dict with keys 'input', 'target', 'time_gaps', 'mask'.

        Returns:
            Loss value for this step.
        """
        self.model.train()

        x = batch["input"].to(self.device)
        target = batch["target"].to(self.device)
        time_gaps = batch["time_gaps"].to(self.device)
        mask = batch["mask"].to(self.device)

        self.optimizer.zero_grad()

        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            means, log_vars, _ = self.model(x, time_gaps, mask)
            loss = gaussian_nll_loss(means, log_vars, target, mask)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set.

        Returns:
            Average validation loss.
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            time_gaps = batch["time_gaps"].to(self.device)
            mask = batch["mask"].to(self.device)

            means, log_vars, _ = self.model(x, time_gaps, mask)
            loss = gaussian_nll_loss(means, log_vars, target, mask)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, path: Path, *, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            path: File path for checkpoint.
            is_best: Whether this is the best model so far.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.model.config,
        }
        torch.save(checkpoint, path)
        logger.info("checkpoint_saved", path=str(path), step=self.global_step, is_best=is_best)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: File path for checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("checkpoint_loaded", path=str(path), step=self.global_step)

    def train(self) -> dict[str, list[float]]:
        """Run full training loop.

        Returns:
            Dict with training history (losses, val_losses, learning_rates).
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            # Get next batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss = self.train_step(batch)
            lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(loss)
            history["lr"].append(lr)

            # Logging
            if self.global_step % 100 == 0:
                logger.info(
                    "training_step",
                    step=self.global_step,
                    loss=f"{loss:.4f}",
                    lr=f"{lr:.2e}",
                )

            # Evaluation
            if self.global_step % self.config.eval_every == 0:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                logger.info(
                    "evaluation",
                    step=self.global_step,
                    val_loss=f"{val_loss:.4f}",
                )

                if val_loss < self.best_val_loss and self.checkpoint_dir:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        self.checkpoint_dir / "best.pt", is_best=True
                    )

            # Regular checkpointing
            if self.global_step % self.config.checkpoint_every == 0 and self.checkpoint_dir:
                self.save_checkpoint(
                    self.checkpoint_dir / f"step_{self.global_step}.pt"
                )

        return history
