"""Configuration management via YAML + dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    """AeroGPT model architecture configuration."""

    name: str = "aerogpt"
    input_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    feedforward_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    output_type: str = "gaussian"
    output_dim: int = 16

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Load model config from YAML file."""
        data = load_yaml(path)
        return cls(**data["model"])


@dataclass
class BBox:
    """Geographic bounding box."""

    west: float = 6.5
    south: float = 44.0
    east: float = 13.5
    north: float = 47.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as (west, south, east, north) for pyopensky."""
        return (self.west, self.south, self.east, self.north)


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    fir: str = "LIMM"
    bbox: BBox = field(default_factory=BBox)
    context_length: int = 128
    min_flight_observations: int = 20
    gap_threshold_minutes: int = 30
    features: list[str] = field(
        default_factory=lambda: [
            "latitude",
            "longitude",
            "baro_altitude",
            "velocity",
            "sin_track",
            "cos_track",
            "vertical_rate",
            "on_ground",
        ]
    )

    @classmethod
    def from_yaml(cls, path: Path) -> DataConfig:
        """Load data config from YAML file."""
        data = load_yaml(path)
        d = data["data"]
        bbox = BBox(**d.pop("bbox"))
        return cls(bbox=bbox, **d)


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    scheduler: str = "cosine_with_warmup"
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_every: int = 5000
    eval_every: int = 1000

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load training config from YAML file."""
        data = load_yaml(path)
        return cls(**data["training"])


@dataclass
class GraphConfig:
    """Graph attention layer configuration."""

    conv_type: str = "GATv2Conv"
    num_layers: int = 2
    hidden_dim: int = 128
    num_heads: int = 4
    edge_dim: int = 5
    dropout: float = 0.1
    residual: bool = True
    output_dim: int = 64
    joint_loss_weight: float = 0.1

    @classmethod
    def from_yaml(cls, path: Path) -> GraphConfig:
        """Load graph config from YAML file."""
        data = load_yaml(path)
        return cls(**data["graph"])


@dataclass
class ConformalConfig:
    """Conformal prediction configuration."""

    alpha: float = 0.01
    buffer_size: int = 2000
    decay: float = 0.995
    score_weight: float = 0.5

    @classmethod
    def from_yaml(cls, path: Path) -> ConformalConfig:
        """Load conformal config from YAML file."""
        data = load_yaml(path)
        return cls(**data.get("conformal", data))
