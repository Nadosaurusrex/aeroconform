"""Central configuration for AeroConform.

All hyperparameters and paths are defined here via pydantic-settings.
Values can be overridden via environment variables prefixed with AEROCONFORM_.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class AeroConformConfig(BaseSettings):
    """Central configuration. All values overridable via env vars (prefix AEROCONFORM_)."""

    model_config = {"env_prefix": "AEROCONFORM_"}

    # Data
    target_fir: str = "LIMM"
    bbox: tuple[float, float, float, float] = (43.5, 47.0, 6.5, 14.0)
    seq_len: int = 128
    patch_len: int = 8
    input_dim: int = 6
    features: list[str] = [
        "latitude",
        "longitude",
        "baro_altitude",
        "velocity",
        "true_track",
        "vertical_rate",
    ]
    min_trajectory_len: int = 30
    window_stride_divisor: int = 2

    # Model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    n_components: int = 5
    dropout: float = 0.1
    max_patches: int = 64

    # Graph
    graph_hidden: int = 128
    graph_heads: int = 4
    graph_layers: int = 2
    edge_dim: int = 4
    proximity_threshold_nm: float = 50.0
    altitude_band_ft: float = 5000.0
    max_edges_per_node: int = 10

    # Conformal
    alpha: float = 0.01
    cal_window: int = 500
    adapt_lr: float = 0.01

    # Training — pretrain
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 128
    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 0.01
    pretrain_warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    checkpoint_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    val_fraction: float = 0.1

    # Training — graph
    graph_epochs: int = 30
    graph_batch_size: int = 16
    graph_lr: float = 1e-3
    graph_weight_decay: float = 0.01
    freeze_foundation: bool = True

    # Paths
    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
    output_dir: Path = Path("./outputs")

    # OpenSky
    opensky_base_url: str = "https://opensky-network.org/api"
    opensky_poll_interval: int = 10

    # Serving
    serve_host: str = "0.0.0.0"
    serve_port: int = 8000

    @property
    def output_dim(self) -> int:
        """Output dimension for the Gaussian mixture head (patch_len * input_dim)."""
        return self.patch_len * self.input_dim

    @property
    def num_patches(self) -> int:
        """Number of patches in a sequence."""
        return self.seq_len // self.patch_len

    @property
    def window_stride(self) -> int:
        """Stride for windowing (seq_len // stride_divisor)."""
        return self.seq_len // self.window_stride_divisor
