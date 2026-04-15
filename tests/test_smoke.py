"""Smoke tests to verify project structure and imports."""

from __future__ import annotations

from pathlib import Path


def test_package_imports() -> None:
    """Verify all packages can be imported."""
    import src.api
    import src.data
    import src.evaluation
    import src.inference
    import src.models
    import src.training
    import src.utils

    assert src.data is not None
    assert src.models is not None
    assert src.training is not None
    assert src.inference is not None
    assert src.api is not None
    assert src.utils is not None
    assert src.evaluation is not None


def test_utils_imports() -> None:
    """Verify utility modules can be imported."""
    from src.utils.constants import FEATURE_NAMES, INPUT_DIM, LIMM_BBOX

    assert INPUT_DIM == 8
    assert len(FEATURE_NAMES) == 8
    assert LIMM_BBOX == (6.5, 44.0, 13.5, 47.0)


def test_config_from_yaml() -> None:
    """Verify config loading from YAML files."""
    from src.utils.config import DataConfig, GraphConfig, ModelConfig, TrainingConfig

    configs_dir = Path(__file__).parent.parent / "configs"

    model_cfg = ModelConfig.from_yaml(configs_dir / "model.yaml")
    assert model_cfg.input_dim == 8
    assert model_cfg.hidden_dim == 256
    assert model_cfg.num_layers == 6
    assert model_cfg.num_heads == 8
    assert model_cfg.output_dim == 16

    data_cfg = DataConfig.from_yaml(configs_dir / "data.yaml")
    assert data_cfg.fir == "LIMM"
    assert data_cfg.bbox.as_tuple() == (6.5, 44.0, 13.5, 47.0)
    assert data_cfg.context_length == 128

    train_cfg = TrainingConfig.from_yaml(configs_dir / "train.yaml")
    assert train_cfg.max_steps == 100000
    assert train_cfg.batch_size == 256

    graph_cfg = GraphConfig.from_yaml(configs_dir / "graph.yaml")
    assert graph_cfg.num_heads == 4
    assert graph_cfg.edge_dim == 5
