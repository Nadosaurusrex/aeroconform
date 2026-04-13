# AGENTS.md — AeroConform

> Cross-agent instructions compatible with Claude Code, Cursor, Copilot, Windsurf, and other AI coding assistants.

## Project

AeroConform: conformalized trajectory foundation model for European airspace anomaly detection. Combines a causal transformer pre-trained on ADS-B state vectors, Graph Attention Networks for multi-aircraft interaction, and adaptive conformal prediction for certified anomaly detection with guaranteed false alarm rates.

## Stack

- Python 3.11+, PyTorch 2.2+, PyTorch Geometric 2.5+
- httpx (async HTTP), pydantic (config), scipy (conformal math)
- Testing: pytest, ruff (lint), mypy (types)
- Environment: Google Colab Pro+ with A100 40GB

## Commands

```bash
pip install -e ".[dev]"       # Install
pytest tests/ -v              # Test
ruff check src/               # Lint
mypy src/ --ignore-missing-imports  # Type check
python scripts/train.py --phase pretrain  # Train foundation model
python scripts/evaluate.py    # Run benchmarks
```

## Conventions

- Type hints on every function. Google-style docstrings on every public API.
- All config via pydantic BaseSettings in `src/aeroconform/config.py`. No magic numbers.
- No hardcoded file paths. No print statements (use structlog). No silent exception swallowing.
- PyTorch only. No TensorFlow or JAX.
- Tests mirror source structure. Coverage target: 80%+.

## Architecture

4-layer pipeline: Data Ingestion → Trajectory Transformer (6L, 8H, d=256, ~5M params) → GATv2 Airspace Graph → Adaptive Conformal Prediction. See `.claude/rules/architecture.md` for full spec.

## Build Order

Phase 0 (scaffolding) → Phase 1 (data) → Phase 2 (foundation model) → Phase 3 (graph) → Phase 4 (conformal) → Phase 5 (training) → Phase 6 (pipeline+eval) → Phase 7 (inference) → Phase 8 (notebooks) → Phase 9 (docs). Each phase has test gates. Do not skip ahead.