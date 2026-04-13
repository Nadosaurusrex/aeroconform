# AeroConform

Conformalized trajectory foundation model for European airspace anomaly detection.

AeroConform combines a causal transformer pre-trained on ADS-B state vectors, Graph Attention Networks (GATv2) for multi-aircraft interaction modeling, and adaptive conformal prediction for certified anomaly detection with guaranteed false alarm rates.

## Architecture

```
Data Ingestion (OpenSky API / Parquet)
        |
        v
Trajectory Foundation Model (6L Causal Transformer, ~5M params)
  - Patch tokenization (8 timesteps -> 1 patch)
  - Self-supervised next-patch prediction
  - Gaussian mixture output head (5 components)
        |
        v
Airspace Graph (GATv2, 2 layers, 4 heads)
  - Aircraft as nodes with trajectory embeddings
  - Proximity/altitude-based edges (50nm, 5000ft)
  - Attention-weighted interactions
        |
        v
Adaptive Conformal Prediction
  - Non-conformity scores from mixture NLL
  - Sliding calibration window
  - Guaranteed FAR <= alpha
        |
        v
Anomaly Alerts with confidence levels
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
# or
uv sync --all-extras
```

### Run Tests

```bash
pytest tests/ -v
```

### Train on Synthetic Data

```bash
python scripts/train.py --phase pretrain --epochs 1 --synthetic
```

### Collect Data

```bash
python scripts/download_data.py --fir LIMM --duration 1.0
```

### Evaluate

```bash
python scripts/evaluate.py --n-clean 100 --anomalies-per-type 20
```

### Start MCP Server

```bash
python scripts/serve.py --port 8000
```

## Project Structure

```
src/aeroconform/
  config.py          - Pydantic configuration
  data/              - OpenSky client, preprocessing, datasets
  models/            - Transformer, GATv2, conformal, pipeline
  training/          - Pre-training, graph training, calibration
  evaluation/        - Metrics, baselines, visualization
  inference/         - Live monitor, MCP server
  utils/             - Geo calculations, airspace definitions
```

## Key Features

- **Foundation Model**: Patch-based causal transformer (d=256, 6L, 8H) with Gaussian mixture output
- **Graph Attention**: GATv2 captures multi-aircraft interaction patterns
- **Conformal Guarantees**: FAR <= alpha with finite-sample coverage proof
- **5 Attack Types**: GPS spoofing, position jumps, ghost aircraft, replay attacks, altitude manipulation
- **Real-time**: Async OpenSky polling with per-aircraft trajectory buffers
- **MCP Integration**: 3 tools for Claude (airspace status, trajectory, graph)

## Dependencies

- Python 3.11+
- PyTorch 2.2+
- PyTorch Geometric 2.5+
- httpx, pydantic, scipy, scikit-learn, structlog

## Configuration

All hyperparameters are centralized in `src/aeroconform/config.py` via pydantic-settings. Override via environment variables prefixed with `AEROCONFORM_`.

## API Reference

### Pipeline

```python
from aeroconform.models.pipeline import AeroConformPipeline

pipeline = AeroConformPipeline(model, detector, config)
result = pipeline.detect_anomalies(trajectory)
# result: {"is_anomalous": bool, "max_score": float, ...}
```

### Conformal Detector

```python
from aeroconform.models.conformal import AdaptiveConformalDetector

detector = AdaptiveConformalDetector(alpha=0.01)
detector.calibrate(clean_scores)
result = detector.predict(observed, means, log_vars, log_weights)
# result: {"p_value": float, "is_anomaly": bool, "confidence": float}
```

### MCP Tools

- `get_airspace_status(bbox, alpha)` - Current anomaly status
- `get_aircraft_trajectory(icao24, window_minutes)` - Per-aircraft data
- `get_airspace_graph(bbox)` - Interaction graph

## License

MIT
