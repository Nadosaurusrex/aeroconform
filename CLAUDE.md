# AeroConform

Conformalized trajectory foundation model for European airspace anomaly detection. Combines three cutting-edge AI threads never before applied together to aviation: (1) a trajectory foundation model pre-trained on ADS-B data, (2) dynamic GATv2 graph attention for multi-aircraft interaction, (3) adaptive conformal prediction for certified anomaly scores with guaranteed false alarm rates.

Read @ARCHITECTURE.md before building any component. Read @tasks/todo.md for the ordered implementation plan and success criteria.

## Stack

- Python 3.11+, PyTorch 2.x, PyTorch Geometric
- Google Colab Pro+ (A100 80GB) for training
- FastAPI for serving, React (Vite) for dashboard
- OpenSky Network: REST API (live), Trino (historical), pyopensky library
- Key libraries: pyopensky, traffic, torch, torch_geometric, crepes (conformal), polars, httpx

## Project structure

```
aeroconform/
├── CLAUDE.md
├── ARCHITECTURE.md
├── tasks/
│   └── todo.md
├── src/
│   ├── data/           # Data pipeline: ingestion, cleaning, tokenization
│   ├── models/         # Foundation model, GNN, conformal wrapper
│   ├── training/       # Training loops, configs, checkpoints
│   ├── inference/      # Real-time pipeline, anomaly scoring
│   ├── api/            # FastAPI endpoints
│   └── utils/          # Shared utilities, constants, logging
├── notebooks/          # Colab training notebooks
├── dashboard/          # React frontend
├── tests/
├── configs/            # YAML configs for model, training, data
├── .env                # Credentials (never commit)
├── pyproject.toml
└── lessons.md          # Document every correction here
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Test
pytest tests/ -x -q

# Lint
ruff check src/ --fix && ruff format src/

# Type check
pyright src/

# Run API server
uvicorn src.api.main:app --reload --port 8000

# Run live ingestion
python -m src.data.live_ingest

# Run training (local, small)
python -m src.training.train --config configs/train_small.yaml
```

## Credentials

OpenSky account: username `nadosaurusrex`, password `Pescespada28`. The REST API now uses OAuth2 client credentials flow. On first setup:
1. Log into opensky-network.org, go to Account page, create an API client
2. Store `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` in `.env`
3. For Trino historical access, apply separately at My OpenSky > Request Data Access
4. If Trino is unavailable, use REST API + public datasets as fallback (see ARCHITECTURE.md)

IMPORTANT: Never commit `.env`. Add it to `.gitignore` immediately.

## Conventions

- All coordinates in WGS84 decimal degrees. Altitudes in meters (barometric).
- State vectors are 7-dim: [lat, lon, baro_altitude, velocity, true_track, vertical_rate, on_ground]
- Delta encoding: model predicts change-in-state, not absolute state
- Use polars over pandas for dataframes. Use httpx over requests.
- Type hints on every function. Docstrings on every public function.
- Configs via YAML + dataclasses, never hardcoded hyperparameters.
- All model dimensions, layer counts, learning rates live in config files.
- Logging via structlog. No print statements.
- Tests: at minimum one test per module. Use pytest fixtures for data.
- Git: conventional commits (feat:, fix:, refactor:, docs:, test:)

## Gotchas

- OpenSky REST API rate limit: 1 req/5s authenticated. Batch wisely.
- OpenSky Trino queries MUST include `hour` partition filter or they scan the full table and you get banned.
- pyopensky Trino class requires separate data access approval. REST class works with OAuth2 API client.
- ADS-B `on_ground` is boolean but encode as 0/1 float for the model.
- `true_track` is heading in degrees [0, 360). Handle wraparound: use sin/cos encoding (two channels) instead of raw degrees.
- Vertical rate can be None when aircraft is on ground. Impute as 0.0.
- ICAO24 addresses are hex strings, always lowercase.
- Conformal prediction calibration set must NEVER include the test point itself. Use proper split.

## Task management

1. Read tasks/todo.md for the full ordered plan
2. Work through phases sequentially. Do not skip ahead.
3. Mark items done as completed in todo.md
4. After every phase, run tests to verify before moving on
5. When corrected on a mistake, add a lesson to lessons.md

## The directive

Keep working on this until you are completely done. Do not stop. Do not pause. No hacks, no shortcuts, no giving up. If something is hard, figure it out. If something fails, debug it and fix it. If you hit an API limit, implement the fallback. If a test fails, fix the code until it passes. Add this to your todo list as a reminder: "I will not stop or pause until every phase is complete and every success criterion passes. No hacks. No shortcuts. No giving up."