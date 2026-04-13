# CLAUDE.md — AeroConform

## Prime Directive

**Keep working on this until you're completely done. Do not stop. Do not pause. No hacks, no shortcuts, no giving up. Add this to your todo list as a reminder.**

You are not done until every success criterion at the bottom is met with passing tests.

## What This Is

AeroConform: the first system combining a trajectory foundation model + Graph Attention Networks + conformal prediction for European airspace anomaly detection. Detects ADS-B spoofing, rogue drones, GPS attacks with statistical false alarm guarantees.

## Commands

- Install: `uv sync` or `pip install -e ".[dev]"`
- Test: `pytest tests/ -v`
- Lint: `ruff check src/`
- Type check: `mypy src/ --ignore-missing-imports`
- Train (foundation): `python scripts/train.py --phase pretrain`
- Train (graph): `python scripts/train.py --phase graph`
- Calibrate: `python scripts/train.py --phase calibrate`
- Evaluate: `python scripts/evaluate.py`
- Serve: `python scripts/serve.py`

## Architecture (Summary)

4-layer pipeline. Read `@.claude/rules/architecture.md` for full spec.

1. **Data ingestion**: OpenSky REST API (live) + Parquet (historical)
2. **Trajectory foundation model**: Patch-based causal transformer, self-supervised next-state prediction with Gaussian mixture head. ~5M params.
3. **Airspace graph (GATv2)**: Aircraft as nodes, proximity/conflict as edges, attention-weighted interactions.
4. **Conformal prediction**: Adaptive conformal anomaly detection. Non-conformity scores yield p-values with guaranteed FAR ≤ α.

## Detailed Specs (Read Before Building)

Before starting any phase, read the relevant rules file:

@.claude/rules/architecture.md
@.claude/rules/model-specs.md
@.claude/rules/data-pipeline.md
@.claude/rules/training-and-eval.md
@.claude/rules/conventions.md

Module-specific instructions are in subdirectory CLAUDE.md files that load automatically when you access those directories.

## Build Order

YOU MUST build in this order. Each phase depends on the previous.

1. **Phase 0**: Scaffolding — `pyproject.toml`, `config.py`, `utils/geo.py`, `utils/airspace.py`. Run `pytest tests/test_geo.py`.
2. **Phase 1**: Data layer — `data/*.py`. Run `pytest tests/test_preprocessing.py tests/test_synthetic_anomalies.py`.
3. **Phase 2**: Foundation model — `models/tokenizer.py`, `models/gaussian_head.py`, `models/trajectory_model.py`. Run `pytest tests/test_tokenizer.py tests/test_gaussian_head.py tests/test_trajectory_model.py`.
4. **Phase 3**: Graph layer — `models/graph_attention.py`. Run `pytest tests/test_graph_attention.py`.
5. **Phase 4**: Conformal layer — `models/conformal.py`. Run `pytest tests/test_conformal.py`. **Coverage guarantee must pass.**
6. **Phase 5**: Training infra — `training/*.py`. Verify 1-epoch synthetic run.
7. **Phase 6**: Pipeline + eval — `models/pipeline.py`, `evaluation/*.py`. Run `pytest tests/test_pipeline.py`.
8. **Phase 7**: Inference — `inference/*.py`, `scripts/*.py`.
9. **Phase 8**: Notebooks — all 6 in `notebooks/`.
10. **Phase 9**: Docs — `README.md`, docstrings, type hints, linting.

## Project Layout

```
aeroconform/
├── CLAUDE.md                    # This file (project root, ~150 lines)
├── AGENTS.md                    # Cross-agent compatible instructions
├── .claude/
│   ├── rules/                   # Auto-loaded project rules
│   │   ├── architecture.md      # Full 4-layer architecture spec
│   │   ├── model-specs.md       # Transformer, GATv2, conformal details
│   │   ├── data-pipeline.md     # OpenSky API, preprocessing, datasets
│   │   ├── training-and-eval.md # Hyperparams, procedures, metrics
│   │   └── conventions.md       # Code style, naming, testing patterns
│   └── skills/                  # Slash commands
│       ├── train/SKILL.md       # /train — run training phases
│       ├── evaluate/SKILL.md    # /evaluate — run benchmarks
│       ├── serve/SKILL.md       # /serve — start MCP server
│       └── test-all/SKILL.md    # /test-all — full test suite
├── src/aeroconform/
│   ├── data/CLAUDE.md           # Data module specifics (lazy-loaded)
│   ├── models/CLAUDE.md         # Model module specifics (lazy-loaded)
│   └── training/CLAUDE.md       # Training specifics (lazy-loaded)
├── tests/CLAUDE.md              # Testing patterns (lazy-loaded)
└── ...                          # (full tree in architecture.md)
```

## Success Criteria

- [ ] All files in project structure exist and are non-empty
- [ ] `pip install -e .` succeeds
- [ ] `pytest tests/ -v` shows 0 failures
- [ ] Conformal coverage test: empirical FAR ≤ α + 0.01 in 95% of runs
- [ ] Foundation model forward pass: (32, 128, 6) input produces correct shapes
- [ ] GATv2 handles variable-size graphs
- [ ] Full pipeline: raw state vectors in, anomaly alerts out
- [ ] Training runs 1 epoch on synthetic data without crash
- [ ] MCP server has all 3 tools defined
- [ ] All 6 notebooks have complete cell structure
- [ ] Every public function has docstring + type hints
- [ ] `ruff check src/` and `mypy src/ --ignore-missing-imports` produce 0 errors
- [ ] No `pass`, `...`, or `# TODO` in production code
- [ ] No hardcoded paths — all from config
- [ ] No silent exception swallowing

## TODO

- [ ] REMINDER: No stops, no pauses, no hacks, no shortcuts, no giving up.
- [ ] Phase 0-9 as listed above. Track completion per phase.
- [ ] FINAL: All success criteria met. Every test passes.