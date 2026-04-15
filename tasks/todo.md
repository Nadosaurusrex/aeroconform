# AeroConform - Implementation Plan

**REMINDER: I will not stop or pause until every phase is complete and every success criterion passes. No hacks. No shortcuts. No giving up.**

---

## Phase 0: Project scaffold
- [x] Initialize git repo, pyproject.toml with all dependencies
- [x] Create full directory structure per CLAUDE.md
- [x] Set up .env with OpenSky placeholder credentials, .gitignore
- [x] Create configs/model.yaml, configs/train.yaml, configs/data.yaml from ARCHITECTURE.md specs
- [x] Set up ruff, pyright config in pyproject.toml
- [x] Create src/__init__.py, src/utils/logging.py (structlog), src/utils/constants.py
- [x] Write a smoke test that imports all modules
- [x] **Verify**: `pip install -e ".[dev]"` succeeds, `pytest tests/ -x` passes, `ruff check src/` clean

## Phase 1: Data pipeline
- [x] Implement `src/data/opensky_client.py`: TrinoClient (pyopensky wrapper) + RESTClient (httpx async, OAuth2)
- [x] Implement `src/data/schemas.py`: dataclass models for StateVector, Flight, TrajectoryWindow, NormStats
- [x] Implement `src/data/preprocessing.py`: 8-dim encoding (sin/cos heading), delta encoding, mean/std normalization, polars
- [x] Implement `src/data/flight_segmentation.py`: group by icao24, segment by 30-min gaps, filter min 20 observations
- [x] Implement `src/data/dataset.py`: PyTorch IterableDataset + MapDataset, random window sampling (128 steps), collate
- [x] Implement `src/data/synthetic.py`: anomaly injection (4 types: spoofing, ghost, GPS drift, impossible maneuver)
- [x] Implement `src/data/download.py`: Trino bulk download script with daily chunks, norm stats computation
- [x] Write tests for each module (test_preprocessing, test_segmentation, test_dataset, test_synthetic)
- [ ] Download and preprocess a minimum viable dataset: 1 week of LIMM FIR data, saved as Parquet
- [ ] **Verify**: dataset loads, shapes are correct, normalization stats computed, 1000+ flights extracted

## Phase 2: Foundation model (AeroGPT)
- [x] Implement `src/models/embeddings.py`: StateEmbedding (linear R^8 -> R^256), TimeEncoding (sinusoidal with actual elapsed seconds)
- [x] Implement `src/models/aerogpt.py`: causal decoder transformer (6 layers, 8 heads, 256 dim, pre-norm, GELU, GaussianHead, ~4.7M params)
- [x] Implement `src/models/heads.py`: GaussianHead (mean + log_var, clamped [-10, 10])
- [x] Implement `src/models/losses.py`: Gaussian NLL loss with masking
- [x] Implement `src/training/trainer.py`: step-based training (AdamW, cosine warmup, bf16, gradient clipping, checkpointing)
- [x] Implement `src/training/masked_trainer.py`: 15% random masking, predict masked states
- [x] Implement `src/training/metrics.py`: ADE/FDE in meters
- [x] Create `notebooks/01_pretrain_aerogpt.ipynb`: Colab notebook for 100K steps + 20K masked fine-tuning
- [ ] Train the model on Colab
- [ ] **Verify**: eval ADE < 500m on 5-min horizon, loss converges

## Phase 3: Graph attention layer (AirGraph)
- [x] Implement `src/models/graph_builder.py`: PyG Data from concurrent states, proximity/conflict edges, 5-dim edge features
- [x] Implement `src/models/airgraph.py`: GATv2Conv (2 layers, 4 heads, edge_dim=5), residual + LayerNorm
- [x] Implement `src/models/combined.py`: AeroConformModel (AeroGPT -> AirGraph -> head, joint loss)
- [x] Implement `src/utils/geo.py`: haversine_km, bearing_rad, closing_speed_mps, time_to_cpa
- [x] Create `notebooks/02_train_airgraph.ipynb`: joint training 30K steps
- [ ] Train and validate on Colab
- [ ] **Verify**: graph model ADE <= foundation-only ADE, attention weights interpretable

## Phase 4: Conformal prediction (AeroConformal)
- [x] Implement `src/models/conformal.py`: Mahalanobis score, AdaptiveConformal (buffer=2000, decay=0.995), weighted quantile, p-value
- [x] Implement `src/models/scoring.py`: combined scoring, AlertLevel (RED/AMBER/YELLOW), Alert dataclass
- [ ] Calibrate on held-out normal traffic data
- [ ] **Verify**: FAR within alpha +/- 0.02, detection >95% for spoofing/maneuver, >80% for ghost/drift

## Phase 5: Inference pipeline
- [x] Implement `src/inference/pipeline.py`: end-to-end real-time pipeline (poll REST, AeroGPT + AirGraph + AeroConformal)
- [x] Implement `src/inference/buffer_manager.py`: per-icao24 rolling windows, stale GC
- [ ] Test end-to-end latency on live data
- [ ] **Verify**: pipeline runs continuously, latency < 5s, handles 200+ aircraft

## Phase 6: API and dashboard
- [x] Implement `src/api/main.py`: FastAPI (GET /health, /aircraft, /alerts, WebSocket /ws/alerts)
- [x] Implement `src/api/models.py`: Pydantic response models
- [x] Build `dashboard/`: React + Vite + Leaflet map, alert panel, aircraft detail, WebSocket
- [ ] **Verify**: dashboard renders live aircraft, alerts appear in < 5s

## Phase 7: Evaluation and documentation
- [x] Implement `src/evaluation/metrics.py`: detection rate, FAR, calibration error, detection latency
- [x] Implement `src/evaluation/benchmark.py`: full evaluation harness for all 4 anomaly types
- [x] Create `notebooks/03_evaluation.ipynb`: reproducible evaluation notebook
- [ ] Run full evaluation protocol
- [ ] **Verify**: all success criteria met

---

## Success criteria (ALL must pass)

1. **Foundation model quality**: ADE < 500m on 5-minute prediction horizon for LIMM FIR test set
2. **Conformal calibration**: empirical false alarm rate within alpha +/- 0.02 on held-out test data
3. **Detection - spoofing**: recall > 95% at alpha = 0.01
4. **Detection - impossible maneuver**: recall > 95% at alpha = 0.01
5. **Detection - ghost injection**: recall > 80% at alpha = 0.01
6. **Detection - GPS drift**: recall > 80% at alpha = 0.01 (within 60s of drift start)
7. **Latency**: end-to-end < 5 seconds from OpenSky API poll to anomaly score
8. **Scale**: handles 200+ concurrent aircraft on CPU inference
9. **Graph interpretability**: GATv2 attention weights rank high-conflict pairs above low-conflict pairs on labeled test scenarios
10. **Dashboard**: live map with color-coded aircraft and real-time alert panel functional

---

**REMINDER: I will not stop or pause until every phase is complete and every success criterion passes. No hacks. No shortcuts. No giving up.**
