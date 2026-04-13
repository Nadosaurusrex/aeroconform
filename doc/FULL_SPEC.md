# CLAUDE.md — AeroConform

## Prime Directive

**Keep working on this until you're completely done. Do not stop. Do not pause. No hacks, no shortcuts, no giving up. Add this to your todo list as a reminder.**

Success criteria are defined at the bottom of this document. You are not done until every single criterion is met with passing tests. If something is hard, figure it out. If a dependency breaks, fix it. If data is messy, clean it. If a model doesn't converge, debug the training loop. You are building production-grade research software, not a prototype.

---

## Project Overview

**AeroConform** is the first system to combine three cutting-edge AI research threads for aviation:

1. A **trajectory foundation model** pre-trained on OpenSky ADS-B state vectors via self-supervised next-state prediction
2. A **Graph Attention Network (GATv2)** that models dynamic multi-aircraft interactions in real-time airspace
3. An **adaptive conformal prediction layer** that provides distribution-free, statistically guaranteed anomaly detection with controllable false alarm rates

The system detects ADS-B spoofing, rogue drone incursions, GPS attacks, and trajectory anomalies in European airspace with certified confidence levels. It is designed to be EU AI Act compliant through explainable uncertainty quantification.

### Why This Matters

Europe is under active drone siege (Copenhagen, Munich, Brussels airports shut down in 2025). The EU launched the European Drone Defence Initiative and a Counter-UAS Action Plan in February 2026. Yet all existing ADS-B anomaly detection uses supervised methods requiring labeled attack data. AeroConform is the first unsupervised, foundation-model-based approach with statistical guarantees.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Data Ingestion                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ OpenSky Live │  │ OpenSky Historic  │  │ Context Streams  │  │
│  │  REST API    │  │  (Trino/Parquet)  │  │ METAR, aircraft  │  │
│  │  ~1Hz state  │  │  Pre-train corpus │  │ type metadata    │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
└─────────┼──────────────────┼──────────────────────┼─────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Layer 2: Trajectory Foundation Model                │
│                                                                 │
│  State Vector ──► Delta Encoding ──► Patch Embedding            │
│  [lat,lon,alt,   [Δlat,Δlon,Δalt,   [Linear projection         │
│   vel,hdg,vr]     Δvel,Δhdg,Δvr]     to d_model=256]           │
│                                                                 │
│  ──► Causal Transformer (6 layers, 8 heads, d_model=256)        │
│  ──► Gaussian Mixture Head (next-state distribution)            │
│                                                                 │
│  Pre-training: self-supervised next-state prediction            │
│  Loss: negative log-likelihood of observed next state           │
│  Output: per-aircraft trajectory embedding + predicted dist.    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│          Layer 3: Dynamic Airspace Graph (GATv2)                │
│                                                                 │
│  Nodes: aircraft with trajectory embeddings as features         │
│  Edges: pairs within proximity threshold (50nm / alt overlap)   │
│  Edge features: distance, closing_speed, alt_diff, bearing      │
│                                                                 │
│  GATv2 (2 layers, 4 heads) ──► context-enriched embeddings     │
│  Captures: conflict geometry, traffic density, interaction      │
│  Output: graph-enhanced trajectory embeddings                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        Layer 4: Adaptive Conformal Prediction                   │
│                                                                 │
│  Non-conformity score: Mahalanobis distance between predicted   │
│    distribution and observed state, using model's covariance    │
│                                                                 │
│  Calibration: sliding window of recent normal observations      │
│  Adaptive quantile: weighted quantile with learned weights      │
│    (handles distribution shift from changing traffic patterns)  │
│                                                                 │
│  Output per aircraft per timestep:                              │
│    - p-value (interpretable as false alarm probability)         │
│    - binary anomaly flag at user-specified α (e.g., 0.01)      │
│    - prediction region (set of plausible next states)           │
│                                                                 │
│  Guarantee: FAR ≤ α on exchangeable data                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output / Serving Layer                        │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Anomaly Alerts   │  │  MCP Server  │  │   Dashboard      │  │
│  │ with confidence  │  │  for Claude  │  │   Visualization  │  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Compute Environment

**Google Colab Pro+**:
- GPU: NVIDIA A100 40GB (request via Runtime > Change runtime type > A100)
- RAM: 51GB (High-RAM runtime)
- Disk: ~160GB
- Session limit: up to 24 hours
- Compute units: ~50/month

**Implications for model design**:
- Model must fit in 40GB VRAM with batch training
- Use mixed precision (fp16/bf16) throughout
- Use gradient accumulation if batch doesn't fit
- Save checkpoints to Google Drive to survive session restarts
- Use streaming data loading (don't load full dataset into RAM)

---

## Project Structure

```
aeroconform/
├── CLAUDE.md                          # This file
├── README.md                          # Project overview, setup, usage
├── pyproject.toml                     # Project metadata and dependencies (use uv)
├── src/
│   └── aeroconform/
│       ├── __init__.py                # Version, public API exports
│       ├── config.py                  # Pydantic settings, all hyperparameters
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── opensky_client.py      # OpenSky REST API wrapper (live data)
│       │   ├── historical.py          # Historical data download, Parquet I/O
│       │   ├── preprocessing.py       # Delta encoding, normalization, windowing
│       │   ├── dataset.py             # PyTorch Dataset/DataLoader classes
│       │   ├── graph_builder.py       # Build dynamic airspace graphs from states
│       │   └── synthetic_anomalies.py # Inject realistic anomalies for evaluation
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── tokenizer.py           # Continuous-to-patch tokenization
│       │   ├── trajectory_model.py    # Transformer foundation model
│       │   ├── gaussian_head.py       # Mixture density network output head
│       │   ├── graph_attention.py     # GATv2 airspace interaction model
│       │   ├── conformal.py           # Adaptive conformal prediction module
│       │   └── pipeline.py            # Full AeroConform inference pipeline
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── pretrain.py            # Foundation model pre-training loop
│       │   ├── train_graph.py         # Graph attention layer training
│       │   ├── calibrate.py           # Conformal calibration procedure
│       │   └── utils.py              # LR scheduling, gradient clipping, logging
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # AUC-ROC, F1, FAR, detection delay
│       │   ├── benchmark.py           # Compare against LSTM/isolation forest baselines
│       │   └── plots.py              # Matplotlib visualization of results
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── live_monitor.py        # Real-time monitoring loop
│       │   └── mcp_server.py          # MCP server for Claude integration
│       │
│       └── utils/
│           ├── __init__.py
│           ├── geo.py                 # Haversine, bearing, geodesic calculations
│           ├── airspace.py            # FIR/TMA boundary polygons (EGTT, LIMM)
│           └── logging.py            # Structured logging setup
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Understand OpenSky data characteristics
│   ├── 02_pretraining.ipynb           # Train foundation model on Colab
│   ├── 03_graph_training.ipynb        # Train GATv2 layer
│   ├── 04_conformal_calibration.ipynb # Calibrate conformal layer
│   ├── 05_evaluation.ipynb            # Full benchmark evaluation
│   └── 06_live_demo.ipynb             # Real-time monitoring demo
│
├── configs/
│   ├── model.yaml                     # Model architecture hyperparameters
│   ├── training.yaml                  # Training hyperparameters
│   └── inference.yaml                 # Inference/serving configuration
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures (synthetic data generators)
│   ├── test_preprocessing.py          # Test delta encoding, normalization
│   ├── test_tokenizer.py              # Test patch tokenization
│   ├── test_trajectory_model.py       # Test model forward pass, shapes
│   ├── test_gaussian_head.py          # Test mixture density output
│   ├── test_graph_attention.py        # Test GATv2 with variable-size graphs
│   ├── test_conformal.py              # Test conformal guarantees empirically
│   ├── test_synthetic_anomalies.py    # Test anomaly injection realism
│   ├── test_pipeline.py               # End-to-end pipeline integration test
│   └── test_geo.py                    # Test geodesic calculations
│
└── scripts/
    ├── download_data.py               # CLI script to download historical data
    ├── train.py                       # CLI entry point for training
    ├── evaluate.py                    # CLI entry point for evaluation
    └── serve.py                       # Start MCP server / live monitor
```

---

## Phase 1: Data Layer

### 1.1 OpenSky Live API Client (`data/opensky_client.py`)

```python
# Key interface
class OpenSkyClient:
    """Wrapper around OpenSky REST API for live state vectors."""

    BASE_URL = "https://opensky-network.org/api"

    async def get_states(
        self,
        bbox: tuple[float, float, float, float] | None = None,  # min_lat, max_lat, min_lon, max_lon
        icao24: list[str] | None = None,
        time_secs: int = 0,  # 0 = most recent
    ) -> pd.DataFrame:
        """Fetch current state vectors, return as DataFrame.

        Columns: icao24, callsign, origin_country, time_position, last_contact,
                 longitude, latitude, baro_altitude, on_ground, velocity,
                 true_track, vertical_rate, geo_altitude, squawk, spi, position_source

        Rate limits (unauthenticated): 1 request / 10 seconds
        Rate limits (authenticated): 1 request / 5 seconds
        """
        ...
```

**Implementation notes**:
- Use `httpx` with async for non-blocking I/O
- Implement exponential backoff for rate limiting (429 responses)
- Cache responses with TTL to avoid redundant calls
- Parse the JSON array response into a typed DataFrame
- Handle None values (common for lat/lon/alt when aircraft on ground or no position fix)
- Add a `stream_states` async generator that polls every N seconds

**Bounding boxes for key FIRs** (define in `utils/airspace.py`):
- EGTT (London FIR): lat [49.0, 61.0], lon [-12.0, 2.0]
- LIMM (Milan FIR): lat [43.5, 47.0], lon [6.5, 14.0]
- LFFF (Paris FIR): lat [45.0, 51.5], lon [-5.5, 8.5]

### 1.2 Historical Data (`data/historical.py`)

**Data source**: OpenSky 

Username:Nadosaurusrex
Password:Pescespada28

**Fallback strategy** (if Trino access not available):
1. Poll live API every 10 seconds for a target FIR
2. Accumulate state vectors into Parquet files (one per hour)
3. Run this continuously for 2-4 weeks to build training corpus
4. This yields ~8,640 snapshots/day x ~200-500 aircraft per snapshot x 7 features = substantial corpus

**Parquet schema**:
```
timestamp: int64 (Unix seconds)
icao24: string
callsign: string (nullable)
latitude: float64 (nullable)
longitude: float64 (nullable)
baro_altitude: float64 (nullable)
velocity: float64 (nullable)
true_track: float64 (nullable)
vertical_rate: float64 (nullable)
on_ground: bool
geo_altitude: float64 (nullable)
origin_country: string
```

**Data collection script** (`scripts/download_data.py`):
- Accept target FIR bbox, duration, output directory as CLI args
- Use asyncio event loop with 10-second polling interval
- Rotate output files hourly
- Log collection statistics (aircraft count, null rates, gaps)
- Handle session restarts gracefully (append mode)

### 1.3 Preprocessing (`data/preprocessing.py`)

**Step 1: Trajectory extraction**
- Group state vectors by `icao24` within each hourly file
- Sort by timestamp within each group
- Filter: remove aircraft with fewer than 30 state updates
- Filter: remove aircraft that are `on_ground=True` for the entire trajectory
- Filter: remove entries with null lat/lon/alt (no position fix)

**Step 2: Delta encoding**
```python
def delta_encode(trajectory: np.ndarray) -> np.ndarray:
    """Convert absolute state vectors to deltas.

    Input shape: (T, 6) where features are [lat, lon, alt, vel, hdg, vrate]
    Output shape: (T-1, 6) where features are [Δlat, Δlon, Δalt, Δvel, Δhdg, Δvrate]

    Special handling for heading (true_track):
    - Wrap-around: Δhdg should be the shortest angular difference
    - Use: Δhdg = ((hdg[t+1] - hdg[t] + 180) % 360) - 180
    """
    ...
```

**Step 3: Robust normalization**
- Compute per-feature median and IQR from training set
- Normalize: `x_norm = (x - median) / (IQR + eps)`
- Store normalization statistics as a JSON sidecar file
- Use robust stats (median/IQR) not mean/std to handle outliers

**Step 4: Windowing**
- Slice trajectories into fixed-length windows of `seq_len=128` timesteps
- Use stride of `seq_len // 2 = 64` for overlapping windows (more training data)
- Pad shorter sequences with a dedicated padding token (all zeros + mask)

### 1.4 Dataset Classes (`data/dataset.py`)

```python
class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch dataset for pre-training the foundation model.

    Each sample is a windowed trajectory:
    - input: delta-encoded, normalized state vectors, shape (seq_len, 6)
    - target: shifted by 1 (next-state prediction), shape (seq_len, 6)
    - mask: boolean mask for valid timesteps, shape (seq_len,)
    - metadata: dict with icao24, start_time, origin_country
    """
    ...

class AirspaceSnapshotDataset(torch.utils.data.Dataset):
    """Dataset for graph layer training.

    Each sample is a full airspace snapshot at time t:
    - node_features: trajectory embeddings for all aircraft, shape (N, d_model)
    - edge_index: connectivity, shape (2, E)
    - edge_attr: edge features, shape (E, 4) [distance, closing_speed, alt_diff, bearing]
    - targets: next-state for each aircraft (for conformal calibration)
    """
    ...
```

**DataLoader configuration**:
- `TrajectoryDataset`: batch_size=128, shuffle=True, num_workers=4, pin_memory=True
- `AirspaceSnapshotDataset`: use `torch_geometric.loader.DataLoader` with batch_size=16

### 1.5 Graph Builder (`data/graph_builder.py`)

```python
class AirspaceGraphBuilder:
    """Build a PyTorch Geometric graph from a set of aircraft states.

    Parameters:
        proximity_threshold_nm: float = 50.0  # Connect aircraft within this range
        altitude_band_ft: float = 5000.0      # Only connect if altitude bands overlap
        max_edges_per_node: int = 10           # Cap for computational efficiency
    """

    def build_graph(
        self,
        states: pd.DataFrame,
        embeddings: torch.Tensor,  # (N, d_model) from foundation model
    ) -> torch_geometric.data.Data:
        """
        Returns:
            Data object with:
            - x: node features (N, d_model) — trajectory embeddings
            - edge_index: (2, E) — COO format connectivity
            - edge_attr: (E, 4) — [distance_nm, closing_speed_kts, alt_diff_ft, bearing_deg]
        """
        ...
```

**Edge feature computation**:
- `distance_nm`: haversine distance between aircraft positions, converted to nautical miles
- `closing_speed_kts`: rate of change of distance (negative = converging), using velocity vectors
- `alt_diff_ft`: absolute altitude difference in feet
- `bearing_deg`: bearing from aircraft A to aircraft B in degrees

### 1.6 Synthetic Anomalies (`data/synthetic_anomalies.py`)

Implement 5 realistic attack scenarios for evaluation:

```python
class AnomalyInjector:
    """Inject realistic anomalies into clean trajectories for evaluation."""

    def inject_gps_spoofing(self, traj: np.ndarray, start_idx: int, offset_nm: float = 5.0) -> np.ndarray:
        """Gradually shift position by offset_nm starting at start_idx.
        Simulates GPS spoofing that pulls aircraft off course."""

    def inject_position_jump(self, traj: np.ndarray, idx: int, jump_nm: float = 10.0) -> np.ndarray:
        """Instantaneous position teleportation. Simulates crude spoofing."""

    def inject_ghost_aircraft(self, clean_traj: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """Generate a synthetic trajectory that looks plausible but doesn't correspond
        to any real aircraft. Uses interpolation between random waypoints with small noise."""

    def inject_replay_attack(self, traj: np.ndarray, replay_traj: np.ndarray, start_idx: int) -> np.ndarray:
        """Replace part of traj with a previously recorded trajectory from a different aircraft.
        Position is shifted to start from current position."""

    def inject_altitude_manipulation(self, traj: np.ndarray, start_idx: int, alt_offset_ft: float = 2000.0) -> np.ndarray:
        """Gradually manipulate reported altitude. Dangerous because it could cause
        loss of separation in altitude-based conflict detection."""
```

**Evaluation protocol**:
- Generate 1000 clean test trajectories
- For each anomaly type, inject anomalies into 200 trajectories at random positions
- Create a mixed test set: 1000 clean + 1000 anomalous (200 per type)
- Labels: per-timestep binary (0=normal, 1=anomalous) + anomaly type

---

## Phase 2: Foundation Model

### 2.1 Tokenizer (`models/tokenizer.py`)

```python
class PatchTokenizer(nn.Module):
    """Convert continuous delta-encoded state vectors into patch embeddings.

    Rather than tokenizing each timestep independently, group P consecutive
    timesteps into a patch (like ViT patches for images, or TimesFM patches
    for time series). This captures short-term dynamics within each patch.

    Parameters:
        input_dim: int = 6          # Number of features per timestep
        patch_len: int = 8          # Timesteps per patch
        d_model: int = 256          # Embedding dimension
        dropout: float = 0.1
    """

    def __init__(self, input_dim, patch_len, d_model, dropout):
        super().__init__()
        self.patch_len = patch_len
        # Linear projection: (patch_len * input_dim) -> d_model
        self.projection = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) — delta-encoded state vectors
        Returns:
            (batch, num_patches, d_model) — patch embeddings
        """
        B, T, D = x.shape
        assert T % self.patch_len == 0, f"seq_len {T} must be divisible by patch_len {self.patch_len}"
        num_patches = T // self.patch_len
        # Reshape: (B, num_patches, patch_len * D)
        x = x.reshape(B, num_patches, self.patch_len * D)
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x
```

**Why patches**: At 1Hz sampling, 8 timesteps = 8 seconds of flight. This is long enough to capture a heading change or altitude transition, but short enough to preserve temporal resolution. It also reduces sequence length from 128 to 16, making attention computationally cheap.

### 2.2 Trajectory Transformer (`models/trajectory_model.py`)

```python
class TrajectoryTransformer(nn.Module):
    """Causal transformer for next-patch prediction on ADS-B trajectories.

    Architecture inspired by GPT-2/TrajGPT with modifications:
    - Continuous input (not discrete tokens)
    - Patch-based input (not per-timestep)
    - Sinusoidal positional encoding (not learned, for length generalization)
    - Gaussian mixture output head (not softmax)

    Parameters:
        d_model: int = 256
        n_heads: int = 8
        n_layers: int = 6
        d_ff: int = 1024         # Feed-forward inner dimension
        dropout: float = 0.1
        max_patches: int = 64    # Maximum number of patches in sequence
        input_dim: int = 6       # Features per timestep
        patch_len: int = 8       # Timesteps per patch
    """

    def __init__(self, ...):
        super().__init__()
        self.tokenizer = PatchTokenizer(input_dim, patch_len, d_model, dropout)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_patches)

        # Causal transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = GaussianMixtureHead(d_model, input_dim * patch_len, n_components=5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, input_dim) — raw delta-encoded states
            mask: (batch, seq_len) — True for valid timesteps
        Returns:
            means: (batch, num_patches, n_components, output_dim)
            log_vars: (batch, num_patches, n_components, output_dim)
            log_weights: (batch, num_patches, n_components)
            embeddings: (batch, num_patches, d_model) — for graph layer
        """
        patches = self.tokenizer(x)  # (B, P, d_model)
        patches = patches + self.pos_encoding(patches)

        # Causal mask: each patch can only attend to itself and previous patches
        causal_mask = nn.Transformer.generate_square_subsequent_mask(patches.size(1))
        causal_mask = causal_mask.to(patches.device)

        # Use decoder with causal self-attention (memory=patches, since no encoder)
        # For decoder-only: pass same tensor as both memory and target
        hidden = self.transformer(
            tgt=patches,
            memory=patches,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )  # (B, P, d_model)

        means, log_vars, log_weights = self.output_head(hidden)
        return means, log_vars, log_weights, hidden

    def get_trajectory_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the last hidden state as the trajectory embedding.
        Used as node features for the graph layer."""
        _, _, _, hidden = self.forward(x)
        return hidden[:, -1, :]  # (B, d_model) — last patch embedding
```

**Parameter count estimate**:
- Tokenizer: 6 * 8 * 256 + 256 = ~12.5K
- Per transformer layer: ~4 * 256^2 (attention) + 2 * 256 * 1024 (FFN) + norms = ~780K
- 6 layers: ~4.7M
- Output head: ~256 * 5 * (48 + 48 + 1) = ~124K
- **Total: ~5M parameters** — easily fits in A100 with large batches

### 2.3 Gaussian Mixture Head (`models/gaussian_head.py`)

```python
class GaussianMixtureHead(nn.Module):
    """Mixture density network output head.

    Predicts a mixture of Gaussians for the next patch.
    This gives us both a point prediction (mixture mean) and uncertainty
    (mixture variance), which the conformal layer needs.

    Parameters:
        d_model: int = 256
        output_dim: int = 48  # patch_len * input_dim = 8 * 6
        n_components: int = 5
    """

    def __init__(self, d_model, output_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.output_dim = output_dim

        self.mean_head = nn.Linear(d_model, n_components * output_dim)
        self.logvar_head = nn.Linear(d_model, n_components * output_dim)
        self.weight_head = nn.Linear(d_model, n_components)

    def forward(self, hidden):
        """
        Args:
            hidden: (batch, num_patches, d_model)
        Returns:
            means: (batch, num_patches, n_components, output_dim)
            log_vars: (batch, num_patches, n_components, output_dim)
            log_weights: (batch, num_patches, n_components) — log softmax
        """
        B, P, D = hidden.shape
        means = self.mean_head(hidden).reshape(B, P, self.n_components, self.output_dim)
        log_vars = self.logvar_head(hidden).reshape(B, P, self.n_components, self.output_dim)
        log_vars = torch.clamp(log_vars, min=-10, max=2)  # Numerical stability
        log_weights = F.log_softmax(self.weight_head(hidden), dim=-1)
        return means, log_vars, log_weights

    def nll_loss(self, means, log_vars, log_weights, targets):
        """Negative log-likelihood loss for mixture of Gaussians.

        Args:
            means: (B, P, K, D)
            log_vars: (B, P, K, D)
            log_weights: (B, P, K)
            targets: (B, P, D) — actual next patches

        Returns:
            scalar loss
        """
        targets = targets.unsqueeze(2)  # (B, P, 1, D)
        # Per-component log-likelihood
        var = torch.exp(log_vars)
        log_probs = -0.5 * (
            log_vars + (targets - means) ** 2 / (var + 1e-8) + math.log(2 * math.pi)
        )  # (B, P, K, D)
        log_probs = log_probs.sum(dim=-1)  # (B, P, K) — sum over features
        # Mixture log-likelihood via log-sum-exp
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)  # (B, P)
        return -log_mixture.mean()
```

### 2.4 Positional Encoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE, not learned, for length generalization."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]
```

---

## Phase 3: Graph Attention Layer

### 3.1 GATv2 Model (`models/graph_attention.py`)

```python
import torch_geometric
from torch_geometric.nn import GATv2Conv

class AirspaceGATv2(nn.Module):
    """Graph attention network for multi-aircraft interaction modeling.

    Takes trajectory embeddings from the foundation model as node features
    and learns which aircraft interactions are operationally significant.

    Parameters:
        in_channels: int = 256    # d_model from foundation model
        hidden_channels: int = 128
        out_channels: int = 256   # Same as d_model for residual connection
        edge_dim: int = 4         # Edge feature dimension
        heads: int = 4
        n_layers: int = 2
        dropout: float = 0.1
    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads, n_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GATv2Conv(
            in_channels, hidden_channels, heads=heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        ))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_channels * heads, hidden_channels, heads=heads,
                edge_dim=edge_dim, dropout=dropout, concat=True,
            ))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Final layer (mean aggregation over heads)
        self.convs.append(GATv2Conv(
            hidden_channels * heads, out_channels, heads=1,
            edge_dim=edge_dim, dropout=dropout, concat=False,
        ))
        self.norms.append(nn.LayerNorm(out_channels))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (N, in_channels) — node features (trajectory embeddings)
            edge_index: (2, E) — graph connectivity
            edge_attr: (E, edge_dim) — edge features

        Returns:
            x: (N, out_channels) — context-enriched embeddings
            attention_weights: list of (E, heads) — per-layer attention
        """
        attention_weights = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x if x.size(-1) == norm.normalized_shape[0] else None
            x, alpha = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            attention_weights.append(alpha)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
            if x_res is not None:
                x = x + x_res  # Residual connection
        return x, attention_weights
```

---

## Phase 4: Conformal Prediction Layer

### 4.1 Adaptive Conformal Anomaly Detection (`models/conformal.py`)

This is the critical differentiator. Follow the methodology from the ICLR 2026 paper on adaptive conformal anomaly detection with foundation models.

```python
class AdaptiveConformalDetector:
    """Distribution-free anomaly detection with guaranteed false alarm rates.

    The core idea:
    1. The foundation model predicts a distribution over next states
    2. We compute a non-conformity score: how "surprising" is the actual observation
       relative to the prediction?
    3. We calibrate a threshold on clean data such that the false alarm rate is ≤ α
    4. We adapt the threshold over time using a sliding calibration window

    The guarantee: under the exchangeability assumption, the probability of a false
    alarm at any single timestep is at most α. This is a finite-sample guarantee
    that holds regardless of the underlying data distribution.

    Parameters:
        alpha: float = 0.01        # Significance level (1% false alarm rate)
        cal_window: int = 500      # Sliding calibration window size
        adapt_lr: float = 0.01     # Learning rate for adaptive quantile weights
    """

    def __init__(self, alpha: float = 0.01, cal_window: int = 500, adapt_lr: float = 0.01):
        self.alpha = alpha
        self.cal_window = cal_window
        self.adapt_lr = adapt_lr
        self.calibration_scores: deque = deque(maxlen=cal_window)
        self.quantile_weights: np.ndarray | None = None

    def compute_nonconformity_score(
        self,
        observed: np.ndarray,      # (D,) — actual next state
        means: np.ndarray,         # (K, D) — mixture means
        log_vars: np.ndarray,      # (K, D) — mixture log-variances
        log_weights: np.ndarray,   # (K,) — mixture log-weights
    ) -> float:
        """Compute Mahalanobis-like non-conformity score.

        For each mixture component k, compute the Mahalanobis distance
        between observed and mean_k using the diagonal covariance.
        The score is the negative log-likelihood under the full mixture.

        A higher score means the observation is MORE surprising (more anomalous).
        """
        vars = np.exp(log_vars)  # (K, D)
        # Per-component log-likelihood
        log_probs = -0.5 * np.sum(
            log_vars + (observed - means) ** 2 / (vars + 1e-8) + np.log(2 * np.pi),
            axis=-1
        )  # (K,)
        # Mixture log-likelihood
        log_mixture = scipy.special.logsumexp(log_weights + log_probs)
        # Non-conformity score = negative log-likelihood (higher = more anomalous)
        return -log_mixture

    def calibrate(self, clean_scores: np.ndarray):
        """Initial calibration on a set of clean (normal) non-conformity scores.

        Sets the quantile threshold such that at most α fraction of clean
        scores exceed it.
        """
        self.calibration_scores = deque(clean_scores.tolist(), maxlen=self.cal_window)
        n = len(self.calibration_scores)
        # Initialize uniform weights
        self.quantile_weights = np.ones(n) / n

    def get_threshold(self) -> float:
        """Compute the adaptive conformal threshold.

        Uses weighted quantile: find the smallest t such that the weighted
        fraction of calibration scores ≤ t is at least (1 - α).
        """
        scores = np.array(self.calibration_scores)
        weights = self.quantile_weights[:len(scores)]
        weights = weights / weights.sum()
        # Weighted quantile
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumulative_weights = np.cumsum(sorted_weights)
        # Find the (1-α) quantile
        quantile_idx = np.searchsorted(cumulative_weights, 1.0 - self.alpha)
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        return sorted_scores[quantile_idx]

    def update(self, score: float, is_normal: bool = True):
        """Update the calibration set with a new observation.

        If the observation is flagged as normal (not anomalous), add its
        score to the calibration window. Adapt weights based on recent
        prediction accuracy.
        """
        if is_normal:
            self.calibration_scores.append(score)
            n = len(self.calibration_scores)
            if self.quantile_weights is None or len(self.quantile_weights) < n:
                self.quantile_weights = np.ones(n) / n
            # Exponential recency weighting
            decay = np.exp(-self.adapt_lr * np.arange(n)[::-1])
            self.quantile_weights = decay / decay.sum()

    def predict(
        self,
        observed: np.ndarray,
        means: np.ndarray,
        log_vars: np.ndarray,
        log_weights: np.ndarray,
    ) -> dict:
        """Run anomaly detection on a single observation.

        Returns:
            {
                "score": float,          # Non-conformity score
                "threshold": float,      # Current adaptive threshold
                "p_value": float,        # Conformal p-value (higher = more normal)
                "is_anomaly": bool,      # True if score > threshold
                "confidence": float,     # 1 - p_value (confidence in anomaly)
            }
        """
        score = self.compute_nonconformity_score(observed, means, log_vars, log_weights)
        threshold = self.get_threshold()

        # Conformal p-value: fraction of calibration scores >= observed score
        cal_scores = np.array(self.calibration_scores)
        p_value = (np.sum(cal_scores >= score) + 1) / (len(cal_scores) + 1)

        is_anomaly = score > threshold

        return {
            "score": score,
            "threshold": threshold,
            "p_value": p_value,
            "is_anomaly": is_anomaly,
            "confidence": 1.0 - p_value,
        }
```

---

## Phase 5: Training

### 5.1 Pre-training (`training/pretrain.py`)

**Objective**: self-supervised next-patch prediction on historical ADS-B trajectories.

**Hyperparameters**:
```yaml
# configs/training.yaml
pretrain:
  epochs: 50
  batch_size: 128
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
  scheduler: cosine_with_warmup
  mixed_precision: true  # Use torch.cuda.amp with bf16
  gradient_accumulation_steps: 2  # Effective batch = 256
  checkpoint_every_n_epochs: 5
  early_stopping_patience: 10
  val_fraction: 0.1
```

**Training loop pseudocode**:
```python
def pretrain(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, ...)
    scaler = torch.amp.GradScaler()  # For mixed precision

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            x = batch['input'].cuda()    # (B, seq_len, 6)
            target = batch['target'].cuda()  # (B, num_patches, patch_len * 6)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                means, log_vars, log_weights, _ = model(x)
                loss = model.output_head.nll_loss(means, log_vars, log_weights, target)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

        # Validation
        val_loss = evaluate(model, val_loader)
        log(epoch, epoch_loss / len(train_loader), val_loss)

        # Early stopping + checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, '/content/drive/MyDrive/aeroconform/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break
```

### 5.2 Graph Layer Training (`training/train_graph.py`)

**Approach**: Freeze the foundation model, train the GATv2 layer to improve next-state prediction using multi-aircraft context.

**Loss**: Same NLL loss as pre-training, but the prediction now uses graph-enhanced embeddings instead of raw trajectory embeddings.

**Hyperparameters**:
```yaml
graph:
  epochs: 30
  batch_size: 16  # Airspace snapshots (variable-size graphs)
  learning_rate: 1e-3
  weight_decay: 0.01
  freeze_foundation: true  # Keep foundation model frozen
```

### 5.3 Conformal Calibration (`training/calibrate.py`)

**Procedure**:
1. Run the trained model (foundation + graph) on a held-out calibration set of **clean** trajectories
2. Compute non-conformity scores for every timestep
3. Initialize the `AdaptiveConformalDetector` with these scores
4. Validate: on a separate clean validation set, check that the empirical false alarm rate is ≤ α
5. Save the calibrated detector (calibration scores + weights) alongside the model

```python
def calibrate_conformal(model, graph_model, cal_loader, alpha=0.01):
    """Run calibration procedure.

    Returns a calibrated AdaptiveConformalDetector.
    """
    model.eval()
    graph_model.eval()
    all_scores = []

    with torch.no_grad():
        for batch in cal_loader:
            # Get predictions
            means, log_vars, log_weights, embeddings = model(batch['input'].cuda())
            # Optionally apply graph layer here if using graph-enhanced predictions

            # Compute non-conformity scores for each timestep
            for i in range(batch['input'].size(0)):
                for p in range(means.size(1) - 1):  # Skip last patch (no target)
                    score = detector.compute_nonconformity_score(
                        batch['target'][i, p].numpy(),
                        means[i, p].cpu().numpy(),
                        log_vars[i, p].cpu().numpy(),
                        log_weights[i, p].cpu().numpy(),
                    )
                    all_scores.append(score)

    detector = AdaptiveConformalDetector(alpha=alpha)
    detector.calibrate(np.array(all_scores))

    # Validate coverage guarantee
    # ... (compute FAR on separate validation set, assert FAR <= alpha + tolerance)

    return detector
```

---

## Phase 6: Evaluation

### 6.1 Metrics (`evaluation/metrics.py`)

Implement these metrics:

1. **False Alarm Rate (FAR)**: fraction of clean timesteps flagged as anomalous. Must be ≤ α.
2. **Detection Rate (DR)**: fraction of anomalous timesteps correctly flagged.
3. **Detection Delay**: average number of timesteps between anomaly onset and first detection.
4. **AUC-ROC**: area under the ROC curve using p-values as continuous scores.
5. **F1 Score**: at the operating threshold (α).
6. **Per-anomaly-type detection rates**: separate DR for each of the 5 anomaly types.

### 6.2 Baselines (`evaluation/benchmark.py`)

Compare AeroConform against:

1. **Isolation Forest**: sklearn.ensemble.IsolationForest on raw state vector features
2. **LSTM Autoencoder**: reconstruct trajectory, flag high reconstruction error
3. **One-Class SVM**: on trajectory features (mean velocity, heading variance, etc.)
4. **Threshold-based**: simple physics rules (max climb rate, max speed, position jump > X nm)

Each baseline gets the same train/test split. Report all metrics for fair comparison.

### 6.3 Plots (`evaluation/plots.py`)

Generate these visualizations:

1. **ROC curves**: AeroConform vs all baselines, one plot per anomaly type + aggregate
2. **Conformal coverage plot**: empirical FAR vs nominal α across a range [0.001, 0.1]
3. **Detection delay histogram**: distribution of detection delays per anomaly type
4. **Attention heatmap**: GATv2 attention weights on a sample airspace graph (which aircraft interact?)
5. **Trajectory anomaly overlay**: plot a trajectory on a map, color-code timesteps by anomaly score
6. **Non-conformity score distribution**: histogram of scores for clean vs anomalous timesteps

---

## Phase 7: Inference & Serving

### 7.1 Live Monitor (`inference/live_monitor.py`)

```python
class LiveAirspaceMonitor:
    """Real-time anomaly detection on live OpenSky data.

    Polls the OpenSky API, maintains per-aircraft trajectory buffers,
    runs inference, and emits anomaly alerts.
    """

    def __init__(self, model, graph_model, detector, config):
        self.model = model
        self.graph_model = graph_model
        self.detector = detector
        self.client = OpenSkyClient()
        self.trajectory_buffers: dict[str, deque] = {}  # icao24 -> state history
        self.graph_builder = AirspaceGraphBuilder()

    async def run(self, bbox, poll_interval=10):
        """Main monitoring loop."""
        while True:
            states = await self.client.get_states(bbox=bbox)
            alerts = self.process_snapshot(states)
            for alert in alerts:
                self.emit_alert(alert)
            await asyncio.sleep(poll_interval)

    def process_snapshot(self, states: pd.DataFrame) -> list[dict]:
        """Process one airspace snapshot."""
        # 1. Update trajectory buffers
        # 2. For each aircraft with enough history, run foundation model
        # 3. Build airspace graph, run GATv2
        # 4. Run conformal prediction on each aircraft
        # 5. Return list of anomaly alerts
        ...
```

### 7.2 MCP Server (`inference/mcp_server.py`)

Implement a Model Context Protocol server that exposes AeroConform to Claude.

**Tools to expose**:

```python
@mcp_tool
def get_airspace_status(bbox: str, alpha: float = 0.01) -> dict:
    """Get current airspace anomaly status for a bounding box.

    Returns:
        {
            "timestamp": "2026-04-13T14:30:00Z",
            "total_aircraft": 247,
            "anomalous_aircraft": 3,
            "alpha": 0.01,
            "alerts": [
                {
                    "icao24": "4b1814",
                    "callsign": "SWR256",
                    "position": {"lat": 46.2, "lon": 8.9, "alt_ft": 12500},
                    "anomaly_type": "trajectory_deviation",
                    "confidence": 0.997,
                    "p_value": 0.003,
                    "description": "Aircraft deviating from predicted trajectory by 4.2nm"
                }
            ]
        }
    """

@mcp_tool
def get_aircraft_trajectory(icao24: str, window_minutes: int = 10) -> dict:
    """Get recent trajectory and anomaly scores for a specific aircraft."""

@mcp_tool
def get_airspace_graph(bbox: str) -> dict:
    """Get the current airspace interaction graph with attention weights."""
```

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "aeroconform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torch-geometric>=2.5",
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.12",
    "httpx>=0.27",
    "pyarrow>=15.0",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "scikit-learn>=1.4",
    "tqdm>=4.66",
    "structlog>=24.1",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "mypy>=1.8",
]
colab = [
    "pyopensky>=2.14",
]
serve = [
    "fastapi>=0.109",
    "uvicorn>=0.27",
    "mcp>=1.0",
]
```

---

## Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class AeroConformConfig(BaseSettings):
    """Central configuration. All values overridable via env vars or YAML."""

    # Data
    target_fir: str = "LIMM"
    bbox: tuple[float, float, float, float] = (43.5, 47.0, 6.5, 14.0)
    seq_len: int = 128
    patch_len: int = 8
    input_dim: int = 6
    features: list[str] = ["latitude", "longitude", "baro_altitude", "velocity", "true_track", "vertical_rate"]

    # Model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    n_components: int = 5
    dropout: float = 0.1

    # Graph
    graph_hidden: int = 128
    graph_heads: int = 4
    graph_layers: int = 2
    proximity_threshold_nm: float = 50.0
    altitude_band_ft: float = 5000.0

    # Conformal
    alpha: float = 0.01
    cal_window: int = 500
    adapt_lr: float = 0.01

    # Training
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 128
    pretrain_lr: float = 3e-4
    graph_epochs: int = 30
    graph_batch_size: int = 16
    graph_lr: float = 1e-3

    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
```

---

## Geodesic Utilities (`utils/geo.py`)

```python
def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in nautical miles."""

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 to point 2 in degrees."""

def closing_speed_kts(lat1, lon1, vel1, hdg1, lat2, lon2, vel2, hdg2) -> float:
    """Rate of change of distance between two aircraft in knots.
    Negative = converging, Positive = diverging."""

def point_in_bbox(lat, lon, bbox) -> bool:
    """Check if a point is within a bounding box."""
```

---

## Testing Strategy

### Unit Tests
- `test_preprocessing.py`: delta encoding correctness, heading wrap-around, normalization invertibility
- `test_tokenizer.py`: patch shapes, padding handling, round-trip consistency
- `test_trajectory_model.py`: forward pass shapes, causal mask correctness (future tokens don't leak), gradient flow
- `test_gaussian_head.py`: NLL loss is finite and positive, gradient flows through all components, sum of mixture weights = 1
- `test_graph_attention.py`: variable-size graph handling, attention weight shapes, edge feature integration
- `test_conformal.py`: **critical** — empirically verify coverage guarantee:
  - Generate 10,000 samples from a known distribution
  - Calibrate at α = 0.05
  - Verify empirical FAR is within [0.03, 0.07] (allow some variance)
  - Repeat 100 times and check that 95% of runs satisfy FAR ≤ α + 0.01
- `test_geo.py`: haversine against known city-pair distances, bearing symmetry, edge cases (poles, antimeridian)

### Integration Tests
- `test_pipeline.py`: end-to-end from raw state vectors to anomaly alerts
  - Feed 100 clean trajectories through full pipeline, check FAR ≤ α
  - Inject 10 anomalies, check at least 8 are detected (DR ≥ 0.8)
  - Check that graph layer doesn't degrade performance vs. foundation-only

### Test Fixtures (`conftest.py`)
```python
@pytest.fixture
def synthetic_trajectory():
    """Generate a synthetic but realistic aircraft trajectory.
    Straight-and-level flight with gentle turns and altitude changes."""

@pytest.fixture
def synthetic_airspace_snapshot():
    """Generate a snapshot with 20 aircraft at various positions/altitudes."""

@pytest.fixture
def trained_model():
    """Load a small pre-trained model for integration tests.
    If no checkpoint exists, train for 2 epochs on synthetic data."""
```

---

## Build Order (Sequential Phases)

**You must build these in order. Each phase depends on the previous one.**

### Phase 0: Project scaffolding
1. Create the full directory structure
2. Write `pyproject.toml` with all dependencies
3. Write `config.py` with all hyperparameters
4. Write `utils/geo.py` with geodesic functions
5. Write `utils/airspace.py` with FIR bounding boxes
6. Run `pytest tests/test_geo.py` — must pass

### Phase 1: Data layer
1. Implement `data/opensky_client.py`
2. Implement `data/preprocessing.py` (delta encoding, normalization, windowing)
3. Implement `data/dataset.py` (PyTorch datasets)
4. Implement `data/graph_builder.py`
5. Implement `data/synthetic_anomalies.py`
6. Write and run `tests/test_preprocessing.py` — must pass
7. Write and run `tests/test_synthetic_anomalies.py` — must pass

### Phase 2: Foundation model
1. Implement `models/tokenizer.py`
2. Implement `models/gaussian_head.py`
3. Implement `models/trajectory_model.py`
4. Write and run `tests/test_tokenizer.py` — must pass
5. Write and run `tests/test_gaussian_head.py` — must pass
6. Write and run `tests/test_trajectory_model.py` — must pass

### Phase 3: Graph layer
1. Implement `models/graph_attention.py`
2. Write and run `tests/test_graph_attention.py` — must pass

### Phase 4: Conformal layer
1. Implement `models/conformal.py`
2. Write and run `tests/test_conformal.py` — **critical, must pass all coverage checks**

### Phase 5: Training infrastructure
1. Implement `training/utils.py` (LR scheduling, logging, checkpointing)
2. Implement `training/pretrain.py`
3. Implement `training/train_graph.py`
4. Implement `training/calibrate.py`
5. Verify training runs for 1 epoch on synthetic data without errors

### Phase 6: Full pipeline
1. Implement `models/pipeline.py` (connects all layers)
2. Write and run `tests/test_pipeline.py` — must pass
3. Implement `evaluation/metrics.py`
4. Implement `evaluation/benchmark.py`
5. Implement `evaluation/plots.py`

### Phase 7: Inference and serving
1. Implement `inference/live_monitor.py`
2. Implement `inference/mcp_server.py`
3. Write `scripts/download_data.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/serve.py`

### Phase 8: Notebooks
1. Write all 6 notebooks with complete, runnable cells
2. Notebooks should work on Colab Pro+ with A100 runtime
3. Include `!pip install` cells at the top of each notebook
4. Include Google Drive mount for checkpoint persistence

### Phase 9: Documentation
1. Write comprehensive `README.md` with:
   - Project overview and motivation
   - Architecture diagram (ASCII)
   - Quick start (install, data collection, training, inference)
   - API reference
   - Evaluation results format
   - Citation information
2. Add docstrings to every public function and class
3. Add type hints to every function signature

---

## Success Criteria

**You are not done until ALL of these are true:**

- [ ] All files in the project structure exist and are non-empty
- [ ] `pyproject.toml` is valid and `uv sync` / `pip install -e .` succeeds
- [ ] All unit tests pass: `pytest tests/ -v` shows 0 failures
- [ ] The conformal coverage test passes: empirical FAR ≤ α + 0.01 in 95% of runs
- [ ] The foundation model can complete a forward pass on synthetic data: input (32, 128, 6) produces correctly shaped outputs
- [ ] The GATv2 model handles variable-size graphs correctly
- [ ] The full pipeline runs end-to-end: raw state vectors in, anomaly alerts out
- [ ] Training loop runs for at least 1 epoch on synthetic data without crashing
- [ ] The MCP server definition is complete with all 3 tools
- [ ] All 6 notebooks have complete, runnable cell structure
- [ ] Every public function has a docstring and type hints
- [ ] `ruff check src/` produces 0 errors
- [ ] `mypy src/ --ignore-missing-imports` produces 0 errors
- [ ] README.md contains setup instructions, architecture overview, and usage examples
- [ ] No placeholder functions (no `pass`, no `...`, no `# TODO` in production code)
- [ ] No hardcoded paths (all paths come from config)
- [ ] No silent exception swallowing (every except block logs or re-raises)

---

## TODO List (Claude Code must track this)

- [ ] REMINDER: Keep working until completely done. No stops, no pauses, no hacks, no shortcuts, no giving up.
- [ ] Phase 0: Scaffolding + geo utils + tests passing
- [ ] Phase 1: Data layer complete + tests passing
- [ ] Phase 2: Foundation model complete + tests passing
- [ ] Phase 3: Graph layer complete + tests passing
- [ ] Phase 4: Conformal layer complete + coverage tests passing
- [ ] Phase 5: Training infrastructure complete + 1-epoch synthetic run
- [ ] Phase 6: Full pipeline + evaluation + tests passing
- [ ] Phase 7: Inference + MCP server + CLI scripts
- [ ] Phase 8: All 6 notebooks complete and runnable
- [ ] Phase 9: README + docstrings + type hints + linting clean
- [ ] FINAL: All success criteria met. Every test passes. Zero TODOs remain.