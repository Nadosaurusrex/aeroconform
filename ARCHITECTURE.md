# Architecture Reference

This document is the technical source of truth. Read the relevant section before implementing each component.

## 1. Data pipeline

### State vector schema

Each ADS-B state vector from OpenSky contains:

| Field | Type | Description | Model input? |
|---|---|---|---|
| icao24 | str | Aircraft ICAO24 address (hex, lowercase) | Identifier only |
| callsign | str | Flight callsign | Metadata |
| origin_country | str | Country of registration | Metadata |
| time_position | int | Unix timestamp of last position update | Timing |
| last_contact | int | Unix timestamp of last message | Timing |
| longitude | float | WGS84 degrees | Yes |
| latitude | float | WGS84 degrees | Yes |
| baro_altitude | float | Barometric altitude in meters | Yes |
| on_ground | bool | Whether aircraft is on ground | Yes (as 0/1) |
| velocity | float | Ground speed in m/s | Yes |
| true_track | float | Heading in degrees clockwise from north [0, 360) | Yes (sin/cos) |
| vertical_rate | float | Climb/descent rate in m/s | Yes |
| geo_altitude | float | Geometric altitude in meters | Optional |
| squawk | str | Transponder code | Metadata (alert detection) |

### Model input vector (8-dim after encoding)

```
[latitude, longitude, baro_altitude, velocity, sin(true_track), cos(true_track), vertical_rate, on_ground]
```

Heading is encoded as `(sin(track_rad), cos(track_rad))` to handle wraparound. This makes the input 8-dimensional.

### Delta encoding

The model predicts change in state between consecutive observations for the same aircraft:

```
delta_t = [lat_t - lat_{t-1}, lon_t - lon_{t-1}, alt_t - alt_{t-1}, vel_t - vel_{t-1},
           sin_track_t - sin_track_{t-1}, cos_track_t - cos_track_{t-1},
           vrate_t - vrate_{t-1}, on_ground_t - on_ground_{t-1}]
```

Normalize deltas per-feature using training set statistics (mean, std). Store normalization params in checkpoint.

### Data sources (priority order)

1. **Trino historical** (if access granted): Query `state_vectors_data4` with `hour` partition. Scope to LIMM FIR (Milan) bounding box: `lat [44.0, 47.0], lon [6.5, 13.5]`. Pull 3-6 months for pre-training.

2. **REST API + public datasets** (fallback): Use pyopensky REST for live data. For historical bulk, use OpenSky's public COVID-19 flight dataset or the crowdsourced datasets at opensky-network.org/datasets. These provide cleaned flight tables you can download as CSV/Parquet.

3. **Synthetic generation**: For anomaly injection testing, generate synthetic attack trajectories: spoofing (duplicate icao24 at different position), ghost injection (plausible but non-existent flights), GPS manipulation (gradual position drift), impossible maneuvers (exceeding aircraft performance limits).

### Flight segmentation

Group state vectors by `icao24`. Segment into individual flights using gaps > 30 minutes between consecutive `time_position` values. Discard flights with fewer than 20 observations. Each flight becomes one training sequence.

### Data loading

Use PyTorch `IterableDataset` with streaming from Parquet files. For training, randomly sample fixed-length windows (context_length = 128 timesteps) from flights. For inference, use sliding window over live stream.

## 2. Trajectory foundation model

### Architecture: AeroGPT

A causal decoder-only transformer. Small enough for Colab Pro+ A100, large enough to learn flight dynamics.

```yaml
# configs/model.yaml
model:
  name: aerogpt
  input_dim: 8           # After sin/cos encoding
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  head_dim: 32           # hidden_dim / num_heads
  feedforward_dim: 1024  # 4x hidden_dim
  dropout: 0.1
  max_seq_len: 512
  output_type: gaussian   # Predict mean + log_var per feature
  output_dim: 16          # 8 means + 8 log_variances
```

**Parameter count**: ~8M parameters. Fits easily on A100.

### Input embedding

Continuous-value projection, not discrete tokenization:

```python
class StateEmbedding(nn.Module):
    # Linear projection: R^8 -> R^256
    # Plus learnable time encoding based on actual time gaps (seconds between observations)
    # Time gaps are encoded via sinusoidal positional encoding with learned frequency basis
```

The time encoding is critical. ADS-B observations are irregularly spaced (1s to 60s gaps). Standard positional encoding assumes uniform spacing. Instead, use:

```
time_enc(t) = [sin(t / 10^(2i/d)), cos(t / 10^(2i/d))] for i in 0..d/2
```

where `t` is the actual elapsed seconds since the first observation in the window.

### Output head

The model outputs a Gaussian distribution per feature:

```python
class GaussianHead(nn.Module):
    # Linear: R^256 -> R^16 (8 means + 8 log_variances)
    # Clamp log_var to [-10, 10] for numerical stability
```

Loss function: negative log-likelihood of the observed delta under the predicted Gaussian:

```
NLL = 0.5 * sum_i [ log_var_i + (delta_i - mu_i)^2 / exp(log_var_i) ]
```

This is better than MSE because it gives us calibrated uncertainty estimates needed for conformal prediction.

### Pre-training

```yaml
# configs/train.yaml
training:
  batch_size: 256
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_steps: 2000
  max_steps: 100000
  scheduler: cosine_with_warmup
  gradient_clip: 1.0
  mixed_precision: true    # bf16 on A100
  checkpoint_every: 5000
  eval_every: 1000
```

Teacher forcing during training: feed ground truth deltas, predict next delta. At inference, can optionally autoregress for multi-step prediction (not needed for anomaly detection, but useful for trajectory forecasting).

### Masked pre-training variant

After initial next-step pre-training, fine-tune with masked prediction: randomly mask 15% of states in the sequence, predict the masked states from context. This teaches the model to fill gaps (common in ADS-B due to coverage holes) and builds richer internal representations.

## 3. Graph attention layer (AirGraph)

### Dynamic graph construction

Every T seconds (T=10 for inference), construct a graph from all currently active aircraft:

- **Nodes**: Each aircraft with at least 5 recent observations
- **Node features**: Concatenation of (a) trajectory embedding from AeroGPT (last hidden state, R^256) and (b) current kinematic state (R^8). Total: R^264.
- **Edges**: Connect aircraft pairs where 3D Euclidean distance < 50 NM (92.6 km) OR altitude separation < 1000 ft (305 m) when lateral distance < 10 NM. This captures operationally relevant interactions.
- **Edge features** (R^5): [distance_km, closing_speed_mps, altitude_diff_m, relative_bearing_rad, time_to_closest_approach_s]

### Architecture

```yaml
graph:
  conv_type: GATv2Conv     # From PyTorch Geometric
  num_layers: 2
  hidden_dim: 128
  num_heads: 4
  edge_dim: 5
  dropout: 0.1
  residual: true
  output_dim: 64           # Per-aircraft context embedding
```

GATv2 (Brody et al., 2022) is strictly more expressive than GAT. It computes dynamic attention using:

```
alpha_ij = softmax_j( a^T * LeakyReLU(W * [h_i || h_j || e_ij]) )
```

This means attention weights depend on both the query AND key node features, unlike GAT where they only depend on the query projection.

### Training the graph layer

Train jointly with a reconstruction objective: given the graph-enriched embedding for aircraft i, predict aircraft i's next state. The graph layer's loss is added to the foundation model's NLL loss with a weighting factor (0.1x). This encourages the graph to capture information from neighboring aircraft that helps predict each aircraft's future state.

### Interpretability

The attention weights from GATv2 are directly interpretable: high attention between aircraft A and B means B's state is important for predicting A's behavior. This gives per-pair interaction importance scores, crucial for explaining anomaly alerts ("This aircraft was flagged because its trajectory is inconsistent with the traffic pattern formed by aircraft X, Y, Z").

## 4. Adaptive conformal prediction (AeroConformal)

### Non-conformity score

For each aircraft at each timestep, compute the Mahalanobis distance between the observed and predicted state:

```
s_t = sqrt( sum_i (delta_i - mu_i)^2 / exp(log_var_i) )
```

where mu_i and log_var_i come from the model's Gaussian output. This is essentially the model's own assessment of how surprising the observation is, weighted by its confidence in each feature.

For the graph-enriched version, combine the per-aircraft foundation model score with a graph-based contextual score:

```
s_combined = alpha * s_individual + (1 - alpha) * s_contextual
```

where s_contextual uses the graph-enriched prediction head. alpha is tuned on validation data.

### Sliding calibration

Maintain a calibration buffer of the last N non-conformity scores from "normal" traffic (N=2000). Use exponentially weighted quantiles to handle distribution shift (traffic patterns change by time of day, weather, etc.):

```python
class AdaptiveConformal:
    def __init__(self, alpha=0.01, buffer_size=2000, decay=0.995):
        self.buffer = deque(maxlen=buffer_size)
        self.alpha = alpha  # Target false alarm rate
        self.decay = decay

    def score(self, observation, prediction):
        s = mahalanobis(observation, prediction)
        # Weighted quantile of calibration buffer
        q = weighted_quantile(self.buffer, 1 - self.alpha, self.decay)
        p_value = (sum(c >= s for c in self.buffer) + 1) / (len(self.buffer) + 1)
        is_anomaly = s > q
        # Only add to buffer if not anomaly (don't contaminate calibration)
        if not is_anomaly:
            self.buffer.append(s)
        return AnomalyScore(s=s, p_value=p_value, threshold=q, is_anomaly=is_anomaly)
```

### Guarantees

Under the assumption that calibration data and test data are exchangeable, conformal prediction provides:

```
P(s_new > q) <= alpha
```

The adaptive weighting relaxes this to approximate validity under distribution shift. Empirically validate: run on held-out test data and verify that the observed false alarm rate is within [alpha - 0.02, alpha + 0.02].

### Multi-level alerting

- **p < 0.01**: RED alert. Trajectory is highly anomalous. Possible spoofing, rogue drone, or GPS attack.
- **p < 0.05**: AMBER alert. Trajectory is unusual. Could be weather deviation, unusual routing, or data quality issue.
- **p < 0.10**: YELLOW advisory. Worth monitoring but likely benign.

## 5. Inference pipeline

```
OpenSky REST API (poll every 10s)
    -> State vector buffer (per-aircraft rolling window, 128 steps)
    -> AeroGPT inference (batch all active aircraft)
    -> AirGraph construction + GATv2 forward pass
    -> AeroConformal scoring
    -> Alert generation + WebSocket push to dashboard
```

Target latency: < 2 seconds from API response to alert. The model inference (AeroGPT + GATv2) should be < 500ms on CPU for up to 200 concurrent aircraft.

## 6. Evaluation protocol

### Synthetic anomaly injection

Generate 4 attack types to evaluate detection:

1. **Position spoofing**: Clone an existing aircraft's icao24, place it at a random valid position. The graph layer should detect the spatial inconsistency.
2. **Ghost injection**: Create a new icao24 with a plausible but fabricated trajectory. The foundation model should detect subtle kinematic inconsistencies (wrong climb rates for type, impossible turn rates).
3. **GPS drift**: Gradually shift an existing aircraft's reported position by 0.01 deg/min. The model should detect the trajectory deviating from its learned dynamics.
4. **Impossible maneuver**: Inject a sudden 90-degree turn or 5000 ft/min climb into an otherwise normal trajectory. Should trigger immediate high non-conformity score.

### Metrics

- **Detection rate** (recall) at each alpha level
- **False alarm rate** (should match alpha within +/- 0.02)
- **Detection latency**: number of anomalous timesteps before first alert
- **ADE/FDE**: Average/Final Displacement Error of the foundation model on normal trajectories (validates that the model actually learned flight dynamics)

## 7. Colab Pro+ training plan

A100 80GB, ~50 compute units/month. Budget allocation:

| Phase | GPU hours | Compute units |
|---|---|---|
| Data download + preprocessing | 2h | ~1 |
| AeroGPT pre-training (100K steps) | 8-12h | ~6 |
| AeroGPT masked fine-tune (20K steps) | 3h | ~2 |
| AirGraph joint training (30K steps) | 4h | ~2 |
| Evaluation + tuning | 3h | ~2 |
| **Total** | **~24h** | **~13** |

Well within Colab Pro+ monthly budget. Create notebooks in `notebooks/` with clear cell structure: setup, data loading, training loop, checkpointing to Google Drive, evaluation.