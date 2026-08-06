# CAMPUS-SenseRL — Research Report

**Working title:** Semantic-Aware Constrained Multi-Agent Reinforcement Learning for Autonomous IoT Sensing in Smart Campuses

**Status:** Implementation and initial experimental pipeline. Numerical RL/MARL claims remain provisional until multi-seed runs complete.

---

## Dataset

**Source:** University of Oulu Smart Campus IoT release `oulu-smartcampus-release-2021062801`.

| Asset | Facts |
|---|---|
| `application.csv` | 10,531,250 rows; physical measurements |
| `lora.csv` | 10,666,727 rows; RSSI, LSNR, channel, seqn/fcnt |
| `devices.json` (JSON Lines) | 429 devices |

**Coverage:** 2020-07-01 → 2021-05-31 (UTC). Nominal uplink interval **15 minutes** (empirical median inter-arrival **900 s**; ~86% of sampled intervals near 15 min).

**Device types:**

- Elsys ERS CO2: **299** (primary research subset)
- Elsys ERS Sound: 113
- Elsys ELT-2 with soil moisture: 17

**Floors:** −1 (6), 1 (314), 2 (59), 3 (26), 4 (21), 5 (3). Full lat/lon and location descriptions present. Building IDs / HVAC topology **not** available.

**Coordinate note:** Release README claims Web Mercator EPSG:3785; observed values match **WGS84** lat/lon for Linnanmaa, Oulu. Implementation treats them as WGS84 and documents the discrepancy.

**CO₂:** Present only on ERS CO2 devices (~7.3M non-null application values). Extreme outliers exist (e.g. values ≫ 5000 ppm); preprocessing flags values outside **[200, 5000] ppm** as implausible for modelling while retaining raw values (~74k flagged on the CO₂-device event stream).

**Network:** RSSI and LSNR available. Spreading factor, payload size, gateway IDs, and measured joules/TX **not** available → report **transmission reduction (%)**, not measured battery-life gains.

Full audit: `reports/data_audit.md`, `outputs/data_summary.json`.

---

## Preprocessing

1. Normalize deveui to uppercase hex without separators.
2. Parse epoch-ms timestamps → UTC; sort; drop exact `(time, deveui)` duplicates (none found).
3. Restrict primary panel to 299 ERS CO2 devices.
4. Merge LoRa metrics on exact `(time, deveui)`.
5. Build regular **15-min** panel with `observed` / `natural_missing` indicators; `rl_skipped` reserved for the simulator.
6. Calendar features + **past-only** rolling features (`shift(1)` before rolling; `co2_delta1 = lag1 − lag2`).
7. Chronological split: train → 2021-01-17, val → 2021-03-25, test → 2021-05-31 (~60/20/20 of slots).
8. Scalers/correlations fit on **train only**.

Outputs: `data/processed/co2_panel_15min.parquet`, `co2_wide_observed.parquet`.

Observed rate on panel ≈ **77%**; given an observation, CO₂ present ≈ **98%** after quality filtering.

---

## Experimental protocol

- **No random shuffle** of time series.
- Trace-driven **counterfactual** RL: TRANSMIT delivers the historical observation to the server; SKIP withholds it. Ground truth is used only for reward/evaluation.
- Agents may use **local** measurements for decisions; the server never sees skipped values.
- Communication budgets: 100/80/60/40/20% transmission fractions (matched comparisons).
- Event thresholds (configurable): 800 / 1000 / 1200 / 1500 ppm + rapid-increase / persistent-high definitions.
- Multi-seed evaluation (target ≥5; ideally 10 for final comparisons).

---

## Models

### Reconstruction baselines (val, 40% random mask, 30 sensors)

| Model | MAE (ppm) | Notes |
|---|---:|---|
| LOCF | ~10.1 | Strong causal baseline |
| Linear extrapolation | ~18.8 | |
| Historical mean/median / seasonal | ~24–26 | Weak for short gaps |
| KNN neighbours | ~73.6 | Needs denser simultaneous coverage |
| ExtraTrees (past features) | ~8.2 | After fixing current-value leakage in delta |
| LightGBM | ~9.5 | |
| Non-causal interpolation | ~8.0 | **Offline upper bound only** |
| Masked ST-GNN (20 sensors, residual-LOCF) | ~16.4 | Beats historical mean; **does not yet beat LOCF/ExtraTrees** — needs longer training / hyperparameter search on validation |

Honest note: improving the neural reconstructor above LOCF on short gaps is non-trivial; tree models with past-only features currently lead. Further legitimate improvements: longer training, multi-scheme masking curriculum, neighbour message features, calibration tuning — **not** evaluation leakage.

### Proposed reconstruction

Masked Spatio-Temporal Graph Network (sensor embeddings, GAT/GraphSAGE, GRU, AoI, heteroscedastic mean + log-variance, Gaussian NLL). Trained with masked self-supervision.

### Graphs (train-only correlation)

- Spatial: 299 nodes, 1436 edges
- Correlation: 295 nodes, 2118 edges
- Hybrid: 299 nodes, 1319 edges

Edges are **statistical/spatial relationships**, not ventilation topology.

---

## RL environment

- Gymnasium-compatible `TraceDrivenCampusEnv`
- Actions: SKIP / TRANSMIT
- AoI increments on skip, resets on successful transmit
- Safety shield: force transmit on max AoI / uncertainty / CO₂ / rate / neighbour disagreement
- Reward: configurable weighted combination of TX cost, reconstruction error, AoI, uncertainty, missed/preserved events
- Communication cost proxy: TX=1, SKIP=0

---

## Results

### Completed

- Dataset audit + EDA figures (`figures/eda/`)
- Preprocessing + chronological splits (299 CO₂ sensors, 32 160 slots)
- Graph construction (spatial / correlation / hybrid)
- Classical reconstruction baselines (val) — ExtraTrees currently best among causal models tested
- Masked ST-GNN training (subset) — improves over historical mean; LOCF/ExtraTrees still stronger on short gaps
- Fixed + heuristic RL baselines with correct budget accounting
- Short PPO and MAPPO training runs (smoke / debugging scale)
- Ablation / robustness / paper-figure generation scripts
- Unit tests including **leakage** tests (**44 passed**)

### Pending for paper-strength claims

- Full-sensor neural reconstruction with validation-tuned hyperparameters (must beat or match ExtraTrees/LOCF on held-out masks)
- Multi-seed PPO/MAPPO at matched communication budgets on the full CO₂ cohort
- Event-recall Pareto fronts with bootstrap CIs
- Full ablation matrix with true component removal (not smoke stubs)
- Robustness sweeps on the real panel (packet loss, outages, threshold sensitivity)

---

## Ablations

Configured in `configs/experiments.yaml` (no-graph, no-motion, no-network, no-uncertainty, no-AoI, no-semantic, no-shield, graph variants, fixed vs heuristic vs RL, etc.). Execute via `scripts/09_run_ablations.py` after models train.

---

## Limitations

1. Counterfactual evaluation: historical policy was fixed ~15-min uplink; RL actions did not occur in reality.
2. No measured energy per transmission.
3. No HVAC / corridor topology — spatial graphs ≠ airflow.
4. CRS documentation mismatch (README vs observed WGS84).
5. CO₂ sensor faults / extreme outliers require quality control.
6. MARL may be unstable; centralized / parameter-sharing PPO retained if needed.
7. Large panel (~9.6M rows) — full-sensor neural + MARL training is computationally heavy; staged subset → full runs required.

---

## Potential paper contributions (only if supported by final results)

1. Trace-driven framework for adaptive IoT communication using real campus measurements.
2. Masked spatiotemporal graph reconstruction with uncertainty.
3. Semantic-aware constrained MARL scheduling (information value, AoI, events).
4. Safety shield preserving critical-event monitoring under communication reduction.
5. Comprehensive multi-budget, outage, and degradation evaluation on a real deployment.

Novelty claims require a completed literature review (not asserted here).

---

## Reproducibility

```text
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python scripts/01_audit_data.py
python scripts/02_preprocess.py
python scripts/03_build_graph.py
python scripts/05b_run_reconstruction_baselines.py
python scripts/04_train_reconstruction.py
python scripts/08_run_baselines.py
python scripts/06_train_rl.py
python scripts/07_train_marl.py
python scripts/09_run_ablations.py
python scripts/10_run_robustness.py
python scripts/11_generate_paper_results.py
```

Configs: `configs/*.yaml`. Seeds, package versions, and experiment folders under `outputs/experiments/`.
