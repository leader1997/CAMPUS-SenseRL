# CAMPUS-SenseRL Finalization Audit

**Date:** 2026-08-06  
**Purpose:** Scientific inventory of the repository *before* correctness fixes and final training.  
**Rule:** Do not treat current PPO/MAPPO/paper “proposed” curves as final publication results.

---

## 1. Executive verdict

The pipeline **runs end-to-end**, but the scientific experiment is **not paper-ready**.

| Area | Status | Verdict |
|------|--------|---------|
| Data splits / past-only features | Implemented + tested | Keep |
| Server vs GT separation on SKIP | Implemented + tested | Keep; extend sequential history test |
| Reconstruction train/eval | Runs; GNN weaker than ExtraTrees | Improve / select honestly |
| Trace Gymnasium env | Runs | Fix event semantics, reconstructor/graph defaults, state |
| PPO / MAPPO | Short single-seed smoke | Retrain after fixes |
| Paper Figs 5–6 “proposed” | Heuristic **proxy** | Never label as final method |
| Ablations / robustness CSVs | Invalid / synthetic smoke | Archived as provisional |
| PDR in robustness | Can exceed 1 | Must fix |

---

## 2. Current checkpoints (provisional)

Archived to `results/legacy_provisional/` (not deleted).

### PPO (`legacy_provisional/ppo_60k/`)

| Field | Value |
|-------|-------|
| File | `final_model.pt` (241 392 B), `metrics.json` |
| Actual timesteps | **60 000** (config default 200 000; `2h` profile) |
| Sensors | **30** (actor weight `(30, 128)`, flat obs `300 = 30×10`) |
| Seeds | **1** (seed 42) |
| Reconstructor used | Default **LOCF** (no `reconstruction_model` passed) |
| Graph used | Default **identity** |
| Status | **Preliminary — not final** |

### MAPPO (`legacy_provisional/mappo_50k/`)

| Field | Value |
|-------|-------|
| File | `final_model.pt` (651 122 B), `metrics.json` |
| Actual timesteps | **50 000** (config default 300 000) |
| Sensors | **30** (critic input dim 300 ⇒ tied to 30 agents) |
| Seeds | **1** |
| Reconstructor / graph | LOCF + identity |
| Status | **Preliminary — not final**; critic not campus-scale |

### Reconstruction (`results/models/reconstruction/`)

| Field | Value |
|-------|-------|
| Checkpoint | `best_model.pt` |
| Sensors | **40** |
| Epochs logged | 0–10; best val loss ≈ 12.01 @ epoch 5 |
| Val MAE (masked ST-GNN) | **16.84** |
| LOCF MAE (same eval file) | **10.31** |
| ExtraTrees MAE (baselines file, mask 0.4) | **10.93** |
| LightGBM MAE | **11.56** |
| Status | GNN **not competitive** on point MAE; protocols not fully aligned |

---

## 3. Current metrics snapshot (honest)

### Reconstruction (not identical protocol)

- `eval/metrics_val.json`: ST-GNN MAE 16.84 vs LOCF 10.31 (n≈82k).  
- `reconstruction_baselines/baselines_val_mask0.4.json`: ExtraTrees 10.93, LightGBM 11.56 (n_sensors=40, mask=0.4).  
- **Problem 3:** different masks / settings → Table 2 is scientifically weak until a shared mask manifest exists.

### Policy paper eval (`results/tables/`)

- Cohort: **20 sensors**, **1200** steps, event threshold **800** ppm / rapid **80** (≠ `rl.yaml` primary 1000 / 150).  
- Proposed row: **`Semantic + safety shield (proxy)`** (`proposed_proxy`).  
- Explicit note: trained MAPPO multi-seed **not available**.

### Ablations (`legacy_provisional/ablations.csv`)

- Nearly identical rows; evaluated with **FixedIntervalPolicy(interval_steps=4)** for every “ablation”.  
- Several named ablations are **no-ops**.  
- **Invalid for paper.**

### Robustness (`legacy_provisional/robustness.csv`)

- Synthetic trace: **6 sensors**, **64** steps.  
- Fixed-interval policy only.  
- **`packet_delivery_ratio` ≈ 1.63–1.85** (impossible).  
- **Smoke only — invalid for paper.**

---

## 4. Critical environment inconsistencies

### 4.1 Event semantics (BLOCKING)

**Training reward** (`trace_environment.py` ≈ L418–419):

```text
events = event_detector.is_event(gt)          # cross-section misuse of temporal detectors
missed_events = events & ~transmit_history[t] # SKIP ∧ true event ⇒ automatic miss
```

**Paper eval** (`paper_eval.py`): SKIP can still be TP if reconstruction flags the event.

**Required science:**

- \(E^{true}_{i,t}\) from GT threshold / rapid rise vs **previous time** for same sensor.  
- \(\tilde y_{i,t} = y\) if TX else \(\hat y\).  
- \(E^{server}\) from \(\tilde y\); TP = true∧server; FN = true∧¬server.  
- Skipped + correctly reconstructed event = **TP**, not FN.

Also: `detect_events_series` on a length-`n_sensors` vector treats **sensor index as time** for rapid/persistent masks — scientific bug.

### 4.2 Reconstructor default (BLOCKING)

- `reconstruction_model is None` → `LocfReconstructor`.  
- PPO/MAPPO `make_env` never pass a trained model.  
- Final proposed RL must load selectable backends: `locf | extratrees | lightgbm | stgnn | proposed_reconstructor`.

### 4.3 Graph default (BLOCKING)

- `adjacency is None` → `np.eye(n_sensors)`.  
- Neighbour disagreement / graph message-passing effectively disabled.  
- Artifacts exist: `results/graphs/adjacency_{spatial,correlation,hybrid}.npy` + `node_order.json`.

### 4.4 Threshold inconsistency

| Location | Primary | Rapid | Shield CO₂ |
|----------|---------|-------|------------|
| `configs/rl.yaml` | 1000 | 150 | 1200 |
| Paper eval defaults | 800 | 80 | (shield on for proxy) |
| Sensitivity list | 800, 1000, 1200, 1500 | — | — |

**Decision:** primary paper threshold = **1000 / 150** from YAML; sensitivity on val only; never call thresholds medical limits.

### 4.5 Reward `w_event`

Currently **penalizes presence of GT events**, not rewarding correct server detection \(D_t\). Must align with miss FN + detect TP formulation.

### 4.6 Battery / RSSI / LSNR

- Present in `co2_panel_15min.parquet` (`battery`, `rssi`, `lsnr`).  
- Loaded in `load_trace_tensors` for battery/rssi but **not attached to env state / observation**.  
- LSNR not loaded into RL tensors.  
- Causality: use **last known** RSSI/LSNR + age after TX; never invent current gateway metrics on SKIP.

### 4.7 PDR

Robustness counts **pre-shield** policy TX attempts vs **post-shield** `transmit_count` → PDR can exceed 1.  
Must use final requested TX after shield vs delivered after loss, with \(0 \le \mathrm{PDR} \le 1\).

### 4.8 MAPPO critic scalability

Centralized critic input = `n_agents × obs_dim`. 30-agent checkpoint cannot transfer to full campus. Needs mean/attention pooling (Phase 8).

---

## 5. Leakage risks (current)

| Risk | Status |
|------|--------|
| SKIP current GT in `server_values` | Tested — OK |
| Chronological splits / rolling shift(1) | Tested — OK |
| Correlation graph train-only | Intended; keep |
| Sequential reconstructor history silently re-inserting skipped GT | **Needs explicit test** |
| Test-set hyperparameter tuning | Protocol freeze not yet done |
| Different recon masks across methods | Unfair benchmark |

---

## 6. Invalid / provisional artifacts (archived)

Moved (not deleted):

- `results/legacy_provisional/ppo_60k/`
- `results/legacy_provisional/mappo_50k/`
- `results/legacy_provisional/ablations.csv`
- `results/legacy_provisional/robustness.csv`
- Copy of `results/tables/` → `results/legacy_provisional/tables/`

New empty directories for final runs:

`results/cohorts/`, `reconstruction_final/`, `rl_final/{ppo,mappo,semantic_constrained_mappo}/`, `ablations_final/`, `robustness_final/`, `final_test/`.

---

## 7. Files planned for change (Phase 1–2 correctness)

| File | Change |
|------|--------|
| `src/campus_senserl/environment/event_detector.py` | Per-sensor temporal true/server events |
| `src/campus_senserl/environment/trace_environment.py` | Event TP/FN; load battery/rssi/lsnr; obs schema; graph/recon defaults from cfg |
| `src/campus_senserl/rl/reward.py` | Miss = FN; event = +TP detect; not penalize event presence |
| `src/campus_senserl/environment/reconstructors.py` | **New** backend factory |
| `src/campus_senserl/environment/graph_utils.py` | **New** load/reorder adjacency |
| `src/campus_senserl/evaluation/pdr.py` | **New** PDR helper + clamp |
| `src/campus_senserl/rl/ppo.py`, `mappo.py` | Pass reconstructor + graph from cfg |
| `scripts/10_run_robustness.py` | Fix PDR counting (archive note; real robustness later) |
| `configs/rl.yaml` | `environment.reconstructor`, `environment.graph`, primary thresholds clarified |
| `tests/test_events_semantics.py` | **New** |
| `tests/test_pdr.py` | **New** |
| `tests/test_reconstructor_wiring.py` | **New** |
| `tests/test_graph_wiring.py` | **New** |
| `tests/test_leakage.py` | Sequential skipped-GT history test |
| `reports/rl_state_definition.md` | State schema documentation |
| `reports/finalization_audit.md` | This file |

**Do not modify:** raw release CSVs under `oulu-smartcampus-release-2021062801/`.

---

## 8. Execution gate

Before any long RL training:

1. ✅ This audit  
2. ✅ Archive provisional outputs  
3. ⬜ Event semantics + tests  
4. ⬜ PDR + tests  
5. ⬜ Reconstructor wiring + tests  
6. ⬜ Graph wiring + tests  
7. ⬜ Sequential no-leakage + tests  
8. ⬜ State availability docs + causal features  
9. ⬜ Full unit test suite green  

Only then: fair reconstruction benchmark → reconstructor selection → PPO → MAPPO → multi-seed → matched budgets → real ablations/robustness → freeze → **one** test pass → paper figures.

---

## 9. Hypotheses (pre-experiment classification)

| Hypothesis | Pre-fix status |
|-----------|-------------------|
| ST-GNN beats causal trees on fair MAE | **At risk** (currently worse) |
| MAPPO + semantic/safety beats fixed/heuristic at matched TX budget | **Unsupported** until trained |
| Reconstruction-aware event recall justifies SKIP | **Not implemented in reward** |
| Graph improves RL | **Unsupported** (identity default) |
| Proxy ≈ proposed method | **Rejected** — proxy is heuristic only |
