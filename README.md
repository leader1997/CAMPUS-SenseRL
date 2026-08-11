# CAMPUS-SenseRL

**Semantic-Aware Constrained Multi-Agent Reinforcement Learning for Autonomous IoT Sensing in Smart Campuses**

CAMPUS-SenseRL learns **when each campus CO₂ sensor should transmit** under bandwidth and reliability constraints. The system combines:

- A **trace-driven counterfactual Gymnasium environment** (real uplink history, simulated scheduling decisions)
- **Spatiotemporal graph reconstruction** with uncertainty estimates
- **Centralized PPO** and **parameter-sharing MAPPO** communication policies
- **Safety shields** and **semantic event detection** for indoor air-quality monitoring

Research data: Oulu Smart Campus LoRaWAN release (CO₂ subset).

---

## Research objective

Learn adaptive communication schedules that **reduce transmissions** while preserving **reconstruction accuracy**, **Age of Information (AoI)**, and **event capture** (elevated CO₂ episodes). Policies are evaluated counterfactually on historical traces: the environment knows ground truth internally, but the server only receives values the policy chooses to transmit.

---

## Installation

```powershell
cd CAMPUS-SenseRL
python -m venv .venv
.venv\Scripts\activate
pip install -e .
# or: pip install -r requirements.txt
```

Python **3.10+** required. GPU optional (PyTorch auto-detects CUDA).

---

## Data layout

The Oulu release ships at the repository root (immutable):

```
oulu-smartcampus-release-2021062801/
  application.csv
  lora.csv
  devices.json   # JSON Lines despite .json extension
  README.md
```

Configured via `configs/data.yaml` → `paths.raw_release`.
Do **not** modify source release files. Preprocessing writes to `data/interim/` and `data/processed/`.

`devices.json` is JSON Lines (one device object per line), not a single JSON array.

---

## One command — full pipeline (~2h on GPU)

Your setup: **RTX 2070** + PyTorch **CUDA 12.1** (`2.5.1+cu121`).

```powershell
cd C:\Users\test\Desktop\CAMPUS-SenseRL
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py --force --profile 2h --device cuda
```

`--profile 2h` uses a paper-useful budget (~2 hours): 40 recon sensors, 30 RL sensors, PPO 60k / MAPPO 50k steps.

Other options:

```powershell
# longer higher-quality run
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py --force --profile full --device cuda

# quick smoke (~45 min)
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py --force --profile fast --device cuda

# only rebuild manuscript pack
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py --paper-only
```

This runs audit → preprocess → graphs → reconstruction → baselines → RL/MARL → ablations → **`results/`**.

---

## Pipeline scripts

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `scripts/01_audit_data.py` | Schema audit, coverage, leakage checks |
| 02 | `scripts/02_preprocess.py` | Clean traces, 15-min panel, chronological splits |
| 03 | `scripts/03_build_graph.py` | Spatial, correlation (train-only), hybrid graphs |
| 04 | `scripts/04_train_reconstruction.py` | Train masked spatiotemporal graph reconstructor |
| 05 | `scripts/05_evaluate_reconstruction.py` | Neural reconstruction + LOCF/mean metrics |
| 05b | `scripts/05b_run_reconstruction_baselines.py` | Classical causal baselines |
| 06 | `scripts/06_train_rl.py` | Centralized PPO (negative / early baseline) |
| 07 | `scripts/07_train_marl.py` | Parameter-sharing MAPPO (negative baseline) |
| 08 | `scripts/08_run_baselines.py` | Fixed-interval and heuristic baselines |
| 10 | `scripts/10_run_robustness.py` | MAPPO packet-loss smoke (seed_42) |
| 12 | `scripts/12_generate_results.py` | Early manuscript helpers (legacy pack) |
| 13 | `scripts/13_run_finalization.py` | Cohort freeze + fair recon benchmark |
| 16 | `scripts/16_distill_expert_policy.py` | BC distillation from semantic expert |
| 17 | `scripts/17_paper_asap_pipeline.py` | Expert + BC multi-seed ASAP pipeline |
| 19 | `scripts/19_train_cmappo.py` | Train KL-CMAPPO from a BC checkpoint |
| 21 | `scripts/21_final_paper_results.py` | **Frozen 5-seed KL-CMAPPO + test** |
| 22 | `scripts/22_scientific_validation.py` | **Matched-budget / robustness / ablation / held-out** |

Authoritative manuscript numbers/figures:

```
results/figures/   # 300 DPI PNG + PDF
results/tables/
reports/final_paper_results.md
reports/scientific_validation.md
```

Example (core data stack):

```powershell
.venv\Scripts\python.exe scripts/02_preprocess.py
.venv\Scripts\python.exe scripts/03_build_graph.py
.venv\Scripts\python.exe scripts/16_distill_expert_policy.py
.venv\Scripts\python.exe scripts/19_train_cmappo.py --bc-checkpoint results/rl_final/paper_asap/bc/seed_42/final_model.pt
.venv\Scripts\python.exe scripts/22_scientific_validation.py --figures-only
```

---

## Reproducibility

- Global seed in `configs/data.yaml`, `configs/rl.yaml`, `configs/experiments.yaml` (default **42**)
- `campus_senserl.utils.set_seed()` seeds Python, NumPy, and PyTorch
- Chronological **train / val / test** splits — no random row shuffling across time
- Correlation graph edges fit on **training slots only**
- Rolling features use **past-only** windows (`shift(1)` before rolling)
- Random evaluation masks (`apply_mask_scheme`, `make_random_mask`) are seed-deterministic

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q --tb=short
```

---

## Scientific principles (no leakage)

1. **Counterfactual RL**: Ground truth exists in the simulator; **skipped measurements are never injected** into server-visible state (`ServerState.server_values`, `input_mask`).
2. **Natural vs RL missingness**: `natural_missing` reflects real uplink gaps; `rl_skipped` records policy decisions only.
3. **Information separation**: Edge agents may observe local readings when physically available; reconstruction and rewards at the server use **transmitted** data only.
4. **Chronological evaluation**: Splits and rolling-origin folds respect time order.
5. **Train-only graph statistics**: Pearson correlation edges never use validation or test periods.

---

## Project structure

```
configs/           YAML experiment and model settings
data/              Raw release + processed parquet (generated)
results/           All experiment artifacts, frozen tables, figures, checkpoints
reports/           Claim-ready result notes
scripts/           Pipeline entry points
src/campus_senserl/
  data/            Preprocessing, splits, features, cohorts
  environment/     Trace-driven Gymnasium env, events
  evaluation/      Metrics, RL eval, fair reconstruction
  graph/           Sensor graph construction
  models/          Reconstruction and baselines
  rl/              PPO, MAPPO, KL-CMAPPO, BC, expert policies
  visualization/   Figure helpers
tests/             Unit tests (synthetic traces)
```

---

## Citation

If you use this codebase, cite the CAMPUS-SenseRL research report (`reports/research_report.md`) and the Oulu Smart Campus dataset release.

---

## License

MIT
