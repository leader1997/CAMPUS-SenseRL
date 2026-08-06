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

This runs audit → preprocess → graphs → reconstruction → baselines → RL/MARL → ablations → **`paper_outputs/`**.

---

## Pipeline scripts (01–12)

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `scripts/01_audit_data.py` | Schema audit, coverage, leakage checks |
| 02 | `scripts/02_preprocess.py` | Clean traces, 15-min panel, chronological splits |
| 03 | `scripts/03_build_graph.py` | Spatial, correlation (train-only), hybrid graphs |
| 04 | `scripts/04_train_reconstruction.py` | Train masked spatiotemporal graph reconstructor |
| 05 | `scripts/05_evaluate_reconstruction.py` | Neural reconstruction + LOCF/mean metrics |
| 05b | `scripts/05b_run_reconstruction_baselines.py` | Full classical causal baselines (+ non-causal upper bound) |
| 06 | `scripts/06_train_rl.py` | Centralized PPO communication policy |
| 07 | `scripts/07_train_marl.py` | Parameter-sharing MAPPO (CTDE) |
| 08 | `scripts/08_run_baselines.py` | Fixed-interval and heuristic baselines |
| 09 | `scripts/09_run_ablations.py` | Ablation configs → `outputs/tables/ablations.csv` |
| 10 | `scripts/10_run_robustness.py` | Packet loss / missingness / threshold sweeps |
| 11 | `scripts/11_generate_paper_results.py` | Aggregate metrics into `outputs/` (legacy helper) |
| 12 | `scripts/12_generate_paper_outputs.py` | **Manuscript figures/tables** → `paper_outputs/` |

Publication figures and tables are written only to:

```
paper_outputs/figures/   # fig01–fig10 (300 DPI PNG + PDF)
paper_outputs/tables/    # CSV + LaTeX
paper_outputs/Figure_Index.md
paper_outputs/Main_Results.md
```

Exploratory EDA plots stay under `figures/eda/` / `outputs/` and are **not** manuscript figures.

Example:

```powershell
.venv\Scripts\python.exe scripts/02_preprocess.py
.venv\Scripts\python.exe scripts/03_build_graph.py
.venv\Scripts\python.exe scripts/06_train_rl.py --max-sensors 20 --timesteps 50000
.venv\Scripts\python.exe scripts/08_run_baselines.py --split val
.venv\Scripts\python.exe scripts/09_run_ablations.py --synthetic
.venv\Scripts\python.exe scripts/11_generate_paper_results.py
```

Use `--synthetic` on scripts 09–10 when the full dataset is not yet preprocessed.

Generate manuscript pack:

```powershell
.venv\Scripts\python.exe scripts/12_generate_paper_outputs.py
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
figures/paper/     Publication figures (generated)
outputs/           Models, baselines, tables, experiment logs
reports/           Research report template and findings
scripts/           Pipeline entry points 01–11
src/campus_senserl/
  data/            Preprocessing, splits, features
  environment/     Trace-driven Gymnasium env, shield, events
  evaluation/      Metrics, calibration, statistics
  graph/           Sensor graph construction
  models/          Reconstruction and baselines
  rl/              PPO, MAPPO, rewards, policies
  visualization/   Paper figures and LaTeX tables
tests/             Unit tests (synthetic traces, no 800 MB dependency)
```

---

## Citation

If you use this codebase, cite the CAMPUS-SenseRL research report (`reports/research_report.md`) and the Oulu Smart Campus dataset release.

---

## License

MIT
