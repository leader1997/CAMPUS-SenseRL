# Phase: BC-init Constrained MAPPO + evaluation corrections

## Reviewer verdict accepted

- Current win is **adaptive semantic sensing + BC**, not end-to-end RL.
- Next contribution: **Semantic expert → BC → Constrained MAPPO (Lagrangian)**.

## Fixes shipped

1. **Precision/F1**: env exposes `server_events` + `false_positive_events`; evaluator uses real FP (was always 0).
2. **AoI**: `aoi_state` clipped for RL features; `aoi_raw` unbounded for reporting (`mean_aoi_raw`, `p95_aoi`).
3. **Event categories**: `recall_high_co2`, `recall_rapid_rise` (+ counts).
4. **ET baselines**: CO₂-only, send-on-delta, AoI heartbeat, delta+heartbeat.
5. **Scripts**:
   - `scripts/18_corrected_scientific_eval.py` — corrected val comparison + threshold sensitivity + robustness
   - `scripts/19_train_cmappo.py` + `src/campus_senserl/rl/cmappo.py` + `configs/rl_cmappo.yaml`
6. **Test discipline**: CMAPPO selects checkpoints on **VAL only** (test already peeked once).

## Constraints (val-defined)

- `eps_miss=0.02` → recall ≳ 0.98
- `eps_mae=10` ppm
- `eps_aoi=4` mean raw AoI

## Target Pareto (aspirational, not forced)

Improve beyond expert/BC gap: ~77–82% TX↓ with recall ≈98%+ and MAE < 9–10.
