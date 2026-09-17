# Revision evaluation — implementation plan

Audit date: 2026-09-17. No training. Frozen KL-CMAPPO checkpoints only.

## What already exists

- Evaluation core: `src/campus_senserl/evaluation/rl_policy_eval.py` (`evaluate_policy`, `make_final_env`).
- TX reduction already uses locally available slots only: `1 - N_tx_attempts / N_locally_available`.
- Events already use server reconstruction (skipped events can still be TP).
- Packet loss already occurs **after** a TRANSMIT attempt (`trace_environment.py`).
- Frozen KL-CMAPPO: `results/rl_final/cmappo_kl/seed_{42,123,2024,3407,9999}/best_model.pt`.
- Frozen single test seed: 123 (`test_freeze_meta.json`).
- Existing CSVs: val/test summaries, held-out transfer, packet-loss (KL/BC/expert only), ablation (seed 42), matched-budget 75/78/80.
- **BC checkpoints are not in this clone.** BC will be imported from frozen CSVs where available, not retrained.
- Ablation checkpoints are not in this clone. Existing `ablation_kl_cmappo_val.csv` will be exported as single-run.

## Gaps vs supervisor request

- Periodic grid missing Fixed-45/75/90; test/held-out/packet-loss omit most classical methods.
- Test reports one KL seed, not five-run aggregate.
- Packet-loss RNG is per-attempt (not a shared slot mask).
- Delta+heartbeat is a single (50 ppm, 4) point, not a validation Pareto grid.
- Fig 3 neighbour links can be read as a communication mesh.
- Fig 5 is used as if it were primary evidence.

## Plan (minimum change)

1. Terminology only: relation graph ≠ LoRaWAN/mesh/links.
2. Add optional `packet_loss_mode=independent_slots` (common `(t,sensor)` mask). Default `per_attempt` unchanged.
3. New pipeline `scripts/29_revision_evaluation.py` → `results/revision/` (does not overwrite `results/rl_final/`).
4. Figures `scripts/30_revision_figures.py` → `results/figures/fig_revision_*.png`.
5. Reuse frozen matched-budget and ablation CSVs; do not retrain.
6. Select Delta+heartbeat on **validation only**, then freeze for test/held-out/packet-loss.

No architecture, reward, split, cohort, or checkpoint changes.
