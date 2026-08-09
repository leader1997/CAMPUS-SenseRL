# Final figure review

No training was performed. All numerical figures read frozen CSVs/JSONs under `outputs/rl_final/`.

## fig01_campus_deployment — MAIN
1. Real distributed campus evaluation?
2. Grey campus map + colored RL/held-out cohorts; bar counts.
3. `devices.json`, cohort JSONs.
4. Main.
5. Spatial view is coordinates, not a building floorplan.

## fig02_fixed_vs_adaptive_timeline — MAIN
1. How adaptive TX differs from Fixed-15 on real CO₂.
2. Dense Fixed-15 markers vs sparse adaptive TX with recon on skips; annotated counts.
3. Frozen KL-CMAPPO seed 123 rollout on validation (see `fig02_meta.json`).
4. Main.
5. One representative window (deterministic selection); not a statistical summary.

## fig03_method_workflow — MAIN
1. What is the method?
2. Expert→BC→KL-CMAPPO; shield-free; KL as regularization.
3. Schematic only.
4. Main.
5. Not an empirical result plot.

## fig04_overall_results — MAIN
1. Overall policy comparison.
2. KL-CMAPPO ~78% TX↓ with lower MAE than Fixed-60; expert best MAE.
3. `full_val_summary.csv` (validation).
4. Main.
5. Fixed-15 omitted from MAE panel (no skips). Event recall left to tables for clarity.

## fig05_matched_budget — MAIN
1. Is the RL gain only from more TX?
2. At 75/78/80%, KL-CMAPPO MAE ≤ BC.
3. `matched_budget_summary.csv` (validation, 5 seeds).
4. Main.
5. Threshold matching approximates budgets; not hard rate constraints.

## fig06_rl_training_dynamics — MAIN
1. Do constraints become feasible while keeping savings?
2. MAE/recall enter feasible bands; feasible-seed count rises; TX↓ remains high.
3. `cmappo_kl/seed_*/metrics.json` validation rows only.
4. Main.
5. Validation every ~8k steps; best checkpoints may occur mid-training.

## fig07_kl_constraint_dynamics — MAIN/METHOD
1. How duals and β evolve?
2. λ's adapt; β decreases (stronger early BC anchoring).
3. Training log fields `lam_*`, `kl_beta`.
4. Main or method/supplement if space-limited.
5. β is a coefficient — **not** measured KL divergence.

## fig08_kl_constraint_ablation — MAIN
1. Are KL and constraints complementary?
2. Constraints-only: best MAE, lower TX↓; full KL-CMAPPO balances.
3. `ablation_kl_cmappo_val.csv` (**seed 42 only**).
4. Main (label as mechanistic single-seed).
5. Not multi-seed significance; catastrophic no-shield MAPPO off-axis.

## fig09_transmission_heatmap — MAIN
1. Is communication sensor-/time-specific?
2. Dense Fixed-15 grid vs structured KL-CMAPPO TX/SKIP pattern.
3. Validation rollouts (meta JSON).
4. Main (or supplement if page limit).
5. 24 h window; natural missing masked separately from SKIP.

## fig10_heldout_generalization — MAIN
1. Transfer to unseen sensors?
2. KL-CMAPPO improves MAE vs BC on held-out **test**.
3. `heldout_transfer_summary.csv`.
4. Main.
5. Temporal test for held-out sensors; development sensors excluded by construction.

## fig11_packet_loss_robustness — MAIN
1. Graceful degradation under loss?
2. MAE rises / recall falls smoothly; ordering preserved.
3. `robustness_summary.csv` packet_loss.
4. Main.
5. Expert has n=1 (no fake error bars). Other stress tests → S2.

## fig12_cumulative_transmissions — OPTIONAL MAIN
1. How savings accumulate?
2. Diverging cumulative TX curves.
3. Validation week rollouts.
4. Optional.
5. Transmission counts only — **not** battery lifetime.

## Supplementary
- S1: reproducibility of training trajectories.
- S2: broader robustness conditions.
- S3: skipped (no existing per-sensor error raw table without new inference).
- S4: reconstruction appendix (fair benchmark).
