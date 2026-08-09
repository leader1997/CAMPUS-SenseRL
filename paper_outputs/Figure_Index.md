# Figure Index — CAMPUS-SenseRL (final manuscript set)

All figures in `paper_outputs/figures/` are regenerated from frozen
`outputs/rl_final/paper_final/` and `outputs/rl_final/scientific_validation/`.
Obsolete plots are in `paper_outputs/legacy_provisional/figures/`.

## Figure 1 — `fig01_sensor_network`
Campus IoT deployment with **RL development cohort (40)** and **held-out transfer cohort (40)** over the full 429-sensor map; floor composition panel.
**Message:** method developed on a geographic subset, then evaluated on unseen campus sensors.

## Figure 2 — `fig02_co2_example`
Representative real CO₂ trajectory (final cohort, validation) with stable vs rising intervals highlighted.
**Message:** information value varies over time → adaptive communication.

## Figure 3 — `fig03_framework`
Final method: Semantic expert → Behavior cloning → **KL-CMAPPO (shield OFF)** → TX/SKIP → server reconstruction/monitoring. CTDE noted.
**Message:** methodological progression; no safety-shield dependence in the proposed policy.

## Figure 4 — `fig04_policy_pareto`
Validation Pareto: TX reduction vs MAE for Fixed-30/60, delta+heartbeat, old MAPPO+shield, expert, BC, KL-CMAPPO (error bars for multi-seed).
**Message:** adaptive policies dominate periodic scheduling in the communication–information plane.

## Figure 5 — `fig05_matched_budget`
BC vs KL-CMAPPO threshold curves with matched operating points at **75 / 78 / 80%** TX reduction (5-seed mean±std).
**Message:** RL MAE gain is not explained solely by transmitting more.

## Figure 6 — `fig06_kl_cmappo_timeline`
Qualitative rollout of **frozen KL-CMAPPO (seed 123, shield OFF)** on a real sensor: true CO₂, TX markers, skip reconstructions, decisions, AoI.
**Message:** learned policy adapts TX frequency to environmental dynamics.

## Figure 7 — `fig07_heldout_transfer`
Held-out **test** transfer (40 never-trained sensors): MAE and event recall for BC vs KL-CMAPPO (5 seeds).
**Message:** parameter-shared actor generalizes across campus sensors.

## Figure 8 — `fig08_ablation_pareto`
Ablation Pareto (seed 42): BC, BC+MAPPO, BC+KL, BC+constraints, full KL-CMAPPO. Catastrophic no-shield MAPPO omitted from axes (reported in caption/table).
**Message:** constraints preserve quality; KL keeps communication near the BC budget.

## Figure 9 — `fig09_robustness`
Packet loss 0–40%: MAE and event recall for Expert / BC / KL-CMAPPO.
**Message:** operational resilience under communication failures.

## Intentionally not in the main set
- Reconstruction benchmark bars → appendix / table
- Event-recall vs TX curve with tiny 0.989–0.999 range → table/robustness
- AoI-vs-TX curve → table / supplement
- Training reward curves → supplement if requested
