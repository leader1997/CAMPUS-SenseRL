# Final clear figures — index

All figures use frozen artifacts only (no retraining). Method names are manuscript display names.

| File | MS # | Scientific question | Sources | Metrics | Main conclusion |
| --- | --- | --- | --- | --- | --- |
| `fig01_campus_deployment` | 1 | Real distributed campus evaluation? | `devices.json`, `final_cohort.json`, `heldout_cohort.json` | sensor counts / coordinates | Developed on 40 sensors; transferred to 40 held-out sensors. |
| `fig02_fixed_vs_adaptive_timeline` | 2 | How does adaptive TX differ from Fixed-15 on real CO₂? | frozen KL-CMAPPO seed 123 rollout + FixedInterval(1), val | interval TX counts, CO₂, recon | KL-CMAPPO suppresses stable uplinks and transmits during change. |
| `fig03_method_workflow` | 3 | What is the method? | schematic | — | Expert→BC→KL-CMAPPO (shield-free) with constraints + KL anchor. |
| `fig04_overall_results` | 4 | How do policies compare? | `paper_final/full_val_summary.csv` | TX↓, MAE (±std) | KL-CMAPPO ~78% TX↓ with much lower MAE than Fixed-60. |
| `fig05_matched_budget` | 5 | Is RL gain only from sending more? | `matched_budget_summary.csv` | MAE, recall at 75/78/80% | KL-CMAPPO still improves BC at matched budgets. |
| `fig06_rl_training_dynamics` | 6 | Do constraints become feasible during training? | `cmappo_kl/seed_*/metrics.json` val rows | val MAE/recall/TX↓, #feasible seeds | Constraints are reached across seeds while keeping large TX savings. |
| `fig07_kl_constraint_dynamics` | 7 | How do KL β and duals evolve? | same metrics train rows | λ_event, λ_MAE, λ_AoI, β | Duals adapt; β anneals from strong BC anchor to more RL freedom. |
| `fig08_kl_constraint_ablation` | 8 | Are KL and constraints complementary? | `ablation_kl_cmappo_val.csv` (**seed 42**) | TX↓, MAE, recall | Mechanistic: constraints protect quality; KL preserves BC-like TX. |
| `fig09_transmission_heatmap` | 9 | Is communication sensor- and time-specific? | val rollouts Fixed-15 vs KL-CMAPPO | TX/SKIP/missing map | KL-CMAPPO varies decisions across sensors/time (not a new fixed period). |
| `fig10_heldout_generalization` | 10 | Does the policy transfer? | `heldout_transfer_summary.csv` **test** | MAE, recall (±std) | Parameter-shared actor transfers to unseen campus sensors. |
| `fig11_packet_loss_robustness` | 11 | Is degradation graceful under loss? | `robustness_summary.csv` packet_loss | MAE, recall | Gradual degradation; KL-CMAPPO remains competitive. |
| `fig12_cumulative_transmissions` | 12 (opt.) | How do savings accumulate? | val week rollouts | cumulative TX | Continuous accumulation of uplink savings vs Fixed-15/60. |

## Supplementary
- `supplementary/figS01_seed_training_dynamics` — individual seeds + mean
- `supplementary/figS02_full_robustness` — outage / edge-drop conditions
- `supplementary/figS04_reconstruction_benchmark` — fair recon MAE (appendix)

## Draft captions (short)
1. University of Oulu campus IoT deployment with RL development and held-out transfer cohorts.
2. Same real CO₂ window under Fixed 15 min vs shield-free KL-CMAPPO (validation).
3. CAMPUS-SenseRL workflow: semantic expert, BC initialization, KL-CMAPPO fine-tuning (shield-free).
4. Validation transmission reduction and skipped-slot MAE across policies (mean±std).
5. Matched-budget comparison of BC and KL-CMAPPO at 75/78/80% TX reduction (validation, 5 seeds).
6. Five-seed KL-CMAPPO validation dynamics and constraint feasibility during training.
7. Adaptive dual variables and KL coefficient schedule (regularization coefficient β, not measured KL).
8. Mechanistic KL/constraint ablation on validation (seed 42).
9. 24 h campus-wide TX/SKIP heatmap: Fixed 15 vs KL-CMAPPO (validation).
10. Held-out sensor test transfer: MAE and event recall (5 seeds).
11. Packet-loss robustness of Expert, BC, and KL-CMAPPO (validation).
12. Cumulative transmissions over one validation week.
