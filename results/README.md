# Results

```text
results/
  figures/       15 manuscript figures (PNG, 600 dpi) + figure_values.csv
  rl_final/      frozen evaluation CSVs + KL-CMAPPO checkpoints (5 seeds)
  cohorts/       final (40 RL agents) and held-out (40 unseen sensors)
  graphs/        hybrid adjacency + node order
```

Regenerate: `python scripts/28_manuscript_figures.py` (inference only, never trains).
Revision comparison pack: `python scripts/29_revision_evaluation.py` then `python scripts/30_revision_figures.py`.

## Figures

| # | File | Question it answers |
|---|------|---------------------|
| 1 | `fig01_campus_deployment` | Where are the sensors, and which are used for development vs transfer? |
| 2 | `fig02_sensors_by_floor` | How are CO₂ sensors distributed across floors? |
| 3 | `fig03_agent_neighbour_network` | Spatial/statistical contextual relations among the 40 agents (not communication links). |
| 4 | `fig04_kl_cmappo_framework` | How does the method work, end to end? |
| 5 | `fig05_adaptive_communication` | Illustrative behaviour example: skip when flat, transmit during an event (not primary quantitative evidence). |
| 6 | `fig06_agent_transmission_schedule` | Do agents act independently rather than on a fixed schedule? |
| 7 | `fig07_agent_budget_vs_information` | Do agents spend communication where information is (r = 0.86)? |
| 8 | `fig08_diurnal_transmission_profile` | Does transmission follow campus occupancy? |
| 9 | `fig09_cumulative_transmissions` | How many uplinks are saved over a week? |
| 10 | `fig10_reconstruction_error_vs_fixed60` | At the same budget, is accuracy better than a fixed schedule? |
| 11 | `fig11_event_preservation` | Are important CO₂ events still detected? |
| 12 | `fig12_unseen_sensor_generalization` | Does it transfer to 40 sensors never seen in training? |
| 13 | `fig13_packet_loss_robustness` | Does it degrade gracefully under packet loss? |
| 14 | `fig14_training_validation_mae` | Does constrained training converge inside the feasible region? |
| 15 | `fig15_kl_gain_over_bc_anchor` | What does the KL-regularized RL stage add over its BC initialization? |

Every plotted number is logged in `results/figures/figure_values.csv`.
Figures 5–9 are produced by rolling out frozen checkpoints on the validation
split; figures 10–15 read only frozen CSV/JSON summaries.

## Revision comparison pack (supervisor response)

| File | Question |
|---|---|
| `fig_revision_contextual_relations` | Spatial/statistical contextual relations — not communication links |
| `fig_revision_mae_vs_reduction` | Reconstruction MAE vs communication reduction (full frontier) |
| `fig_revision_recall_vs_reduction` | Event recall vs communication reduction |
| `fig_revision_unseen_comparison` | All major methods on the held-out 40-sensor cohort |
| `fig_revision_packet_loss_comparison` | MAE under 0/10/20/40% packet loss for several methods |
| `fig_revision_packet_loss_recall` | Recall under the same packet-loss levels |

Tables and master CSV: `results/revision/`.
Reproduce: `python scripts/29_revision_evaluation.py` then `python scripts/30_revision_figures.py`.
