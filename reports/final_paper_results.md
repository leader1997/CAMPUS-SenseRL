# Final paper-ready results

Protocol: 5 seeds × 60k KL-CMAPPO steps, feasible-only checkpoints, shield OFF, VAL selection then one frozen TEST.

## Validation (mean ± std)

| Method | TX↓ % | MAE | Recall | Precision | AoI raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 30 | 50.0 | 9.66 | 0.931 | 0.903 | 0.60 |
| Fixed 60 | 75.3 | 11.34 | 0.887 | 0.856 | 1.89 |
| Delta+heartbeat | 79.6 | 10.60 | 0.966 | 0.902 | 1.99 |
| **Semantic expert** | 73.9 | **7.22** | **0.999** | **0.988** | **1.75** |
| BC (5 seeds) | 80.0±0.13 | 9.19±0.03 | 0.993±0.001 | 0.975±0.002 | 2.16±0.02 |
| **KL-CMAPPO (5 seeds)** | **77.8±0.78** | **8.66±0.16** | **0.995±0.002** | **0.987±0.006** | **1.94±0.09** |
| Old MAPPO+shield | 78.9 | 13.08 | 0.935 | 0.844 | 3.35 |

**RL contribution:** KL-CMAPPO improves BC on MAE (9.19→8.66), recall, precision, and AoI at similar TX reduction.

## Frozen test (CMAPPO seed 123 selected on val)

| Method | TX↓ % | MAE | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Fixed 60 | 75.1 | 10.55 | **0.265** | 0.300 |
| Delta+heartbeat | 79.6 | 10.01 | 0.902 | 0.736 |
| Semantic expert | 74.5 | **7.22** | **0.980** | 0.935 |
| BC (5-seed mean) | 80.4±0.12 | 8.97±0.02 | 0.925±0.005 | 0.882±0.005 |
| **KL-CMAPPO** | **77.5** | **8.37** | **0.961** | **0.925** |

Fixed intervals collapse on test event recall; proposed stack does not.

## Artifacts
- Raw: `outputs/rl_final/paper_final/`
- Tables: `paper_outputs/tables/table3_*.csv`
- Figures: `paper_outputs/figures/fig05_*`, `fig06_*`, `fig08_*`, `fig10_*`
- Checkpoints: `outputs/rl_final/cmappo_kl/seed_*/best_model.pt`
