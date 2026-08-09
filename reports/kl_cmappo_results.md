# KL-CMAPPO results (Path B — RL included)

## Method
Semantic expert → BC residual actor → **KL-regularized Constrained MAPPO**
(feasible-only checkpoints; shield OFF; VAL selection).

## Validation (full split, 826 events)

| Method | TX↓ % | MAE | Recall | Precision | AoI raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 60 | 75.3 | 11.34 | 0.887 | 0.856 | 1.89 |
| **Semantic expert** | 73.9 | **7.22** | **0.999** | **0.988** | **1.75** |
| BC (5 seeds) | 80.0 | 9.19 | 0.993 | 0.975 | 2.16 |
| **KL-CMAPPO** | **78.7** | **8.87** | **0.993** | **0.977** | **2.03** |
| Old MAPPO+shield | 78.9 | 13.08 | 0.935 | 0.844 | 3.35 |
| Delta+heartbeat | 79.6 | 10.60 | 0.966 | 0.902 | 1.99 |

## RL contribution
KL-CMAPPO **improves BC on MAE** (9.19 → 8.87) and **AoI** (2.16 → 2.03) at similar TX reduction (~79%), while keeping recall 0.993.

It does **not** beat the hand-designed expert on MAE (expert remains best quality), but it is the best **learned** neural policy and beats Fixed-60 / old shield-MAPPO clearly.

## Artifacts
- Checkpoint: `outputs/rl_final/cmappo_kl/seed_42/best_model.pt`
- Tables: `paper_outputs/tables/table3_policy_comparison.csv`, `table3_policy_summary.csv`
- Figures: `paper_outputs/figures/fig05_*`, `fig06_*`, `fig08_*`, `fig10_*`
