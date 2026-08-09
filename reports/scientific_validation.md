# Scientific validation (complete)

Core 5-seed KL-CMAPPO was **not** retrained. All proofs below are post-freeze.

## 1. Matched communication budget (VAL, 5 seeds)

At the same TX-reduction targets, KL-CMAPPO still has lower MAE:

| Target TX↓ | BC MAE | KL-CMAPPO MAE | Δ MAE |
| ---: | ---: | ---: | ---: |
| 75% | 7.94±0.02 | **7.90±0.03** | −0.03 |
| 78% | 8.87±0.01 | **8.72±0.07** | **−0.16** |
| 80% | 9.20±0.02 | **9.13±0.04** | −0.07 |

So the gain is **not only** “send more packets.” Strongest gap near the operating point (~78%).

## 2. Robustness (packet loss, VAL)

KL-CMAPPO stays better than BC on MAE at every loss rate; expert remains best absolute quality. Fig 10 / `table5_robustness.csv` include KL-CMAPPO.

## 3. Ablation (seed 42)

| Variant | TX↓ % | MAE | Recall |
| --- | ---: | ---: | ---: |
| BC only | 79.8 | 9.17 | 0.992 |
| BC+MAPPO (no KL, no constraints) | 81.3 | 9.53 | 0.993 |
| BC+KL only | 80.7 | 9.36 | 0.993 |
| BC+constraints only | 70.5 | **7.31** | 0.998 |
| **Full KL-CMAPPO** | **78.7** | **8.87** | 0.993 |
| Old MAPPO+shield | 78.9 | 13.08 | 0.935 |
| Old MAPPO no shield | 97.6 | 74.0 | 0.027 |

Constraints drive quality; KL keeps the policy near BC’s communication budget so you don’t collapse to “just TX more.”

## 4. Held-out sensors (40 never used in RL)

| Method | Held-out test TX↓ | MAE | Recall |
| --- | ---: | ---: | ---: |
| BC | 79.9±0.10 | 8.93±0.05 | 0.996 |
| **KL-CMAPPO** | **77.8±0.70** | **8.43±0.12** | **0.997** |
| Expert (val) | 73.8 | 7.25 | 0.999 |

Parameter-shared actor transfers; KL-CMAPPO still improves BC on unseen sensors.

## Claim wording
> KL-CMAPPO trades ~2 pp extra communication for better MAE/recall/precision/AoI vs BC; at matched TX budgets it still improves MAE.

## Algorithm wording
> KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility.

## Artifacts
`outputs/rl_final/scientific_validation/` · `paper_outputs/figures/` (300 DPI) · `paper_outputs/tables/`
