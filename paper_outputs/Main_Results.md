# Main Results — CAMPUS-SenseRL (claim-ready)

Sources: `outputs/rl_final/paper_final/`, `outputs/rl_final/scientific_validation/`.
Algorithm wording: **KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility** (shield OFF).

## Dataset / cohorts
- University of Oulu Smart Campus LoRaWAN CO₂ traces.
- **RL development:** frozen 40-sensor cohort (`final_cohort.json`).
- **Transfer:** 40 held-out eligible sensors never used to train/tune RL (`heldout_cohort.json`).

## Validation policy comparison (mean ± std)

| Method | TX↓ % | MAE | Recall | Precision | AoI raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 30 | 50.0 | 9.66 | 0.931 | 0.903 | 0.60 |
| Fixed 60 | 75.3 | 11.34 | 0.887 | 0.856 | 1.89 |
| Delta+heartbeat | 79.6 | 10.60 | 0.966 | 0.902 | 1.99 |
| Semantic expert | 73.9 | **7.22** | **0.999** | 0.988 | **1.75** |
| BC (5 seeds) | 80.0±0.13 | 9.19±0.03 | 0.993±0.001 | 0.975±0.002 | 2.16±0.02 |
| **KL-CMAPPO (5 seeds)** | **77.8±0.78** | **8.66±0.16** | **0.995±0.002** | **0.987±0.006** | **1.94±0.09** |
| Old MAPPO+shield | 78.9 | 13.08 | 0.935 | 0.844 | 3.35 |

**Claim (not domination):** KL-CMAPPO trades ~2.2 pp extra communication vs BC for better MAE, recall, precision, and AoI.

## Matched communication budgets (VAL, 5 seeds)

| Target TX↓ | BC MAE | KL-CMAPPO MAE |
| ---: | ---: | ---: |
| 75% | 7.94±0.02 | **7.90±0.03** |
| 78% | 8.87±0.01 | **8.72±0.07** |
| 80% | 9.20±0.02 | **9.13±0.04** |

## Frozen temporal test (KL-CMAPPO seed 123)
TX↓ 77.5%, MAE 8.37, recall 0.961, precision 0.925 (improves BC; Fixed-60 recall collapses).

## Held-out sensors (test, 5 seeds)
| Method | TX↓ % | MAE | Recall |
| --- | ---: | ---: | ---: |
| BC | 79.95±0.10 | 8.93±0.05 | 0.996 |
| KL-CMAPPO | 77.77±0.70 | **8.43±0.12** | **0.997** |

## Ablation (seed 42)
Constraints alone → best MAE but lower TX↓ (~70.5%). KL alone ≈ BC. Full KL-CMAPPO balances quality and budget (~78.7%, MAE 8.87). Old MAPPO without shield collapses (MAE≈74).

## Reconstruction note
Causal reconstructors were benchmarked; **LOCF** is used online in the RL environment for causal sequential suitability. Detailed recon numbers belong in the appendix — not the main contribution.
