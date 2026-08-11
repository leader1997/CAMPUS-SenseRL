# Main Results — CAMPUS-SenseRL / KL-CMAPPO

**Proposed method:** KL-CMAPPO (shield-free).  
**BC** is a **policy initialization stage** (not a competing final method).  
**Semantic expert** is the demonstration teacher / strong reference.

Algorithm wording: **KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility**, initialized from expert-guided behavior cloning.

Sources: `results/csv_json/` (copied from `results/rl_final/`).

## Dataset / cohorts
- University of Oulu Smart Campus LoRaWAN CO₂ traces.
- **RL development:** frozen 40-sensor cohort (`final_cohort.json`).
- **Transfer:** 40 held-out sensors never used to train/tune RL (`heldout_cohort.json`).

## Main validation comparison (practical baselines)

| Category | Method | TX↓ % | MAE | Recall |
| --- | --- | ---: | ---: | ---: |
| Periodic | Fixed 30 | 50.0 | 9.66 | 0.931 |
| Periodic | Fixed 60 | 75.3 | 11.34 | 0.887 |
| Heuristic | Delta + heartbeat | 79.6 | 10.60 | 0.966 |
| Expert reference | Semantic expert | 73.9 | **7.22** | **0.999** |
| **Proposed** | **KL-CMAPPO (5 seeds)** | **77.8±0.72** | **8.74±0.14** | **0.994±0.001** |

**Claim:** At ~75–80% communication reduction, KL-CMAPPO substantially improves reconstruction quality relative to Fixed-60 and Delta+heartbeat, while approaching (but not beating) the handcrafted semantic expert on MAE.

## Ablation — RL fine-tuning beyond BC initialization (seed 42)

| Variant | TX↓ % | MAE |
| --- | ---: | ---: |
| BC initialization only | 79.6 | 9.17 |
| Init + MAPPO (no KL/constraints) | 81.2 | 9.56 |
| Init + KL only | 80.6 | 9.36 |
| Init + constraints only | 73.2 | 7.59 |
| **Full KL-CMAPPO** | **78.6** | **8.92** |

Constraints protect information quality; KL helps retain communication-efficient BC behavior; full KL-CMAPPO balances both.

## Matched-budget RL vs initialization (supplementary / VAL, 5 seeds)

| Target TX↓ | BC initialization MAE | KL-CMAPPO MAE |
| ---: | ---: | ---: |
| 75% | 7.97±0.02 | **7.94±0.02** |
| 78% | 8.92±0.01 | **8.79±0.07** |
| 80% | 9.25±0.01 | **9.17±0.04** |

## Held-out sensors (proposed method, test, 5 seeds)
KL-CMAPPO: TX↓ **77.83±0.63%**, MAE **8.50±0.12**, recall **0.996**.

## Figures
15 manuscript figures in `results/figures/` (PNG, 600 dpi) — regenerate with
`python scripts/28_manuscript_figures.py` (inference only, no training).
Plotted values are logged to `results/figures/figure_values.csv`.
