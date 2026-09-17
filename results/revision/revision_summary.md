# Revision evaluation summary

No training was performed. Frozen KL-CMAPPO checkpoints were evaluated as-is.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -u scripts\29_revision_evaluation.py --device auto
.\.venv\Scripts\python.exe -u scripts\30_revision_figures.py
```

Resume is supported: existing `results/revision/all_method_evaluations.csv` rows are skipped unless `--force`.

## Training / checkpoints

- Training performed: **false**
- KL-CMAPPO frozen runs: seeds 42, 123, 2024, 3407, 9999  
  `results/rl_final/cmappo_kl/seed_*/best_model.pt`
- Original single-seed test report remains seed **123**
- BC checkpoints: **missing in this clone**. BC val/test/held-out/packet-loss numbers were imported from frozen CSVs and labelled `legacy_frozen_summary` / `legacy_frozen_csv`.
- Ablation: existing seed-42 CSV exported to `results/revision/ablation.csv` (single-run; no significance claims)

## Delta+heartbeat selection (validation only)

- Grid: delta ∈ {25,50,75,100} ppm × heartbeat ∈ {3,4,5,6} intervals (16 configs)
- **No grid point satisfied all constraints** (MAE≤9, recall≥0.985, AoI≤3.5)
- Selected: `delta_ppm=25,heartbeat=3` (minimum constraint-violation score on the Pareto set)
- VAL: TX↓ 72.83%, MAE 9.224 ppm, recall 0.990, AoI 1.44
- This selection was frozen before test / held-out / packet-loss evaluation

## Constraint thresholds

- MAE ≤ 9 ppm (skipped/unavailable-observation MAE)
- Event miss rate ≤ 0.015 ⇒ recall ≥ 0.985
- Raw AoI ≤ 3.5 decision intervals
- Fixed-15 MAE is **N/A** under zero packet loss (no skipped locally available observations)

## Runtime (this machine, CUDA)

| Stage | Seconds |
|---|---:|
| VAL baselines + delta grid | 306 |
| VAL KL-CMAPPO (5 seeds) | 142 |
| Frozen temporal test | 251 |
| Held-out 40-sensor test | 222 |
| Packet-loss (shared masks) | 645 |
| **Total evaluation** | **1568 (~26 min)** |
| Figures | ~15 s |

## Unfavorable / honest findings (kept)

- Semantic expert still has the lowest MAE (~7.22 ppm) and transmits more than KL-CMAPPO.
- On the **frozen temporal test**, KL-CMAPPO 5-seed mean recall is **0.953**, below 0.985, so test constraints are **not** satisfied. Fixed periodic recall collapses (Fixed-60 recall 0.265). This is a small-event-count test split (102 events), not a retuned result.
- VAL-selected Delta+heartbeat does **not** meet the 9 ppm MAE constraint (9.22 ppm).
- BC often transmits slightly less than KL-CMAPPO; constrained RL spends a bit more communication to improve MAE/recall.
- Under 20–40% packet loss, no compared method except sometimes the semantic expert stays inside the MAE=9 / recall=0.985 box.

## Figure replacements

| Old | Role | Replace with |
|---|---|---|
| fig03_agent_neighbour_network | can look like a mesh | **fig_revision_contextual_relations.png** |
| fig05_adaptive_communication | keep | keep, labelled **illustrative only** |
| fig10 (vs Fixed-60 only) | too narrow | **fig_revision_mae_vs_reduction.png** |
| fig11 event window-style | supplementary | keep illustrative; use tables for evidence |
| fig12 unseen KL-only | too narrow | **fig_revision_unseen_comparison.png** |
| fig13 packet-loss KL-only | too narrow | **fig_revision_packet_loss_comparison.png** (+ recall companion) |

Primary quantitative evidence: `results/revision/` CSVs, not single windows.
