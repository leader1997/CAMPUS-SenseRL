# Scientific validation (post KL-CMAPPO freeze)

Core 5-seed KL-CMAPPO models were **not** retrained.

Claim phrasing: KL-CMAPPO trades ~2.24 pp extra communication for better MAE/recall/precision/AoI vs BC — not strict domination.

Algorithm description: **KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility**.

Artifacts: `C:/Users/MossabBatal/Desktop/OULU_RL/outputs/rl_final/scientific_validation`

## Outputs
- Matched budget: `matched_budget_summary.csv` → Fig 5/6/8
- Robustness: `robustness_summary.csv` → Fig 10 / table5
- Ablation: `ablation_kl_cmappo_val.csv` → Fig 9 / table4
- Held-out sensors: `heldout_transfer_summary.csv`
