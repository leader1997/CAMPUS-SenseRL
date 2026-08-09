# Paper outputs — FINAL

Ready for manuscript drafting. See also `reports/final_paper_results.md` and `reports/scientific_validation.md`.

## Tables
- `tables/table3_policy_comparison.csv` / `table3_policy_summary.csv` — main val comparison
- `tables/table3_policy_comparison_test.csv` / `table3_policy_summary_test.csv` — frozen test
- `tables/table4_ablation.csv` — KL-CMAPPO component ablation
- `tables/table5_robustness.csv` — packet loss / outages (includes KL-CMAPPO)
- `tables/table_matched_budget.csv` — matched TX-budget Pareto
- `tables/table_heldout_transfer.csv` — unseen-sensor transfer

## Figures (300 DPI PNG + PDF)
- `fig05_*` matched-budget reconstruction Pareto
- `fig06_*` event recall vs TX
- `fig08_*` raw AoI
- `fig09_*` algorithmic ablation
- `fig10_*` robustness

## Canonical checkpoints (under `outputs/`)
- `rl_final/cmappo_kl/seed_*/best_model.pt` — frozen 5-seed KL-CMAPPO
- `rl_final/paper_asap/bc/seed_*/` — BC baselines
- `rl_final/cmappo_ablations/` — ablation variants
- `rl_final/scientific_validation/` — post-freeze proof CSVs
- `rl_final/mappo/seed_42/` — old shield MAPPO (negative baseline only)
