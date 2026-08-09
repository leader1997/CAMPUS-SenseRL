# Paper outputs — FINAL manuscript pack

Regenerate with (no training):

```powershell
.venv\Scripts\python.exe scripts/23_manuscript_figures.py
```

## Main figures (only these belong in the manuscript)

| Fig | File | Content |
| --- | --- | --- |
| 1 | `fig01_sensor_network` | Deployment + RL / held-out cohorts |
| 2 | `fig02_co2_example` | Why adaptive communication |
| 3 | `fig03_framework` | Expert → BC → KL-CMAPPO (shield OFF) |
| 4 | `fig04_policy_pareto` | Overall TX↓ vs MAE Pareto |
| 5 | `fig05_matched_budget` | BC vs KL-CMAPPO at matched budgets |
| 6 | `fig06_kl_cmappo_timeline` | Real KL-CMAPPO TX/SKIP behaviour |
| 7 | `fig07_heldout_transfer` | Unseen-sensor generalization |
| 8 | `fig08_ablation_pareto` | KL / constraint ablation |
| 9 | `fig09_robustness` | Packet-loss MAE + recall |

## Docs
- `Figure_Index.md` — captions / scientific messages
- `Main_Results.md` — claim-ready tables
- `reports/scientific_validation.md` — validation proofs

## Legacy
Obsolete plots live in `legacy_provisional/figures/` (do not use in the paper).
