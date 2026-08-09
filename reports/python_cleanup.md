# Python cleanup notes

Scanned imports/scripts carefully. Removed only high-confidence dead or superseded files.

## Removed

### Unused library wrappers (nothing imported them)
- `src/campus_senserl/evaluation/calibration.py`
- `src/campus_senserl/evaluation/statistics.py`
- `src/campus_senserl/evaluation/metrics.py` (re-export only; metrics live in `evaluation/__init__.py`)

### Superseded / footgun scripts
- `scripts/09_run_ablations.py` — old reward ablations (invalid for paper; replaced by `22`)
- `scripts/11_generate_paper_results.py` — legacy aggregator
- `scripts/14_evaluate_trained_policies.py` — evaluated deleted PPO/multi-seed MAPPO dumps
- `scripts/15_train_mappo_boosted.py` — unstable boost experiment (outputs already deleted)
- `scripts/18_corrected_scientific_eval.py` — replaced by `22_scientific_validation.py`
- `scripts/20_kl_cmappo_paper_pipeline.py` — replaced by `21_final_paper_results.py`

## Kept on purpose (even if “entry-point only”)

| Path | Why keep |
| --- | --- |
| `01`–`05b`, `13` | Data / recon reproducibility |
| `06` PPO, `07` MAPPO, `rl/ppo.py`, `rl/mappo.py` | Paper narrative: naïve RL fails; seed_42 MAPPO negative baseline |
| `08`, `10` | Baselines / light robustness helper |
| `12` + `visualization/paper_*` | Still wired into `run_full_pipeline.py` for early fig pack |
| `16`, `17`, `19` | BC + KL-CMAPPO training reproducibility |
| `21`, `22` | Canonical paper freeze + scientific validation |
| `rl/info_value.py`, `heuristic_policies.py` | Used by baselines / paper_eval |
| `rl/mappo_boosted.py` | Residual actor + critic used by KL-CMAPPO / BC |

Do **not** delete `mappo_boosted.py` just because script 15 is gone.
