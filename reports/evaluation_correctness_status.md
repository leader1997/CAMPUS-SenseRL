# Evaluation correctness pass (response to scientific review)

## Verdict

Kept the codebase. Fixed wiring/evaluation bugs. **Did not retrain.**

Full validation comparison is complete: `outputs/rl_final/policy_eval/`.

## Fixes shipped

| Issue | Fix |
| --- | --- |
| Graph silent identity | `graph_fail_fast: true`; raise if hybrid fails; `make_final_env` always passes frozen `sensor_ids` |
| Unfair recon n | Trees scored only on shared hide masks; removed fake `stgnn_pending_aligned`; real ST-GNN forward |
| TX reduction wrong | `evaluate_policy`: rate = TX / locally_available; Fixed-15 → **0%** |
| No RL in tables | `scripts/14_evaluate_trained_policies.py` — all 5 PPO + 5 MAPPO seeds |
| ChangeThreshold crash | `__post_init__` initializes `_last_local` |
| Packet loss = measurement fail | Env drops delivery with `local_available` unchanged |
| Synthetic robustness | `scripts/10_run_robustness.py` → real val + MAPPO + `packet_loss_rate` |
| Stale paper figs/tables | → `paper_outputs/legacy_provisional/` |

## Full validation (entire val split, n=40, hybrid, 826 events @ 1000 ppm)

### MAPPO + shield (requested table)

| Seed | Tx reduction % | MAE skipped | Event recall | Event F1 | Mean AoI | Shield overrides |
| ---: | -------------: | ----------: | -----------: | -------: | -------: | ---------------: |
| 42 | 78.91 | 13.08 | 0.935 | 0.966 | 3.35 | 41075 |
| 123 | 77.16 | 13.26 | 0.915 | 0.956 | 3.39 | 42145 |
| 2024 | 79.42 | 13.57 | 0.921 | 0.959 | 3.53 | 46329 |
| 3407 | 75.94 | 13.50 | 0.959 | 0.979 | 3.34 | 43835 |
| 9999 | 76.48 | 13.64 | 0.923 | 0.960 | 3.22 | 44009 |

### PPO + shield

| Seed | Tx reduction % | MAE skipped | Event recall | Event F1 | Mean AoI | Shield overrides |
| ---: | -------------: | ----------: | -----------: | -------: | -------: | ---------------: |
| 42 | 39.15 | 12.29 | 0.975 | 0.987 | 1.59 | 23543 |
| 123 | 40.43 | 11.30 | 0.969 | 0.984 | 1.68 | 30654 |
| 2024 | 40.20 | 11.03 | 0.988 | 0.994 | 1.65 | 22955 |
| 3407 | 40.67 | 11.81 | 0.950 | 0.975 | 1.64 | 25166 |
| 9999 | 42.54 | 14.22 | 0.915 | 0.956 | 1.71 | 21534 |

### Same-trace baselines

| Method | Tx red % | MAE skipped | Recall | F1 | Mean AoI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 15 | **0.0** | — | 1.000 | 1.000 | 0.00 |
| Fixed 30 | 44.65 | 9.55 | 0.982 | 0.991 | 0.54 |
| Fixed 45 | 59.58 | 10.25 | 0.966 | 0.983 | 1.06 |
| Fixed 60 | 66.98 | 10.99 | 0.964 | 0.982 | 1.57 |
| Change threshold | 69.70 | 10.79 | 0.958 | 0.978 | 2.39 |
| Uncertainty thr. | 38.94 | 11.08 | 1.000 | 1.000 | 1.72 |
| Semantic heuristic | 78.72 | 12.35 | 0.931 | 0.964 | 3.26 |
| AoI threshold | 79.70 | 13.56 | 0.923 | 0.960 | 3.55 |

### MAPPO without shield (eval-only ablation)

Collapses: ~97–100% TX reduction, recall ≈ 0–0.63, MAE 30–90. **The trained MAPPO policy alone does not transmit meaningfully; the shield carries performance.**

## Fair reconstruction (re-run)

Approx. same n (~82.7k hidden cells). LOCF MAE ≈ 10.34 still best. Real ST-GNN MAE ≈ 17.53 (no longer LOCF-labeled). Trees ≈ 12.2–12.4 on the **same** masks.

## Interpretation (before any retrain)

1. Graph wiring and TX accounting are now trustworthy.
2. MAPPO+shield ≈ semantic heuristic / AoI threshold — **not a clear RL win**.
3. PPO ≈ Fixed-30 bandwidth with worse MAE than Fixed-30.
4. Learning signal remains weak; shield dominates MAPPO eval.

## Still open

1. Real-val robustness sweep (`10_run_robustness.py` full period) — not yet run after rewrite
2. True ablations (feature / reward / graph off) — `outputs/ablations_final/` still empty
3. Clean train-time MAPPO±shield (current no-shield is eval-only on shield-trained ckpt)
4. Slight fair-recon n gaps (ST-GNN warm-up; tree merge) — tighten if publishing recon table

## Do not put in paper yet

`paper_outputs/legacy_provisional/`, old proxy figures, fake ablation table.
