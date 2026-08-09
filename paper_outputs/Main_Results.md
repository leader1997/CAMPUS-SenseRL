# Main Results — CAMPUS-SenseRL

Only results considered sufficiently rigorous for manuscript reporting are emphasised below. Weak or preliminary runs are explicitly labelled.

## Rigorous enough to report

### Dataset

- Real University of Oulu Smart Campus LoRaWAN traces (2020-07-01 to 2021-05-31).
- Primary CO₂ subset: **299** ERS CO₂ sensors; 15-minute nominal cadence; chronological 60/20/20 split.

### Reconstruction (validation, 40% random causal mask)

| Model | MAE | RMSE | R² | Event recall |
|---|---:|---:|---:|---:|
| Last observation (LOCF) | 12.97 | 64.79 | 0.792 | 0.87 |
| Linear extrapolation | 23.36 | 131.82 | 0.14 | 0.863 |
| KNN neighbours | 76.37 | 151.21 | -0.131 | 0.0 |
| ExtraTrees | 10.93 | 51.73 | 0.875 | 0.889 |
| LightGBM | 11.56 | 53.51 | 0.866 | 0.848 |
| Masked ST-GNN | 16.84 | 58.33 | --- | --- |

**Finding:** Among causal reconstructors evaluated, **ExtraTrees** achieves the lowest MAE (10.93 ppm) on the validation masked-reconstruction task.

### Communication scheduling (validation subset evaluation)

Policies evaluated on the trace-driven environment with LOCF server reconstruction. Metrics are computed from actual rollouts (not invented).

| Method | Tx reduction (%) | MAE on skips | Event recall | Mean AoI |
|---|---:|---:|---:|---:|
| Fixed 15 min | 0.0 | 0.00 | 100.0% | 0.01 |
| Fixed 30 min | 50.2 | 10.66 | 55.0% | 0.72 |
| Random (matched 30min) | 50.9 | 11.95 | 55.0% | 1.00 |
| Change threshold | 58.7 | 10.58 | 100.0% | 1.42 |
| Fixed 45 min | 66.6 | 12.14 | 30.0% | 1.41 |
| Random (matched 45min) | 67.1 | 13.85 | 30.0% | 1.90 |
| Fixed 60 min | 75.3 | 13.51 | 30.0% | 2.17 |
| Random (matched 60min) | 75.9 | 14.87 | 32.5% | 2.76 |
| Semantic + safety shield (proxy) | 84.7 | 12.50 | 65.0% | 3.31 |
| Info-value heuristic | 99.7 | 32.54 | 47.5% | 7.88 |
| Uncertainty heuristic | 100.0 | 71.95 | 5.0% | 8.00 |
| AoI threshold | 100.0 | 71.95 | 5.0% | 8.00 |
| CO2 threshold | 100.0 | 71.95 | 5.0% | 8.00 |

## Not yet rigorous enough for strong claims

- Figure 10 / Table 5 robustness results are synthetic smoke tests (small environment) and are not yet campus-panel multi-seed results.
- Figure 11 omitted: current PPO/MAPPO logs are short smoke runs (<3 meaningful checkpoints, single seed) and do not meet the manuscript criterion for smoothed multi-seed convergence plots.
- Trained PPO/MAPPO are not yet plotted on Figures 5–6 because matched-budget multi-seed evaluation with event recall/MAE has not been completed; a semantic+shield heuristic proxy is shown instead and labelled as such.
- Do not claim measured battery-life gains; communication cost is a transmission proxy.

## Hypotheses status (honest)

- **H1 (adaptive reduces TX while keeping accuracy):** Partially supported by fixed vs heuristic trade-offs on the evaluated subset; full multi-seed MARL confirmation pending.
- **H2 (semantic scheduling preserves events better at matched budgets):** Requires matched-budget comparison including trained MARL; proxy semantic+shield is reported but not definitive.
- **H3–H5:** Not confirmed yet — trained multi-seed PPO/MAPPO Pareto evaluation incomplete.
