# Figure source audit — manuscript_figures_v2

No training. All quantitative points read from frozen CSV/JSON under `outputs/rl_final/`.
Illustrative rollouts use frozen checkpoint `cmappo_kl/seed_123/best_model.pt` only.

## `main/fig01_campus_deployment.png`
- **Scientific question:** Is KL-CMAPPO evaluated on a real distributed smart-campus deployment?
- **Source:** `devices + final_cohort.json + heldout_cohort.json`
- **Split:** metadata (spatial)
- **Methods:** other / RL development / held-out
- **Seeds:** N/A
- **Metric:** sensor coordinates
- **Exact plotted numbers:** other=349, RL=40, heldout=40

## `main/fig02_real_adaptive_communication.png`
- **Scientific question:** What does adaptive TX/SKIP scheduling look like on a real CO₂ series?
- **Source:** `frozen KL-CMAPPO seed_123 val rollout + FixedInterval(1)`
- **Split:** validation (illustrative window)
- **Methods:** KL-CMAPPO (+ Fixed15 count annotation)
- **Seeds:** policy seed 123; rollout seed 42
- **Metric:** CO2 ppm + TX markers
- **Exact plotted numbers:** Fixed15 TX=192, KL TX=55, red=71.35%

## `main/fig03_kl_cmappo_framework.png`
- **Scientific question:** How is the proposed KL-CMAPPO policy trained?
- **Source:** `schematic`
- **Split:** N/A
- **Methods:** expert → BC init → KL-CMAPPO → TX/SKIP
- **Seeds:** N/A
- **Metric:** N/A
- **Exact plotted numbers:** BC = initialization; KL-CMAPPO = final shield-free method

## `main/fig04_main_mae_comparison.png`
- **Scientific question:** At similar high communication savings, which strategy preserves CO₂ best?
- **Source:** `outputs/rl_final/paper_final/full_val_summary.csv`
- **Split:** validation
- **Methods:** Fixed 60, Delta+heartbeat, KL-CMAPPO
- **Seeds:** Fixed60/Delta n=1; KL-CMAPPO n=5 (mean±std)
- **Metric:** skipped-observation CO2 MAE (ppm)
- **Exact plotted numbers:** Fixed 60=11.335 (75.3% TX↓); Delta + heartbeat=10.597 (79.6% TX↓); KL-CMAPPO=8.742 (77.8% TX↓)

## `main/fig05_event_recall_test.png`
- **Scientific question:** Does communication reduction cause missed important CO₂ events?
- **Source:** `outputs/rl_final/paper_final/full_test_summary.csv`
- **Split:** frozen TEST (development cohort)
- **Methods:** Fixed 60, Delta+heartbeat, KL-CMAPPO
- **Seeds:** all n=1 on this frozen test report (KL = seed 123 checkpoint)
- **Metric:** important-event recall (%)
- **Exact plotted numbers:** Fixed 60=26.47%; Delta + heartbeat=90.20%; KL-CMAPPO=96.08%

## `main/fig06_rl_training_validation_mae.png`
- **Scientific question:** Did KL-CMAPPO learning enter the predefined feasible reconstruction-quality region?
- **Source:** `outputs/rl_final/cmappo_kl/seed_*/metrics.json (val rows)`
- **Split:** validation checkpoints during training
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** validation CO2 MAE (ppm)
- **Exact plotted numbers:** mean±std; constraint MAE=9 ppm

## `main/fig07_kl_constraint_ablation.png`
- **Scientific question:** What do KL anchoring and explicit constraints contribute?
- **Source:** `outputs/rl_final/scientific_validation/ablation_kl_cmappo_val.csv`
- **Split:** validation; single-seed mechanistic ablation (seed 42)
- **Methods:** BC init / no KL-no constr. / KL only / constr. only / full KL-CMAPPO
- **Seeds:** 1 (seed 42) — not multi-seed statistics
- **Metric:** CO2 MAE (ppm) + TX↓ annotation
- **Exact plotted numbers:** BC initialization only=9.166; RL fine-tune no KL / no constr.=9.564; + KL only=9.356; + constraints only=7.595; Full KL-CMAPPO=8.923

## `main/fig08_unseen_sensor_generalization.png`
- **Scientific question:** Does KL-CMAPPO work on sensors unused during policy development?
- **Source:** `paper_final/full_test_summary.csv; scientific_validation/heldout_transfer_summary.csv`
- **Split:** TEST vs TEST (development frozen report vs held-out transfer)
- **Methods:** KL-CMAPPO only
- **Seeds:** dev n=1 (frozen seed 123); heldout n=5 mean±std
- **Metric:** CO2 MAE (ppm)
- **Exact plotted numbers:** dev=8.454; heldout=8.495±0.116

## `main/fig09_packet_loss_robustness.png`
- **Scientific question:** How does KL-CMAPPO behave as wireless packet delivery deteriorates?
- **Source:** `outputs/rl_final/scientific_validation/robustness_summary.csv`
- **Split:** validation stress (packet loss)
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** CO2 MAE (ppm)
- **Exact plotted numbers:** 0%→8.742; 10%→9.001; 20%→9.254; 40%→9.988

## `main/fig10_cumulative_transmissions.png`
- **Scientific question:** What does ~78% TX reduction mean operationally over time?
- **Source:** `val rollouts start=3552 length=672; KL seed_123`
- **Split:** validation
- **Methods:** Fixed 15, Fixed 60, KL-CMAPPO
- **Seeds:** illustrative (KL seed 123)
- **Metric:** cumulative transmissions
- **Exact plotted numbers:** end Fixed15=24424, Fixed60=6021, KL=5606

## `supplementary/figS01_bc_initialization_vs_kl_matched_budget.png`
- **Scientific question:** Effect of KL-CMAPPO fine-tuning relative to BC initialization at matched budgets?
- **Source:** `matched_budget_summary.csv`
- **Split:** validation
- **Methods:** BC initialization, KL-CMAPPO
- **Seeds:** 5
- **Metric:** CO2 MAE
- **Exact plotted numbers:** targets 75/78/80%

## `supplementary/figS02_training_event_recall.png`
- **Scientific question:** Does validation event recall stay near the constraint during training?
- **Source:** `cmappo_kl metrics.json val_recall`
- **Split:** validation checkpoints
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** event recall (%)
- **Exact plotted numbers:** mean±std; threshold 98.5%; axis 90–100%

## `supplementary/figS03_training_tx_reduction.png`
- **Scientific question:** Did reaching the MAE constraint simply force always-transmit?
- **Source:** `cmappo_kl metrics.json val_tx_reduction`
- **Split:** validation checkpoints
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** TX reduction (%)
- **Exact plotted numbers:** mean±std

## `supplementary/figS04_feasible_seeds.png`
- **Scientific question:** How many seeds satisfy constraints at each checkpoint?
- **Source:** `metrics.json feasible`
- **Split:** validation checkpoints
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** feasible count 0–5
- **Exact plotted numbers:** Boolean feasible field sum

## `supplementary/figS05_kl_beta_schedule.png`
- **Scientific question:** How does the KL regularization coefficient β anneal?
- **Source:** `metrics.json kl_beta (coefficient, NOT D_KL)`
- **Split:** training logs
- **Methods:** KL-CMAPPO
- **Seeds:** 5
- **Metric:** kl_beta coefficient
- **Exact plotted numbers:** β≈0.080→0.026

## `supplementary/figS06_adaptive_lagrange.png`
- **Scientific question:** How do adaptive constraint multipliers evolve?
- **Source:** `metrics.json lam_e/lam_m/lam_a`
- **Split:** training logs
- **Methods:** KL-CMAPPO (λ_event, λ_MAE, λ_AoI)
- **Seeds:** 5
- **Metric:** Lagrange multipliers
- **Exact plotted numbers:** lam_e→λ_event, lam_m→λ_MAE, lam_a→λ_AoI

## `supplementary/figS07_complete_policy_benchmark.png`
- **Scientific question:** Complete validation MAE benchmark across reported policies?
- **Source:** `full_val_summary.csv`
- **Split:** validation
- **Methods:** Fixed30/60, Delta, Expert, BC init, KL-CMAPPO
- **Seeds:** mixed (1 or 5)
- **Metric:** MAE ppm
- **Exact plotted numbers:** prefer main TABLE for completeness

## `supplementary/figS08a_packet_loss_mae_all_methods.png`
- **Scientific question:** Full packet-loss MAE comparison (Expert / BC init / KL)?
- **Source:** `robustness_summary.csv`
- **Split:** validation stress
- **Methods:** Semantic expert, BC initialization, KL-CMAPPO
- **Seeds:** Expert n=1; BC/KL n=5
- **Metric:** MAE
- **Exact plotted numbers:** MAE only

## `supplementary/figS08b_packet_loss_recall_all_methods.png`
- **Scientific question:** Full packet-loss recall comparison?
- **Source:** `robustness_summary.csv`
- **Split:** validation stress
- **Methods:** Semantic expert, BC initialization, KL-CMAPPO
- **Seeds:** Expert n=1; BC/KL n=5
- **Metric:** event recall (%)
- **Exact plotted numbers:** recall only (separate from MAE)

## `supplementary/figS09_transmission_heatmap.png`
- **Scientific question:** Do TX/SKIP decisions vary across sensors and time?
- **Source:** `KL-CMAPPO seed_123 val rollout`
- **Split:** validation
- **Methods:** KL-CMAPPO only
- **Seeds:** illustrative
- **Metric:** TX / SKIP / missing map
- **Exact plotted numbers:** 40×96

## `supplementary/figS10_reconstruction_benchmark.png`
- **Scientific question:** Fair causal reconstruction benchmark (not central contribution)?
- **Source:** `outputs\reconstruction_final\fair_benchmark\fair_results_summary.csv`
- **Split:** fair reconstruction benchmark
- **Methods:** LOCF, ExtraTrees, LightGBM, Linear extrapolation, ST-GNN, Historical mean
- **Seeds:** 5
- **Metric:** masked MAE
- **Exact plotted numbers:** LOCF=10.34; ExtraTrees=12.16; LightGBM=12.37; Linear extrapolation=19.72; ST-GNN=17.53; Historical mean=24.75

