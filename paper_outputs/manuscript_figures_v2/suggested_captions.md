# Suggested captions (manuscript_figures_v2)

**Fig. 1** Spatial layout of the smart-campus deployment (metadata). Light markers: remaining campus sensors; dark markers: 40 RL development sensors; diamonds: 40 held-out transfer sensors. Counts are for caption use only.

**Fig. 2** Illustrative validation window: true CO₂, KL-CMAPPO TX markers, causal reconstruction during SKIP, and 1000 ppm threshold. Fixed-15 TX count and reduction verified from this frozen rollout (see `fig02_meta.json`).

**Fig. 3** KL-CMAPPO training workflow. BC is an initialization stage; the final policy is shield-free KL-CMAPPO.

**Fig. 4** Validation skipped-observation CO₂ MAE for Fixed 60, Delta+heartbeat, and KL-CMAPPO. TX↓ annotations from the same validation summary. KL-CMAPPO: mean ± std over five seeds; baselines deterministic (n=1).

**Fig. 5** Frozen TEST important-event recall (development cohort). Methods: Fixed 60, Delta+heartbeat, KL-CMAPPO (frozen seed-123 checkpoint). Y-axis 0–100%.

**Fig. 6** Validation reconstruction quality during KL-CMAPPO training (mean ± std over five seeds). Dashed line: predefined MAE = 9 ppm feasible region. Not a raw-reward plot.

**Fig. 7** Mechanistic ablation, seed 42 only. Bars: CO₂ MAE; annotations: TX reduction. BC framed as initialization. Do not interpret as multi-seed significance.

**Fig. 8** KL-CMAPPO generalization: development-cohort frozen TEST MAE vs held-out TEST MAE (five seeds mean ± std on held-out). BC omitted.

**Fig. 9** Packet-loss robustness of KL-CMAPPO alone (validation stress; five-seed mean ± std).

**Fig. 10** Cumulative transmissions over a fixed validation interval for Fixed 15, Fixed 60, and KL-CMAPPO (illustrative frozen rollout).
