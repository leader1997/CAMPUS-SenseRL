# Figure Index — CAMPUS-SenseRL

All figures live in `paper_outputs/figures/`. Exploratory plots remain under `figures/eda/` or `outputs/` and are **not** manuscript figures.

## Figure 1: `fig01_sensor_network.png`

- **Manuscript figure number:** Figure 1
- **Purpose:** Describe the real campus sensing deployment used in experiments.
- **Variables shown:** Longitude/latitude; floor; device type counts; CO2 vs non-CO2.
- **Experiment / configuration:** devices.json metadata; primary experiments use 299 ERS CO2 sensors.
- **Suggested caption:** University of Oulu campus IoT deployment: spatial layout of sensors (left) and floor-wise device-type composition (right). CO2 experiments use the ERS CO2 subset.
- **Main result:** The network comprises 429 sensors including 299 CO2 devices across floors −1 to 5, providing the spatiotemporal substrate for reconstruction and scheduling experiments.
- **Status:** Final (real metadata)

## Figure 2: `fig02_co2_example.png`

- **Manuscript figure number:** Figure 2
- **Purpose:** Motivate adaptive communication: stable periods vs rapid CO2 changes.
- **Variables shown:** CO2 (ppm) vs time; optional rapid-change markers; 1000 ppm reference.
- **Experiment / configuration:** Validation split; sensor 03101B; window 2021-02-16 08:30:00+00:00 → 2021-02-19 11:45:00+00:00.
- **Suggested caption:** Example campus CO2 trajectory containing both stable intervals and rapid changes. Fixed-period uplinks are often redundant during stable regimes but information-critical during transitions.
- **Main result:** Real CO2 dynamics are intermittent: long near-stationary periods punctuated by rises, supporting value-based rather than purely periodic transmission.
- **Status:** Final (real measurements)

## Figure 3: `fig03_framework.png`

- **Manuscript figure number:** Figure 3
- **Purpose:** Present the CAMPUS-SenseRL methodological framework.
- **Variables shown:** Conceptual components (sensing, semantic value, RL decision, shield, reconstruction).
- **Experiment / configuration:** Architecture diagram (not an empirical plot).
- **Suggested caption:** CAMPUS-SenseRL framework: local measurements inform a semantic/uncertainty/AoI-aware transmission decision; skipped values are reconstructed at the server under a safety shield.
- **Main result:** The figure clarifies the counterfactual information split between edge decisions and server-visible data.
- **Status:** Final (schematic)

## Figure 4: `fig04_reconstruction_comparison.png`

- **Manuscript figure number:** Figure 4
- **Purpose:** Compare causal CO2 reconstruction methods under masked validation.
- **Variables shown:** MAE (ppm) by model.
- **Experiment / configuration:** Val split; 40% random mask; up to 30 sensors for classical baselines; ST-GNN on 20-sensor subset.
- **Suggested caption:** Causal CO2 reconstruction error (MAE) under a 40% random observation mask on the validation period. Non-causal interpolation is excluded from this comparison.
- **Main result:** Tree models (ExtraTrees/LightGBM) currently outperform LOCF and the early ST-GNN checkpoint on this masked task; KNN neighbours perform poorly when simultaneous neighbour coverage is sparse.
- **Status:** Final for reported baselines (single-run; multi-seed pending)

## Figure 5: `fig05_tradeoff_reconstruction.png`

- **Manuscript figure number:** Figure 5
- **Purpose:** Show communication–reconstruction trade-off across scheduling policies.
- **Variables shown:** Transmission reduction (%) vs MAE on skipped observations (ppm).
- **Experiment / configuration:** Val split; 20 sensors; 1200 steps; LOCF server reconstruction; important event = CO2>=800 ppm OR rise>=80 ppm/interval.
- **Suggested caption:** Trade-off between transmission reduction and CO2 reconstruction MAE on skipped slots for fixed, random, heuristic, and semantic+shield proxy policies.
- **Main result:** Higher transmission reduction generally increases skip-MAE; change-threshold and the semantic+shield proxy are more efficient than matched random at comparable reductions. Trained multi-seed MARL points are not yet included.
- **Status:** Subset evaluation (real); full MARL multi-seed pending

## Figure 6: `fig06_tradeoff_event_recall.png`

- **Manuscript figure number:** Figure 6
- **Purpose:** Assess preservation of important CO2 events under communication reduction.
- **Variables shown:** Transmission reduction (%) vs important-event recall (%).
- **Experiment / configuration:** Same as Figure 5; event = CO2>=800 ppm OR rise>=80 ppm/interval.
- **Suggested caption:** Important-event recall versus transmission reduction for communication scheduling policies on the campus validation traces.
- **Main result:** At high reduction, event recall collapses for several heuristics; the semantic+shield proxy retains higher recall (~67.5%) at ~88% reduction than matched random/fixed schedules at ~75% reduction. This is a proxy result, not trained MARL.
- **Status:** Subset evaluation (real); matched-budget MARL pending

## Figure 8: `fig08_aoi_comparison.png`

- **Manuscript figure number:** Figure 8
- **Purpose:** Compare Age of Information behaviour across policy families.
- **Variables shown:** Transmission reduction (%) vs mean AoI (intervals).
- **Experiment / configuration:** Same rollout setting as Figures 5–6.
- **Suggested caption:** Mean Age of Information as a function of transmission reduction for fixed, heuristic, random, and semantic+shield proxy policies.
- **Main result:** Mean AoI rises with aggressive skipping; policy family influences the AoI–reduction curve.
- **Status:** Subset evaluation (real)

## Figure 7: `fig07_adaptive_transmission_example.png`

- **Manuscript figure number:** Figure 7
- **Purpose:** Illustrate adaptive TRANSMIT/SKIP behaviour on a real CO2 trajectory.
- **Variables shown:** True CO2; transmitted points; reconstructed skips; uncertainty; TX/SKIP markers; event threshold.
- **Experiment / configuration:** Info-value heuristic + safety shield; val split; one sensor; 160 steps after warm-up.
- **Suggested caption:** Example adaptive transmission timeline: true CO2, transmitted observations, reconstructions during skips, and uncertainty. The policy tends to spare transmissions during stable periods and activate around informative changes.
- **Main result:** The qualitative pattern matches the intended semantic scheduling behaviour on a real campus segment (proxy policy, not final trained MARL).
- **Status:** Illustrative (real trace + proxy policy)

## Figure 10: `fig10_robustness.png`

- **Manuscript figure number:** Figure 10
- **Purpose:** Probe degradation under packet loss / missingness stress.
- **Variables shown:** Stress level (%) vs mean episode cost.
- **Experiment / configuration:** Synthetic smoke robustness script (small env), not full campus panel sweep.
- **Suggested caption:** Preliminary robustness trends under increasing packet loss and missingness.
- **Main result:** Costs worsen under higher stress in the smoke test; campus-scale robustness remains to be reported.
- **Status:** Preliminary smoke test

## Figure 11: `fig11_training_convergence.png`

- **Manuscript figure number:** Figure 11
- **Purpose:** Show RL training stability (reward, TX rate, event-miss).
- **Variables shown:** Training steps vs smoothed metrics across seeds.
- **Experiment / configuration:** Insufficient: only short single-seed smoke runs available.
- **Suggested caption:** N/A — figure not generated.
- **Main result:** No manuscript-ready convergence figure yet.
- **Status:** OMITTED (insufficient evidence)
