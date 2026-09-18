# Paper-ready revision summary

No training was performed. Splits, cohorts, and KL-CMAPPO checkpoints are frozen.
Every numeric claim below is copied from `results/revision/` tables.

## Protocol

- Primary event: high-CO2 (**≥1000 ppm**) **OR** rapid rise (**≥150 ppm**) on ground truth vs the same rule on the server monitor (TX GT or reconstruction). SKIP is not an automatic FN.
- TX reduction: `1 - N_tx_attempts / N_locally_available`. Packet-loss cost is counted on attempts.
- Periodic Fixed-k is aligned to the 15-minute time grid; missingness does not shift later TX times.
- Dispersion is **sample SD (ddof=1)**. Legacy `results/rl_final/` matched-budget files used ddof=0 and were not rewritten.
- Fixed-15 MAE is **N/A (full-transmission reference)**.

## Delta+heartbeat (VAL freeze, not re-selected)

- Selected: `delta_ppm=25,heartbeat=3`
- Reason: no VAL config met all constraints; selected min violation on Pareto set
- Feasible VAL configs: 0 / 16

## Validation (development cohort)

- CAMPUS-SenseRL: TX↓ 77.79 ± 0.72%; MAE 8.742 ± 0.138 ppm; recall 0.9944 ± 0.0014; constraints 5/5; n=5; source=new_evaluation
- Fixed-60: TX↓ 75.27%; MAE 11.335 ppm; recall 0.8874; constraints 0/1; n=1; source=new_evaluation
- Fixed-75: TX↓ 80.02%; MAE 11.755 ppm; recall 0.8850; constraints 0/1; n=1; source=new_evaluation
- Delta+heartbeat: TX↓ 72.83%; MAE 9.224 ppm; recall 0.9903; constraints 0/1; n=1; source=new_evaluation
- Semantic expert: TX↓ 73.89%; MAE 7.219 ppm; recall 0.9988; constraints 1/1; n=1; source=new_evaluation
- BC initialization: TX↓ 79.76%; MAE 9.205 ppm; recall 0.9930; constraints 0/1; n=1; source=legacy_frozen

## Frozen temporal test (development cohort)

- CAMPUS-SenseRL: TX↓ 78.23 ± 0.71%; MAE 8.613 ± 0.109 ppm; recall 0.9529 ± 0.0082; constraints 0/5; n=5; source=new_evaluation
- Fixed-60: TX↓ 75.14%; MAE 10.551 ppm; recall 0.2647; constraints 0/1; n=1; source=new_evaluation
- Semantic expert: TX↓ 74.46%; MAE 7.216 ppm; recall 0.9804; constraints 0/1; n=1; source=new_evaluation

**Unsupported on test:** constraint feasibility. Event recall on this slice is below 0.985 for KL and for the communicating baselines. Do not claim the test operating point is feasible.

## Held-out sensors (same test window)

- CAMPUS-SenseRL: TX↓ 77.83 ± 0.70%; MAE 8.495 ± 0.130 ppm; recall 0.9964 ± 0.0007; constraints 5/5; n=5; source=new_evaluation
- Fixed-60: TX↓ 75.10%; MAE 14.769 ppm; recall 0.7433; constraints 0/1; n=1; source=new_evaluation
- Delta+heartbeat: TX↓ 72.79%; MAE 9.441 ppm; recall 0.9878; constraints 0/1; n=1; source=new_evaluation
- Semantic expert: TX↓ 74.34%; MAE 7.230 ppm; recall 0.9984; constraints 1/1; n=1; source=new_evaluation

The generalization figure compares **development TEST vs held-out TEST**, not VAL vs held-out.

## Claims that are not supported

- Beating the semantic expert on MAE (expert MAE is lower on VAL, test, and held-out).
- Always transmitting less than BC (at tau=0.5, KL TX reduction is lower than BC; BC VAL/test rows may be legacy_frozen).
- Test-set constraint feasibility.
- Interpolated matched-budget numbers as if they were direct KL rollouts. Only the VAL tau-sweep rows labelled `direct_measured` are primary; BC remains `interpolated_legacy`. Do not present mixed KL-direct / BC-interpolated matched-budget as primary evidence.
- That the deployed KL actor is a pure neural policy with no expert-derived logic. `SemanticExpertPolicy` is not called, but a frozen expert-informed residual component is still evaluated inside `ResidualSharedActor`.
- That the relation graph is a wireless mesh or that sensors exchange packets with neighbours.

## Figures

- `fig_revision_tradeoff_mae.png` — communication reduction vs reconstruction MAE.
- `fig_revision_event_miss.png` — horizontal bars of union event-miss rate (%); 1.5% line.
- `fig_revision_generalization_dumbbell.png` — MAE, development TEST vs unseen TEST.
- `fig_revision_packet_loss_mae.png` / `fig_revision_packet_loss_event_miss.png` — shared-mask replicates; BC omitted.
- `fig_revision_contextual_relations.png` — spatial/statistical relations, **not** communication links.
- `fig_revision_constraint_matrix.png` — VAL Pass/Fail/N/A supplement.
- Fig05 remains illustrative only.

- Training performed: False
