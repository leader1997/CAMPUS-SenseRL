# Making RL work — diagnosis and fix

## Why old MAPPO/PPO failed

1. **Safety shield stole credit:** policy learned SKIP; shield forced TX; reward attributed to SKIP.
2. **Always-SKIP was near-optimal** under LOCF + AoI shield + `w_tx=1`.
3. **No behavioral prior** — Bernoulli collapsed to 0 TX without shield.

## Trick that works

1. **Semantic expert** (local rules): TX if large ΔCO₂ **or** AoI≥3 **or** CO₂≥900 **or** |local−recon|≥25.
2. **Residual actor:** `logit = f_θ(s) + α · h(s)` with differentiable expert bias `h(s)`.
3. **Train without shield** so the policy owns decisions.
4. **BC distillation** into the neural actor (pure imitation). Unconstrained RL fine-tune was unstable (drifted to always-TX); keep BC as the primary learner for now.

## Full-val results (no shield, frozen cohort, hybrid graph, 826 events)

From `outputs/rl_final/policy_eval/boosted_vs_baselines_val.csv`:

| Method | TX red % | MAE skip | Recall | F1 | Mean AoI |
| --- | ---: | ---: | ---: | ---: | ---: |
| **mappo_boosted_bc** | **73.85** | **9.24** | **0.992** | **0.996** | 1.44 |
| semantic_expert (rules) | 71.68 | **7.87** | **0.999** | **0.999** | 1.40 |
| Fixed 30 | 49.95 | 9.66 | 0.931 | 0.964 | 0.60 |
| Fixed 60 | 75.27 | 11.34 | 0.887 | 0.940 | 1.84 |
| Old MAPPO+shield | ~77 | ~13 | ~0.93 | ~0.96 | ~3.4 |

**Win vs old RL / Fixed-60:** far better MAE and recall at similar TX reduction, **without** the shield.

Note: unconstrained RL fine-tune after BC drifted to always-TX; **use BC distill checkpoint**. Optional next step: constrained fine-tune with a hard MAE/recall gate.

## Code

- `src/campus_senserl/rl/expert_policy.py`
- `src/campus_senserl/rl/mappo_boosted.py`
- `configs/rl_learn.yaml`
- `scripts/15_train_mappo_boosted.py` (BC+RL; RL phase unstable)
- `scripts/16_distill_expert_policy.py` (**use this**)
