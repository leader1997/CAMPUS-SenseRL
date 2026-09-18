# Constraint-implementation audit

Training was not changed. This document records how λ updates are actually
computed in `campus_senserl.rl.cmappo.CMAPPOTrainer`.

## Multipliers

`lam_e` (event miss), `lam_m` (MAE), `lam_a` (AoI).

Floors/caps from `configs/rl_cmappo.yaml`:

- `eps_miss = 0.015`, `eps_mae = 9.0`, `eps_aoi = 3.5`
- `lr_lambda = 0.08`
- `lambda_max = 25`
- `lambda_min_e = 0.5`, `lambda_min_m = 1.0`, `lambda_min_a = 1.5`

## Per-step rollout estimates (`_step_costs`)

On each environment step, with `local` = locally available sensors:

```
c_tx   = mean( final_action == TRANSMIT | local )
j_miss = (# union-event FN on local) / (# true union events on local)   if any true events else 0
j_mae  = mean |GT − recon| over local SKIP slots with finite GT/recon    else 0
j_aoi  = mean raw AoI over local
```

`j_miss` uses `info["missed_events"]`, which is the **union** event
(high-CO₂ OR rapid-rise) false-negative mask from `EventDetector.classify_detection`
on **server monitoring values**, not "skip ⇒ FN".

These are **per-step / rollout estimates**, not exact full-dataset MAE or
recall after a complete split evaluation.

## Reward

If constraints are enabled:

```
viol_e = max(0, j_miss − ε_miss)
viol_m = max(0, (j_mae − ε_mae) / 10)
viol_a = max(0, (j_aoi − ε_aoi) / 4)
lag    = c_tx + λ_e viol_e + λ_m viol_m + λ_a viol_a
reward = −lag
```

PPO/GAE then uses this scalar reward. Minibatches do **not** recompute the
final dataset-level MAE/recall equations.

## Dual update (once per rollout of `n_steps`)

```
mean_miss = mean_t(j_miss)
mean_mae  = mean_t(j_mae)
mean_aoi  = mean_t(j_aoi)

λ_e ← clip( λ_e + lr_λ (mean_miss − ε_miss),          λmin_e, λmax )
λ_m ← clip( λ_m + lr_λ (mean_mae  − ε_mae)/10,        λmin_m, λmax )
λ_a ← clip( λ_a + lr_λ (mean_aoi  − ε_aoi)/4,         λmin_a, λmax )
```

## Manuscript wording that matches the code

Adaptive Lagrange multipliers are updated from rollout estimates of the
monitoring-constraint violations.

Do **not** claim that PPO directly optimizes the final dataset-level MAE/recall
equation at every minibatch.
