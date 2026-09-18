# Deployment actor audit

Inspected frozen KL-CMAPPO checkpoints and the exact inference call path.
No retraining. Residual mechanism left unchanged.

## Call path

```
observation (14-D per logical per-sensor agent)
  → ResidualSharedActor.forward()
      neural = f_θ(obs)
      logit  = neural + residual_scale * heuristic_logits_torch(obs)
  → p = σ(logit)
  → TRANSMIT iff p > τ else SKIP
```

Loader: `campus_senserl.evaluation.rl_policy_eval.load_mappo_policy`.
When `actor_type == "residual_heuristic"`, it instantiates `ResidualSharedActor`
and loads the checkpoint `actor` state dict. Default τ = 0.5.

`heuristic_logits_torch` lives in `campus_senserl.rl.expert_policy` and is a
differentiable soft score over ΔCO₂, AoI, CO₂ level, local-vs-server
disagreement, and uncertainty. It is **not** `SemanticExpertPolicy.act`.

## Distinction required for the manuscript

| | Statement | Status |
|---|---|---|
| **A** | The separate semantic-expert policy is not called | **True.** `SemanticExpertPolicy` is never instantiated by `load_mappo_policy`. No external expert object is required at deployment. |
| **B** | Expert-derived heuristic logic is still embedded/evaluated inside the final actor | **True.** Every forward pass evaluates `heuristic_logits_torch(obs)` and adds `residual_scale` times that vector to the neural logits. |

Preferred wording:

- "fixed expert-derived residual prior"
- "expert-informed residual component"

Do **not** call this an external expert policy.
Do **not** call the deployed actor a pure neural actor.

## Checkpoints (`results/rl_final/cmappo_kl/seed_*/best_model.pt`)

All five `best_model.pt` files have `actor_type = residual_heuristic`, `obs_dim = 14`,
and a `residual_scale` tensor in the actor state dict.

| seed | residual_scale | trainable during KL-CMAPPO | evaluated at deployment |
|---:|---:|---|---|
| 42 | 1.631554 | frozen (`configs/rl_cmappo.yaml`: `freeze_residual_scale: true`; `requires_grad_(False)` in `cmappo.py`) | yes |
| 123 | 1.631184 | frozen | yes |
| 2024 | 1.630958 | frozen | yes |
| 3407 | 1.630747 | frozen | yes |
| 9999 | 1.630735 | frozen | yes |

`residual_scale` is an `nn.Parameter`. It was trainable during BC warm-start and
**frozen** for KL-CMAPPO. The five values therefore differ slightly because each
seed's BC pretrain produced a different scale; KL did not update it. At
inference the parameter is loaded and used; no gradient is taken.

A separate `SemanticExpertPolicy` object is **not** required at deployment.

## Comments / README

Searched for "expert is no longer required", "expert removed at inference",
and "pure neural actor". No such statements were found in README or code.
Actor/loader docstrings now state the residual explicitly.
