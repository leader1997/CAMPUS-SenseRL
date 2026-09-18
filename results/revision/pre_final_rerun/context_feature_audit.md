# Context-feature / relation-graph audit

The graph itself was not changed.

## What the edges mean

`src/campus_senserl/graph/__init__.py` builds spatial, correlation, and hybrid
adjacencies. Edges are **spatial/statistical contextual relations** among
logical per-sensor decision agents.

They are **not**:

- LoRaWAN or other radio links
- sensor-to-sensor packet exchange
- a wireless mesh
- ventilation / airflow topology

Correlation edges are fit on **training data only**.

## Where the neighbour feature is computed

`TraceDrivenCampusEnv._agent_observation` (observation index 13) calls
`_neighbor_summary(recon)`.

```
neigh_sum[i] = mean( recon[j] for j in neighbors(i) if recon[j] is finite )
```

`recon` is the current server reconstruction / last-delivered monitoring value,
not the hidden current ground truth of a skipped neighbour.

## Causality

- A skipped neighbour contributes only its **server-available** reconstructed
  value (typically LOCF of the last successful delivery).
- Direct current local measurements of other sensors are **not** written into
  this feature.
- `_neighbor_disagreement` was removed from `TraceDrivenCampusEnv`.
  It mixed a neighbour's current local GT when that neighbour was locally
  available. Production observations never called it. The deployed/evaluated
  observation path uses only `_neighbor_summary(recon)`.

## Preferred technical interpretation

The graph defines spatial/statistical contextual relations.
It is not a wireless mesh.
The contextual feature is derived from server-available monitoring states.
Agents are logical per-sensor decision agents; they do not exchange packets
with physical neighbours.
