# RL State Definition (CAMPUS-SenseRL)

**Status:** Correctness-phase schema after finalization audit (2026-08-06).  
**Implementation:** `TraceDrivenCampusEnv._agent_observation` (`obs_dim = 14`).

Primary thresholds for events (from `configs/rl.yaml`):

- `primary_threshold_ppm: 1000`
- `rapid_increase_ppm: 150`

These are **monitoring / event thresholds**, not medical danger limits.

---

## Observation vector \(s_{i,t}\) (per sensor)

| Idx | Feature | Source | Local / server | Current or last-known | Available after SKIP? | Normalization | Missing handling |
|-----|---------|--------|----------------|----------------------|----------------------|---------------|------------------|
| 0 | Local CO₂ | Device uplink payload | Local | Current if `local_available` | Yes if physically observed (edge), else 0 | `/1500` | 0 if unavailable |
| 1 | Last transmitted CO₂ | Server `last_transmitted` | Server | Last successful TX | Yes (last known) | `/1500` | 0 if never TX |
| 2 | Local ΔCO₂ | `y_t - y_{t-1}` when both finite locally | Local | Current delta | Only if both slots locally available | `/1500` | 0 |
| 3 | AoI | Server AoI counter | Server | Current | Yes | `/max_aoi` | starts at max |
| 4 | Reconstruction \(\hat y\) | Reconstructor on **ServerState only** | Server | Current estimate | Yes | `/1500` | 0 if non-finite |
| 5 | Uncertainty \(\sigma\) | Reconstructor | Server | Current | Yes | `/5` | 1.0 if unknown |
| 6 | Motion | Device payload | Local | Current if in panel | If present in panel row | `/50` | 0 |
| 7 | Battery | Device payload | Local | Current if observed | If present (local health) | `(b-2.5)/1.5` | 0 if NaN |
| 8 | Last RSSI | Gateway after delivered TX | Server/link | **Last known only** | Yes (stale allowed) | `(rssi+120)/60` | 0 if never TX |
| 9 | Link AoI | Steps since last delivered TX | Server/link | Current age of link info | Yes | `/max_aoi` | starts at max |
| 10 | Hour sin | Calendar | Global | Current slot | Always | already in [-1,1] | 0 |
| 11 | Hour cos | Calendar | Global | Current slot | Always | already in [-1,1] | 0 |
| 12 | Local available | Trace mask | Meta | Current | Flag | {0,1} | — |
| 13 | Neighbour recon summary | Mean finite neighbour \(\hat y\) via adjacency | Server/graph | Current | Yes | `/1500` | 0 if no neighbours |

---

## Causality rules (non-negotiable)

1. **Skipped ground-truth CO₂ never enters** `ServerState.server_values` / `input_mask`.
2. **RSSI / LSNR** are gateway measurements of a delivered packet. On SKIP, the agent may only use **last known** RSSI/LSNR and **link AoI**, never a fabricated current gateway sample.
3. **Battery** is treated as locally measurable when present in the uplink panel for that slot; if NaN, feature is 0 (unavailable).
4. **Reconstruction** consumes only server-visible history (transmitted values + masks + AoI). Sequential history must not re-insert skipped GT (see `tests/test_leakage.py`).

---

## Action space

- `0 = SKIP`, `1 = TRANSMIT` (binary per sensor).
- Safety shield may override SKIP → TRANSMIT; reward and PDR use **final** actions.

---

## Event / reward coupling

- True event: \(y \ge 1000\) or \(\Delta y \ge 150\) (temporal, per sensor).
- Server event: same rule on \(\tilde y\) (TX → \(y\), else \(\hat y\)).
- Miss penalty: FN = true ∧ ¬server.
- Event bonus: TP = true ∧ server.

---

## Config wiring

```yaml
environment:
  reconstructor: locf   # or stgnn / proposed_reconstructor after fair selection
  graph: hybrid         # or identity / spatial / correlation
```

Until the fair reconstruction benchmark selects a winner, default reconstructor remains **LOCF** for honesty; graph default is **hybrid** when artifacts exist.
