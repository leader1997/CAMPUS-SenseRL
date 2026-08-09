"""Trace-driven counterfactual Gymnasium environment for communication scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from campus_senserl.environment.communication_model import SKIP, TRANSMIT, CommunicationCostModel
from campus_senserl.environment.event_detector import EventDetector
from campus_senserl.rl.reward import RewardComputer
from campus_senserl.utils import load_yaml, repo_root


def load_trace_tensors(
    split: str = "train",
    max_sensors: int | None = None,
    sensor_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load panel trace as numpy arrays for simulation (vectorized).

    If ``sensor_ids`` is provided, use that exact cohort order (preferred).
    Else fall back to sorted unique + optional ``max_sensors`` prefix.
    """
    root = repo_root()
    panel_path = root / "data" / "processed" / "co2_panel_15min.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Run scripts/02_preprocess.py first.")

    want_cols = [
        "slot",
        "deveui",
        "split",
        "observed",
        "co2",
        "motion",
        "battery",
        "rssi",
        "lsnr",
        "floor",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    # Read available columns only (lsnr may be absent in older panels).
    import pyarrow.parquet as pq

    available = set(pq.read_schema(panel_path).names)
    cols = [c for c in want_cols if c in available]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel = panel[panel["split"] == split]
    if sensor_ids is not None:
        sensors = [str(s) for s in sensor_ids]
        panel = panel[panel["deveui"].astype(str).isin(sensors)]
    else:
        sensors = sorted(panel["deveui"].unique().tolist())
        if max_sensors is not None:
            sensors = sensors[:max_sensors]
            panel = panel[panel["deveui"].isin(sensors)]
    slots = sorted(panel["slot"].unique())
    panel = panel[panel["slot"].isin(slots)]

    def _pivot(col: str) -> np.ndarray:
        if col not in panel.columns:
            return np.full((len(slots), len(sensors)), np.nan, dtype=np.float32)
        wide = panel.pivot_table(index="slot", columns="deveui", values=col, aggfunc="last")
        wide = wide.reindex(index=slots, columns=sensors)
        return wide.to_numpy(dtype=np.float32)

    ground_truth = _pivot("co2")
    observed = _pivot("observed")
    observed = np.nan_to_num(observed, nan=0.0) > 0.5
    natural_missing = ~observed
    motion = np.nan_to_num(_pivot("motion"), nan=0.0)
    battery = _pivot("battery")
    rssi = _pivot("rssi")
    lsnr = _pivot("lsnr")

    meta = panel.drop_duplicates("deveui").set_index("deveui").reindex(sensors)
    floors = (
        pd.to_numeric(meta["floor"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    cal = panel.drop_duplicates("slot").set_index("slot").reindex(slots)
    time_feats = np.stack(
        [
            cal["hour_sin"].fillna(0).to_numpy(dtype=np.float32) if "hour_sin" in cal else np.zeros(len(slots), dtype=np.float32),
            cal["hour_cos"].fillna(0).to_numpy(dtype=np.float32) if "hour_cos" in cal else np.zeros(len(slots), dtype=np.float32),
            cal["dow_sin"].fillna(0).to_numpy(dtype=np.float32) if "dow_sin" in cal else np.zeros(len(slots), dtype=np.float32),
            cal["dow_cos"].fillna(0).to_numpy(dtype=np.float32) if "dow_cos" in cal else np.zeros(len(slots), dtype=np.float32),
        ],
        axis=-1,
    )
    # Only treat as locally available when observation existed and CO2 is finite
    ground_truth = np.where(observed, ground_truth, np.nan)
    local_available = observed & np.isfinite(ground_truth)
    return _pack_trace(
        ground_truth.astype(np.float32),
        natural_missing,
        local_available,
        motion.astype(np.float32),
        battery.astype(np.float32),
        rssi.astype(np.float32),
        lsnr.astype(np.float32),
        floors,
        time_feats,
        sensors,
        slots,
    )


def _pack_trace(
    ground_truth: np.ndarray,
    natural_missing: np.ndarray,
    local_available: np.ndarray,
    motion: np.ndarray,
    battery: np.ndarray,
    rssi: np.ndarray,
    lsnr: np.ndarray,
    floors: np.ndarray,
    time_feats: np.ndarray,
    sensors: list[str],
    slots: list[Any],
) -> dict[str, Any]:
    n_t, n_s = ground_truth.shape
    return {
        "ground_truth": ground_truth,
        "natural_missing": natural_missing,
        "local_available": local_available,
        "motion": motion,
        "battery": battery,
        "rssi": rssi,
        "lsnr": lsnr,
        "floors": floors,
        "time_feats": time_feats,
        "sensors": sensors,
        "slots": slots,
        "n_steps": n_t,
        "n_sensors": n_s,
    }


def _subset_trace(trace: dict[str, Any], max_sensors: int | None) -> dict[str, Any]:
    if max_sensors is None or trace["n_sensors"] <= max_sensors:
        return trace
    n = int(max_sensors)
    out = dict(trace)
    for key in (
        "ground_truth",
        "natural_missing",
        "local_available",
        "motion",
        "battery",
        "rssi",
        "lsnr",
    ):
        if key in trace:
            out[key] = trace[key][:, :n]
    out["floors"] = trace["floors"][:n]
    out["sensors"] = trace["sensors"][:n]
    out["n_sensors"] = n
    return out


def build_synthetic_trace(
    *,
    n_steps: int = 24,
    n_sensors: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a tiny in-memory trace panel for tests and debugging."""
    rng = np.random.default_rng(seed)
    sensors = [f"S{i:02d}" for i in range(n_sensors)]
    slots = pd.date_range("2024-01-01", periods=n_steps, freq="15min", tz="UTC")
    ground_truth = rng.uniform(450.0, 900.0, size=(n_steps, n_sensors)).astype(np.float32)
    natural_missing = rng.random((n_steps, n_sensors)) < 0.08
    natural_missing[:, 0] = False  # keep at least one always-available sensor
    local_available = ~natural_missing & np.isfinite(ground_truth)
    motion = rng.uniform(0.0, 20.0, size=(n_steps, n_sensors)).astype(np.float32)
    battery = rng.uniform(2.8, 3.6, size=(n_steps, n_sensors)).astype(np.float32)
    rssi = rng.uniform(-110.0, -70.0, size=(n_steps, n_sensors)).astype(np.float32)
    lsnr = rng.uniform(-5.0, 10.0, size=(n_steps, n_sensors)).astype(np.float32)
    floors = np.arange(n_sensors, dtype=int) % 3
    time_feats = np.zeros((n_steps, 4), dtype=np.float32)
    for t, slot in enumerate(slots):
        hour = slot.hour + slot.minute / 60.0
        dow = float(slot.dayofweek)
        time_feats[t] = np.array(
            [
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * dow / 7.0),
                np.cos(2 * np.pi * dow / 7.0),
            ],
            dtype=np.float32,
        )
    return _pack_trace(
        ground_truth,
        natural_missing,
        local_available,
        motion,
        battery,
        rssi,
        lsnr,
        floors,
        time_feats,
        sensors,
        list(slots),
    )


@dataclass
class ServerState:
    """Server-visible state — never includes RL-skipped ground truth."""

    server_values: np.ndarray
    input_mask: np.ndarray
    aoi: np.ndarray  # clipped for RL state / shield features
    aoi_raw: np.ndarray  # unbounded physical AoI for reporting
    rl_skipped: np.ndarray
    last_transmitted: np.ndarray


class LocfReconstructor:
    """Last observation carried forward on server-visible stream only."""

    def predict(
        self,
        server_state: ServerState,
        time_feats: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_s = server_state.server_values.shape[0]
        mean = np.full(n_s, np.nan, dtype=np.float32)
        unc = np.full(n_s, 1.0, dtype=np.float32)
        for i in range(n_s):
            if server_state.input_mask[i] and np.isfinite(server_state.server_values[i]):
                mean[i] = server_state.server_values[i]
                unc[i] = 0.25
            elif np.isfinite(server_state.last_transmitted[i]):
                mean[i] = server_state.last_transmitted[i]
                unc[i] = 1.0 + server_state.aoi[i] * 0.1
            else:
                mean[i] = 400.0
                unc[i] = 5.0
        return mean, unc


class TraceDrivenCampusEnv(gym.Env):
    """Multi-agent-ready trace environment with strict information separation.

    Agents may observe local measurements when physically available at the edge.
    The server / reconstruction module only receives TRANSMIT actions.
    Skipped ground truth is never injected into server-side observations.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        split: str = "train",
        max_sensors: int | None = None,
        sensor_ids: list[str] | None = None,
        trace: dict[str, Any] | None = None,
        reconstruction_model: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        adjacency: np.ndarray | None = None,
        multi_agent: bool = True,
    ) -> None:
        super().__init__()
        root = repo_root()
        cfg = cfg or load_yaml(root / "configs" / "rl.yaml")
        self.cfg = cfg
        self.split = split
        self.multi_agent = multi_agent

        env_cfg = cfg.get("environment", {})
        if sensor_ids is None:
            cohort_name = env_cfg.get("cohort")
            if cohort_name:
                try:
                    from campus_senserl.data.cohort import load_cohort

                    sensor_ids = load_cohort(str(cohort_name))
                    if max_sensors is not None:
                        sensor_ids = sensor_ids[: int(max_sensors)]
                except Exception as exc:
                    print(f"[env] cohort load failed ({exc}); falling back to max_sensors slice")

        if trace is not None:
            self.trace = _subset_trace(trace, max_sensors)
        else:
            self.trace = load_trace_tensors(split, max_sensors=max_sensors, sensor_ids=sensor_ids)
        self.n_sensors = self.trace["n_sensors"]
        self.n_steps = self.trace["n_steps"]
        self.ground_truth = self.trace["ground_truth"]
        self.natural_missing = self.trace["natural_missing"]
        self.local_available = self.trace["local_available"]
        self.motion = self.trace["motion"]
        self.battery = self.trace.get("battery", np.full_like(self.ground_truth, np.nan))
        self.rssi = self.trace.get("rssi", np.full_like(self.ground_truth, np.nan))
        self.lsnr = self.trace.get("lsnr", np.full_like(self.ground_truth, np.nan))
        self.time_feats = self.trace["time_feats"]
        self.floors = self.trace["floors"]
        self.sensor_ids = [str(s) for s in self.trace.get("sensors", list(range(self.n_sensors)))]

        self.comm = CommunicationCostModel.from_config(cfg)
        self.reward_computer = RewardComputer.from_config(cfg)
        self.event_detector = EventDetector.from_config(cfg)
        self.max_aoi = float(cfg.get("env", {}).get("max_aoi", 8))

        env_cfg = cfg.get("environment", {})
        self.reconstructor_name = str(env_cfg.get("reconstructor", "locf")).lower()
        self.graph_type = str(env_cfg.get("graph", "identity")).lower()

        if reconstruction_model is not None:
            self.reconstruction_fn = reconstruction_model
        else:
            from campus_senserl.environment.reconstructors import build_reconstructor

            self.reconstruction_fn = build_reconstructor(
                self.reconstructor_name,
                cfg=cfg,
                n_sensors=self.n_sensors,
                sensor_ids=self.sensor_ids,
            )
        self.locf = LocfReconstructor()

        if adjacency is not None:
            self.adjacency = np.asarray(adjacency, dtype=np.float32)
        elif self.graph_type not in {"identity", "none", "eye", ""}:
            from campus_senserl.environment.graph_utils import (
                assert_not_identity,
                load_adjacency_for_sensors,
            )

            # Final experiments: never silently fall back to identity.
            fail_fast = bool(env_cfg.get("graph_fail_fast", True))
            try:
                self.adjacency = load_adjacency_for_sensors(self.graph_type, self.sensor_ids)
                assert_not_identity(self.adjacency)
            except Exception as exc:
                if fail_fast:
                    raise RuntimeError(
                        f"Required graph '{self.graph_type}' failed to load for "
                        f"{len(self.sensor_ids)} sensors. Pass exact cohort sensor_ids. "
                        f"Original error: {exc}"
                    ) from exc
                print(f"[env] graph '{self.graph_type}' load failed ({exc}); using identity")
                self.adjacency = np.eye(self.n_sensors, dtype=np.float32)
        else:
            self.adjacency = np.eye(self.n_sensors, dtype=np.float32)

        self._t = 0
        self.server_state: ServerState | None = None
        self.rl_skipped = np.zeros((self.n_steps, self.n_sensors), dtype=bool)
        self.transmit_history = np.zeros((self.n_steps, self.n_sensors), dtype=bool)
        # Last-known link quality (updated only on successful TX)
        self.last_rssi = np.full(self.n_sensors, np.nan, dtype=np.float32)
        self.last_lsnr = np.full(self.n_sensors, np.nan, dtype=np.float32)
        self.link_aoi = np.full(self.n_sensors, self.max_aoi, dtype=np.float32)
        self._prev_server_monitor = np.full(self.n_sensors, np.nan, dtype=np.float32)
        # Post-shield wireless packet loss (communication failure, not measurement failure)
        self.packet_loss_rate = float(env_cfg.get("packet_loss_rate", 0.0))
        stress_seed = int(env_cfg.get("stress_seed", cfg.get("seed", 42)))
        self._loss_rng = np.random.default_rng(stress_seed)

        # Robustness stress: permanent / temporary sensor outages + neighbour (edge) loss
        outage_frac = float(env_cfg.get("sensor_outage_fraction", 0.0))
        self.forced_outage = np.zeros(self.n_sensors, dtype=bool)
        if outage_frac > 0.0:
            n_out = max(1, int(round(outage_frac * self.n_sensors)))
            idx = self._loss_rng.choice(self.n_sensors, size=min(n_out, self.n_sensors), replace=False)
            self.forced_outage[idx] = True

        self.temp_outage_frac = float(env_cfg.get("temp_outage_fraction", 0.0))
        self.temp_outage_duration_frac = float(env_cfg.get("temp_outage_duration_frac", 0.1))
        self.temp_outage_mask = np.zeros(self.n_sensors, dtype=bool)
        self.temp_outage_start = 0
        self.temp_outage_end = 0
        if self.temp_outage_frac > 0.0:
            n_tmp = max(1, int(round(self.temp_outage_frac * self.n_sensors)))
            idx = self._loss_rng.choice(self.n_sensors, size=min(n_tmp, self.n_sensors), replace=False)
            self.temp_outage_mask[idx] = True
            dur = max(1, int(round(self.temp_outage_duration_frac * self.n_steps)))
            start = int(self._loss_rng.integers(0, max(1, self.n_steps - dur)))
            self.temp_outage_start = start
            self.temp_outage_end = start + dur

        edge_drop = float(env_cfg.get("edge_drop_fraction", 0.0))
        if edge_drop > 0.0 and self.adjacency is not None:
            adj = np.array(self.adjacency, dtype=np.float32, copy=True)
            # Drop random off-diagonal edges (neighbour loss); keep self-loops
            off = ~np.eye(self.n_sensors, dtype=bool)
            candidates = np.argwhere(off & (adj > 0))
            if len(candidates):
                n_drop = int(round(edge_drop * len(candidates)))
                if n_drop > 0:
                    pick = self._loss_rng.choice(len(candidates), size=min(n_drop, len(candidates)), replace=False)
                    for r, c in candidates[pick]:
                        adj[r, c] = 0.0
            self.adjacency = adj

        # Observation schema — see reports/rl_state_definition.md
        obs_dim = 14
        if multi_agent:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_sensors, obs_dim), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([2] * self.n_sensors)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_sensors * obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.MultiBinary(self.n_sensors)

        self._obs_dim = obs_dim

    def _init_server_state(self) -> ServerState:
        return ServerState(
            server_values=np.full(self.n_sensors, np.nan, dtype=np.float32),
            input_mask=np.zeros(self.n_sensors, dtype=bool),
            aoi=np.full(self.n_sensors, self.max_aoi, dtype=np.float32),
            aoi_raw=np.full(self.n_sensors, self.max_aoi, dtype=np.float32),
            rl_skipped=np.zeros(self.n_sensors, dtype=bool),
            last_transmitted=np.full(self.n_sensors, np.nan, dtype=np.float32),
        )

    def _reconstruct(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.server_state is not None
        if self.reconstruction_fn is not None:
            return self.reconstruction_fn(self.server_state, self.time_feats[self._t], self.adjacency)
        return self.locf.predict(self.server_state, self.time_feats[self._t], self.adjacency)

    def _neighbor_summary(self, recon: np.ndarray) -> np.ndarray:
        """Mean reconstructed neighbour CO2 (graph-aware); 0 if identity/no neighbours."""
        out = np.zeros(self.n_sensors, dtype=np.float32)
        for i in range(self.n_sensors):
            neigh = np.where(self.adjacency[i] > 0)[0]
            neigh = neigh[neigh != i]
            if len(neigh) == 0:
                continue
            vals = recon[neigh]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                out[i] = float(np.mean(vals))
        return out

    def _neighbor_disagreement(self, local_co2: np.ndarray, recon: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_sensors, dtype=np.float32)
        for i in range(self.n_sensors):
            neigh = np.where(self.adjacency[i] > 0)[0]
            neigh = neigh[neigh != i]
            if len(neigh) == 0:
                continue
            vals = []
            for j in neigh:
                if np.isfinite(local_co2[j]):
                    vals.append(local_co2[j])
                elif np.isfinite(recon[j]):
                    vals.append(recon[j])
            if vals and np.isfinite(local_co2[i]):
                out[i] = abs(local_co2[i] - float(np.mean(vals)))
        return out

    def _effective_local_available(self, t: int | None = None) -> np.ndarray:
        """Local availability after applying robustness outage masks."""
        t = self._t if t is None else int(t)
        avail = np.asarray(self.local_available[t], dtype=bool).copy()
        if getattr(self, "forced_outage", None) is not None and self.forced_outage.any():
            avail[self.forced_outage] = False
        if (
            getattr(self, "temp_outage_mask", None) is not None
            and self.temp_outage_mask.any()
            and self.temp_outage_start <= t < self.temp_outage_end
        ):
            avail[self.temp_outage_mask] = False
        return avail

    def _agent_observation(self, recon: np.ndarray, unc: np.ndarray) -> np.ndarray:
        assert self.server_state is not None
        t = self._t
        obs = np.zeros((self.n_sensors, self._obs_dim), dtype=np.float32)
        gt = self.ground_truth[t]
        prev_gt = self.ground_truth[t - 1] if t > 0 else np.full(self.n_sensors, np.nan)
        neigh_sum = self._neighbor_summary(recon)
        local_avail = self._effective_local_available(t)
        for i in range(self.n_sensors):
            local_val = gt[i] if local_avail[i] else np.nan
            last_tx = self.server_state.last_transmitted[i]
            local_delta = (
                local_val - prev_gt[i]
                if np.isfinite(local_val) and np.isfinite(prev_gt[i])
                else np.nan
            )
            batt = self.battery[t, i]
            # 0 local_co2, 1 last_tx, 2 local_delta, 3 aoi, 4 recon, 5 unc,
            # 6 motion, 7 battery, 8 last_rssi, 9 link_aoi, 10 hour_sin, 11 hour_cos,
            # 12 local_available, 13 neighbor_summary
            obs[i, 0] = 0.0 if not np.isfinite(local_val) else local_val / 1500.0
            obs[i, 1] = 0.0 if not np.isfinite(last_tx) else last_tx / 1500.0
            obs[i, 2] = 0.0 if not np.isfinite(local_delta) else local_delta / 1500.0
            obs[i, 3] = self.server_state.aoi[i] / self.max_aoi
            obs[i, 4] = recon[i] / 1500.0 if np.isfinite(recon[i]) else 0.0
            obs[i, 5] = unc[i] / 5.0 if np.isfinite(unc[i]) else 1.0
            obs[i, 6] = self.motion[t, i] / 50.0
            obs[i, 7] = 0.0 if not np.isfinite(batt) else (batt - 2.5) / 1.5
            obs[i, 8] = (
                0.0
                if not np.isfinite(self.last_rssi[i])
                else (self.last_rssi[i] + 120.0) / 60.0
            )
            obs[i, 9] = self.link_aoi[i] / self.max_aoi
            obs[i, 10] = self.time_feats[t, 0]
            obs[i, 11] = self.time_feats[t, 1]
            obs[i, 12] = float(local_avail[i])
            obs[i, 13] = neigh_sum[i] / 1500.0 if np.isfinite(neigh_sum[i]) else 0.0
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        self.server_state = self._init_server_state()
        self.rl_skipped[:] = False
        self.transmit_history[:] = False
        self.last_rssi[:] = np.nan
        self.last_lsnr[:] = np.nan
        self.link_aoi[:] = self.max_aoi
        self._prev_server_monitor[:] = np.nan
        recon, unc = self._reconstruct()
        obs = self._agent_observation(recon, unc)
        if not self.multi_agent:
            obs = obs.reshape(-1)
        info = {
            "timestep": self._t,
            "reconstruction": recon,
            "uncertainty": unc,
            "reconstructor": self.reconstructor_name,
            "graph": self.graph_type,
        }
        return obs, info

    def step(self, action: np.ndarray | int):
        if self._t >= self.n_steps - 1:
            obs, info = self.reset()
            return obs, 0.0, True, False, info

        assert self.server_state is not None
        t = self._t
        actions = np.asarray(action, dtype=int).reshape(-1)
        if actions.size == 1 and self.n_sensors > 1:
            actions = np.full(self.n_sensors, actions[0], dtype=int)

        gt = self.ground_truth[t]
        local_avail = self._effective_local_available(t)
        prev_gt = self.ground_truth[t - 1] if t > 0 else np.full(self.n_sensors, np.nan, dtype=np.float32)

        # Policy actions are applied directly (no safety-shield override).
        final_actions = np.asarray(actions, dtype=int).reshape(-1)

        # Age link info every step; refresh only on successful TX
        self.link_aoi = np.minimum(self.link_aoi + 1.0, self.max_aoi)

        tx_count = 0
        tx_requested = 0
        tx_dropped = 0
        for i in range(self.n_sensors):
            if not local_avail[i]:
                continue
            if final_actions[i] == TRANSMIT:
                tx_requested += 1
                # Packet loss: sensor measured and requested TX, but server never receives it.
                # Local availability stays True; only delivery fails.
                if self.packet_loss_rate > 0.0 and self._loss_rng.random() < self.packet_loss_rate:
                    tx_dropped += 1
                    self.server_state.rl_skipped[i] = True
                    self.rl_skipped[t, i] = True
                    self.server_state.aoi_raw[i] = self.server_state.aoi_raw[i] + 1.0
                    self.server_state.aoi[i] = min(self.server_state.aoi_raw[i], self.max_aoi)
                    self.transmit_history[t, i] = False
                    self.server_state.input_mask[i] = False
                    self.server_state.server_values[i] = np.nan
                    continue
                self.server_state.server_values[i] = gt[i]
                self.server_state.input_mask[i] = True
                self.server_state.last_transmitted[i] = gt[i]
                self.server_state.aoi_raw[i] = 0.0
                self.server_state.aoi[i] = 0.0
                self.server_state.rl_skipped[i] = False
                self.transmit_history[t, i] = True
                # Causal link features: only known after a delivered uplink
                if np.isfinite(self.rssi[t, i]):
                    self.last_rssi[i] = self.rssi[t, i]
                if np.isfinite(self.lsnr[t, i]):
                    self.last_lsnr[i] = self.lsnr[t, i]
                self.link_aoi[i] = 0.0
                tx_count += 1
            else:
                self.server_state.rl_skipped[i] = True
                self.rl_skipped[t, i] = True
                self.server_state.aoi_raw[i] = self.server_state.aoi_raw[i] + 1.0
                self.server_state.aoi[i] = min(self.server_state.aoi_raw[i], self.max_aoi)
                self.transmit_history[t, i] = False
                # Clear current input slot so skipped GT cannot remain masked-in
                self.server_state.input_mask[i] = False
                self.server_state.server_values[i] = np.nan

        post_recon, post_unc = self._reconstruct()

        # Server monitoring value: GT if transmitted, else reconstruction
        server_y = np.where(
            self.transmit_history[t] & local_avail,
            gt,
            post_recon,
        ).astype(np.float32)
        server_y = np.where(local_avail, server_y, np.nan)

        det = self.event_detector.classify_detection(
            true_values=gt,
            server_values=server_y,
            prev_true=prev_gt if t > 0 else None,
            prev_server=self._prev_server_monitor if t > 0 else None,
            valid=local_avail,
        )
        true_events = det["true"]
        missed_events = det["fn"]
        detected_events = det["tp"]  # TP only — for reward credit
        server_events = det["server"]
        false_positive_events = det["fp"]
        self._prev_server_monitor = server_y.copy()

        reward, reward_parts = self.reward_computer.compute_step(
            actions=final_actions,
            ground_truth=gt,
            reconstruction=post_recon,
            uncertainty=post_unc,
            aoi=self.server_state.aoi,
            local_available=local_avail,
            events=true_events,
            missed_event_mask=missed_events,
            detected_event_mask=detected_events,
            comm_model=self.comm,
        )

        self._t += 1
        recon, unc = self._reconstruct()
        obs = self._agent_observation(recon, unc)

        terminated = self._t >= self.n_steps - 1
        if not self.multi_agent:
            obs = obs.reshape(-1)

        info = {
            "timestep": self._t,
            "reward_parts": reward_parts,
            "transmit_count": tx_count,
            "tx_requested": tx_requested,
            "tx_dropped": tx_dropped,
            "reconstruction": post_recon,
            "uncertainty": post_unc,
            "server_monitor": server_y.copy(),
            "true_events": true_events.copy(),
            "server_events": server_events.copy(),
            "detected_events": detected_events.copy(),  # TP
            "missed_events": missed_events.copy(),
            "false_positive_events": false_positive_events.copy(),
            "true_high_co2": det["true_high_co2"].copy(),
            "true_rapid_rise": det["true_rapid_rise"].copy(),
            "tp_high_co2": det["tp_high_co2"].copy(),
            "fn_high_co2": det["fn_high_co2"].copy(),
            "tp_rapid_rise": det["tp_rapid_rise"].copy(),
            "fn_rapid_rise": det["fn_rapid_rise"].copy(),
            "rl_skipped": self.rl_skipped[t].copy(),
            "natural_missing": self.natural_missing[t].copy(),
            "final_actions": final_actions.copy(),
            "local_available": local_avail.copy(),
            "aoi_state": self.server_state.aoi.copy(),
            "aoi_raw": self.server_state.aoi_raw.copy(),
        }
        return obs, float(reward), terminated, False, info

    def get_episode_metrics(self) -> dict[str, Any]:
        # Only count steps actually executed in this episode
        t_end = max(int(self._t), 1)
        tx_hist = self.transmit_history[:t_end]
        skip_hist = self.rl_skipped[:t_end]
        natural = self.natural_missing[:t_end]
        avail = self.local_available[:t_end]
        tx_rate = float(tx_hist[avail].mean()) if avail.any() else 0.0
        rl_skip_rate = float(skip_hist[avail].mean()) if avail.any() else 0.0
        return {
            "transmit_rate": tx_rate,
            "rl_skip_rate": rl_skip_rate,
            "natural_missing_rate": float(natural.mean()),
            "n_steps_executed": t_end,
            "n_steps": self.n_steps,
            "n_sensors": self.n_sensors,
            "mean_aoi": float(self.server_state.aoi.mean()) if self.server_state is not None else float("nan"),
            "mean_aoi_raw": float(self.server_state.aoi_raw.mean()) if self.server_state is not None else float("nan"),
            "reconstructor": self.reconstructor_name,
            "graph": self.graph_type,
        }
