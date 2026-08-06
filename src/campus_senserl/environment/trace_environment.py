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
from campus_senserl.environment.safety_shield import SafetyShield
from campus_senserl.rl.reward import RewardComputer
from campus_senserl.utils import load_yaml, repo_root


def load_trace_tensors(
    split: str = "train",
    max_sensors: int | None = None,
) -> dict[str, Any]:
    """Load panel trace as numpy arrays for simulation (vectorized)."""
    root = repo_root()
    panel_path = root / "data" / "processed" / "co2_panel_15min.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Run scripts/02_preprocess.py first.")

    cols = [
        "slot",
        "deveui",
        "split",
        "observed",
        "co2",
        "motion",
        "battery",
        "rssi",
        "floor",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    panel = pd.read_parquet(panel_path, columns=[c for c in cols if True])
    panel = panel[panel["split"] == split]
    sensors = sorted(panel["deveui"].unique().tolist())
    if max_sensors is not None:
        sensors = sensors[:max_sensors]
        panel = panel[panel["deveui"].isin(sensors)]
    slots = sorted(panel["slot"].unique())
    panel = panel[panel["slot"].isin(slots)]

    def _pivot(col: str, fill=np.nan) -> np.ndarray:
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

    meta = panel.drop_duplicates("deveui").set_index("deveui").reindex(sensors)
    floors = (
        pd.to_numeric(meta["floor"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    cal = panel.drop_duplicates("slot").set_index("slot").reindex(slots)
    time_feats = np.stack(
        [
            cal["hour_sin"].fillna(0).to_numpy(dtype=np.float32),
            cal["hour_cos"].fillna(0).to_numpy(dtype=np.float32),
            cal["dow_sin"].fillna(0).to_numpy(dtype=np.float32),
            cal["dow_cos"].fillna(0).to_numpy(dtype=np.float32),
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
    ):
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
    aoi: np.ndarray
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
        if trace is not None:
            self.trace = _subset_trace(trace, max_sensors)
        else:
            self.trace = load_trace_tensors(split, max_sensors=max_sensors)
        self.n_sensors = self.trace["n_sensors"]
        self.n_steps = self.trace["n_steps"]
        self.ground_truth = self.trace["ground_truth"]
        self.natural_missing = self.trace["natural_missing"]
        self.local_available = self.trace["local_available"]
        self.motion = self.trace["motion"]
        self.time_feats = self.trace["time_feats"]
        self.floors = self.trace["floors"]

        self.comm = CommunicationCostModel.from_config(cfg)
        self.reward_computer = RewardComputer.from_config(cfg)
        self.event_detector = EventDetector.from_config(cfg)
        self.shield = SafetyShield.from_config(cfg)
        self.max_aoi = float(cfg.get("env", {}).get("max_aoi", 8))
        self.reconstruction_fn = reconstruction_model
        self.locf = LocfReconstructor()
        self.adjacency = adjacency if adjacency is not None else np.eye(self.n_sensors, dtype=np.float32)

        self._t = 0
        self.server_state: ServerState | None = None
        self.rl_skipped = np.zeros((self.n_steps, self.n_sensors), dtype=bool)
        self.transmit_history = np.zeros((self.n_steps, self.n_sensors), dtype=bool)

        obs_dim = 10
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
            rl_skipped=np.zeros(self.n_sensors, dtype=bool),
            last_transmitted=np.full(self.n_sensors, np.nan, dtype=np.float32),
        )

    def _reconstruct(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.server_state is not None
        if self.reconstruction_fn is not None:
            return self.reconstruction_fn(self.server_state, self.time_feats[self._t], self.adjacency)
        return self.locf.predict(self.server_state, self.time_feats[self._t], self.adjacency)

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

    def _agent_observation(self, recon: np.ndarray, unc: np.ndarray) -> np.ndarray:
        assert self.server_state is not None
        t = self._t
        obs = np.zeros((self.n_sensors, self._obs_dim), dtype=np.float32)
        gt = self.ground_truth[t]
        for i in range(self.n_sensors):
            local_val = gt[i] if self.local_available[t, i] else np.nan
            last_tx = self.server_state.last_transmitted[i]
            obs[i, 0] = 0.0 if not np.isfinite(local_val) else local_val / 1500.0
            obs[i, 1] = 0.0 if not np.isfinite(last_tx) else last_tx / 1500.0
            obs[i, 2] = self.server_state.aoi[i] / self.max_aoi
            obs[i, 3] = recon[i] / 1500.0 if np.isfinite(recon[i]) else 0.0
            obs[i, 4] = unc[i] / 5.0 if np.isfinite(unc[i]) else 1.0
            obs[i, 5] = float(self.natural_missing[t, i])
            obs[i, 6] = float(self.local_available[t, i])
            obs[i, 7] = self.time_feats[t, 0]
            obs[i, 8] = self.time_feats[t, 1]
            obs[i, 9] = self.motion[t, i] / 50.0
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        self.server_state = self._init_server_state()
        self.rl_skipped[:] = False
        self.transmit_history[:] = False
        recon, unc = self._reconstruct()
        obs = self._agent_observation(recon, unc)
        if not self.multi_agent:
            obs = obs.reshape(-1)
        info = {"timestep": self._t, "reconstruction": recon, "uncertainty": unc}
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
        local_avail = self.local_available[t]
        local_co2 = np.where(local_avail, gt, np.nan)
        prev_recon, prev_unc = self._reconstruct()

        co2_rate = np.zeros(self.n_sensors, dtype=np.float32)
        if t > 0:
            prev = self.ground_truth[t - 1]
            co2_rate = np.where(
                local_avail & np.isfinite(prev),
                gt - prev,
                0.0,
            )

        neighbor_dis = self._neighbor_disagreement(local_co2, prev_recon)
        final_actions, shield_decisions = self.shield.apply_vector(
            actions,
            aoi=self.server_state.aoi,
            uncertainty=prev_unc,
            local_co2=local_co2,
            co2_rate=co2_rate,
            neighbor_disagreement=neighbor_dis,
            local_available=local_avail,
        )

        tx_count = 0
        for i in range(self.n_sensors):
            if not local_avail[i]:
                continue
            if final_actions[i] == TRANSMIT:
                self.server_state.server_values[i] = gt[i]
                self.server_state.input_mask[i] = True
                self.server_state.last_transmitted[i] = gt[i]
                self.server_state.aoi[i] = 0.0
                self.server_state.rl_skipped[i] = False
                self.transmit_history[t, i] = True
                tx_count += 1
            else:
                self.server_state.rl_skipped[i] = True
                self.rl_skipped[t, i] = True
                self.server_state.aoi[i] = min(self.server_state.aoi[i] + 1.0, self.max_aoi)
                self.transmit_history[t, i] = False

        post_recon, post_unc = self._reconstruct()
        events = self.event_detector.is_event(gt)
        missed_events = events & ~self.transmit_history[t]

        reward, reward_parts = self.reward_computer.compute_step(
            actions=final_actions,
            ground_truth=gt,
            reconstruction=post_recon,
            uncertainty=post_unc,
            aoi=self.server_state.aoi,
            local_available=local_avail,
            events=events,
            missed_event_mask=missed_events,
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
            "shield_overrides": sum(d.overridden for d in shield_decisions),
            "reconstruction": post_recon,
            "uncertainty": post_unc,
            "rl_skipped": self.rl_skipped[t].copy(),
            "natural_missing": self.natural_missing[t].copy(),
            "final_actions": final_actions.copy(),
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
        }
