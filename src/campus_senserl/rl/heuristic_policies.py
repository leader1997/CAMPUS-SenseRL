"""Heuristic threshold policies for transmission scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.rl.info_value import InfoValueWeights, information_value


def _n_agents(obs: np.ndarray) -> int:
    return obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 10


@dataclass
class ChangeThresholdPolicy:
    delta_ppm: float = 50.0
    name: str = "change_threshold"

    def reset(self) -> None:
        self._last_local = None

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = _n_agents(obs)
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        local = obs2[:, 0] * 1500.0
        actions = np.full(n, SKIP, dtype=int)
        if self._last_local is None:
            actions[:] = TRANSMIT
        else:
            delta = np.abs(local - self._last_local)
            actions[delta >= self.delta_ppm] = TRANSMIT
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        self._last_local = local.copy()
        return actions


@dataclass
class UncertaintyThresholdPolicy:
    threshold: float = 1.5
    name: str = "uncertainty_threshold"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = _n_agents(obs)
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        unc = obs2[:, 4] * 5.0
        actions = np.where(unc >= self.threshold, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class AoIThresholdPolicy:
    threshold: float = 4.0
    name: str = "aoi_threshold"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = _n_agents(obs)
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        aoi = obs2[:, 2] * 8.0
        actions = np.where(aoi >= self.threshold, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class CO2ThresholdPolicy:
    threshold_ppm: float = 1000.0
    name: str = "co2_threshold"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = _n_agents(obs)
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        local = obs2[:, 0] * 1500.0
        actions = np.where(local >= self.threshold_ppm, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class InfoValuePolicy:
    threshold: float = 0.45
    weights: InfoValueWeights | None = None
    name: str = "info_value"

    def reset(self) -> None:
        self._last_local = None

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = _n_agents(obs)
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        actions = np.full(n, SKIP, dtype=int)
        for i in range(n):
            local = obs2[i, 0] * 1500.0
            baseline = obs2[i, 1] * 1500.0
            rate = 0.0 if self._last_local is None else local - self._last_local[i]
            iv = information_value(
                local_value=local,
                baseline=baseline,
                rate_of_change=rate,
                motion=obs2[i, 9] * 50.0,
                uncertainty=obs2[i, 4] * 5.0,
                aoi=obs2[i, 2] * 8.0,
                weights=self.weights,
            )["total"]
            if iv >= self.threshold:
                actions[i] = TRANSMIT
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        self._last_local = obs2[:, 0] * 1500.0
        return actions


def make_heuristic_policies(cfg: dict | None = None) -> dict[str, object]:
    return {
        "change_threshold": ChangeThresholdPolicy(),
        "uncertainty_threshold": UncertaintyThresholdPolicy(),
        "aoi_threshold": AoIThresholdPolicy(),
        "co2_threshold": CO2ThresholdPolicy(),
        "info_value": InfoValuePolicy(),
    }
