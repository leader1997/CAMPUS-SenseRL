"""Strong local expert for BC warm-start / residual MAPPO.

Uses only on-sensor observation features (no future leakage).
Transmit when the uplink is likely informative for monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from campus_senserl.environment.communication_model import SKIP, TRANSMIT


@dataclass
class SemanticExpertPolicy:
    """OR of change / AoI / level / local-vs-server disagreement rules."""

    delta_ppm: float = 35.0
    aoi_threshold: float = 3.5
    co2_ppm: float = 1000.0
    disagreement_ppm: float = 20.0
    max_aoi: float = 8.0
    name: str = "semantic_expert"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 14
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        local = obs2[:, 0] * 1500.0
        last_tx = obs2[:, 1] * 1500.0
        delta = np.abs(obs2[:, 2] * 1500.0)
        aoi = obs2[:, 3] * self.max_aoi
        recon = obs2[:, 4] * 1500.0
        disagree = np.abs(local - recon)

        need = (
            (delta >= self.delta_ppm)
            | (aoi >= self.aoi_threshold)
            | (local >= self.co2_ppm)
            | (disagree >= self.disagreement_ppm)
        )
        actions = np.where(need, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class CO2OnlyPolicy:
    """Event-triggered baseline: TX iff local CO2 >= threshold."""

    co2_ppm: float = 1000.0
    name: str = "co2_threshold_only"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 14
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        local = obs2[:, 0] * 1500.0
        actions = np.where(local >= self.co2_ppm, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class SendOnDeltaPolicy:
    """Classic send-on-delta: TX iff |ΔCO2| >= threshold."""

    delta_ppm: float = 50.0
    name: str = "send_on_delta"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 14
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        delta = np.abs(obs2[:, 2] * 1500.0)
        actions = np.where(delta >= self.delta_ppm, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class AoIHeartbeatPolicy:
    """Heartbeat: TX iff server AoI >= threshold."""

    aoi_threshold: float = 4.0
    max_aoi: float = 8.0
    name: str = "aoi_heartbeat"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 14
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        aoi = obs2[:, 3] * self.max_aoi
        actions = np.where(aoi >= self.aoi_threshold, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


@dataclass
class DeltaPlusHeartbeatPolicy:
    """Send-on-delta OR AoI heartbeat (classic ET + freshness)."""

    delta_ppm: float = 50.0
    aoi_threshold: float = 4.0
    max_aoi: float = 8.0
    name: str = "delta_plus_heartbeat"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, *, local_available: np.ndarray | None = None) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 14
        obs2 = obs.reshape(n, -1) if obs.ndim == 1 else obs
        delta = np.abs(obs2[:, 2] * 1500.0)
        aoi = obs2[:, 3] * self.max_aoi
        need = (delta >= self.delta_ppm) | (aoi >= self.aoi_threshold)
        actions = np.where(need, TRANSMIT, SKIP).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


def heuristic_logits_torch(obs: torch.Tensor, *, max_aoi: float = 8.0) -> torch.Tensor:
    """Differentiable soft expert score → logit bias favoring informative TX.

    obs: (..., 14) normalized features matching env schema.
    """
    local = obs[..., 0] * 1500.0
    delta = (obs[..., 2] * 1500.0).abs()
    aoi = obs[..., 3] * max_aoi
    recon = obs[..., 4] * 1500.0
    unc = obs[..., 5]
    disagree = (local - recon).abs()
    avail = obs[..., 12]

    # Soft scores in ~[0, 1]
    s_delta = torch.sigmoid((delta - 35.0) / 15.0)
    s_aoi = torch.sigmoid((aoi - 3.5) / 0.75)
    s_co2 = torch.sigmoid((local - 1000.0) / 80.0)
    s_dis = torch.sigmoid((disagree - 20.0) / 10.0)
    s_unc = torch.sigmoid((unc - 0.3) / 0.2)
    score = 0.30 * s_delta + 0.25 * s_aoi + 0.20 * s_co2 + 0.20 * s_dis + 0.05 * s_unc
    # Map score→logit; unavailable sensors strongly prefer SKIP
    logits = 6.0 * (score - 0.35)
    logits = torch.where(avail > 0.5, logits, torch.full_like(logits, -8.0))
    return logits
