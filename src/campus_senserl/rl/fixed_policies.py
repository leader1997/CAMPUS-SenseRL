"""Fixed-interval transmission baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT


@dataclass
class FixedIntervalPolicy:
    """Transmit every k intervals (1 => every step)."""

    interval_steps: int = 1
    name: str = "fixed_15"

    def reset(self) -> None:
        self._counter = 0

    def act(
        self,
        obs: np.ndarray,
        *,
        local_available: np.ndarray | None = None,
    ) -> np.ndarray:
        n = obs.shape[0] if obs.ndim == 2 else obs.shape[0] // 10
        if not hasattr(self, "_counter"):
            self.reset()
        actions = np.full(n, SKIP, dtype=int)
        if self._counter % max(self.interval_steps, 1) == 0:
            actions[:] = TRANSMIT
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        self._counter += 1
        return actions

    @property
    def interval_minutes(self) -> int:
        return int(self.interval_steps * 15)


def make_fixed_policies(intervals_min: list[int] | None = None) -> dict[str, FixedIntervalPolicy]:
    intervals_min = intervals_min or [15, 30, 45, 60]
    out: dict[str, FixedIntervalPolicy] = {}
    for m in intervals_min:
        steps = max(1, m // 15)
        name = f"fixed_{m}min"
        out[name] = FixedIntervalPolicy(interval_steps=steps, name=name)
    return out
