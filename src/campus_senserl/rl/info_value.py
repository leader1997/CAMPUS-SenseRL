"""Semantic information-value estimation for transmission scheduling.

All components are normalized. Weights are configurable and must be tuned
on validation data only (never test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InfoValueWeights:
    absolute: float = 0.15
    deviation: float = 0.20
    rate: float = 0.20
    motion: float = 0.10
    uncertainty: float = 0.15
    neighbor_disagreement: float = 0.10
    aoi: float = 0.10
    network_anomaly: float = 0.00

    def as_dict(self) -> dict[str, float]:
        return {
            "absolute": self.absolute,
            "deviation": self.deviation,
            "rate": self.rate,
            "motion": self.motion,
            "uncertainty": self.uncertainty,
            "neighbor_disagreement": self.neighbor_disagreement,
            "aoi": self.aoi,
            "network_anomaly": self.network_anomaly,
        }


def _norm(x: float, scale: float) -> float:
    if not np.isfinite(x) or scale <= 0:
        return 0.0
    return float(np.clip(x / scale, 0.0, 1.0))


def information_value(
    *,
    local_value: float,
    baseline: float,
    rate_of_change: float,
    motion: float = 0.0,
    uncertainty: float = 0.0,
    neighbor_disagreement: float = 0.0,
    aoi: float = 0.0,
    rssi: float | None = None,
    rssi_baseline: float | None = None,
    weights: InfoValueWeights | None = None,
    scales: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute semantic importance in [0, 1] approximately.

    Returns components and total for logging/ablations.
    """
    w = weights or InfoValueWeights()
    scales = scales or {
        "absolute": 1500.0,
        "deviation": 400.0,
        "rate": 200.0,
        "motion": 50.0,
        "uncertainty": 200.0,
        "neighbor_disagreement": 200.0,
        "aoi": 8.0,
        "network_anomaly": 20.0,
    }
    comps = {
        "absolute": _norm(local_value, scales["absolute"]),
        "deviation": _norm(abs(local_value - baseline), scales["deviation"]),
        "rate": _norm(abs(rate_of_change), scales["rate"]),
        "motion": _norm(motion, scales["motion"]),
        "uncertainty": _norm(uncertainty, scales["uncertainty"]),
        "neighbor_disagreement": _norm(neighbor_disagreement, scales["neighbor_disagreement"]),
        "aoi": _norm(aoi, scales["aoi"]),
        "network_anomaly": 0.0,
    }
    if rssi is not None and rssi_baseline is not None and np.isfinite(rssi):
        comps["network_anomaly"] = _norm(abs(rssi - rssi_baseline), scales["network_anomaly"])

    total = (
        w.absolute * comps["absolute"]
        + w.deviation * comps["deviation"]
        + w.rate * comps["rate"]
        + w.motion * comps["motion"]
        + w.uncertainty * comps["uncertainty"]
        + w.neighbor_disagreement * comps["neighbor_disagreement"]
        + w.aoi * comps["aoi"]
        + w.network_anomaly * comps["network_anomaly"]
    )
    return {"total": float(total), "components": comps, "weights": w.as_dict()}


def ablation_weight_sets() -> dict[str, InfoValueWeights]:
    """Weight configurations for component ablations."""
    base = InfoValueWeights()
    out = {"full": base}
    for name in base.as_dict():
        d = base.as_dict()
        d[name] = 0.0
        out[f"no_{name}"] = InfoValueWeights(**d)
    return out
