"""Configurable wrapper around evaluation event detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from campus_senserl.evaluation.events import detect_events_series


@dataclass
class EventDetector:
    """Detect important CO2 events with configurable thresholds."""

    co2_thresholds_ppm: list[float] = field(default_factory=lambda: [800, 1000, 1200, 1500])
    primary_threshold_ppm: float = 1000.0
    rapid_increase_ppm: float = 150.0
    persistent_high_intervals: int = 3

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EventDetector":
        events = cfg.get("events", cfg)
        return cls(
            co2_thresholds_ppm=list(events.get("co2_thresholds_ppm", [800, 1000, 1200, 1500])),
            primary_threshold_ppm=float(events.get("primary_threshold_ppm", 1000)),
            rapid_increase_ppm=float(events.get("rapid_increase_ppm", 150)),
            persistent_high_intervals=int(events.get("persistent_high_intervals", 3)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "co2_thresholds_ppm": self.co2_thresholds_ppm,
            "primary_threshold_ppm": self.primary_threshold_ppm,
            "rapid_increase_ppm": self.rapid_increase_ppm,
            "persistent_high_intervals": self.persistent_high_intervals,
        }

    def detect(
        self,
        values: np.ndarray,
        baseline: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        return detect_events_series(values, self.as_dict(), baseline=baseline)

    def is_event(self, values: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
        return self.detect(values, baseline=baseline)["any"]

    def threshold_masks(self, values: np.ndarray) -> dict[str, np.ndarray]:
        v = np.asarray(values, dtype=float)
        out: dict[str, np.ndarray] = {}
        for thr in self.co2_thresholds_ppm:
            out[f"high_{int(thr)}"] = (v >= thr) & np.isfinite(v)
        return out
