"""Configurable wrapper around evaluation event detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from campus_senserl.evaluation.events import detect_events_series


@dataclass
class EventDetector:
    """Detect important CO2 events with configurable thresholds.

    Primary scientific definition (per sensor, per time):

    - True event: y >= C OR (y - y_prev) >= D when both finite
    - Server event: same rule on server monitoring value \\tilde y
      (transmitted GT or reconstruction)
    """

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

    def detect_step(
        self,
        current: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-sensor event mask at one time index (temporal rapid rise).

        Unlike ``detect()`` / ``detect_events_series``, this does **not** treat
        the sensor axis as a time series.
        """
        cur = np.asarray(current, dtype=float)
        thr = float(self.primary_threshold_ppm)
        rapid = float(self.rapid_increase_ppm)
        high = (cur >= thr) & np.isfinite(cur)
        if previous is None:
            return high
        prev = np.asarray(previous, dtype=float)
        if prev.shape != cur.shape:
            raise ValueError(f"previous shape {prev.shape} != current shape {cur.shape}")
        rise = (cur - prev) >= rapid
        rapid_m = rise & np.isfinite(cur) & np.isfinite(prev)
        return high | rapid_m

    def is_event_step(
        self,
        current: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.detect_step(current, previous)

    def detect(
        self,
        values: np.ndarray,
        baseline: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Series detector for 1-D temporal arrays (offline evaluation).

        Do **not** pass a cross-sensor vector here for RL step rewards.
        """
        return detect_events_series(values, self.as_dict(), baseline=baseline)

    def is_event(self, values: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
        """Legacy helper for 1-D series. Prefer ``is_event_step`` in the env."""
        return self.detect(values, baseline=baseline)["any"]

    def threshold_masks(self, values: np.ndarray) -> dict[str, np.ndarray]:
        v = np.asarray(values, dtype=float)
        out: dict[str, np.ndarray] = {}
        for thr in self.co2_thresholds_ppm:
            out[f"high_{int(thr)}"] = (v >= thr) & np.isfinite(v)
        return out

    def detect_step_categories(
        self,
        current: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Split primary event into high-CO2 vs rapid-rise (and their OR)."""
        cur = np.asarray(current, dtype=float)
        thr = float(self.primary_threshold_ppm)
        rapid = float(self.rapid_increase_ppm)
        high = (cur >= thr) & np.isfinite(cur)
        if previous is None:
            rapid_m = np.zeros_like(high, dtype=bool)
        else:
            prev = np.asarray(previous, dtype=float)
            if prev.shape != cur.shape:
                raise ValueError(f"previous shape {prev.shape} != current shape {cur.shape}")
            rapid_m = ((cur - prev) >= rapid) & np.isfinite(cur) & np.isfinite(prev)
        return {
            "high_co2": high,
            "rapid_rise": rapid_m,
            "any": high | rapid_m,
        }

    def classify_detection(
        self,
        *,
        true_values: np.ndarray,
        server_values: np.ndarray,
        prev_true: np.ndarray | None = None,
        prev_server: np.ndarray | None = None,
        valid: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return true/server/TP/FN/FP masks under reconstruction-aware semantics."""
        true_cats = self.detect_step_categories(true_values, prev_true)
        server_cats = self.detect_step_categories(server_values, prev_server)
        true_e = true_cats["any"]
        server_e = server_cats["any"]
        if valid is None:
            valid = np.isfinite(true_values)
        else:
            valid = np.asarray(valid, dtype=bool) & np.isfinite(true_values)
        tp = true_e & server_e & valid
        fn = true_e & ~server_e & valid
        fp = ~true_e & server_e & valid
        # Category-specific TP/FN (server uses same category detector)
        tp_high = true_cats["high_co2"] & server_cats["high_co2"] & valid
        fn_high = true_cats["high_co2"] & ~server_cats["high_co2"] & valid
        fp_high = ~true_cats["high_co2"] & server_cats["high_co2"] & valid
        tp_rapid = true_cats["rapid_rise"] & server_cats["rapid_rise"] & valid
        fn_rapid = true_cats["rapid_rise"] & ~server_cats["rapid_rise"] & valid
        fp_rapid = ~true_cats["rapid_rise"] & server_cats["rapid_rise"] & valid
        return {
            "true": true_e & valid,
            "server": server_e & valid,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "valid": valid,
            "true_high_co2": true_cats["high_co2"] & valid,
            "true_rapid_rise": true_cats["rapid_rise"] & valid,
            "tp_high_co2": tp_high,
            "fn_high_co2": fn_high,
            "fp_high_co2": fp_high,
            "tp_rapid_rise": tp_rapid,
            "fn_rapid_rise": fn_rapid,
            "fp_rapid_rise": fp_rapid,
        }
