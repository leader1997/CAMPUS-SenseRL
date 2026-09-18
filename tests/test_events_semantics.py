"""Event semantics: SKIP + correctly reconstructed event is TP, not FN."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.event_detector import EventDetector
from campus_senserl.environment.reconstructors import make_constant_reconstructor
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace


def test_detect_step_is_per_sensor_temporal():
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    prev = np.array([400.0, 400.0, 900.0])
    cur = np.array([400.0, 1100.0, 920.0])  # high on sensor1; rapid would need +150
    mask = det.detect_step(cur, prev)
    assert mask.tolist() == [False, True, False]

    # Rapid rise on sensor 0
    cur2 = np.array([600.0, 400.0, 400.0])
    prev2 = np.array([400.0, 400.0, 400.0])
    mask2 = det.detect_step(cur2, prev2)
    assert mask2[0]


def test_skipped_but_reconstructed_event_is_tp_not_fn(minimal_rl_cfg):
    """Critical: SKIP ∧ true event ∧ recon detects ⇒ TP (not automatic FN)."""
    n_steps, n_sensors = 8, 2
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=0)
    # Force a clear high-CO2 event at t=1 for sensor 0, with calm previous value
    trace["ground_truth"][:, :] = 400.0
    trace["ground_truth"][1, 0] = 1200.0
    trace["natural_missing"][:] = False
    trace["local_available"][:] = True

    cfg = dict(minimal_rl_cfg)
    cfg["events"] = {
        "primary_threshold_ppm": 1000,
        "rapid_increase_ppm": 150,
        "co2_thresholds_ppm": [800, 1000, 1200, 1500],
    }

    # Reconstructor that always reports 1200 → server detects event even on SKIP
    recon = make_constant_reconstructor(mean_value=1200.0, unc_value=0.5)
    env = TraceDrivenCampusEnv(
        cfg=cfg,
        trace=trace,
        reconstruction_model=recon,
        multi_agent=True,
    )
    env.reset(seed=0)

    # t=0: transmit to establish last_tx / monitor baseline
    env.step(np.full(n_sensors, TRANSMIT, dtype=int))
    # t=1: SKIP while GT is an event; recon still high → TP
    _, _, _, _, info = env.step(np.full(n_sensors, SKIP, dtype=int))

    assert info["true_events"][0]
    assert info["detected_events"][0]
    assert not info["missed_events"][0]


def test_skipped_and_missed_reconstruction_is_fn(minimal_rl_cfg):
    n_steps, n_sensors = 8, 2
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=1)
    trace["ground_truth"][:, :] = 400.0
    trace["ground_truth"][1, 0] = 1200.0
    trace["natural_missing"][:] = False
    trace["local_available"][:] = True

    cfg = dict(minimal_rl_cfg)
    cfg["events"] = {
        "primary_threshold_ppm": 1000,
        "rapid_increase_ppm": 150,
        "co2_thresholds_ppm": [800, 1000, 1200, 1500],
    }

    # Reconstructor always low → FN on SKIP
    recon = make_constant_reconstructor(mean_value=400.0, unc_value=0.5)
    env = TraceDrivenCampusEnv(
        cfg=cfg,
        trace=trace,
        reconstruction_model=recon,
        multi_agent=True,
    )
    env.reset(seed=0)
    env.step(np.full(n_sensors, TRANSMIT, dtype=int))
    _, _, _, _, info = env.step(np.full(n_sensors, SKIP, dtype=int))

    assert info["true_events"][0]
    assert info["missed_events"][0]
    assert not info["detected_events"][0]


def test_classify_detection_helper():
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    true_v = np.array([1200.0, 400.0])
    server_v = np.array([1150.0, 400.0])  # reconstructs event on sensor 0
    out = det.classify_detection(true_values=true_v, server_values=server_v)
    assert out["tp"][0] and not out["fn"][0]
    assert not out["true"][1]


def test_classify_transmitted_event_is_tp():
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    out = det.classify_detection(
        true_values=np.array([1200.0]),
        server_values=np.array([1200.0]),
        prev_true=np.array([400.0]),
        prev_server=np.array([400.0]),
    )
    assert out["true"][0] and out["server"][0] and out["tp"][0]
    assert not out["fn"][0] and not out["fp"][0]


def test_classify_skip_locf_still_high_is_tp():
    """Skip + LOCF still ≥1000: combined primary event is TP, not FN."""
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    out = det.classify_detection(
        true_values=np.array([1100.0]),
        server_values=np.array([1100.0]),  # LOCF of last TX
        prev_true=np.array([1100.0]),
        prev_server=np.array([1100.0]),
    )
    assert out["true"][0] and out["server"][0] and out["tp"][0]
    assert not out["fn"][0]


def test_classify_skip_wrong_recon_is_fn():
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    out = det.classify_detection(
        true_values=np.array([1200.0]),
        server_values=np.array([400.0]),
        prev_true=np.array([400.0]),
        prev_server=np.array([400.0]),
    )
    assert out["true"][0] and out["fn"][0]
    assert not out["tp"][0] and not out["fp"][0]


def test_classify_false_server_event_is_fp():
    det = EventDetector(primary_threshold_ppm=1000, rapid_increase_ppm=150)
    out = det.classify_detection(
        true_values=np.array([400.0]),
        server_values=np.array([1200.0]),
        prev_true=np.array([400.0]),
        prev_server=np.array([400.0]),
    )
    assert not out["true"][0] and out["server"][0] and out["fp"][0]
    assert not out["tp"][0] and not out["fn"][0]
