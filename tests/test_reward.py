"""Reward component finiteness and configurability."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT, CommunicationCostModel
from campus_senserl.rl.reward import RewardComputer, compute_reward


def _sample_step(n: int = 4) -> dict:
    rng = np.random.default_rng(0)
    gt = rng.uniform(400, 1200, size=n)
    recon = gt + rng.normal(0, 50, size=n)
    unc = rng.uniform(0.1, 3.0, size=n)
    aoi = rng.uniform(0, 8, size=n)
    local = np.ones(n, dtype=bool)
    events = gt > 1000
    missed = events & (rng.random(n) < 0.3)
    actions = rng.integers(0, 2, size=n)
    return {
        "actions": actions,
        "ground_truth": gt,
        "reconstruction": recon,
        "uncertainty": unc,
        "aoi": aoi,
        "local_available": local,
        "events": events,
        "missed_event_mask": missed,
    }


def test_reward_components_finite():
    rc = RewardComputer()
    total, parts = rc.compute_step(**_sample_step(), comm_model=CommunicationCostModel())
    assert np.isfinite(total)
    for k, v in parts.items():
        assert np.isfinite(v), f"component {k} not finite: {v}"


def test_reward_respects_clip():
    rc = RewardComputer(w_tx=100.0, w_err=100.0, clip=2.0, normalize_components=False)
    total, _ = rc.compute_step(**_sample_step(), comm_model=CommunicationCostModel())
    assert -2.0 <= total <= 2.0


def test_reward_weights_configurable():
    base = _sample_step()
    rc_low = RewardComputer(w_tx=0.01, w_err=0.01, w_aoi=0.01, w_unc=0.01, w_miss=0.01, w_event=0.01)
    rc_high = RewardComputer(w_tx=10.0, w_err=10.0, w_aoi=10.0, w_unc=10.0, w_miss=10.0, w_event=10.0)
    t_low, _ = rc_low.compute_step(**base, comm_model=CommunicationCostModel())
    t_high, _ = rc_high.compute_step(**base, comm_model=CommunicationCostModel())
    assert abs(t_high) >= abs(t_low) or np.isclose(t_high, t_low)


def test_from_config_reads_weights():
    cfg = {"reward": {"w_tx": 2.0, "w_aoi": 3.0, "clip": 5.0}}
    rc = RewardComputer.from_config(cfg)
    assert rc.w_tx == 2.0
    assert rc.w_aoi == 3.0
    assert rc.clip == 5.0


def test_compute_reward_helper():
    total, parts = compute_reward(**_sample_step(), cfg={"reward": {"w_tx": 1.0}})
    assert np.isfinite(total)
    assert set(parts.keys()) == {"tx", "err", "aoi", "unc", "miss", "event"}


def test_transmit_cost_affects_tx_component():
    base = _sample_step()
    base["actions"] = np.full(4, TRANSMIT)
    rc = RewardComputer(w_tx=1.0, normalize_components=False)
    _, parts_tx = rc.compute_step(**base, comm_model=CommunicationCostModel(transmit_cost=1.0))
    _, parts_free = rc.compute_step(**base, comm_model=CommunicationCostModel(transmit_cost=0.0))
    assert parts_tx["tx"] <= parts_free["tx"]
