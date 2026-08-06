"""Evaluate communication policies for paper metrics on real traces.

Computes transmission reduction, reconstruction MAE, event recall, and AoI
from the trace-driven environment. Does not invent numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.evaluation import mae as mae_metric
from campus_senserl.rl.fixed_policies import make_fixed_policies
from campus_senserl.rl.heuristic_policies import InfoValuePolicy, make_heuristic_policies
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json


class RandomBudgetPolicy:
    """Random TRANSMIT with target rate (matched-budget baseline)."""

    def __init__(self, rate: float, seed: int = 42, name: str = "random"):
        self.rate = float(rate)
        self.rng = np.random.default_rng(seed)
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs, *, local_available=None):
        n = obs.shape[0] if obs.ndim == 2 else max(1, obs.shape[0] // 10)
        actions = (self.rng.random(n) < self.rate).astype(int)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions


def evaluate_policy(
    env: TraceDrivenCampusEnv,
    policy,
    *,
    max_steps: int | None = None,
    event_threshold: float = 800.0,
    rapid_delta_ppm: float = 80.0,
) -> dict[str, Any]:
    """Run one episode and accumulate scientifically relevant metrics.

    Important events = (CO2 >= threshold) OR (local rise >= rapid_delta_ppm).
    """
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    max_steps = max_steps or (env.n_steps - 1)
    y_true_skip: list[float] = []
    y_pred_skip: list[float] = []
    n_true_events = 0
    n_detected_events = 0
    n_pred_events = 0
    aoi_samples: list[float] = []
    tx = 0
    avail_n = 0
    steps = 0
    prev_gt = np.full(env.n_sensors, np.nan, dtype=np.float32)

    while steps < max_steps:
        t = env._t
        local = env.local_available[t]
        actions = policy.act(obs, local_available=local)
        obs, reward, terminated, truncated, info = env.step(actions)
        final = info.get("final_actions", actions)
        recon = info["reconstruction"]
        gt = env.ground_truth[t]

        for i in range(env.n_sensors):
            if not local[i] or not np.isfinite(gt[i]):
                continue
            avail_n += 1
            is_tx = int(final[i]) == TRANSMIT
            if is_tx:
                tx += 1
            elif np.isfinite(recon[i]):
                y_true_skip.append(float(gt[i]))
                y_pred_skip.append(float(recon[i]))

            rapid = bool(np.isfinite(prev_gt[i]) and (gt[i] - prev_gt[i]) >= rapid_delta_ppm)
            true_event = bool(gt[i] >= event_threshold) or rapid

            if is_tx:
                # Server observes ground truth
                pred_event = true_event
                detected = true_event
            else:
                pred_high = bool(np.isfinite(recon[i]) and recon[i] >= event_threshold)
                pred_rapid = bool(
                    np.isfinite(prev_gt[i])
                    and np.isfinite(recon[i])
                    and (recon[i] - prev_gt[i]) >= rapid_delta_ppm
                )
                pred_event = pred_high or pred_rapid
                detected = true_event and pred_event

            if true_event:
                n_true_events += 1
                if detected:
                    n_detected_events += 1
            if pred_event:
                n_pred_events += 1

            prev_gt[i] = float(gt[i])

        aoi_samples.extend(env.server_state.aoi.tolist())
        steps += 1
        if terminated or truncated:
            break

    tx_rate = tx / avail_n if avail_n else 0.0
    reduction = (1.0 - tx_rate) * 100.0
    yt = np.asarray(y_true_skip, dtype=float)
    yp = np.asarray(y_pred_skip, dtype=float)

    recall = (n_detected_events / n_true_events) if n_true_events else float("nan")
    precision = (n_detected_events / n_pred_events) if n_pred_events else float("nan")
    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = float("nan")

    aoi_arr = np.asarray(aoi_samples, dtype=float)
    if len(yt):
        mae_skipped = float(mae_metric(yt, yp))
    elif tx_rate > 0.999:
        mae_skipped = 0.0
    else:
        mae_skipped = float("nan")

    return {
        "transmit_rate": float(tx_rate),
        "transmission_reduction_pct": float(reduction),
        "mae_skipped": mae_skipped,
        "n_skipped_eval": int(len(yt)),
        "event_recall": float(recall),
        "event_precision": float(precision),
        "event_f1": float(f1),
        "n_true_events": int(n_true_events),
        "n_detected_events": int(n_detected_events),
        "mean_aoi": float(np.mean(aoi_arr)) if len(aoi_arr) else float("nan"),
        "max_aoi": float(np.max(aoi_arr)) if len(aoi_arr) else float("nan"),
        "p90_aoi": float(np.percentile(aoi_arr, 90)) if len(aoi_arr) else float("nan"),
        "steps": steps,
        "n_sensors": env.n_sensors,
        "n_available_decisions": avail_n,
        "transmit_events": tx,
        "event_threshold_ppm": event_threshold,
        "rapid_delta_ppm": rapid_delta_ppm,
    }


def collect_policy_results(
    *,
    split: str = "val",
    max_sensors: int = 20,
    max_steps: int = 800,
    event_threshold: float = 800.0,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Evaluate fixed, random (matched), and heuristic policies on real traces."""
    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl.yaml")
    cfg_no_shield = dict(cfg)
    cfg_no_shield["safety_shield"] = {**cfg.get("safety_shield", {}), "enabled": False}
    cfg_shield = dict(cfg)
    cfg_shield["safety_shield"] = {**cfg.get("safety_shield", {}), "enabled": True}

    env = TraceDrivenCampusEnv(cfg=cfg_no_shield, split=split, max_sensors=max_sensors, multi_agent=True)
    results: dict[str, dict[str, Any]] = {}

    policies: dict[str, Any] = {}
    policies.update(make_fixed_policies([15, 30, 45, 60]))
    policies.update(make_heuristic_policies(cfg))

    for name, pol in policies.items():
        print(f"[paper-eval] {name}...")
        results[name] = evaluate_policy(env, pol, max_steps=max_steps, event_threshold=event_threshold)
        results[name]["display_name"] = _display_name(name)
        results[name]["family"] = _family(name)
        results[name]["safety_shield"] = False

    for fixed_name in ["fixed_30min", "fixed_45min", "fixed_60min"]:
        if fixed_name not in results:
            continue
        rate = results[fixed_name]["transmit_rate"]
        rname = f"random_match_{fixed_name}"
        print(f"[paper-eval] {rname} (rate={rate:.3f})...")
        rpol = RandomBudgetPolicy(rate=rate, seed=seed, name=rname)
        results[rname] = evaluate_policy(env, rpol, max_steps=max_steps, event_threshold=event_threshold)
        results[rname]["display_name"] = f"Random (matched {fixed_name.replace('fixed_', '')})"
        results[rname]["family"] = "random"
        results[rname]["safety_shield"] = False

    env_s = TraceDrivenCampusEnv(cfg=cfg_shield, split=split, max_sensors=max_sensors, multi_agent=True)
    print("[paper-eval] proposed_proxy (info-value + safety shield)...")
    prop = InfoValuePolicy(threshold=0.35)
    results["proposed_proxy"] = evaluate_policy(
        env_s, prop, max_steps=max_steps, event_threshold=event_threshold
    )
    results["proposed_proxy"]["display_name"] = "Semantic + safety shield (proxy)"
    results["proposed_proxy"]["family"] = "proposed"
    results["proposed_proxy"]["safety_shield"] = True
    results["proposed_proxy"]["note"] = (
        "Proxy for proposed method: semantic info-value heuristic with safety shield. "
        "Full trained MAPPO multi-seed evaluation is not yet available."
    )

    out = ensure_dir(root / "paper_outputs" / "tables")
    save_json(
        {
            "split": split,
            "max_sensors": max_sensors,
            "max_steps": max_steps,
            "event_threshold_ppm": event_threshold,
            "results": results,
        },
        out / "policy_eval_raw.json",
    )
    return results


def _display_name(name: str) -> str:
    mapping = {
        "fixed_15min": "Fixed 15 min",
        "fixed_30min": "Fixed 30 min",
        "fixed_45min": "Fixed 45 min",
        "fixed_60min": "Fixed 60 min",
        "change_threshold": "Change threshold",
        "uncertainty_threshold": "Uncertainty heuristic",
        "aoi_threshold": "AoI threshold",
        "co2_threshold": "CO2 threshold",
        "info_value": "Info-value heuristic",
    }
    return mapping.get(name, name)


def _family(name: str) -> str:
    if name.startswith("fixed"):
        return "fixed"
    if name.startswith("random"):
        return "random"
    if name.startswith("proposed"):
        return "proposed"
    return "heuristic"


def collect_adaptive_timeline(
    *,
    split: str = "val",
    sensor_index: int = 0,
    start: int = 200,
    length: int = 192,
    max_sensors: int = 20,
) -> dict[str, Any]:
    """Record true CO2, TX/SKIP, reconstruction, uncertainty for one sensor."""
    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl.yaml")
    cfg["safety_shield"] = {**cfg.get("safety_shield", {}), "enabled": True}
    env = TraceDrivenCampusEnv(cfg=cfg, split=split, max_sensors=max_sensors, multi_agent=True)
    policy = InfoValuePolicy(threshold=0.35)
    obs, _ = env.reset()
    policy.reset()

    for _ in range(start):
        local = env.local_available[env._t]
        actions = policy.act(obs, local_available=local)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break

    times, true_co2, recon_co2, unc, transmitted, actions_hist = [], [], [], [], [], []
    for k in range(length):
        t = env._t
        local = env.local_available[t]
        actions = policy.act(obs, local_available=local)
        obs, _, term, trunc, info = env.step(actions)
        i = sensor_index
        times.append(k)
        true_co2.append(float(env.ground_truth[t, i]) if np.isfinite(env.ground_truth[t, i]) else np.nan)
        r = info["reconstruction"][i]
        u = info["uncertainty"][i]
        recon_co2.append(float(r) if np.isfinite(r) else np.nan)
        unc.append(float(u) if np.isfinite(u) else np.nan)
        final = int(info["final_actions"][i])
        actions_hist.append(final)
        transmitted.append(true_co2[-1] if final == TRANSMIT and local[i] else np.nan)
        if term or trunc:
            break

    return {
        "times": times,
        "true_co2": true_co2,
        "recon_co2": recon_co2,
        "uncertainty": unc,
        "transmitted": transmitted,
        "actions": actions_hist,
        "sensor_index": sensor_index,
        "event_threshold": float(cfg.get("events", {}).get("primary_threshold_ppm", 1000)),
        "policy": "info_value + safety_shield",
        "split": split,
        "start_offset_steps": start,
    }
