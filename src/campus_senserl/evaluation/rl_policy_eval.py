"""Scientifically correct evaluation of trained and baseline policies.

Transmission rate is counted only over locally available measurements:
    TX_rate = N_TX / N_locally_available
    Reduction = 1 - TX_rate

Event metrics use reconstruction-aware TP/FN from the environment info.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.graph_utils import assert_not_identity
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.evaluation import mae as mae_metric
from campus_senserl.evaluation.pdr import packet_delivery_ratio
from campus_senserl.rl.mappo import SharedActor
from campus_senserl.rl.ppo import ActorCritic
from campus_senserl.utils import load_yaml, repo_root, set_seed


def make_final_env(
    *,
    split: str = "val",
    cfg: dict[str, Any] | None = None,
    multi_agent: bool = True,
    packet_loss_rate: float = 0.0,
    packet_loss_mode: str = "per_attempt",
    packet_loss_mask_seed: int | None = None,
    sensor_ids: list[str] | None = None,
    sensor_outage_fraction: float = 0.0,
    temp_outage_fraction: float = 0.0,
    temp_outage_duration_frac: float = 0.1,
    edge_drop_fraction: float = 0.0,
    stress_seed: int = 0,
) -> TraceDrivenCampusEnv:
    """Build env with frozen cohort IDs and fail-fast hybrid graph.

    Robustness knobs (validation stress only; do not tune on these):
      - packet_loss_rate: post-decision uplink drops
      - sensor_outage_fraction: permanent random sensor unavailability
      - temp_outage_*: mid-trace block outage for a sensor subset
      - packet_loss_mode: ``per_attempt`` (legacy) or ``independent_slots`` (shared mask)
      - edge_drop_fraction: random hybrid-graph edge removal (contextual-relation loss)
    """
    root = repo_root()
    cfg = dict(cfg or load_yaml(root / "configs" / "rl.yaml"))
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg["cohort"] = env_cfg.get("cohort", "final")
    env_cfg["graph"] = env_cfg.get("graph", "hybrid")
    env_cfg["graph_fail_fast"] = True
    env_cfg["packet_loss_rate"] = float(packet_loss_rate)
    env_cfg["packet_loss_mode"] = str(packet_loss_mode)
    if packet_loss_mask_seed is not None:
        env_cfg["packet_loss_mask_seed"] = int(packet_loss_mask_seed)
    env_cfg["sensor_outage_fraction"] = float(sensor_outage_fraction)
    env_cfg["temp_outage_fraction"] = float(temp_outage_fraction)
    env_cfg["temp_outage_duration_frac"] = float(temp_outage_duration_frac)
    env_cfg["edge_drop_fraction"] = float(edge_drop_fraction)
    env_cfg["stress_seed"] = int(stress_seed)
    cfg["environment"] = env_cfg

    ids = sensor_ids if sensor_ids is not None else load_cohort(str(env_cfg["cohort"]))
    env = TraceDrivenCampusEnv(
        cfg=cfg,
        split=split,
        sensor_ids=ids,
        multi_agent=multi_agent,
    )
    if env_cfg["graph"] not in {"identity", "none", "eye", ""}:
        assert_not_identity(env.adjacency)
    return env


def load_ppo_policy(checkpoint: str | Path, device: str = "cpu") -> Callable:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    # Infer dims from weights
    obs_dim = state["shared.0.weight"].shape[1]
    act_dim = state["actor.weight"].shape[0]
    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def act(obs, *, local_available=None):
        x = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        logits, _ = model(x)
        actions = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions.astype(int)

    return act


def load_mappo_policy(
    checkpoint: str | Path,
    device: str = "cpu",
    *,
    prob_threshold: float = 0.5,
) -> Callable:
    """Load shared/residual MAPPO actor.

    ``prob_threshold`` controls deterministic TX: transmit iff σ(logit) > τ.
    Sweeping τ yields matched communication-budget comparisons.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", ckpt["actor"]["net.0.weight"].shape[1]))
    actor_type = str(ckpt.get("actor_type", "shared"))
    tau = float(prob_threshold)
    if actor_type == "residual_heuristic":
        from campus_senserl.rl.mappo_boosted import ResidualSharedActor

        actor = ResidualSharedActor(obs_dim).to(device)
    else:
        actor = SharedActor(obs_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    @torch.no_grad()
    def act(obs, *, local_available=None):
        if obs.ndim == 1:
            n = obs.shape[0] // obs_dim
            obs2 = obs.reshape(n, obs_dim)
        else:
            obs2 = obs
        x = torch.as_tensor(obs2, dtype=torch.float32, device=device)
        logits = actor(x)
        actions = (torch.sigmoid(logits) > tau).long().cpu().numpy().reshape(-1)
        if local_available is not None:
            actions = np.where(local_available, actions, SKIP)
        return actions.astype(int)

    return act


def evaluate_policy(
    env: TraceDrivenCampusEnv,
    policy_act: Callable,
    *,
    max_steps: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Full-trace (or truncated) evaluation with correct TX / event metrics."""
    set_seed(seed)
    obs, info0 = env.reset(seed=seed)
    if hasattr(policy_act, "reset"):
        policy_act.reset()

    max_steps = max_steps if max_steps is not None else (env.n_steps - 1)
    y_true_skip: list[float] = []
    y_pred_skip: list[float] = []
    tp = fn = fp = 0
    n_true = 0
    n_server = 0
    # Category counters
    tp_high = fn_high = n_high = 0
    tp_rapid = fn_rapid = n_rapid = 0
    aoi_state_all: list[float] = []
    aoi_raw_all: list[float] = []
    tx = 0
    avail_n = 0
    tx_requested = 0
    tx_delivered = 0
    steps = 0
    reward_sum = 0.0

    while steps < max_steps and env._t < env.n_steps - 1:
        t = env._t
        local = env.local_available[t]
        if callable(policy_act) and not hasattr(policy_act, "act"):
            actions = policy_act(obs, local_available=local)
        else:
            actions = policy_act.act(obs, local_available=local)
        obs, reward, terminated, truncated, info = env.step(actions)
        reward_sum += float(reward)
        final = info.get("final_actions", actions)
        recon = info["reconstruction"]
        gt = env.ground_truth[t]
        local = info.get("local_available", local)

        for i in range(env.n_sensors):
            if not local[i] or not np.isfinite(gt[i]):
                continue
            avail_n += 1
            is_tx_decision = int(final[i]) == TRANSMIT
            delivered = bool(env.transmit_history[t, i])
            if is_tx_decision:
                tx += 1
            if (not delivered) and np.isfinite(recon[i]):
                y_true_skip.append(float(gt[i]))
                y_pred_skip.append(float(recon[i]))

        if "true_events" in info:
            te = np.asarray(info["true_events"], dtype=bool) & local
            me = np.asarray(info["missed_events"], dtype=bool) & local
            # Prefer explicit FP / server masks (correct precision)
            if "false_positive_events" in info:
                fpm = np.asarray(info["false_positive_events"], dtype=bool) & local
                tpm = np.asarray(info["detected_events"], dtype=bool) & local  # TP
            else:
                # Backward compat — do NOT derive FP from TP mask
                tpm = np.asarray(info["detected_events"], dtype=bool) & local
                fpm = np.zeros_like(tpm)
            se = np.asarray(info.get("server_events", tpm | fpm), dtype=bool) & local
            n_true += int(te.sum())
            n_server += int(se.sum())
            tp += int(tpm.sum())
            fn += int(me.sum())
            fp += int(fpm.sum())

            if "true_high_co2" in info:
                th = np.asarray(info["true_high_co2"], dtype=bool) & local
                n_high += int(th.sum())
                tp_high += int((np.asarray(info["tp_high_co2"], dtype=bool) & local).sum())
                fn_high += int((np.asarray(info["fn_high_co2"], dtype=bool) & local).sum())
            if "true_rapid_rise" in info:
                tr = np.asarray(info["true_rapid_rise"], dtype=bool) & local
                n_rapid += int(tr.sum())
                tp_rapid += int((np.asarray(info["tp_rapid_rise"], dtype=bool) & local).sum())
                fn_rapid += int((np.asarray(info["fn_rapid_rise"], dtype=bool) & local).sum())

        tx_requested += int(info.get("tx_requested", info.get("tx_requested_after_shield", 0)))
        tx_delivered += int(info.get("transmit_count", 0))
        if "aoi_raw" in info:
            aoi_raw_all.extend(np.asarray(info["aoi_raw"], dtype=float).tolist())
        if "aoi_state" in info:
            aoi_state_all.extend(np.asarray(info["aoi_state"], dtype=float).tolist())
        elif env.server_state is not None:
            aoi_state_all.extend(env.server_state.aoi.tolist())
        steps += 1
        if terminated or truncated:
            break

    tx_rate = tx / avail_n if avail_n else 0.0
    reduction = (1.0 - tx_rate) * 100.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    def _rec(tpp, fnn, nn):
        if nn <= 0:
            return float("nan")
        return float(tpp / (tpp + fnn)) if (tpp + fnn) else float("nan")

    # Prefer raw AoI for paper metrics
    aoi_report = aoi_raw_all if aoi_raw_all else aoi_state_all

    return {
        "steps": steps,
        "n_sensors": env.n_sensors,
        "n_locally_available": avail_n,
        "n_tx": tx,
        "transmit_rate": float(tx_rate),
        "transmission_reduction_pct": float(reduction),
        "mae_skipped": float(mae_metric(np.asarray(y_true_skip), np.asarray(y_pred_skip))) if y_true_skip else float("nan"),
        "n_skipped_eval": len(y_true_skip),
        "n_true_events": n_true,
        "n_server_events": n_server,
        "event_recall": float(recall) if n_true else float("nan"),
        "event_precision": float(precision) if (tp + fp) else float("nan"),
        "event_f1": float(f1),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "n_high_co2_events": n_high,
        "recall_high_co2": _rec(tp_high, fn_high, n_high),
        "n_rapid_rise_events": n_rapid,
        "recall_rapid_rise": _rec(tp_rapid, fn_rapid, n_rapid),
        "mean_aoi": float(np.mean(aoi_report)) if aoi_report else float("nan"),
        "mean_aoi_state": float(np.mean(aoi_state_all)) if aoi_state_all else float("nan"),
        "mean_aoi_raw": float(np.mean(aoi_raw_all)) if aoi_raw_all else float("nan"),
        "p90_aoi": float(np.percentile(aoi_report, 90)) if aoi_report else float("nan"),
        "p95_aoi": float(np.percentile(aoi_report, 95)) if aoi_report else float("nan"),
        "max_aoi": float(np.max(aoi_report)) if aoi_report else float("nan"),
        "mean_reward_per_step": reward_sum / max(steps, 1),
        "packet_delivery_ratio": packet_delivery_ratio(tx_requested, tx_delivered),
        "tx_requested": tx_requested,
        "tx_delivered": tx_delivered,
        "graph": getattr(env, "graph_type", None),
        "reconstructor": getattr(env, "reconstructor_name", None),
        "adjacency_is_identity": bool(np.allclose(env.adjacency, np.eye(env.n_sensors))),
    }
