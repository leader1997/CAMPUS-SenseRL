#!/usr/bin/env python
"""Manuscript figure pack (15 PNGs) → results/figures/. No training.

All quantitative values are read from frozen CSV/JSON under results/rl_final/.
Qualitative panels use frozen KL-CMAPPO checkpoints for inference only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import TRANSMIT
from campus_senserl.evaluation.rl_policy_eval import load_mappo_policy, make_final_env
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, save_json
from campus_senserl.visualization.paper_style import apply_paper_style

FIG = ROOT / "results" / "figures"
PAPER = ROOT / "results" / "rl_final" / "paper_final"
SCI = ROOT / "results" / "rl_final" / "scientific_validation"
CKPT = ROOT / "results" / "rl_final" / "cmappo_kl"
SEEDS = [42, 123, 2024, 3407, 9999]
DPI = 600

C_KL = "#B91C1C"
C_FIXED = "#6B7280"
C_FIXED15 = "#9CA3AF"
C_BG = "#D9DCE1"
C_DEV = "#1F2937"
C_HELD = "#0369A1"
C_EXPERT = "#166534"
C_BC = "#6A51A3"
FLOOR_COLORS = {"-1": "#7C3AED", "1": "#2563EB", "2": "#0891B2", "3": "#059669", "4": "#D97706", "5": "#DC2626"}

VALUES: list[dict] = []


def save(fig, name: str) -> None:
    ensure_dir(FIG)
    fig.savefig(FIG / f"{name}.png", dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {name}.png", flush=True)


def rec(fig_name: str, key: str, value, note: str = "") -> None:
    VALUES.append({"figure": fig_name, "quantity": key, "value": value, "note": note})


def devices() -> pd.DataFrame:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    d = load_devices(ROOT / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"]).copy()
    d["deveui"] = d["device_id"].astype(str).str.upper()
    d["is_co2"] = d["device_type"].astype(str).str.contains("CO2", case=False, na=False)
    return d


def val_rows(seed: int) -> pd.DataFrame:
    m = json.loads((CKPT / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    return pd.DataFrame([r for r in m if "val_mae" in r])


def kl_policy():
    return load_mappo_policy(CKPT / "seed_123" / "best_model.pt", device="cpu", prob_threshold=0.5)


def new_env():
    return make_final_env(split="val", cfg=load_yaml(ROOT / "configs" / "rl_cmappo.yaml"), multi_agent=True)


def rollout(env, policy, start: int, length: int, seed: int = 42) -> dict:
    """Step a frozen policy through a deterministic validation window."""
    obs, _ = env.reset(seed=seed)
    for _ in range(start):
        obs, _, te, tr, _ = env.step(policy(obs, local_available=env.local_available[env._t]))
        if te or tr:
            obs, _ = env.reset(seed=seed)
            break
    out = {k: [] for k in ("gt", "recon", "actions", "avail")}
    steps = []
    for _ in range(length):
        t = env._t
        loc = env.local_available[t]
        a = policy(obs, local_available=loc)
        obs, _, te, tr, info = env.step(a)
        steps.append(t)
        out["gt"].append(env.ground_truth[t].copy())
        out["recon"].append(np.asarray(info["reconstruction"], dtype=float).copy())
        out["actions"].append(np.asarray(info["final_actions"], dtype=int).copy())
        out["avail"].append(loc.copy())
        if te or tr:
            break
    res = {k: np.asarray(v) for k, v in out.items()}
    res["steps"] = np.asarray(steps)
    res["tx"] = (res["actions"] == TRANSMIT) & res["avail"].astype(bool)
    return res


def fixed_policy(interval: int):
    pol = FixedIntervalPolicy(interval)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    return act


def hour_of_day(env, steps: np.ndarray) -> np.ndarray:
    tf = np.asarray(env.time_feats)
    return (np.arctan2(tf[steps, 0], tf[steps, 1]) / (2 * np.pi) * 24) % 24


def agent_floors(env) -> list[str]:
    d = devices().set_index("deveui")
    return [str(d.loc[s.upper(), "floor"]) if s.upper() in d.index else "?" for s in env.sensor_ids]


def event_mask(values: np.ndarray, threshold: float = 1000.0, rapid: float = 150.0) -> np.ndarray:
    """Event definition used by the environment: high CO2 or a rapid rise."""
    v = np.asarray(values, dtype=float)
    high = np.isfinite(v) & (v >= threshold)
    jump = np.zeros_like(high)
    jump[1:] = np.isfinite(v[1:]) & np.isfinite(v[:-1]) & ((v[1:] - v[:-1]) >= rapid)
    return high | jump


def select_window(env, length: int = 192) -> tuple[int, int]:
    """Deterministic window: quiet baseline plus a genuine threshold-crossing event."""
    gt = env.ground_truth
    best = None
    for t0 in range(200, gt.shape[0] - length - 5, 24):
        for i in range(gt.shape[1]):
            seg = gt[t0 : t0 + length, i]
            fin = np.isfinite(seg)
            if fin.mean() < 0.93:
                continue
            v = seg[fin]
            n_ev = int(event_mask(v).sum())
            if not (8 <= n_ev <= 45) or v.max() < 1000:
                continue
            med = float(np.median(v))
            score = (float(v.max()) - med) - 2.0 * abs(med - 400.0)
            if best is None or score > best[0]:
                best = (score, t0, i)
    return (best[1], best[2]) if best else (800, 0)


# ---------------------------------------------------------------- 1-3 sensors
def fig01():
    apply_paper_style()
    d = devices()
    tr_ids = {s.upper() for s in load_cohort("final")}
    ho_ids = {s.upper() for s in load_cohort("heldout")}
    co2 = d[d["is_co2"]]
    tr, ho = co2[co2["deveui"].isin(tr_ids)], co2[co2["deveui"].isin(ho_ids)]
    bg = co2[~co2["deveui"].isin(tr_ids | ho_ids)]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(bg["longitude"], bg["latitude"], s=14, c=C_BG, alpha=0.75, label=f"Other CO$_2$ sensors (n={len(bg)})", zorder=1)
    ax.scatter(tr["longitude"], tr["latitude"], s=52, c=C_DEV, edgecolors="white", linewidths=0.5, label=f"RL development agents (n={len(tr)})", zorder=3)
    ax.scatter(ho["longitude"], ho["latitude"], s=58, c=C_HELD, marker="D", edgecolors="white", linewidths=0.5, label=f"Held-out transfer sensors (n={len(ho)})", zorder=4)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig01_campus_deployment")
    rec("fig01", "n_co2_total", int(d["is_co2"].sum()))
    rec("fig01", "n_rl_agents", len(tr))
    rec("fig01", "n_heldout", len(ho))


def fig02():
    apply_paper_style()
    d = devices()
    tr_ids = {s.upper() for s in load_cohort("final")}
    ho_ids = {s.upper() for s in load_cohort("heldout")}
    co2 = d[d["is_co2"]].copy()
    co2["fl"] = pd.to_numeric(co2["floor"], errors="coerce")
    floors = [-1, 1, 2, 3, 4, 5]
    total = [int((co2["fl"] == f).sum()) for f in floors]
    rl = [int(((co2["fl"] == f) & co2["deveui"].isin(tr_ids)).sum()) for f in floors]
    ho = [int(((co2["fl"] == f) & co2["deveui"].isin(ho_ids)).sum()) for f in floors]
    x = np.arange(len(floors))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - 0.26, total, width=0.26, color=C_BG, edgecolor="#9CA3AF", linewidth=0.5, label="All CO$_2$ sensors")
    ax.bar(x, rl, width=0.26, color=C_DEV, label="RL development agents")
    ax.bar(x + 0.26, ho, width=0.26, color=C_HELD, label="Held-out sensors")
    for xi, v in zip(x, total):
        ax.text(xi - 0.26, v + 3, str(v), ha="center", fontsize=8.5, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([("B1" if f == -1 else str(f)) for f in floors])
    ax.set_xlabel("Building floor")
    ax.set_ylabel("Number of CO$_2$ sensors")
    ax.set_ylim(0, max(total) * 1.16)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig02_sensors_by_floor")
    for f, t, r_, h in zip(floors, total, rl, ho):
        rec("fig02", f"floor_{f}", f"all={t}, rl={r_}, heldout={h}")


def fig03():
    apply_paper_style()
    d = devices().set_index("deveui")
    ids = [s.upper() for s in load_cohort("final")]
    raw = json.loads((ROOT / "results" / "graphs" / "node_order.json").read_text(encoding="utf-8"))
    node_order = [str(x).upper() for x in (raw.get("node_order") if isinstance(raw, dict) else raw)]
    A = np.load(ROOT / "results" / "graphs" / "adjacency_hybrid.npy")
    idx = [node_order.index(s) for s in ids]
    S = A[np.ix_(idx, idx)]
    np.fill_diagonal(S, 0.0)
    lon = np.array([float(d.loc[s, "longitude"]) for s in ids])
    lat = np.array([float(d.loc[s, "latitude"]) for s in ids])
    fl = [str(d.loc[s, "floor"]) for s in ids]
    deg = (S > 0).sum(1)

    co2 = devices()
    co2 = co2[co2["is_co2"] & ~co2["deveui"].isin(set(ids))]
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(co2["longitude"], co2["latitude"], s=10, c=C_BG, alpha=0.55, zorder=1, label="Other campus CO$_2$ sensors")
    n_edges = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] > 0:
                n_edges += 1
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#F87171", lw=1.0 + 2.2 * float(S[i, j]), alpha=0.75, zorder=2)
    for f in sorted(set(fl), key=lambda v: float(v)):
        m = np.array([x == f for x in fl])
        ax.scatter(
            lon[m],
            lat[m],
            s=60 + 55 * deg[m],
            c=FLOOR_COLORS.get(f, "#374151"),
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=f"Floor {f} ({int(m.sum())} agent{'s' if m.sum() != 1 else ''})",
        )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.99,
        0.01,
        f"{len(ids)} agents · {n_edges} neighbour links\nmarker size scales with neighbour count",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    save(fig, "fig03_agent_neighbour_network")
    rec("fig03", "n_edges", n_edges)
    rec("fig03", "mean_degree", round(float(deg.mean()), 2))


# ---------------------------------------------------------------- 4 method
def fig04():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.2)
    ax.axis("off")

    def box(y, text, fc, ec, h=1.0, fs=10.5, lw=1.4):
        ax.add_patch(FancyBboxPatch((0.7, y), 8.6, h, boxstyle="round,pad=0.03,rounding_size=0.10", lw=lw, ec=ec, fc=fc))
        ax.text(5.0, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(y1, y2):
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.6))

    box(14.6, "40 campus sensor agents\nlocal CO$_2$, freshness, network, temporal & neighbour context", "#F3F4F6", C_FIXED, h=1.35, fs=9.5)
    arrow(14.6, 13.85)
    box(12.7, "Semantic expert demonstrations", "#DCFCE7", C_EXPERT, h=1.0)
    arrow(12.7, 11.95)
    box(10.8, "Behaviour-cloning initialization  →  $\\pi_{BC}$", "#EDE9FE", C_BC, h=1.0)
    arrow(10.8, 9.75)
    box(
        5.6,
        "KL-CMAPPO  (proposed)\n\n"
        "PPO policy improvement\n"
        "KL anchor to $\\pi_{BC}$\n"
        "event · MAE · AoI constraints\n"
        "centralized critic, shared policy",
        "#FEE2E2",
        C_KL,
        h=4.0,
        fs=10.5,
        lw=2.6,
    )
    arrow(5.6, 4.8)
    box(3.5, "Decentralized execution\neach agent outputs TX / SKIP", "#FEE2E2", C_KL, h=1.25, fs=10)
    arrow(3.5, 2.7)
    box(1.4, "Server monitoring\ncausal reconstruction of skipped observations", "#F3F4F6", C_FIXED, h=1.25, fs=9.5)
    ax.text(5.0, 0.55, "shield-free final policy", ha="center", fontsize=9, color="#6B7280", style="italic")
    save(fig, "fig04_kl_cmappo_framework")


# ------------------------------------------------------- 5-8 learned behaviour
def fig05():
    apply_paper_style()
    env = new_env()
    start, sid = select_window(env, 192)
    h15 = rollout(new_env(), fixed_policy(1), start, 192)
    hkl = rollout(new_env(), kl_policy(), start, 192)
    hours = np.arange(len(hkl["gt"])) * 0.25
    gt = hkl["gt"][:, sid]
    avail = hkl["avail"][:, sid].astype(bool)
    tx = hkl["tx"][:, sid]
    recon = hkl["recon"][:, sid]
    n15, nkl = int(h15["avail"][:, sid].sum()), int(tx.sum())
    red = 100 * (1 - nkl / n15)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    lo, hi = float(np.nanmin(gt)), float(np.nanmax(gt))
    ylo, yhi = lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo)
    ev = event_mask(gt)
    ax.fill_between(hours, ylo, yhi, where=ev, color="#FEE2E2", alpha=0.9, zorder=0, label="Event period (≥1000 ppm or rapid rise)")
    ax.plot(hours, gt, color="#111827", lw=1.6, label="True CO$_2$", zorder=2)
    sk = (~tx) & avail
    ax.plot(hours, np.where(sk, recon, np.nan), color="#10B981", lw=1.3, alpha=0.9, label="Server reconstruction while skipping", zorder=1)
    ax.scatter(hours[tx], gt[tx], s=40, c=C_KL, edgecolors="white", linewidths=0.4, zorder=4, label="KL-CMAPPO transmission")
    ax.axhline(1000, color=C_FIXED, ls="--", lw=1.1, label="1000 ppm event threshold")
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CO$_2$ (ppm)")
    ev_tx = int((tx & ev).sum())
    ev_n = int((ev & avail).sum())
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2, columnspacing=1.2)
    ax.text(
        0.985,
        0.94,
        f"Fixed 15-min: {n15} transmissions\nKL-CMAPPO: {nkl} transmissions ({red:.1f}% fewer)\nEvent steps transmitted: {ev_tx}/{ev_n}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"),
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig05_adaptive_communication")
    save_json({"split": "val", "start_step": int(start), "sensor_index": int(sid), "fixed15_tx": n15, "kl_tx": nkl, "reduction_pct": red}, FIG / "fig05_window.json")
    rec("fig05", "fixed15_tx", n15)
    rec("fig05", "kl_tx", nkl)
    rec("fig05", "reduction_pct", round(red, 2))


def fig06():
    apply_paper_style()
    h = rollout(new_env(), kl_policy(), 3552, 96)
    avail = h["avail"].astype(bool)
    order = np.argsort(-h["tx"].sum(0))  # busiest agents first: makes the budget spread visible
    M = np.where(~avail.T[order], 0.0, np.where(h["tx"].T[order], 2.0, 1.0))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#D1D5DB", "#FFFFFF", "#1D4ED8"]), vmin=0, vmax=2)
    ax.set_xticks(np.arange(0, 97, 16))
    ax.set_xticklabels([f"{int(v/4)}h" for v in np.arange(0, 97, 16)])
    ax.set_xlabel("Time over a 24-hour validation window")
    ax.set_ylabel("Sensor agents (sorted by activity)")
    ax.legend(
        handles=[
            Patch(facecolor="#1D4ED8", label="Transmit"),
            Patch(facecolor="#FFFFFF", edgecolor="#9CA3AF", label="Skip"),
            Patch(facecolor="#D1D5DB", label="No local sample"),
        ],
        frameon=False,
        fontsize=8.5,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.11),
    )
    fig.tight_layout()
    save(fig, "fig06_agent_transmission_schedule")
    rec("fig06", "tx_share_pct", round(100 * float(h["tx"].sum() / max(avail.sum(), 1)), 2))


def _long_rollout():
    env = new_env()
    h = rollout(env, kl_policy(), 1000, 1344)
    return env, h


def fig07(env, h):
    apply_paper_style()
    avail = h["avail"].astype(bool)
    tx_rate = 100 * h["tx"].sum(0) / np.maximum(avail.sum(0), 1)
    gt = h["gt"]
    stds = np.array([np.nanstd(gt[:, i][avail[:, i]]) if avail[:, i].sum() > 5 else np.nan for i in range(gt.shape[1])])
    fl = agent_floors(env)
    ok = np.isfinite(stds) & np.isfinite(tx_rate)
    r = float(np.corrcoef(stds[ok], tx_rate[ok])[0, 1])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for f in sorted(set(fl), key=lambda v: float(v)):
        m = np.array([x == f for x in fl]) & ok
        if m.any():
            ax.scatter(stds[m], tx_rate[m], s=64, c=FLOOR_COLORS.get(f, "#374151"), edgecolors="white", linewidths=0.5, zorder=3, label=f"Floor {f}")
    b, a = np.polyfit(stds[ok], tx_rate[ok], 1)
    xs = np.linspace(stds[ok].min(), stds[ok].max(), 50)
    ax.plot(xs, a + b * xs, color=C_KL, lw=2.0, zorder=2, label=f"Linear fit (r = {r:.2f})")
    ax.set_xlabel("Local CO$_2$ variability of the sensor (standard deviation, ppm)")
    ax.set_ylabel("Agent transmission rate (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig07_agent_budget_vs_information")
    rec("fig07", "pearson_r", round(r, 3), "per-agent TX rate vs local CO2 std, 14-day validation window")
    rec("fig07", "tx_rate_range_pct", f"{tx_rate[ok].min():.1f}-{tx_rate[ok].max():.1f}")


def fig08(env, h):
    apply_paper_style()
    avail = h["avail"].astype(bool)
    hod = np.floor(hour_of_day(env, h["steps"])).astype(int)
    rate = 100 * h["tx"].sum(1) / np.maximum(avail.sum(1), 1)
    df = pd.DataFrame({"h": hod, "r": rate}).groupby("h")["r"].agg(["mean", "std", "count"]).reindex(range(24))
    sem = df["std"] / np.sqrt(df["count"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axvspan(8, 17, color="#FEF3C7", alpha=0.7, zorder=0)
    ax.fill_between(df.index, df["mean"] - sem, df["mean"] + sem, color=C_KL, alpha=0.25, zorder=2)
    ax.plot(df.index, df["mean"], color=C_KL, lw=2.3, marker="o", ms=4, zorder=3, label="Mean ± s.e.m. over 14 days")
    peak = int(df["mean"].idxmax())
    ax.set_ylim(df["mean"].min() - 2.5, df["mean"].max() + 3.4)
    ax.text(12.5, ax.get_ylim()[1] - 0.25, "typical occupied hours", ha="center", va="top", fontsize=8.5, color="#92400E")
    ax.annotate(
        f"peak {df['mean'].max():.0f}% at {peak:02d}:00",
        xy=(peak, df["mean"].max()),
        xytext=(peak + 3.0, df["mean"].max() + 1.7),
        fontsize=8.5,
        color="#374151",
        arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.0),
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{v:02d}" for v in range(0, 24, 3)])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Transmission rate (%)")
    ax.set_xlim(-0.5, 23.5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig08_diurnal_transmission_profile")
    rec("fig08", "peak_hour", peak)
    rec("fig08", "peak_rate_pct", round(float(df["mean"].max()), 2))
    rec("fig08", "min_rate_pct", round(float(df["mean"].min()), 2))


# ---------------------------------------------------------------- 9 savings
def fig09():
    apply_paper_style()
    start, length = 3552, 672
    c = {}
    for label, pol in [("Fixed 15", fixed_policy(1)), ("Fixed 60", fixed_policy(4)), ("KL-CMAPPO", kl_policy())]:
        h = rollout(new_env(), pol, start, length)
        c[label] = np.cumsum(h["tx"].sum(axis=1))
    hours = np.arange(len(c["Fixed 15"])) * 0.25
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    offsets = {"Fixed 15": 0.0, "Fixed 60": 1400.0, "KL-CMAPPO": -1400.0}
    for label, color, lw in [("Fixed 15", C_FIXED15, 2.0), ("Fixed 60", C_FIXED, 2.0), ("KL-CMAPPO", C_KL, 2.8)]:
        ax.plot(hours, c[label], color=color, lw=lw)
        ax.text(
            hours[-1] + 4,
            c[label][-1] + offsets[label],
            f"{label}\n{int(c[label][-1]):,} uplinks",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold" if label == "KL-CMAPPO" else "normal",
        )
    saved = 100 * (1 - c["KL-CMAPPO"][-1] / c["Fixed 15"][-1])
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative transmissions")
    ax.set_xlim(0, hours[-1] + 46)
    ax.set_ylim(0, c["Fixed 15"][-1] * 1.08)
    ax.text(0.02, 0.96, f"{saved:.1f}% fewer uplinks than Fixed 15 over one week,\nat the Fixed 60 communication budget", transform=ax.transAxes, va="top", fontsize=9, color="#374151")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig09_cumulative_transmissions")
    for k, v in c.items():
        rec("fig09", f"cum_tx_{k}", int(v[-1]))


# ------------------------------------------------------------ 10-12 headline
def fig10():
    apply_paper_style()
    s = pd.read_csv(PAPER / "full_val_summary.csv").set_index("method")
    methods, labels, colors = ["fixed_60", "cmappo_kl"], ["Fixed 60", "KL-CMAPPO"], [C_FIXED, C_KL]
    mae = [float(s.loc[m, "mae_mean"]) for m in methods]
    std = [float(s.loc[m, "mae_std"]) if int(s.loc[m, "n"]) > 1 else 0.0 for m in methods]
    tx = [float(s.loc[m, "tx_reduction_mean"]) for m in methods]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.arange(2)
    ax.bar(x, mae, color=colors, width=0.46)
    ax.errorbar(x[1], mae[1], yerr=std[1], fmt="none", ecolor="#374151", capsize=5, lw=1.2)
    for i in range(2):
        top = mae[i] + std[i]
        ax.text(i, top + 0.50, f"{mae[i]:.2f} ppm", ha="center", fontsize=10.5, fontweight="bold")
        ax.text(i, top + 0.14, f"{tx[i]:.1f}% fewer uplinks", ha="center", fontsize=8.5, color="#374151")
    gain = 100 * (1 - mae[1] / mae[0])
    ax.text(0.5, max(mae) * 1.19, f"{gain:.0f}% lower error", ha="center", fontsize=10, color=C_KL, fontweight="bold")
    ax.annotate("", xy=(0.12, max(mae) * 1.14), xytext=(0.88, max(mae) * 1.14), arrowprops=dict(arrowstyle="<|-", color=C_KL, lw=1.4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel("Reconstruction error on skipped\nobservations, CO$_2$ MAE (ppm)")
    ax.set_ylim(0, max(mae) * 1.30)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig10_reconstruction_error_vs_fixed60")
    for lab, m, t in zip(labels, mae, tx):
        rec("fig10", f"mae_{lab}", round(m, 3), f"tx_reduction={t:.2f}%")


def fig11():
    apply_paper_style()
    s = pd.read_csv(PAPER / "full_test_summary.csv").set_index("method")
    n_true = 102
    labels = ["Fixed 60", "KL-CMAPPO"]
    keys = ["fixed_60", "cmappo_kl"]
    det = [int(round(float(s.loc[k, "recall_mean"]) * n_true)) for k in keys]
    miss = [n_true - v for v in det]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(2)
    ax.barh(y, det, color=[C_FIXED, C_KL], height=0.5, label="Detected")
    ax.barh(y, miss, left=det, color="#E5E7EB", height=0.5, label="Missed")
    for i in range(2):
        ax.text(det[i] / 2, i, f"{det[i]} detected", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if miss[i] > 6:
            ax.text(det[i] + miss[i] / 2, i, f"{miss[i]} missed", ha="center", va="center", color="#4B5563", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(f"Important CO$_2$ events (n = {n_true}, frozen temporal test)")
    ax.set_xlim(0, n_true)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout()
    save(fig, "fig11_event_preservation")
    for lab, dd in zip(labels, det):
        rec("fig11", f"detected_{lab}", f"{dd}/{n_true}")


def fig12():
    apply_paper_style()
    raw = pd.read_csv(SCI / "heldout_transfer_raw.csv")
    sub = raw[(raw["method"] == "cmappo_kl") & (raw["split"] == "test") & (raw["cohort"] == "heldout")].sort_values("seed")
    summ = pd.read_csv(SCI / "heldout_transfer_summary.csv")
    row = summ[(summ["method"] == "cmappo_kl") & (summ["split"] == "test")].iloc[0]
    mean, sd = float(row["mae_mean"]), float(row["mae_std"])
    dev = float(pd.read_csv(PAPER / "full_test_summary.csv").set_index("method").loc["cmappo_kl", "mae_mean"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(sub))
    ax.axhspan(mean - sd, mean + sd, color=C_KL, alpha=0.12)
    ax.axhline(dev, color=C_DEV, lw=1.6, ls=":", label=f"Development cohort = {dev:.2f} ppm")
    ax.axhline(mean, color=C_KL, lw=1.8, ls="--", label=f"Unseen sensors = {mean:.2f} ± {sd:.2f} ppm")
    ax.scatter(x, sub["mae_skipped"], s=95, c=C_KL, edgecolors="white", linewidths=0.6, zorder=3, label="Individual seed")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(v)}" for v in sub["seed"]], fontsize=9)
    ax.set_ylabel("CO$_2$ MAE on 40 unseen sensors (ppm)")
    lo = min(float(sub["mae_skipped"].min()), dev)
    hi = max(float(sub["mae_skipped"].max()), dev)
    pad = max(0.45, (hi - lo) * 1.2)
    ax.set_ylim(lo - pad, hi + pad * 1.5)
    ax.text(
        0.99,
        0.04,
        f"transmission reduction {float(row['tx_reduction_mean']):.2f}%\nevent recall {100*float(row['recall_mean']):.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"),
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig12_unseen_sensor_generalization")
    rec("fig12", "heldout_mae", f"{mean:.3f}±{sd:.3f}")
    rec("fig12", "heldout_recall_pct", round(100 * float(row["recall_mean"]), 2))


# ---------------------------------------------------------------- 13 robust
def fig13():
    apply_paper_style()
    r = pd.read_csv(SCI / "robustness_summary.csv")
    sub = r[(r["method"] == "cmappo_kl") & (r["condition"] == "packet_loss")].sort_values("level")
    x = 100 * sub["level"].astype(float).values
    y = sub["mae_mean"].astype(float).values
    e = sub["mae_std"].astype(float).values
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.fill_between(x, y - e, y + e, color=C_KL, alpha=0.18)
    ax.plot(x, y, color=C_KL, marker="o", lw=2.4, ms=9, label="KL-CMAPPO (5 seeds)")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.13, f"{yi:.2f}", ha="center", fontsize=9, color="#374151")
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_xticks(x)
    ax.set_ylim(min(y) - 0.8, max(y) + 0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig13_packet_loss_robustness")
    for xi, yi in zip(x, y):
        rec("fig13", f"mae_loss_{int(xi)}pct", round(float(yi), 3))


# ------------------------------------------------------------- 14-15 RL / KL
def fig14():
    apply_paper_style()
    allv = pd.concat([val_rows(s).assign(seed=s) for s in SEEDS])
    g = allv.groupby("step", as_index=False).agg(mean=("val_mae", "mean"), std=("val_mae", "std"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, s in enumerate(SEEDS):
        v = val_rows(s)
        ax.plot(v["step"], v["val_mae"], color="#B9BEC6", lw=0.9, zorder=1, label="Individual seed" if k == 0 else None)
    ax.axhspan(0, 9.0, color="#D1FAE5", alpha=0.45, zorder=0)
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_KL, alpha=0.2, zorder=2)
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2.6, label="Mean over 5 seeds", zorder=3)
    ax.axhline(9.0, color="#047857", ls="--", lw=1.4, label="Quality constraint (MAE ≤ 9 ppm)", zorder=4)
    ax.text(g["step"].iloc[0], 8.9, "feasible region", fontsize=9, color="#047857", va="top")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Validation CO$_2$ MAE (ppm)")
    ax.set_ylim(8.2, max(9.6, float((g["mean"] + g["std"].fillna(0)).max()) + 0.15))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save(fig, "fig14_training_validation_mae")
    rec("fig14", "final_mean_val_mae", round(float(g["mean"].iloc[-1]), 3))
    rec("fig14", "n_feasible_final", int(allv[allv["step"] == allv["step"].max()]["feasible"].sum()))


def fig15():
    """What the KL-regularized RL stage adds on top of its own BC anchor, at equal cost."""
    apply_paper_style()
    m = pd.read_csv(SCI / "matched_budget_summary.csv")
    budgets = sorted(m["target_tx_reduction"].unique())
    bc = m[m["method"] == "campus_senserl_bc"].set_index("target_tx_reduction")
    kl = m[m["method"] == "cmappo_kl"].set_index("target_tx_reduction")
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.19, [bc.loc[b, "mae_mean"] for b in budgets], yerr=[bc.loc[b, "mae_std"] for b in budgets], width=0.38, color=C_BC, capsize=4, error_kw=dict(lw=1.1, ecolor="#374151"), label="BC initialization ($\\pi_{BC}$)")
    ax.bar(x + 0.19, [kl.loc[b, "mae_mean"] for b in budgets], yerr=[kl.loc[b, "mae_std"] for b in budgets], width=0.38, color=C_KL, capsize=4, error_kw=dict(lw=1.1, ecolor="#374151"), label="KL-CMAPPO (proposed)")
    for xi, b in zip(x, budgets):
        v0, v1 = float(bc.loc[b, "mae_mean"]), float(kl.loc[b, "mae_mean"])
        ax.text(xi + 0.19, v1 - 0.32, f"−{v0 - v1:.2f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}%" for b in budgets], fontsize=11)
    ax.set_xlabel("Matched communication budget (transmission reduction)")
    ax.set_ylabel("CO$_2$ MAE (ppm), mean over 5 seeds")
    ax.set_ylim(0, max(m["mae_mean"]) * 1.28)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig15_kl_gain_over_bc_anchor")
    for b in budgets:
        rec("fig15", f"budget_{int(b)}pct", f"bc={float(bc.loc[b,'mae_mean']):.3f}, kl={float(kl.loc[b,'mae_mean']):.3f}")


def main():
    ensure_dir(FIG)
    apply_paper_style()
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    env, h = _long_rollout()
    fig07(env, h)
    fig08(env, h)
    fig09()
    fig10()
    fig11()
    fig12()
    fig13()
    fig14()
    fig15()
    pd.DataFrame(VALUES).to_csv(FIG / "figure_values.csv", index=False)
    print(f"WROTE figure_values.csv ({len(VALUES)} rows)", flush=True)


if __name__ == "__main__":
    main()
