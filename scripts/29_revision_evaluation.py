#!/usr/bin/env python
"""Revision evaluation: broader, fair comparison without retraining.

Uses frozen KL-CMAPPO checkpoints. Does not tune on test or held-out sensors.
Outputs live under results/revision/ (does not overwrite results/rl_final/).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.cohort import load_cohort
from campus_senserl.evaluation.revision_checks import RevisionCheckError, validate_master
from campus_senserl.evaluation.revision_export import (
    N_LOSS_MASKS,
    write_claim_audit,
    write_delta_artifacts,
    write_paper_ready_summary,
    write_protocol_json,
    write_tables,
)
from campus_senserl.evaluation.revision_metrics import (
    MASTER_COLUMNS,
    backfill_master,
    row_key,
    select_delta_heartbeat,
    standardize_eval,
)
from campus_senserl.evaluation.rl_policy_eval import (
    evaluate_policy,
    load_mappo_policy,
    make_final_env,
)
from campus_senserl.rl.expert_policy import DeltaPlusHeartbeatPolicy, SemanticExpertPolicy
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json

SEEDS = [42, 123, 2024, 3407, 9999]
FROZEN_TEST_SEED = 123
FIXED_MINUTES = [15, 30, 45, 60, 75, 90]
DELTA_PPM = [25.0, 50.0, 75.0, 100.0]
HEARTBEAT = [3.0, 4.0, 5.0, 6.0]
PACKET_LOSS = [0.0, 0.1, 0.2, 0.4]
PACKET_LOSS_MASK_SEED = 20260317
TARGET_TX_REDUCTIONS = [0.75, 0.78, 0.80]
TAU_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _wrap(pol):
    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    act.reset = getattr(pol, "reset", lambda: None)
    return act


def _cfg(root: Path) -> dict:
    return load_yaml(root / "configs" / "rl_cmappo.yaml")


def _kl_ckpt(root: Path, seed: int) -> Path:
    return root / "results" / "rl_final" / "cmappo_kl" / f"seed_{seed}" / "best_model.pt"


def _bc_ckpt(root: Path, seed: int) -> Path | None:
    for p in [
        root / "results" / "rl_final" / "paper_asap" / "bc" / f"seed_{seed}" / "final_model.pt",
        root / "results" / "rl_final" / "paper_asap" / "bc" / f"seed_{seed}" / "best_model.pt",
        root / "results" / "rl_final" / "bc" / f"seed_{seed}" / "final_model.pt",
    ]:
        if p.exists():
            return p
    return None


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_master(path: Path) -> pd.DataFrame:
    if path.exists():
        return backfill_master(pd.read_csv(path))
    return pd.DataFrame(columns=MASTER_COLUMNS)


def _save_master(df: pd.DataFrame, path: Path) -> None:
    df = backfill_master(df)
    df = df.reindex(columns=MASTER_COLUMNS)
    df.to_csv(path, index=False)


def _done(df: pd.DataFrame, row: dict) -> bool:
    if df.empty:
        return False
    k = row_key(row)
    for rec in df.to_dict("records"):
        if row_key(rec) == k:
            return True
    return False


def _append(master_path: Path, row: dict) -> pd.DataFrame:
    df = _load_master(master_path)
    if _done(df, row):
        return df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_master(df, master_path)
    return df


def run_eval(
    *,
    root: Path,
    cfg: dict,
    act,
    split: str,
    cohort: str,
    sensor_ids: list[str] | None,
    seed: int,
    packet_loss: float,
    packet_loss_mode: str,
    max_steps: int | None,
    packet_loss_mask_seed: int | None = None,
) -> dict:
    kwargs = {}
    if packet_loss > 0.0:
        kwargs.update(
            packet_loss_rate=float(packet_loss),
            packet_loss_mode=packet_loss_mode,
            packet_loss_mask_seed=int(
                PACKET_LOSS_MASK_SEED if packet_loss_mask_seed is None else packet_loss_mask_seed
            ),
        )
    env = make_final_env(
        split=split,
        cfg=cfg,
        multi_agent=True,
        sensor_ids=sensor_ids,
        **kwargs,
    )
    return evaluate_policy(env, act, max_steps=max_steps, seed=seed)


def timed_record(
    *,
    master_path: Path,
    timings: list[dict],
    scenario: str,
    split: str,
    cohort: str,
    method: str,
    configuration: str,
    run: str,
    checkpoint: str,
    packet_loss: float,
    packet_loss_mode: str,
    notes: str,
    raw: dict,
    source: str = "revision_eval",
    aggregation: str = "single_run",
    target_reduction=None,
    policy_run: str = "",
    loss_mask_id=None,
    result_source: str = "",
    prob_threshold=None,
) -> dict:
    row = standardize_eval(
        raw,
        scenario=scenario,
        split=split,
        cohort=cohort,
        method=method,
        configuration=configuration,
        run=run,
        checkpoint=checkpoint,
        packet_loss=packet_loss,
        target_reduction=target_reduction,
        aggregation=aggregation,
        source=source,
        packet_loss_mode=packet_loss_mode,
        notes=notes,
        policy_run=policy_run,
        loss_mask_id=loss_mask_id,
        result_source=result_source,
        prob_threshold=prob_threshold,
    )
    _append(master_path, row)
    return row


def eval_named(args, master_path, timings, **kw) -> dict | None:
    df = _load_master(master_path)
    probe = {
        "scenario": kw["scenario"],
        "split": kw["split"],
        "cohort": kw["cohort"],
        "method": kw["method"],
        "configuration": kw["configuration"],
        "run": str(kw["run"]),
        "packet_loss": float(kw.get("packet_loss", 0.0)),
        "target_reduction": kw.get("target_reduction", float("nan")),
        "loss_mask_id": kw.get("loss_mask_id", float("nan")),
        "prob_threshold": kw.get("prob_threshold", float("nan")),
    }
    if _done(df, probe) and not args.force:
        print(
            f"  [skip] {probe['method']} {probe['configuration']} {probe['split']} "
            f"loss={probe['packet_loss']} mask={probe['loss_mask_id']} tau={probe['prob_threshold']}"
        )
        return None
    t0 = time.perf_counter()
    raw = run_eval(
        root=args.root,
        cfg=kw["cfg"],
        act=kw["act"],
        split=kw["split"],
        cohort=kw["cohort"],
        sensor_ids=kw.get("sensor_ids"),
        seed=int(kw.get("eval_seed", 42)),
        packet_loss=float(kw.get("packet_loss", 0.0)),
        packet_loss_mode=kw.get("packet_loss_mode", "none"),
        max_steps=args.max_steps,
        packet_loss_mask_seed=kw.get("packet_loss_mask_seed"),
    )
    dt = time.perf_counter() - t0
    row = timed_record(
        master_path=master_path,
        timings=timings,
        scenario=kw["scenario"],
        split=kw["split"],
        cohort=kw["cohort"],
        method=kw["method"],
        configuration=kw["configuration"],
        run=str(kw["run"]),
        checkpoint=str(kw.get("checkpoint", "")),
        packet_loss=float(kw.get("packet_loss", 0.0)),
        packet_loss_mode=kw.get("packet_loss_mode", "none"),
        notes=kw.get("notes", ""),
        raw=raw,
        source=kw.get("source", "revision_eval"),
        aggregation=kw.get("aggregation", "single_run"),
        target_reduction=kw.get("target_reduction"),
        policy_run=str(kw.get("policy_run", kw["run"])),
        loss_mask_id=kw.get("loss_mask_id"),
        result_source=kw.get("result_source", "new_evaluation"),
        prob_threshold=kw.get("prob_threshold"),
    )
    timings.append(
        {
            "method": kw["method"],
            "configuration": kw["configuration"],
            "scenario": kw["scenario"],
            "split": kw["split"],
            "packet_loss": kw.get("packet_loss", 0.0),
            "loss_mask_id": kw.get("loss_mask_id", ""),
            "prob_threshold": kw.get("prob_threshold", ""),
            "seconds": dt,
        }
    )
    mae = row["mae"]
    mae_s = "NA" if not np.isfinite(mae) else f"{mae:.3f}"
    print(
        f"  {kw['method']:22s} {kw['configuration']:28s} {kw['split']:4s} "
        f"loss={kw.get('packet_loss', 0):.1f}  "
        f"TXred={100*row['transmission_reduction']:.2f}% MAE={mae_s} "
        f"R={row['recall']:.4f}  ({dt:.1f}s)"
    )
    return row


def fixed_policies() -> list[tuple[str, str, FixedIntervalPolicy]]:
    out = []
    for minutes in FIXED_MINUTES:
        steps = minutes // 15
        name = f"fixed_{minutes}"
        cfg = f"interval_steps={steps} ({minutes} min)"
        out.append((name, cfg, FixedIntervalPolicy(interval_steps=steps, name=name)))
    return out


def import_legacy_bc(root: Path, master_path: Path) -> None:
    """Import frozen BC rows when checkpoints are absent. Do not retrain."""
    notes = "Imported from frozen CSV; BC checkpoints missing in this clone."
    held_raw = root / "results" / "rl_final" / "scientific_validation" / "heldout_transfer_raw.csv"
    if held_raw.exists():
        raw = pd.read_csv(held_raw)
        sub = raw[raw["method"] == "campus_senserl_bc"]
        for rec in sub.to_dict("records"):
            row = standardize_eval(
                rec,
                scenario="heldout_sensors" if rec.get("cohort") == "heldout" else "unknown",
                split=str(rec.get("split")),
                cohort=str(rec.get("cohort", "heldout")),
                method="campus_senserl_bc",
                configuration="tau=0.5",
                run=str(int(rec["seed"])),
                checkpoint="legacy_frozen_csv",
                packet_loss=0.0,
                source="legacy_frozen_csv",
                packet_loss_mode="none",
                notes=notes,
                result_source="legacy_frozen",
                policy_run=str(int(rec["seed"])),
            )
            _append(master_path, row)

    for split, path in [
        ("val", root / "results" / "rl_final" / "paper_final" / "full_val_summary.csv"),
        ("test", root / "results" / "rl_final" / "paper_final" / "full_test_summary.csv"),
    ]:
        if not path.exists():
            continue
        s = pd.read_csv(path)
        hit = s[s["method"] == "campus_senserl_bc"]
        if hit.empty:
            continue
        r = hit.iloc[0]
        fake = {
            "n_locally_available": np.nan,
            "n_tx": np.nan,
            "tx_delivered": np.nan,
            "mae_skipped": float(r["mae_mean"]),
            "event_precision": float(r["precision_mean"]),
            "event_recall": float(r["recall_mean"]),
            "event_f1": float(r["f1_mean"]),
            "mean_aoi_raw": float(r["aoi_raw_mean"]),
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "n_true_events": np.nan,
        }
        # n_available missing in summary — store reduction from percent
        row = standardize_eval(
            fake,
            scenario="validation_cohort" if split == "val" else "frozen_temporal_test",
            split=split,
            cohort="final",
            method="campus_senserl_bc",
            configuration="tau=0.5 (legacy mean of 5)",
            run="mean_of_5",
            checkpoint="legacy_frozen_summary",
            packet_loss=0.0,
            aggregation="mean_of_n",
            source="legacy_frozen_summary",
            packet_loss_mode="none",
            notes=notes + " Counts (n_available/n_tx) unavailable in summary.",
            result_source="legacy_frozen",
            policy_run="mean_of_5",
        )
        row["transmission_reduction"] = float(r["tx_reduction_mean"]) / 100.0
        row["transmission_rate"] = 1.0 - row["transmission_reduction"]
        _append(master_path, row)

    rob = root / "results" / "rl_final" / "scientific_validation" / "robustness_summary.csv"
    if rob.exists():
        rdf = pd.read_csv(rob)
        sub = rdf[(rdf["method"] == "campus_senserl_bc") & (rdf["condition"] == "packet_loss")]
        for rec in sub.to_dict("records"):
            fake = {
                "mae_skipped": float(rec["mae_mean"]),
                "event_recall": float(rec["recall_mean"]),
                "mean_aoi_raw": float(rec["aoi_raw_mean"]),
                "n_locally_available": np.nan,
                "n_tx": np.nan,
                "tx_delivered": np.nan,
            }
            row = standardize_eval(
                fake,
                scenario="packet_loss",
                split="val",
                cohort="final",
                method="campus_senserl_bc",
                configuration="tau=0.5 (legacy mean of 5)",
                run="mean_of_5",
                checkpoint="legacy_frozen_summary",
                packet_loss=float(rec["level"]),
                aggregation="mean_of_n",
                source="legacy_frozen_summary",
                packet_loss_mode="per_attempt_legacy",
                notes=notes + " Legacy per-attempt loss; not the shared slot mask. Not used in the new packet-loss figure.",
                result_source="legacy_frozen",
                policy_run="mean_of_5",
            )
            row["transmission_reduction"] = float(rec["tx_reduction_mean"]) / 100.0
            row["transmission_rate"] = 1.0 - row["transmission_reduction"]
            _append(master_path, row)


def import_matched_budget(root: Path, out_dir: Path, master_path: Path) -> None:
    src = root / "results" / "rl_final" / "scientific_validation" / "matched_budget_summary.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    df.to_csv(out_dir / "matched_budget_legacy_val.csv", index=False)
    for rec in df.to_dict("records"):
        fake = {
            "mae_skipped": float(rec["mae_mean"]),
            "event_recall": float(rec["recall_mean"]),
            "event_precision": float(rec.get("precision_mean", np.nan)),
            "mean_aoi_raw": float(rec.get("aoi_raw_mean", rec.get("aoi_mean", np.nan))),
        }
        row = standardize_eval(
            fake,
            scenario="matched_budget",
            split="val",
            cohort="final",
            method=str(rec["method"]),
            configuration=f"interpolated_target={float(rec['target_tx_reduction']):.1f}%",
            run="mean_of_5",
            checkpoint="legacy_matched_budget_summary",
            packet_loss=0.0,
            target_reduction=float(rec["target_tx_reduction"]) / 100.0,
            aggregation="mean_of_n",
            source="legacy_frozen_summary",
            packet_loss_mode="none",
            notes=(
                "VAL threshold interpolation from frozen scientific_validation CSV (ddof=0 in source). "
                "Not a direct rollout. Non-primary for KL; BC-only interpolated estimate."
            ),
            result_source="interpolated_legacy",
            policy_run="mean_of_5",
        )
        row["transmission_reduction"] = float(rec["target_tx_reduction"]) / 100.0
        row["transmission_rate"] = 1.0 - row["transmission_reduction"]
        _append(master_path, row)


def export_ablation(root: Path, out_dir: Path) -> None:
    src = root / "results" / "rl_final" / "scientific_validation" / "ablation_kl_cmappo_val.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    df["aggregation"] = "single_run"
    df["n_runs"] = 1
    df["notes"] = (
        "Existing seed-42 ablation. Single-run; no significance claims. "
        "Not retrained. Checkpoint paths may refer to the original machine."
    )
    df.to_csv(out_dir / "ablation.csv", index=False)


def copy_val_tau50_to_sweep(master_path: Path) -> None:
    """Reuse frozen VAL tau=0.5 KL rows instead of re-evaluating them."""
    df = _load_master(master_path)
    src = df[
        (df["scenario"] == "validation_cohort")
        & (df["method"] == "cmappo_kl")
        & (df["split"] == "val")
        & np.isclose(pd.to_numeric(df["packet_loss"], errors="coerce").fillna(0.0), 0.0)
        & df["configuration"].astype(str).str.contains(r"tau=0\.5", regex=True)
        & (df["source"].astype(str) == "revision_eval")
    ]
    for rec in src.to_dict("records"):
        row = dict(rec)
        row["scenario"] = "matched_budget_tau_sweep"
        row["configuration"] = "tau=0.50"
        row["prob_threshold"] = 0.5
        row["target_reduction"] = float("nan")
        row["result_source"] = "direct_measured"
        row["notes"] = "Copied from VAL tau=0.5 frozen-actor eval; not interpolated."
        row["policy_run"] = str(row.get("policy_run") or row.get("run"))
        _append(master_path, row)


def select_direct_matched_budget(master_path: Path, out_dir: Path) -> None:
    df = _load_master(master_path)
    sweep = df[df["scenario"] == "matched_budget_tau_sweep"].copy()
    if sweep.empty:
        return
    sweep["prob_threshold"] = pd.to_numeric(sweep["prob_threshold"], errors="coerce")
    sweep.to_csv(out_dir / "matched_budget_tau_grid.csv", index=False)
    means = sweep.groupby("prob_threshold", as_index=True)["transmission_reduction"].mean()
    chosen = []
    for target in TARGET_TX_REDUCTIONS:
        if means.empty:
            break
        tau_star = float((means - target).abs().idxmin())
        gap = float(abs(means.loc[tau_star] - target))
        rows = sweep[np.isclose(sweep["prob_threshold"].astype(float), tau_star)]
        for rec in rows.to_dict("records"):
            row = dict(rec)
            row["scenario"] = "matched_budget"
            row["target_reduction"] = float(target)
            row["result_source"] = "direct_measured"
            row["configuration"] = f"tau={tau_star:.2f} (nearest to {100*target:.0f}%)"
            row["notes"] = (
                f"Direct VAL rollout at measured tau={tau_star:.2f}; "
                f"mean TX reduction={100*float(means.loc[tau_star]):.2f}% "
                f"(gap {100*gap:.2f} pp to target {100*target:.0f}%). No MAE/recall interpolation."
            )
            _append(master_path, row)
        chosen.append(
            {
                "target_tx_reduction": target,
                "selected_tau": tau_star,
                "measured_mean_tx_reduction": float(means.loc[tau_star]),
                "abs_gap": gap,
                "n_seeds": int(len(rows)),
            }
        )
    save_json({"kl": "direct", "bc": "interpolated_legacy", "selections": chosen}, out_dir / "matched_budget_selection.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--skip-packet-loss", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-matched-budget", action="store_true")
    parser.add_argument("--n-loss-masks", type=int, default=N_LOSS_MASKS)
    args = parser.parse_args()
    args.root = repo_root()
    args.device = _device(args.device)
    cfg = _cfg(args.root)
    out = ensure_dir(args.root / "results" / "revision")
    master_path = out / "all_method_evaluations.csv"
    if master_path.exists():
        _save_master(_load_master(master_path), master_path)
    write_protocol_json(out)
    timings: list[dict] = []
    t_all = time.perf_counter()

    print(f"[revision] device={args.device} max_steps={args.max_steps}")
    final_ids = load_cohort("final")
    held_ids = load_cohort("heldout")
    print(f"[revision] final cohort n={len(final_ids)} heldout n={len(held_ids)}")
    overlap = set(final_ids) & set(held_ids)
    if overlap:
        raise RuntimeError(f"Held-out sensors overlap the development cohort: {len(overlap)}")

    kl_ckpts = {s: _kl_ckpt(args.root, s) for s in SEEDS if _kl_ckpt(args.root, s).exists()}
    bc_ckpts = {s: p for s in SEEDS if (p := _bc_ckpt(args.root, s)) is not None}
    print(f"[revision] KL checkpoints: {sorted(kl_ckpts)}")
    print(f"[revision] BC checkpoints: {sorted(bc_ckpts) if bc_ckpts else 'NONE (will import frozen CSVs)'}")

    common = dict(cfg=cfg, sensor_ids=final_ids, master_path=master_path, timings=timings)

    # ---- Validation: periodic + expert + delta grid ----
    print("[stage] validation deterministic methods")
    t_stage = time.perf_counter()
    for name, conf, pol in fixed_policies():
        eval_named(
            args,
            **common,
            act=_wrap(pol),
            scenario="validation_cohort",
            split="val",
            cohort="final",
            method=name,
            configuration=conf,
            run="deterministic",
            packet_loss=0.0,
            packet_loss_mode="none",
            notes="Periodic TX on locally available slots only",
        )
    eval_named(
        args,
        **common,
        act=_wrap(SemanticExpertPolicy()),
        scenario="validation_cohort",
        split="val",
        cohort="final",
        method="semantic_expert",
        configuration="frozen_expert_thresholds",
        run="deterministic",
        packet_loss=0.0,
        packet_loss_mode="none",
        notes="Frozen semantic expert; not retuned",
    )
    for d in DELTA_PPM:
        for h in HEARTBEAT:
            eval_named(
                args,
                **common,
                act=_wrap(DeltaPlusHeartbeatPolicy(delta_ppm=d, aoi_threshold=h)),
                scenario="validation_cohort",
                split="val",
                cohort="final",
                method="delta_plus_heartbeat",
                configuration=f"delta_ppm={d:g},heartbeat={h:g}",
                run="deterministic",
                packet_loss=0.0,
                packet_loss_mode="none",
                notes="VAL grid; selection uses this split only",
            )
    timings.append({"stage": "val_baselines", "seconds": time.perf_counter() - t_stage})

    # ---- Validation: KL (and BC if present) ----
    print("[stage] validation KL-CMAPPO frozen runs")
    t_stage = time.perf_counter()
    if not args.skip_kl:
        for seed, ckpt in kl_ckpts.items():
            act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
            eval_named(
                args,
                **common,
                act=act,
                scenario="validation_cohort",
                split="val",
                cohort="final",
                method="cmappo_kl",
                configuration="tau=0.5",
                run=str(seed),
                checkpoint=str(ckpt),
                eval_seed=seed,
                packet_loss=0.0,
                packet_loss_mode="none",
                notes="Frozen final KL-CMAPPO; no retraining",
            )
        for seed, ckpt in bc_ckpts.items():
            act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
            eval_named(
                args,
                **common,
                act=act,
                scenario="validation_cohort",
                split="val",
                cohort="final",
                method="campus_senserl_bc",
                configuration="tau=0.5",
                run=str(seed),
                checkpoint=str(ckpt),
                eval_seed=seed,
                packet_loss=0.0,
                packet_loss_mode="none",
                notes="BC initialization checkpoint",
            )
    timings.append({"stage": "val_kl", "seconds": time.perf_counter() - t_stage})

    import_legacy_bc(args.root, master_path)

    # ---- Freeze Delta+heartbeat from VAL ----
    master = _load_master(master_path)
    val_delta = master[
        (master["scenario"] == "validation_cohort")
        & (master["method"] == "delta_plus_heartbeat")
        & np.isclose(master["packet_loss"].astype(float), 0.0)
    ]
    selection = select_delta_heartbeat(val_delta)
    write_delta_artifacts(master, out, selection)
    selected_delta_cfg = selection["selected_configuration"]
    print(f"[select] Delta+heartbeat frozen from VAL: {selected_delta_cfg} ({selection['reason']})")
    d_sel = float(selected_delta_cfg.split("delta_ppm=")[1].split(",")[0])
    h_sel = float(selected_delta_cfg.split("heartbeat=")[1])

    def selected_delta():
        return _wrap(DeltaPlusHeartbeatPolicy(delta_ppm=d_sel, aoi_threshold=h_sel))

    # ---- Frozen temporal test ----
    if not args.skip_test:
        print("[stage] frozen temporal test")
        t_stage = time.perf_counter()
        for name, conf, pol in fixed_policies():
            eval_named(
                args,
                **common,
                act=_wrap(pol),
                scenario="frozen_temporal_test",
                split="test",
                cohort="final",
                method=name,
                configuration=conf,
                run="deterministic",
                packet_loss=0.0,
                packet_loss_mode="none",
                notes="Test after VAL freeze; not used for selection",
            )
        eval_named(
            args,
            **common,
            act=_wrap(SemanticExpertPolicy()),
            scenario="frozen_temporal_test",
            split="test",
            cohort="final",
            method="semantic_expert",
            configuration="frozen_expert_thresholds",
            run="deterministic",
            packet_loss=0.0,
            packet_loss_mode="none",
            notes="Test after VAL freeze",
        )
        eval_named(
            args,
            **common,
            act=selected_delta(),
            scenario="frozen_temporal_test",
            split="test",
            cohort="final",
            method="delta_plus_heartbeat",
            configuration=selected_delta_cfg,
            run="deterministic",
            packet_loss=0.0,
            packet_loss_mode="none",
            notes="VAL-selected Delta+heartbeat, frozen before test",
        )
        if not args.skip_kl:
            for seed, ckpt in kl_ckpts.items():
                act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
                note = "Frozen 5-run test aggregate"
                if seed == FROZEN_TEST_SEED:
                    note += f"; seed {FROZEN_TEST_SEED} is the original single-seed paper report"
                eval_named(
                    args,
                    **common,
                    act=act,
                    scenario="frozen_temporal_test",
                    split="test",
                    cohort="final",
                    method="cmappo_kl",
                    configuration="tau=0.5",
                    run=str(seed),
                    checkpoint=str(ckpt),
                    eval_seed=seed,
                    packet_loss=0.0,
                    packet_loss_mode="none",
                    notes=note,
                )
            for seed, ckpt in bc_ckpts.items():
                act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
                eval_named(
                    args,
                    **common,
                    act=act,
                    scenario="frozen_temporal_test",
                    split="test",
                    cohort="final",
                    method="campus_senserl_bc",
                    configuration="tau=0.5",
                    run=str(seed),
                    checkpoint=str(ckpt),
                    eval_seed=seed,
                    packet_loss=0.0,
                    packet_loss_mode="none",
                    notes="BC on frozen test",
                )
        timings.append({"stage": "test", "seconds": time.perf_counter() - t_stage})

    # ---- Held-out sensors (temporal test split of unseen cohort) ----
    if not args.skip_heldout:
        print("[stage] held-out 40-sensor cohort (test split)")
        t_stage = time.perf_counter()
        held_common = dict(common, sensor_ids=held_ids)
        for name, conf, pol in fixed_policies():
            eval_named(
                args,
                **held_common,
                act=_wrap(pol),
                scenario="heldout_sensors",
                split="test",
                cohort="heldout",
                method=name,
                configuration=conf,
                run="deterministic",
                packet_loss=0.0,
                packet_loss_mode="none",
                notes="Unseen sensors; never used for RL development",
            )
        eval_named(
            args,
            **held_common,
            act=_wrap(SemanticExpertPolicy()),
            scenario="heldout_sensors",
            split="test",
            cohort="heldout",
            method="semantic_expert",
            configuration="frozen_expert_thresholds",
            run="deterministic",
            packet_loss=0.0,
            packet_loss_mode="none",
            notes="Unseen sensors",
        )
        eval_named(
            args,
            **held_common,
            act=selected_delta(),
            scenario="heldout_sensors",
            split="test",
            cohort="heldout",
            method="delta_plus_heartbeat",
            configuration=selected_delta_cfg,
            run="deterministic",
            packet_loss=0.0,
            packet_loss_mode="none",
            notes="VAL-selected Delta+heartbeat transferred to unseen sensors",
        )
        if not args.skip_kl:
            for seed, ckpt in kl_ckpts.items():
                act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
                eval_named(
                    args,
                    **held_common,
                    act=act,
                    scenario="heldout_sensors",
                    split="test",
                    cohort="heldout",
                    method="cmappo_kl",
                    configuration="tau=0.5",
                    run=str(seed),
                    checkpoint=str(ckpt),
                    eval_seed=seed,
                    packet_loss=0.0,
                    packet_loss_mode="none",
                    notes="Parameter-shared actor transfer; no held-out tuning",
                )
        timings.append({"stage": "heldout", "seconds": time.perf_counter() - t_stage})

    # ---- Packet loss (VAL, shared slot masks, 5 replicates) ----
    if not args.skip_packet_loss:
        print("[stage] packet-loss (shared independent_slots masks, 5 replicates)")
        t_stage = time.perf_counter()
        pl_methods = []
        pl_methods.append(("fixed_60", "interval_steps=4 (60 min)", _wrap(FixedIntervalPolicy(4, "fixed_60"))))
        pl_methods.append(("fixed_75", "interval_steps=5 (75 min)", _wrap(FixedIntervalPolicy(5, "fixed_75"))))
        pl_methods.append(("delta_plus_heartbeat", selected_delta_cfg, selected_delta()))
        pl_methods.append(("semantic_expert", "frozen_expert_thresholds", _wrap(SemanticExpertPolicy())))
        n_masks_cfg = int(getattr(args, "n_loss_masks", N_LOSS_MASKS))
        for rate in PACKET_LOSS:
            mode = "none" if rate == 0.0 else "independent_slots"
            n_masks = 1 if rate == 0.0 else n_masks_cfg
            for mask_id in range(n_masks):
                mask_seed = PACKET_LOSS_MASK_SEED + int(mask_id)
                for method, conf, act in pl_methods:
                    eval_named(
                        args,
                        **common,
                        act=act,
                        scenario="packet_loss",
                        split="val",
                        cohort="final",
                        method=method,
                        configuration=conf,
                        run="deterministic",
                        policy_run="deterministic",
                        packet_loss=rate,
                        packet_loss_mode=mode,
                        loss_mask_id=mask_id,
                        packet_loss_mask_seed=mask_seed,
                        result_source="new_evaluation",
                        notes=(
                            f"Shared (t,sensor) drop mask id={mask_id} seed={mask_seed}; "
                            "cost counted on attempts"
                        ),
                    )
                if not args.skip_kl:
                    for seed, ckpt in kl_ckpts.items():
                        act = load_mappo_policy(ckpt, device=args.device, prob_threshold=0.5)
                        eval_named(
                            args,
                            **common,
                            act=act,
                            scenario="packet_loss",
                            split="val",
                            cohort="final",
                            method="cmappo_kl",
                            configuration="tau=0.5",
                            run=str(seed),
                            policy_run=str(seed),
                            checkpoint=str(ckpt),
                            eval_seed=42,
                            packet_loss=rate,
                            packet_loss_mode=mode,
                            loss_mask_id=mask_id,
                            packet_loss_mask_seed=mask_seed,
                            result_source="new_evaluation",
                            notes=(
                                f"Shared slot mask id={mask_id} seed={mask_seed}; "
                                "eval seed 42 for mask fairness; policy identity in policy_run"
                            ),
                        )
        timings.append({"stage": "packet_loss", "seconds": time.perf_counter() - t_stage})

    # ---- Direct KL matched-budget tau sweep (VAL) ----
    if not args.skip_matched_budget and not args.skip_kl:
        print("[stage] matched-budget direct VAL tau sweep (KL only)")
        t_stage = time.perf_counter()
        copy_val_tau50_to_sweep(master_path)
        for tau in TAU_GRID:
            if abs(float(tau) - 0.5) < 1e-12:
                continue
            for seed, ckpt in kl_ckpts.items():
                act = load_mappo_policy(ckpt, device=args.device, prob_threshold=float(tau))
                eval_named(
                    args,
                    **common,
                    act=act,
                    scenario="matched_budget_tau_sweep",
                    split="val",
                    cohort="final",
                    method="cmappo_kl",
                    configuration=f"tau={tau:.2f}",
                    run=str(seed),
                    policy_run=str(seed),
                    checkpoint=str(ckpt),
                    eval_seed=seed,
                    packet_loss=0.0,
                    packet_loss_mode="none",
                    prob_threshold=float(tau),
                    result_source="direct_measured",
                    notes="Direct VAL tau sweep on frozen actor; no interpolation",
                )
        select_direct_matched_budget(master_path, out)
        timings.append({"stage": "matched_budget_tau", "seconds": time.perf_counter() - t_stage})

    import_matched_budget(args.root, out, master_path)
    export_ablation(args.root, out)

    master = _load_master(master_path)
    write_protocol_json(out)
    write_delta_artifacts(master, out, selection)
    summary = write_tables(master, out)
    write_claim_audit(summary, out)
    checks = {
        "ok": True,
        "kl_checkpoints": {str(k): str(v) for k, v in kl_ckpts.items()},
        "bc_checkpoints": {str(k): str(v) for k, v in bc_ckpts.items()},
        "frozen_test_seed": FROZEN_TEST_SEED,
        "delta_selection": selected_delta_cfg,
        "packet_loss_mask_seed": PACKET_LOSS_MASK_SEED,
        "n_loss_masks": int(getattr(args, "n_loss_masks", N_LOSS_MASKS)),
        "training_performed": False,
        "n_master_rows": int(len(master)),
        "tau_grid": TAU_GRID,
        "statistical_dispersion": "sample_sd_ddof_1",
    }
    try:
        warnings = validate_master(master)
        checks["warnings"] = warnings
    except RevisionCheckError as e:
        checks["ok"] = False
        checks["error"] = str(e)
        save_json(checks, out / "checks.json")
        raise
    save_json(checks, out / "checks.json")
    write_paper_ready_summary(summary, out, checks)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    print(f"[done] rows={len(master)} elapsed={time.perf_counter()-t_all:.1f}s -> {out}")


if __name__ == "__main__":
    main()
