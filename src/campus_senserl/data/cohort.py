"""Train-only sensor cohort selection for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_senserl.utils import ensure_dir, repo_root, save_json


def compute_sensor_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor coverage and variability using TRAIN rows only for ranking stats.

    Validation/test coverage are reported for eligibility gates but never used
    to rank sensors by test performance.
    """
    rows = []
    for deveui, g in panel.groupby("deveui"):
        def _rate(split: str) -> float:
            s = g[g["split"] == split]
            if len(s) == 0:
                return 0.0
            return float(s["observed"].mean()) if "observed" in s else float(np.isfinite(s["co2"]).mean())

        train = g[g["split"] == "train"]
        train_co2 = train.loc[train["observed"].astype(bool), "co2"] if "observed" in train else train["co2"]
        train_co2 = pd.to_numeric(train_co2, errors="coerce").dropna()
        rows.append(
            {
                "deveui": str(deveui),
                "train_coverage": _rate("train"),
                "val_coverage": _rate("val"),
                "test_coverage": _rate("test"),
                "train_n_obs": int(len(train_co2)),
                "train_std": float(train_co2.std()) if len(train_co2) > 1 else 0.0,
                "train_mean": float(train_co2.mean()) if len(train_co2) else float("nan"),
                "has_floor": bool(pd.notna(g["floor"].iloc[0])) if "floor" in g.columns else False,
            }
        )
    return pd.DataFrame(rows)


def select_cohort(
    coverage: pd.DataFrame,
    *,
    n: int,
    min_train_coverage: float = 0.40,
    min_val_coverage: float = 0.30,
    min_test_coverage: float = 0.30,
    min_train_n_obs: int = 500,
    min_train_std: float = 5.0,
    prefer_node_order: list[str] | None = None,
) -> pd.DataFrame:
    """Select top-n eligible sensors.

    If ``prefer_node_order`` is provided (graph artifact order), walk that list
    and take the first ``n`` eligible IDs so cohort prefixes match
    ``node_order[:n]`` used by reconstruction training.
    Otherwise rank by train coverage (then train_std).
    """
    elig_ids = set(
        coverage[
            (coverage["train_coverage"] >= min_train_coverage)
            & (coverage["val_coverage"] >= min_val_coverage)
            & (coverage["test_coverage"] >= min_test_coverage)
            & (coverage["train_n_obs"] >= min_train_n_obs)
            & (coverage["train_std"] >= min_train_std)
        ]["deveui"].astype(str)
    )
    cov_idx = coverage.set_index("deveui")

    if prefer_node_order:
        chosen = []
        for sid in prefer_node_order:
            sid = str(sid)
            if sid in elig_ids:
                chosen.append(sid)
            if len(chosen) >= int(n):
                break
        selected = cov_idx.loc[chosen].reset_index()
    else:
        elig = coverage[coverage["deveui"].astype(str).isin(elig_ids)].copy()
        elig = elig.sort_values(["train_coverage", "train_std"], ascending=False)
        selected = elig.head(int(n)).copy()
    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected


def freeze_cohorts(
    *,
    debug_n: int = 30,
    development_n: int = 40,
    final_n: int | None = 80,
    panel_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    panel_path = Path(panel_path) if panel_path else root / "data" / "processed" / "co2_panel_15min.parquet"
    out_dir = ensure_dir(out_dir or root / "results" / "cohorts")

    panel = pd.read_parquet(panel_path, columns=["deveui", "split", "observed", "co2", "floor"])
    coverage = compute_sensor_coverage(panel)
    coverage.to_csv(out_dir / "sensor_coverage.csv", index=False)

    node_order_path = root / "results" / "graphs" / "node_order.json"
    prefer = None
    if node_order_path.exists():
        with open(node_order_path, encoding="utf-8") as f:
            raw = json.load(f)
        prefer = [str(x) for x in (raw.get("node_order") if isinstance(raw, dict) else raw)]

    criteria = {
        "min_train_coverage": 0.40,
        "min_val_coverage": 0.30,
        "min_test_coverage": 0.30,
        "min_train_n_obs": 500,
        "min_train_std": 5.0,
        "rank_by": "node_order walk among eligible (aligns with recon graph prefix)",
        "note": "Val/test used only as eligibility gates, never for ranking by accuracy.",
    }
    min_kwargs = {k: v for k, v in criteria.items() if k.startswith("min_")}

    cohorts = {}
    for name, n in [("debug", debug_n), ("development", development_n)]:
        sel = select_cohort(coverage, n=n, prefer_node_order=prefer, **min_kwargs)
        cohorts[name] = {
            "name": name,
            "n_requested": n,
            "n_selected": int(len(sel)),
            "sensor_ids": sel["deveui"].astype(str).tolist(),
            "criteria": criteria,
        }
        sel.to_csv(out_dir / f"cohort_{name}.csv", index=False)

    all_elig = select_cohort(
        coverage,
        n=len(coverage) if prefer is None else len(prefer),
        prefer_node_order=prefer,
        **min_kwargs,
    )
    if final_n is not None:
        all_elig = all_elig.head(int(final_n))
    cohorts["final"] = {
        "name": "final",
        "n_requested": final_n if final_n is not None else int(len(all_elig)),
        "n_selected": int(len(all_elig)),
        "n_eligible_uncapped": int(len(select_cohort(coverage, n=10**9, prefer_node_order=prefer, **min_kwargs))),
        "sensor_ids": all_elig["deveui"].astype(str).tolist(),
        "criteria": criteria,
        "frozen": True,
        "warning": "Do not change after test evaluation begins.",
    }
    all_elig.to_csv(out_dir / "cohort_final.csv", index=False)

    primary = cohorts["final"]
    save_json(primary, out_dir / "final_cohort.json")
    save_json(cohorts, out_dir / "all_cohorts.json")

    coverage = coverage.copy()
    coverage["in_final"] = coverage["deveui"].astype(str).isin(primary["sensor_ids"])
    coverage["exclusion_reason"] = ""
    for i, row in coverage.iterrows():
        if row["in_final"]:
            coverage.at[i, "exclusion_reason"] = "included"
            continue
        reasons = []
        if row["train_coverage"] < criteria["min_train_coverage"]:
            reasons.append("low_train_coverage")
        if row["val_coverage"] < criteria["min_val_coverage"]:
            reasons.append("low_val_coverage")
        if row["test_coverage"] < criteria["min_test_coverage"]:
            reasons.append("low_test_coverage")
        if row["train_n_obs"] < criteria["min_train_n_obs"]:
            reasons.append("few_train_obs")
        if row["train_std"] < criteria["min_train_std"]:
            reasons.append("low_variability")
        if not reasons:
            reasons.append("beyond_final_n_cap_or_node_order")
        coverage.at[i, "exclusion_reason"] = "|".join(reasons)
    coverage.to_csv(out_dir / "inclusion_exclusion.csv", index=False)

    return {"cohorts": cohorts, "out_dir": str(out_dir)}


def load_cohort(name: str = "final") -> list[str]:
    root = repo_root()
    # Dedicated frozen files take precedence
    dedicated = root / "results" / "cohorts" / f"{name}_cohort.json"
    if dedicated.exists():
        with open(dedicated, encoding="utf-8") as f:
            return list(json.load(f)["sensor_ids"])
    path = root / "results" / "cohorts" / "final_cohort.json"
    if name != "final":
        all_path = root / "results" / "cohorts" / "all_cohorts.json"
        with open(all_path, encoding="utf-8") as f:
            data = json.load(f)
        if name in data:
            return list(data[name]["sensor_ids"])
        raise KeyError(f"Unknown cohort {name!r}")
    with open(path, encoding="utf-8") as f:
        return list(json.load(f)["sensor_ids"])


def freeze_heldout_cohort(
    *,
    n: int = 40,
    exclude_cohort: str = "final",
    out_name: str = "heldout",
) -> dict[str, Any]:
    """Freeze n eligible sensors never used in the RL training cohort.

    Selection uses the same train-coverage eligibility gates and node_order walk,
    excluding ``exclude_cohort`` IDs. Do not re-select based on policy performance.
    """
    root = repo_root()
    out_dir = ensure_dir(root / "results" / "cohorts")
    panel = pd.read_parquet(
        root / "data" / "processed" / "co2_panel_15min.parquet",
        columns=["deveui", "split", "observed", "co2", "floor"],
    )
    coverage = compute_sensor_coverage(panel)
    exclude = set(load_cohort(exclude_cohort))
    node_order_path = root / "results" / "graphs" / "node_order.json"
    prefer = None
    if node_order_path.exists():
        with open(node_order_path, encoding="utf-8") as f:
            raw = json.load(f)
        prefer = [str(x) for x in (raw.get("node_order") if isinstance(raw, dict) else raw)]
        prefer = [s for s in prefer if s not in exclude]

    criteria = {
        "min_train_coverage": 0.40,
        "min_val_coverage": 0.30,
        "min_test_coverage": 0.30,
        "min_train_n_obs": 500,
        "min_train_std": 5.0,
    }
    # Mark excluded as ineligible by filtering coverage rows
    cov = coverage[~coverage["deveui"].astype(str).isin(exclude)].copy()
    sel = select_cohort(cov, n=n, prefer_node_order=prefer, **criteria)
    payload = {
        "name": out_name,
        "n_requested": int(n),
        "n_selected": int(len(sel)),
        "sensor_ids": sel["deveui"].astype(str).tolist(),
        "excluded_cohort": exclude_cohort,
        "n_excluded": len(exclude),
        "criteria": criteria,
        "frozen": True,
        "note": "Held-out sensors never used to train/tune RL; parameter-shared actor transfer eval only.",
    }
    save_json(payload, out_dir / f"{out_name}_cohort.json")
    sel.to_csv(out_dir / f"cohort_{out_name}.csv", index=False)
    return payload

