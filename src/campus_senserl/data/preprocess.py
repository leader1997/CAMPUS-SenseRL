"""Reproducible preprocessing pipeline for Oulu Smart Campus traces.

Rules:
- Never overwrite source release files.
- Parse timestamps correctly (epoch ms → UTC).
- Sort chronologically; drop exact duplicates; retain legitimate repeats.
- Distinguish naturally missing values from later RL-induced missingness.
- Fit nothing that leaks from val/test (scalers live in a later stage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_senserl.data.audit import load_devices
from campus_senserl.data.features import add_calendar_features, add_rolling_features
from campus_senserl.data.splits import chronological_split_bounds, assign_split
from campus_senserl.utils import (
    ensure_dir,
    load_yaml,
    normalize_deveui,
    repo_root,
    save_json,
    set_seed,
)


def _release_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = repo_root()
    release = root / cfg["paths"]["raw_release"]
    return {
        "application": release / cfg["dataset"]["application_file"],
        "lora": release / cfg["dataset"]["lora_file"],
        "devices": release / cfg["dataset"]["devices_file"],
    }


def load_application(
    path: Path,
    chunksize: int = 500_000,
    keep_deveuis: set[str] | None = None,
) -> pd.DataFrame:
    """Load and clean application table.

    Optionally filter to `keep_deveuis` early to reduce memory (e.g. CO2 devices).
    """
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk["deveui"] = chunk["deveui"].map(normalize_deveui)
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time", "deveui"])
        if keep_deveuis is not None:
            chunk = chunk[chunk["deveui"].isin(keep_deveuis)]
        if chunk.empty:
            continue
        chunk["time"] = chunk["time"].astype("int64")
        parts.append(chunk)
    if not parts:
        raise RuntimeError("No application rows loaded after filtering.")
    df = pd.concat(parts, ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["time", "deveui"], keep="first")
    n_dup = n_before - len(df)
    df = df.sort_values(["time", "deveui"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df.attrs["n_exact_duplicates_removed"] = int(n_dup)
    return df


def flag_co2_quality(
    df: pd.DataFrame,
    min_ppm: float = 200.0,
    max_ppm: float = 5000.0,
) -> pd.DataFrame:
    """Flag physically implausible CO2 while retaining raw values.

    Extreme outliers (e.g. 65502) appear in the release and are treated as
    sensor faults for modelling, not as true concentrations.
    """
    out = df.copy()
    if "co2" not in out.columns:
        return out
    out["co2_raw"] = out["co2"]
    bad = out["co2"].notna() & ((out["co2"] < min_ppm) | (out["co2"] > max_ppm))
    out["co2_quality_flag"] = np.where(bad, "implausible", "ok")
    # Modelling column: keep plausible values only
    out.loc[bad, "co2"] = np.nan
    out.attrs["n_co2_implausible"] = int(bad.sum())
    return out


def load_lora(
    path: Path,
    chunksize: int = 500_000,
    keep_deveuis: set[str] | None = None,
) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk["deveui"] = chunk["deveui"].map(normalize_deveui)
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time", "deveui"])
        if keep_deveuis is not None:
            chunk = chunk[chunk["deveui"].isin(keep_deveuis)]
        if chunk.empty:
            continue
        chunk["time"] = chunk["time"].astype("int64")
        parts.append(chunk)
    if not parts:
        raise RuntimeError("No lora rows loaded after filtering.")
    df = pd.concat(parts, ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["time", "deveui"], keep="first")
    df = df.sort_values(["time", "deveui"]).reset_index(drop=True)
    df.attrs["n_exact_duplicates_removed"] = int(n_before - len(df))
    return df


def merge_lora(
    app: pd.DataFrame,
    lora: pd.DataFrame,
    tolerance_ms: int = 2000,
) -> pd.DataFrame:
    """Attach LoRa radio metrics to application rows by deveui + nearest time."""
    lora_cols = ["time", "deveui", "rssi", "lsnr", "chan", "port", "rfch", "seqn", "fcnt"]
    lora_s = lora[[c for c in lora_cols if c in lora.columns]].rename(
        columns={"time": "lora_time"}
    )
    # Exact join on (time, deveui) first — timestamps usually match
    merged = app.merge(
        lora_s,
        left_on=["time", "deveui"],
        right_on=["lora_time", "deveui"],
        how="left",
        suffixes=("", "_lora"),
    )
    merged = merged.drop(columns=["lora_time"], errors="ignore")
    merged["lora_matched"] = merged["rssi"].notna().astype("int8")
    return merged


def build_regular_panel(
    df: pd.DataFrame,
    devices: pd.DataFrame,
    rule: str = "15min",
    co2_only: bool = False,
) -> pd.DataFrame:
    """Create a regular time × sensor panel with availability indicators.

    `observed=1` means a real uplink existed in that bin (natural availability).
    RL-induced missingness is NOT applied here — only natural gaps.
    """
    meta = devices.rename(columns={"device_id": "deveui"})
    df = df.merge(meta, on="deveui", how="left")
    if co2_only:
        df = df[df["device_type"] == "Elsys ERS CO2"].copy()

    # Floor timestamps to interval grid
    df = df.copy()
    df["slot"] = df["timestamp"].dt.floor(rule)

    # Aggregate multiple uplinks in same bin (rare): keep last
    value_cols = [
        c
        for c in [
            "temperature",
            "humidity",
            "light",
            "motion",
            "co2",
            "battery",
            "sound_avg",
            "sound_peak",
            "moisture",
            "pressure",
            "rssi",
            "lsnr",
            "chan",
            "seqn",
            "fcnt",
        ]
        if c in df.columns
    ]
    agg = (
        df.sort_values("timestamp")
        .groupby(["slot", "deveui"], as_index=False)
        .agg(
            {
                **{c: "last" for c in value_cols},
                "timestamp": "last",
                "time": "last",
                "device_type": "last",
                "floor": "last",
                "latitude": "last",
                "longitude": "last",
                "desc": "last",
                "status": "last",
            }
        )
    )
    agg["observed"] = 1
    agg["natural_missing"] = 0

    # Full grid
    slots = pd.date_range(agg["slot"].min(), agg["slot"].max(), freq=rule, tz="UTC")
    sensors = agg["deveui"].unique()
    grid = pd.MultiIndex.from_product([slots, sensors], names=["slot", "deveui"]).to_frame(
        index=False
    )
    panel = grid.merge(agg, on=["slot", "deveui"], how="left")
    panel["observed"] = panel["observed"].fillna(0).astype("int8")
    panel["natural_missing"] = (1 - panel["observed"]).astype("int8")
    # Carry metadata for unobserved slots
    meta_map = (
        devices.rename(columns={"device_id": "deveui"})[
            ["deveui", "device_type", "floor", "latitude", "longitude", "desc", "status"]
        ]
        .drop_duplicates("deveui")
    )
    for col in ["device_type", "floor", "latitude", "longitude", "desc", "status"]:
        if col in panel.columns:
            panel = panel.drop(columns=[col])
    panel = panel.merge(meta_map, on="deveui", how="left")
    panel["rl_skipped"] = 0  # placeholder; set only in RL environment
    panel["ground_truth_available"] = panel["observed"]  # for simulator internals
    return panel.sort_values(["slot", "deveui"]).reset_index(drop=True)


def run_preprocess(cfg: dict[str, Any] | None = None, co2_focus: bool = True) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "data.yaml")
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    paths = _release_paths(cfg)

    interim = ensure_dir(root / cfg["paths"]["data_interim"])
    processed = ensure_dir(root / cfg["paths"]["data_processed"])

    print("[preprocess] Loading devices…")
    devices = load_devices(paths["devices"])
    devices.to_parquet(interim / "devices.parquet", index=False)

    co2_ids = set(
        devices.loc[devices["device_type"] == "Elsys ERS CO2", "device_id"].tolist()
    )
    keep_ids = co2_ids if co2_focus else None
    if keep_ids is not None:
        print(f"[preprocess] Restricting event load to {len(keep_ids)} ERS CO2 devices…")

    print("[preprocess] Loading application.csv…")
    app = load_application(paths["application"], keep_deveuis=keep_ids)
    app = flag_co2_quality(
        app,
        min_ppm=float(cfg.get("preprocess", {}).get("co2_min_ppm", 200)),
        max_ppm=float(cfg.get("preprocess", {}).get("co2_max_ppm", 5000)),
    )
    n_co2_implausible = int(app.attrs.get("n_co2_implausible", 0))
    print(
        f"[preprocess] application cleaned rows={len(app):,} "
        f"dups_removed={app.attrs.get('n_exact_duplicates_removed')} "
        f"co2_implausible={n_co2_implausible}"
    )

    print("[preprocess] Loading lora.csv…")
    lora = load_lora(paths["lora"], keep_deveuis=keep_ids)
    print(f"[preprocess] lora cleaned rows={len(lora):,}")

    if cfg.get("preprocess", {}).get("merge_lora", True):
        print("[preprocess] Merging LoRa metrics…")
        app = merge_lora(
            app,
            lora,
            tolerance_ms=int(cfg["preprocess"].get("lora_join_tolerance_ms", 2000)),
        )

    # Save cleaned event-level tables (CO2 subset when co2_focus)
    print("[preprocess] Writing interim event tables (parquet)…")
    app.to_parquet(interim / "application_clean_co2.parquet", index=False)
    lora.to_parquet(interim / "lora_clean_co2.parquet", index=False)

    # Build regular panel for CO2 sensors (primary research subset)
    print("[preprocess] Building 15-min panel (CO2 devices)…")
    panel = build_regular_panel(
        app,
        devices,
        rule=cfg.get("preprocess", {}).get("resample_rule", "15min"),
        co2_only=co2_focus,
    )
    panel = add_calendar_features(panel, time_col="slot")
    panel = add_rolling_features(
        panel,
        value_col="co2",
        group_col="deveui",
        time_col="slot",
        windows=cfg.get("features", {}).get("rolling_windows", [4, 8, 16]),
    )

    # Chronological split assignment on slots
    bounds = chronological_split_bounds(
        panel["slot"],
        train_frac=cfg["splits"]["train_frac"],
        val_frac=cfg["splits"]["val_frac"],
        test_frac=cfg["splits"]["test_frac"],
    )
    panel["split"] = assign_split(panel["slot"], bounds)

    out_panel = processed / "co2_panel_15min.parquet"
    print(f"[preprocess] Writing {out_panel} ({len(panel):,} rows)…")
    panel.to_parquet(out_panel, index=False)

    # Also a lighter wide matrix for reconstruction experiments
    print("[preprocess] Building CO2 wide matrix…")
    observed = panel[panel["observed"] == 1]
    wide = observed.pivot_table(index="slot", columns="deveui", values="co2", aggfunc="last")
    wide_path = processed / "co2_wide_observed.parquet"
    wide.to_parquet(wide_path)

    obs_mask = panel.pivot_table(index="slot", columns="deveui", values="observed", aggfunc="max")
    obs_mask_path = processed / "co2_observed_mask.parquet"
    obs_mask.to_parquet(obs_mask_path)

    meta = {
        "n_application_rows": int(len(app)),
        "n_lora_rows": int(len(lora)),
        "n_panel_rows": int(len(panel)),
        "n_co2_sensors": int(panel["deveui"].nunique()),
        "n_slots": int(panel["slot"].nunique()),
        "split_bounds": {k: str(v) for k, v in bounds.items()},
        "split_counts": panel["split"].value_counts().to_dict(),
        "observed_rate": float(panel["observed"].mean()),
        "co2_present_given_observed": float(
            panel.loc[panel["observed"] == 1, "co2"].notna().mean()
        ),
        "n_co2_implausible": n_co2_implausible,
        "co2_quality_range_ppm": [
            float(cfg.get("preprocess", {}).get("co2_min_ppm", 200)),
            float(cfg.get("preprocess", {}).get("co2_max_ppm", 5000)),
        ],
        "outputs": {
            "panel": str(out_panel.relative_to(root)),
            "wide": str(wide_path.relative_to(root)),
            "observed_mask": str(obs_mask_path.relative_to(root)),
        },
    }
    save_json(meta, processed / "preprocess_summary.json")
    print("[preprocess] Done.")
    return meta


if __name__ == "__main__":
    run_preprocess()
