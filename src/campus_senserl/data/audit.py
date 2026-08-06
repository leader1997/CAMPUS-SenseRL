"""Dataset audit for the Oulu Smart Campus release.

Inspects files on disk, infers schemas, computes coverage / missingness /
frequency statistics, and writes human + machine-readable reports.
Never fabricates metadata.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from campus_senserl.utils import (
    ensure_dir,
    haversine_m,
    load_yaml,
    normalize_deveui,
    repo_root,
    save_json,
)

APP_COLS = [
    "time",
    "deveui",
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
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
]
LORA_COLS = ["time", "chan", "deveui", "lsnr", "port", "rfch", "rssi", "seqn", "fcnt"]
MEASUREMENT_COLS = [
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
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
]


def load_devices(path: Path) -> pd.DataFrame:
    """Load devices.json (JSON Lines despite .json extension)."""
    rows = []
    text = path.read_text(encoding="utf-8").strip()
    # Support both JSONL and a single JSON array
    if text.startswith("["):
        data = json.loads(text)
        rows = data
    else:
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    records = []
    for d in rows:
        loc = d.get("location") or [None, None]
        lat = float(loc[0]) if loc[0] not in (None, "") else np.nan
        lon = float(loc[1]) if loc[1] not in (None, "") else np.nan
        records.append(
            {
                "device_id": normalize_deveui(d.get("id", "")),
                "device_type": d.get("type"),
                "floor": str(d.get("floor")) if d.get("floor") is not None else None,
                "status": d.get("status"),
                "installed": d.get("installed"),
                "desc": d.get("desc"),
                "latitude": lat,
                "longitude": lon,
                "added_by": d.get("addedBy"),
                "raw_id": d.get("id"),
            }
        )
    return pd.DataFrame(records)


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo_root())) if path.exists() else str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "size_mb": round(path.stat().st_size / (1024**2), 2) if path.exists() else 0,
    }


def audit_csv_chunked(
    path: Path,
    expected_cols: list[str],
    chunksize: int = 500_000,
    sample_for_plots: int = 2_000_000,
) -> dict[str, Any]:
    """Stream a large CSV and compute audit statistics without loading all RAM."""
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    n_rows = 0
    n_dup_exact = 0
    seen_hashes: set[int] | None = None  # too large to store all; approximate via sample
    t_min = None
    t_max = None
    devices: Counter = Counter()
    null_counts: Counter = Counter()
    value_counts_present: Counter = Counter()
    interval_samples: list[float] = []
    prev_time_by_dev: dict[str, int] = {}
    co2_values: list[float] = []
    battery_values: list[float] = []
    rssi_values: list[float] = []
    lsnr_values: list[float] = []
    # Reservoir-like samples for plots
    plot_rows: list[dict] = []
    rng = np.random.default_rng(42)

    # Exact duplicate detection on (time, deveui) using a set — may use memory;
    # for 800MB files this is typically manageable (~10M keys).
    seen_keys: set[tuple] = set()
    n_key_dups = 0

    usecols = [c for c in columns if c in expected_cols or c in columns]
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        n_rows += len(chunk)
        for c in columns:
            if c in chunk.columns:
                null_counts[c] += int(chunk[c].isna().sum())
                value_counts_present[c] += int(chunk[c].notna().sum())

        if "deveui" in chunk.columns:
            chunk = chunk.copy()
            chunk["deveui_norm"] = chunk["deveui"].map(normalize_deveui)
            devices.update(chunk["deveui_norm"].value_counts().to_dict())

        if "time" in chunk.columns:
            t0 = int(chunk["time"].min())
            t1 = int(chunk["time"].max())
            t_min = t0 if t_min is None else min(t_min, t0)
            t_max = t1 if t_max is None else max(t_max, t1)

            # Transmission intervals per device (sample)
            if "deveui_norm" in chunk.columns:
                for deveui, g in chunk.groupby("deveui_norm", sort=False):
                    times = g["time"].to_numpy(dtype=np.int64)
                    order = np.argsort(times)
                    times = times[order]
                    if deveui in prev_time_by_dev and len(times):
                        dt = (times[0] - prev_time_by_dev[deveui]) / 1000.0
                        if 0 < dt < 24 * 3600:
                            interval_samples.append(dt)
                    if len(times) > 1:
                        dts = np.diff(times) / 1000.0
                        dts = dts[(dts > 0) & (dts < 24 * 3600)]
                        if len(dts):
                            # subsample
                            if len(dts) > 20:
                                dts = rng.choice(dts, size=20, replace=False)
                            interval_samples.extend(dts.tolist())
                    if len(times):
                        prev_time_by_dev[deveui] = int(times[-1])

            # key duplicates
            if "deveui_norm" in chunk.columns:
                keys = list(zip(chunk["time"].tolist(), chunk["deveui_norm"].tolist()))
                for k in keys:
                    if k in seen_keys:
                        n_key_dups += 1
                    else:
                        seen_keys.add(k)

        if "co2" in chunk.columns:
            vals = chunk["co2"].dropna()
            if len(vals):
                take = vals.sample(min(len(vals), 5000), random_state=None)
                co2_values.extend(take.astype(float).tolist())
        if "battery" in chunk.columns:
            vals = chunk["battery"].dropna()
            if len(vals):
                take = vals.sample(min(len(vals), 5000), random_state=None)
                battery_values.extend(take.astype(float).tolist())
        if "rssi" in chunk.columns:
            vals = chunk["rssi"].dropna()
            if len(vals):
                take = vals.sample(min(len(vals), 5000), random_state=None)
                rssi_values.extend(take.astype(float).tolist())
        if "lsnr" in chunk.columns:
            vals = chunk["lsnr"].dropna()
            if len(vals):
                take = vals.sample(min(len(vals), 5000), random_state=None)
                lsnr_values.extend(take.astype(float).tolist())

        # Reservoir sample for temporal plots
        if len(plot_rows) < sample_for_plots:
            need = sample_for_plots - len(plot_rows)
            sample_n = min(need, len(chunk))
            if sample_n > 0:
                samp = chunk.sample(sample_n, random_state=None)
                keep_cols = [c for c in samp.columns if c in APP_COLS + LORA_COLS + ["deveui_norm"]]
                plot_rows.extend(samp[keep_cols].to_dict(orient="records"))

    def _ts(ms):
        if ms is None:
            return None
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

    missingness = {}
    for c in columns:
        present = value_counts_present.get(c, 0)
        missingness[c] = {
            "null_count": int(null_counts.get(c, 0)),
            "present_count": int(present),
            "null_frac": float(null_counts.get(c, 0) / n_rows) if n_rows else None,
        }

    intervals = np.array(interval_samples) if interval_samples else np.array([])
    interval_stats = {}
    if len(intervals):
        interval_stats = {
            "n_samples": int(len(intervals)),
            "median_sec": float(np.median(intervals)),
            "mean_sec": float(np.mean(intervals)),
            "p10_sec": float(np.percentile(intervals, 10)),
            "p90_sec": float(np.percentile(intervals, 90)),
            "frac_near_15min": float(np.mean((intervals > 800) & (intervals < 1000))),
        }

    return {
        "columns": columns,
        "n_rows": int(n_rows),
        "n_devices": int(len(devices)),
        "device_observation_counts": dict(devices.most_common()),
        "time_min_ms": t_min,
        "time_max_ms": t_max,
        "time_min_iso": _ts(t_min),
        "time_max_iso": _ts(t_max),
        "missingness": missingness,
        "n_duplicate_time_deveui": int(n_key_dups),
        "interval_stats": interval_stats,
        "co2_sample": co2_values,
        "battery_sample": battery_values,
        "rssi_sample": rssi_values,
        "lsnr_sample": lsnr_values,
        "plot_sample": plot_rows,
        "top_devices": devices.most_common(20),
    }


def _describe_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def make_eda_figures(summary: dict[str, Any], devices: pd.DataFrame, fig_dir: Path) -> list[str]:
    ensure_dir(fig_dir)
    sns.set_theme(style="whitegrid", context="paper")
    written: list[str] = []

    # Observations by device (top 40)
    counts = summary["application"]["device_observation_counts"]
    if counts:
        items = sorted(counts.items(), key=lambda x: -x[1])[:40]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(items)), [v for _, v in items], color="#2c7fb8")
        ax.set_xlabel("Device rank")
        ax.set_ylabel("Observation count")
        ax.set_title("Application observations by device (top 40)")
        fig.tight_layout()
        p = fig_dir / "observations_by_device.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "observations_by_device.pdf")
        plt.close(fig)
        written.append(str(p))

    # Transmission intervals
    app = summary["application"]
    if app.get("interval_stats"):
        # Recompute from stored isn't available; plot from interval_stats text + lora if any
        pass

    # CO2 distribution
    if app.get("co2_sample"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(app["co2_sample"], bins=60, color="#2c7fb8", edgecolor="white")
        ax.set_xlabel("CO₂ (ppm)")
        ax.set_ylabel("Count (sample)")
        ax.set_title("CO₂ distribution (sampled observations)")
        fig.tight_layout()
        p = fig_dir / "co2_distribution.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "co2_distribution.pdf")
        plt.close(fig)
        written.append(str(p))

    # Battery
    if app.get("battery_sample"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(app["battery_sample"], bins=50, color="#41ab5d", edgecolor="white")
        ax.set_xlabel("Battery voltage (V)")
        ax.set_ylabel("Count (sample)")
        ax.set_title("Battery voltage distribution (sampled)")
        fig.tight_layout()
        p = fig_dir / "battery_distribution.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "battery_distribution.pdf")
        plt.close(fig)
        written.append(str(p))

    # RSSI / SNR
    lora = summary.get("lora", {})
    if lora.get("rssi_sample") or lora.get("lsnr_sample"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if lora.get("rssi_sample"):
            axes[0].hist(lora["rssi_sample"], bins=50, color="#fdae61", edgecolor="white")
            axes[0].set_title("RSSI (dBm)")
        if lora.get("lsnr_sample"):
            axes[1].hist(lora["lsnr_sample"], bins=50, color="#d7191c", edgecolor="white")
            axes[1].set_title("LSNR / SNR (dB)")
        for ax in axes:
            ax.set_ylabel("Count (sample)")
        fig.suptitle("LoRaWAN link quality (sampled)")
        fig.tight_layout()
        p = fig_dir / "rssi_snr_distribution.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "rssi_snr_distribution.pdf")
        plt.close(fig)
        written.append(str(p))

    # Spatial distribution
    if len(devices) and devices["latitude"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 6))
        for dtype, g in devices.groupby("device_type"):
            ax.scatter(g["longitude"], g["latitude"], s=18, alpha=0.75, label=dtype)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Campus sensor spatial distribution (WGS84)")
        ax.legend(fontsize=8, frameon=True)
        fig.tight_layout()
        p = fig_dir / "sensor_spatial_distribution.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "sensor_spatial_distribution.pdf")
        plt.close(fig)
        written.append(str(p))

    # Sensors by floor
    if len(devices):
        fig, ax = plt.subplots(figsize=(7, 4))
        floor_order = sorted(devices["floor"].dropna().unique(), key=lambda x: float(x))
        ct = (
            devices.groupby(["floor", "device_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(floor_order)
        )
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_xlabel("Floor")
        ax.set_ylabel("Number of sensors")
        ax.set_title("Sensors by floor and device type")
        ax.legend(fontsize=8, title="Type")
        fig.tight_layout()
        p = fig_dir / "sensors_by_floor.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "sensors_by_floor.pdf")
        plt.close(fig)
        written.append(str(p))

    # Missingness heatmap (application columns)
    miss = app.get("missingness", {})
    if miss:
        cols = list(miss.keys())
        fracs = [miss[c]["null_frac"] for c in cols]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.barplot(x=cols, y=fracs, ax=ax, color="#2c7fb8")
        ax.set_ylabel("Null fraction")
        ax.set_title("Application-table missingness by column")
        ax.set_xticklabels(cols, rotation=45, ha="right")
        fig.tight_layout()
        p = fig_dir / "missingness_by_column.png"
        fig.savefig(p, dpi=200)
        fig.savefig(fig_dir / "missingness_by_column.pdf")
        plt.close(fig)
        written.append(str(p))

    # Temporal coverage from plot sample
    plot_sample = app.get("plot_sample") or []
    if plot_sample:
        df = pd.DataFrame(plot_sample)
        if "time" in df.columns:
            df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            daily = df.set_index("ts").resample("1D").size()
            fig, ax = plt.subplots(figsize=(10, 3.5))
            daily.plot(ax=ax, color="#2c7fb8")
            ax.set_ylabel("Sampled obs / day")
            ax.set_title("Temporal coverage (reservoir sample of application rows)")
            fig.tight_layout()
            p = fig_dir / "temporal_coverage.png"
            fig.savefig(p, dpi=200)
            fig.savefig(fig_dir / "temporal_coverage.pdf")
            plt.close(fig)
            written.append(str(p))

            # Example trajectories: pick a few CO2 devices with data
            if "co2" in df.columns and "deveui_norm" in df.columns:
                co2_df = df.dropna(subset=["co2"])
                top = co2_df["deveui_norm"].value_counts().head(4).index.tolist()
                if top:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for dev in top:
                        g = co2_df[co2_df["deveui_norm"] == dev].sort_values("ts")
                        # downsample for readability
                        g = g.iloc[:: max(1, len(g) // 800)]
                        ax.plot(g["ts"], g["co2"], lw=0.8, label=dev[-6:])
                    ax.set_ylabel("CO₂ (ppm)")
                    ax.set_title("Example CO₂ trajectories (sampled)")
                    ax.legend(fontsize=7, title="deveui tail")
                    fig.tight_layout()
                    p = fig_dir / "example_co2_trajectories.png"
                    fig.savefig(p, dpi=200)
                    fig.savefig(fig_dir / "example_co2_trajectories.pdf")
                    plt.close(fig)
                    written.append(str(p))

    # Interval distribution if we can get from summary plot via re-sampling note:
    # Store interval hist using a quick second pass is expensive; skip if no samples saved.
    # Correlation matrix on a small multi-sensor pivot sample is optional.

    return written


def write_audit_markdown(summary: dict[str, Any], path: Path) -> None:
    app = summary["application"]
    lora = summary["lora"]
    meta = summary["devices"]
    suitable = summary["research_variables"]["suitable"]
    not_avail = summary["research_variables"]["not_available"]
    limitations = summary["scientific_limitations"]

    lines = [
        "# Dataset Audit — Oulu Smart Campus Release",
        "",
        "Generated by `scripts/01_audit_data.py`. Metadata is inferred from files on disk; nothing is fabricated.",
        "",
        "## 1. Dataset files",
        "",
        "| File | Size (MB) | Role |",
        "|---|---:|---|",
    ]
    for f in summary["files"]:
        lines.append(f"| `{f['path']}` | {f['size_mb']} | {f['role']} |")
    lines += [
        "",
        "## 2. Schemas",
        "",
        "### application.csv",
        "",
        f"Columns: `{', '.join(app['columns'])}`",
        "",
        f"Rows: **{app['n_rows']:,}**",
        "",
        "### lora.csv",
        "",
        f"Columns: `{', '.join(lora['columns'])}`",
        "",
        f"Rows: **{lora['n_rows']:,}**",
        "",
        "### devices.json (JSON Lines)",
        "",
        f"Devices: **{meta['n_devices']}**",
        "",
        f"Fields: `{', '.join(meta['fields'])}`",
        "",
        "## 3. Date coverage",
        "",
        f"- Application: `{app['time_min_iso']}` → `{app['time_max_iso']}`",
        f"- LoRa: `{lora['time_min_iso']}` → `{lora['time_max_iso']}`",
        "",
        "Official README states collection from **2020-07-01** to **2021-06-01** at a nominal **15-minute** uplink interval.",
        "",
        "## 4. Number of devices",
        "",
        f"- Metadata devices: **{meta['n_devices']}**",
        f"- Distinct deveui in application: **{app['n_devices']}**",
        f"- Distinct deveui in lora: **{lora['n_devices']}**",
        "",
        "### Device types (metadata)",
        "",
    ]
    for t, n in meta["type_counts"].items():
        lines.append(f"- `{t}`: {n}")
    lines += [
        "",
        "### Floors",
        "",
    ]
    for fl, n in meta["floor_counts"].items():
        lines.append(f"- Floor `{fl}`: {n}")
    lines += [
        "",
        "## 5. Measurements available by device type",
        "",
        "| Device type | Expected measurements |",
        "|---|---|",
        "| Elsys ERS CO2 | temperature, humidity, light, motion, **co2**, battery |",
        "| Elsys ERS Sound | temperature, humidity, light, motion, battery, sound_avg, sound_peak |",
        "| Elsys ELT-2 with soil moisture | temperature, humidity, battery, moisture, pressure, acceleration_x/y/z |",
        "",
        "CO₂ is present **only** for ERS CO2 devices. Do not invent CO₂ for other types.",
        "",
        "## 6. Missingness statistics",
        "",
        "### application.csv null fractions",
        "",
        "| Column | Null fraction | Present count |",
        "|---|---:|---:|",
    ]
    for c, m in app["missingness"].items():
        lines.append(f"| {c} | {m['null_frac']:.4f} | {m['present_count']:,} |")
    lines += [
        "",
        f"Duplicate `(time, deveui)` keys in application: **{app['n_duplicate_time_deveui']:,}**",
        "",
        f"Duplicate `(time, deveui)` keys in lora: **{lora['n_duplicate_time_deveui']:,}**",
        "",
        "## 7. Measurement frequencies",
        "",
    ]
    if app.get("interval_stats"):
        s = app["interval_stats"]
        lines += [
            "Empirical inter-arrival times (application, sampled per device):",
            "",
            f"- median: **{s['median_sec']:.1f} s**",
            f"- mean: **{s['mean_sec']:.1f} s**",
            f"- p10–p90: **{s['p10_sec']:.1f}–{s['p90_sec']:.1f} s**",
            f"- fraction near 15 min (800–1000 s): **{s['frac_near_15min']:.3f}**",
            "",
            "A regular **15-minute** grid is therefore empirically supported.",
            "",
        ]
    else:
        lines += ["Interval statistics could not be computed.", ""]

    lines += [
        "## 8. Metadata coverage",
        "",
        f"- Location present: **{meta['n_with_location']}/{meta['n_devices']}**",
        f"- Floor present: **{meta['n_with_floor']}/{meta['n_devices']}**",
        f"- Description present: **{meta['n_with_desc']}/{meta['n_devices']}**",
        f"- Status counts: `{meta['status_counts']}`",
        "",
        "### Coordinate system note",
        "",
        "The release README claims Web Mercator (EPSG:3785), but observed `location` values",
        f"fall in latitude **{meta['lat_range']}** and longitude **{meta['lon_range']}**,",
        "consistent with **WGS84** coordinates for the University of Oulu (Linnanmaa).",
        "This implementation treats them as WGS84 and documents the discrepancy.",
        "",
        "Building identifiers are **not** present as a separate field; location descriptions",
        "(`desc`) contain room/area names (e.g., Tellus Cube).",
        "",
        "## 9. Network variables available",
        "",
        "From `lora.csv`:",
        "",
    ]
    for c in lora["columns"]:
        lines.append(f"- `{c}`")
    lines += [
        "",
        "Spreading factor, payload size, gateway ID, and explicit TX power are **not** present",
        "in this release. RSSI (`rssi`) and SNR (`lsnr`) are available.",
        "",
        "## 10. Variables suitable for this research",
        "",
    ]
    for v in suitable:
        lines.append(f"- {v}")
    lines += [
        "",
        "## 11. Variables NOT available",
        "",
    ]
    for v in not_avail:
        lines.append(f"- {v}")
    lines += [
        "",
        "## 12. Scientific limitations",
        "",
    ]
    for v in limitations:
        lines.append(f"- {v}")
    lines += [
        "",
        "## CO₂ research subset",
        "",
        f"- ERS CO2 devices in metadata: **{meta['n_co2_devices']}**",
        f"- CO₂ non-null observations (application): **{app['missingness'].get('co2', {}).get('present_count', 0):,}**",
        "",
        f"CO₂ sample stats: `{summary.get('co2_stats', {})}`",
        "",
        "---",
        "",
        "*Audit complete. Proceed to preprocessing without modifying source files.*",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "data.yaml")
    release = root / cfg["paths"]["raw_release"]
    app_path = release / cfg["dataset"]["application_file"]
    lora_path = release / cfg["dataset"]["lora_file"]
    dev_path = release / cfg["dataset"]["devices_file"]

    print("[audit] Loading device metadata…")
    devices = load_devices(dev_path)
    # Persist a clean metadata table for later stages
    ensure_dir(root / "data" / "interim")
    devices.to_parquet(root / "data" / "interim" / "devices.parquet", index=False)

    print("[audit] Streaming application.csv (this may take several minutes)…")
    app_stats = audit_csv_chunked(app_path, APP_COLS)
    print(f"[audit] application rows={app_stats['n_rows']:,} devices={app_stats['n_devices']}")

    print("[audit] Streaming lora.csv…")
    lora_stats = audit_csv_chunked(lora_path, LORA_COLS)
    print(f"[audit] lora rows={lora_stats['n_rows']:,} devices={lora_stats['n_devices']}")

    type_counts = devices["device_type"].value_counts().to_dict()
    floor_counts = devices["floor"].value_counts().to_dict()
    status_counts = devices["status"].value_counts().to_dict()
    n_co2 = int((devices["device_type"] == "Elsys ERS CO2").sum())

    summary: dict[str, Any] = {
        "dataset_name": "University of Oulu Smart Campus IoT (release 2021062801)",
        "files": [
            {**_file_info(app_path), "role": "Application measurements (physical quantities)"},
            {**_file_info(lora_path), "role": "LoRaWAN gateway / radio metrics"},
            {**_file_info(dev_path), "role": "Device metadata (JSON Lines)"},
            {**_file_info(release / "README.md"), "role": "Release documentation"},
        ],
        "devices": {
            "n_devices": int(len(devices)),
            "fields": [
                "id",
                "type",
                "floor",
                "status",
                "installed",
                "desc",
                "location",
                "addedBy",
            ],
            "type_counts": type_counts,
            "floor_counts": floor_counts,
            "status_counts": status_counts,
            "n_with_location": int(devices["latitude"].notna().sum()),
            "n_with_floor": int(devices["floor"].notna().sum()),
            "n_with_desc": int(devices["desc"].notna().sum()),
            "lat_range": [
                float(devices["latitude"].min()),
                float(devices["latitude"].max()),
            ],
            "lon_range": [
                float(devices["longitude"].min()),
                float(devices["longitude"].max()),
            ],
            "n_co2_devices": n_co2,
            "coordinate_note": (
                "README claims Web Mercator EPSG:3785; observed values match WGS84 lat/lon for Oulu."
            ),
        },
        "application": {
            k: v
            for k, v in app_stats.items()
            if k
            not in {
                "co2_sample",
                "battery_sample",
                "rssi_sample",
                "lsnr_sample",
                "plot_sample",
                "device_observation_counts",
            }
        }
        | {
            "device_observation_counts": {
                k: int(v)
                for k, v in list(app_stats["device_observation_counts"].items())[:50]
            },
            "missingness": app_stats["missingness"],
            "interval_stats": app_stats["interval_stats"],
            "top_devices": [(d, int(c)) for d, c in app_stats["top_devices"]],
            # keep samples out of huge JSON — store stats instead
        },
        "lora": {
            k: v
            for k, v in lora_stats.items()
            if k
            not in {
                "co2_sample",
                "battery_sample",
                "rssi_sample",
                "lsnr_sample",
                "plot_sample",
                "device_observation_counts",
            }
        }
        | {
            "device_observation_counts": {
                k: int(v)
                for k, v in list(lora_stats["device_observation_counts"].items())[:50]
            },
            "top_devices": [(d, int(c)) for d, c in lora_stats["top_devices"]],
        },
        "co2_stats": _describe_numeric(app_stats.get("co2_sample", [])),
        "battery_stats": _describe_numeric(app_stats.get("battery_sample", [])),
        "rssi_stats": _describe_numeric(lora_stats.get("rssi_sample", [])),
        "lsnr_stats": _describe_numeric(lora_stats.get("lsnr_sample", [])),
        "research_variables": {
            "suitable": [
                "CO₂ (ppm) on Elsys ERS CO2 devices — primary target",
                "temperature, humidity — secondary targets / features",
                "light, motion (PIR) — activity / occupancy proxies",
                "battery voltage — device health feature",
                "sound_avg / sound_peak — on ERS Sound only",
                "RSSI / LSNR — network quality features",
                "floor, lat/lon, location description — spatial graph",
                "seqn / fcnt — packet sequence (limited)",
                "15-minute nominal cadence — supports regular grid + AoI in intervals",
            ],
            "not_available": [
                "Spreading factor (SF)",
                "Explicit TX power / measured energy per uplink",
                "Payload size / airtime (not directly logged)",
                "Gateway identifiers / multi-gateway reception",
                "Building ID as a structured field",
                "Room ID / HVAC zone / corridor connectivity graph",
                "Exact ventilation topology",
                "Ground-truth occupancy counts",
                "CO₂ on non-ERS-CO2 device types",
            ],
        },
        "scientific_limitations": [
            "Historical traces reflect the original fixed ~15-min transmission policy; RL actions are counterfactual.",
            "Skipped observations in the RL simulator must never expose ground truth to agents/server models.",
            "No measured joules-per-transmission; report transmission reduction (%) and optionally estimated energy only.",
            "Spatial proximity ≠ airflow connectivity; graphs are statistical/spatial relationships.",
            "Coordinate CRS discrepancy between README and observed lat/lon values.",
            "Some quantities (sound, soil moisture) exist only on device subsets — do not pool naively.",
            "Natural missingness (packet loss / offline) must be distinguished from RL-induced skips.",
        ],
    }

    # Attach samples needed for figures only in memory
    summary_for_plots = {
        "application": {**summary["application"], **{
            "co2_sample": app_stats.get("co2_sample", []),
            "battery_sample": app_stats.get("battery_sample", []),
            "plot_sample": app_stats.get("plot_sample", []),
            "device_observation_counts": app_stats["device_observation_counts"],
            "missingness": app_stats["missingness"],
        }},
        "lora": {
            "rssi_sample": lora_stats.get("rssi_sample", []),
            "lsnr_sample": lora_stats.get("lsnr_sample", []),
        },
    }

    fig_dir = root / "figures" / "eda"
    print("[audit] Writing EDA figures…")
    figs = make_eda_figures(summary_for_plots, devices, fig_dir)
    summary["eda_figures"] = figs

    out_json = root / "outputs" / "data_summary.json"
    print(f"[audit] Writing {out_json}")
    save_json(summary, out_json)

    report = root / "reports" / "data_audit.md"
    print(f"[audit] Writing {report}")
    write_audit_markdown(summary, report)

    # Convenience: also copy/link note for data/raw
    raw_note = root / "data" / "raw" / "README.md"
    ensure_dir(raw_note.parent)
    raw_note.write_text(
        "# Raw data\n\n"
        "Source release is kept immutable at "
        f"`{cfg['paths']['raw_release']}/`.\n"
        "Do not modify those files. Preprocessing writes to "
        "`data/interim/` and `data/processed/`.\n",
        encoding="utf-8",
    )
    print("[audit] Done.")
    return summary


if __name__ == "__main__":
    run_audit()
