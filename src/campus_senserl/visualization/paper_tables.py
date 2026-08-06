"""CSV + LaTeX exporters for manuscript tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_senserl.utils import ensure_dir


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def df_to_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = " & ".join(_latex_escape(c) for c in df.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + ("l" + "c" * (len(df.columns) - 1)) + "}",
        "\\toprule",
        cols + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("---")
                else:
                    vals.append(f"{v:.3g}" if abs(v) < 1000 else f"{v:.1f}")
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table1_dataset(devices: pd.DataFrame, pre: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    n_co2 = int((devices["device_type"] == "Elsys ERS CO2").sum())
    rows = [
        {"Item": "Total sensors (metadata)", "Value": int(len(devices))},
        {"Item": "CO2 sensors (ERS CO2)", "Value": n_co2},
        {"Item": "Period", "Value": "2020-07-01 to 2021-05-31 (UTC)"},
        {"Item": "Nominal sampling interval", "Value": "15 min"},
        {"Item": "Primary measurements", "Value": "CO2, temperature, humidity, light, motion, battery"},
        {"Item": "Network measurements", "Value": "RSSI, LSNR"},
        {"Item": "Panel observed rate", "Value": f"{100*float(pre['observed_rate']):.1f}%"},
        {"Item": "Train end", "Value": str(pre["split_bounds"]["train_end"])},
        {"Item": "Validation end", "Value": str(pre["split_bounds"]["val_end"])},
        {"Item": "Test end", "Value": str(pre["split_bounds"]["test_end"])},
        {"Item": "CO2 panel slots", "Value": int(pre["n_slots"])},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1_dataset_summary.csv", index=False)
    df_to_latex(df, out_dir / "table1_dataset_summary.tex", "Dataset summary for CAMPUS-SenseRL experiments.", "tab:dataset")
    return df


def write_table2_reconstruction(baselines: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    keep = {
        "locf": "Last observation (LOCF)",
        "linear_extrapolation": "Linear extrapolation",
        "knn_neighbors": "KNN neighbours",
        "extratrees": "ExtraTrees",
        "lightgbm": "LightGBM",
        "masked_st_graph": "Masked ST-GNN",
    }
    rows = []
    for _, r in baselines.iterrows():
        m = r["model"]
        if m not in keep:
            continue
        er = r["event_recall"] if "event_recall" in baselines.columns else np.nan
        r2 = r["r2"] if "r2" in baselines.columns else np.nan
        rows.append(
            {
                "Model": keep[m],
                "MAE": round(float(r["mae"]), 2),
                "RMSE": round(float(r["rmse"]), 2),
                "R2": round(float(r2), 3) if pd.notna(r2) else "---",
                "Event_Recall": round(float(er), 3) if pd.notna(er) else "---",
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("Model")
    df.to_csv(out_dir / "table2_reconstruction.csv", index=False)
    df_to_latex(
        df,
        out_dir / "table2_reconstruction.tex",
        "Causal CO2 reconstruction results on the validation set (40\\% random mask).",
        "tab:reconstruction",
    )
    return df


def write_table3_policies(policy_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    rows = []
    for _, r in policy_df.iterrows():
        rows.append(
            {
                "Method": r["display_name"],
                "Transmission_Reduction_pct": round(float(r["transmission_reduction_pct"]), 1),
                "CO2_MAE_skips": round(float(r["mae_skipped"]), 2) if pd.notna(r["mae_skipped"]) else "---",
                "Event_Recall_pct": round(100 * float(r["event_recall"]), 1) if pd.notna(r["event_recall"]) else "---",
                "Event_F1": round(float(r["event_f1"]), 3) if pd.notna(r["event_f1"]) else "---",
                "Mean_AoI": round(float(r["mean_aoi"]), 2) if pd.notna(r["mean_aoi"]) else "---",
                "Max_AoI": round(float(r["max_aoi"]), 2) if pd.notna(r["max_aoi"]) else "---",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table3_policy_comparison.csv", index=False)
    df_to_latex(
        df,
        out_dir / "table3_policy_comparison.tex",
        "Communication policy comparison on the validation subset (single-run subset evaluation).",
        "tab:policies",
    )
    return df


def write_table4_ablation(abl: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    df = abl.copy()
    df = df.rename(
        columns={
            "ablation": "Ablation",
            "mean_reward": "Mean_Reward",
            "transmit_rate": "Transmit_Rate",
            "rl_skip_rate": "Skip_Rate",
        }
    )
    keep = [c for c in ["Ablation", "Mean_Reward", "Transmit_Rate", "Skip_Rate"] if c in df.columns]
    df = df[keep]
    df.to_csv(out_dir / "table4_ablation.csv", index=False)
    df_to_latex(
        df,
        out_dir / "table4_ablation.tex",
        "Preliminary ablation results (not multi-seed matched-budget).",
        "tab:ablation",
    )
    return df


def write_table5_robustness(rob: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    df = rob.copy()
    df.to_csv(out_dir / "table5_robustness.csv", index=False)
    # Compact latex
    slim = df.copy()
    if "mean_reward" in slim.columns:
        slim = slim[["condition", "level", "transmit_rate", "mean_reward"]]
    slim.to_csv(out_dir / "table5_robustness_slim.csv", index=False)
    df_to_latex(
        slim,
        out_dir / "table5_robustness.tex",
        "Preliminary robustness smoke-test results.",
        "tab:robustness",
    )
    return df
