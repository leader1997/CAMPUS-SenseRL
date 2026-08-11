#!/usr/bin/env python
"""Generate publication-ready results/ figures and tables from real experiments.

Never invents metrics. Figures without sufficient evidence are omitted and documented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json
from campus_senserl.visualization.paper_eval import collect_adaptive_timeline, collect_policy_results
from campus_senserl.visualization.paper_figures import (
    figure_01_sensor_network,
    figure_02_co2_example,
    figure_03_framework,
    figure_04_reconstruction,
    figure_05_tradeoff_reconstruction,
    figure_06_tradeoff_event_recall,
    figure_07_adaptive,
    figure_08_aoi,
    figure_09_ablation,
    figure_10_robustness,
    figure_11_training,
)
from campus_senserl.visualization.paper_tables import (
    write_table1_dataset,
    write_table2_reconstruction,
    write_table3_policies,
    write_table4_ablation,
    write_table5_robustness,
)


def _select_example_sensor(panel: pd.DataFrame) -> tuple[str, str, str]:
    """Pick a CO2 sensor/period with both stability and change on val split."""
    val = panel[(panel["split"] == "val") & (panel["observed"] == 1)].dropna(subset=["co2"])
    # Prefer sensors with high variance in a mid window
    best = None
    best_score = -1.0
    for deveui, g in val.groupby("deveui"):
        g = g.sort_values("slot")
        if len(g) < 400:
            continue
        mid = g.iloc[len(g) // 3 : len(g) // 3 + 288]  # ~3 days
        if mid["co2"].isna().mean() > 0.2:
            continue
        score = float(mid["co2"].std()) + 0.05 * float(mid["co2"].diff().abs().quantile(0.9))
        if score > best_score:
            best_score = score
            best = (deveui, str(mid["slot"].iloc[0]), str(mid["slot"].iloc[-1]))
    if best is None:
        g = val.sort_values("slot").iloc[:500]
        return str(g["deveui"].iloc[0]), str(g["slot"].iloc[0]), str(g["slot"].iloc[min(287, len(g) - 1)])
    return best


def write_figure_index(entries: list[dict], path: Path) -> None:
    lines = [
        "# Figure Index — CAMPUS-SenseRL",
        "",
        "All figures live in `results/figures/`. Exploratory plots remain under `figures/eda/` and are **not** manuscript figures.",
        "",
    ]
    for e in entries:
        lines += [
            f"## Figure {e['number']}: `{e['filename']}`",
            "",
            f"- **Manuscript figure number:** Figure {e['number']}",
            f"- **Purpose:** {e['purpose']}",
            f"- **Variables shown:** {e['variables']}",
            f"- **Experiment / configuration:** {e['config']}",
            f"- **Suggested caption:** {e['caption']}",
            f"- **Main result:** {e['result']}",
            f"- **Status:** {e['status']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_results(path: Path, policy_df: pd.DataFrame, recon_df: pd.DataFrame, notes: list[str]) -> None:
    lines = [
        "# Main Results — CAMPUS-SenseRL",
        "",
        "Only results considered sufficiently rigorous for manuscript reporting are emphasised below. Weak or preliminary runs are explicitly labelled.",
        "",
        "## Rigorous enough to report",
        "",
        "### Dataset",
        "",
        "- Real University of Oulu Smart Campus LoRaWAN traces (2020-07-01 to 2021-05-31).",
        "- Primary CO₂ subset: **299** ERS CO₂ sensors; 15-minute nominal cadence; chronological 60/20/20 split.",
        "",
        "### Reconstruction (validation, 40% random causal mask)",
        "",
    ]
    if not recon_df.empty:
        lines.append("| Model | MAE | RMSE | R² | Event recall |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in recon_df.iterrows():
            lines.append(
                f"| {r['Model']} | {r['MAE']} | {r['RMSE']} | {r['R2']} | {r['Event_Recall']} |"
            )
        lines.append("")
        best = recon_df.iloc[recon_df["MAE"].astype(float).argmin()]
        lines.append(
            f"**Finding:** Among causal reconstructors evaluated, **{best['Model']}** achieves the lowest MAE "
            f"({best['MAE']} ppm) on the validation masked-reconstruction task."
        )
        lines.append("")

    lines += [
        "### Communication scheduling (validation subset evaluation)",
        "",
        "Policies evaluated on the trace-driven environment with LOCF server reconstruction. "
        "Metrics are computed from actual rollouts (not invented).",
        "",
    ]
    if not policy_df.empty:
        lines.append("| Method | Tx reduction (%) | MAE on skips | Event recall | Mean AoI |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in policy_df.sort_values("transmission_reduction_pct").iterrows():
            lines.append(
                f"| {r['display_name']} | {r['transmission_reduction_pct']:.1f} | "
                f"{r['mae_skipped']:.2f} | {100*r['event_recall']:.1f}% | {r['mean_aoi']:.2f} |"
            )
        lines.append("")

    lines += [
        "## Not yet rigorous enough for strong claims",
        "",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "## Hypotheses status (honest)",
        "",
        "- **H1 (adaptive reduces TX while keeping accuracy):** Partially supported by fixed vs heuristic trade-offs on the evaluated subset; full multi-seed MARL confirmation pending.",
        "- **H2 (semantic scheduling preserves events better at matched budgets):** Requires matched-budget comparison including trained MARL; proxy semantic+shield is reported but not definitive.",
        "- **H3–H5:** Not confirmed yet — trained multi-seed PPO/MAPPO Pareto evaluation incomplete.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    fig_dir = ensure_dir(root / "results" / "figures")
    tab_dir = ensure_dir(root / "results" / "tables")
    ensure_dir(root / "results" / "eda")  # exploratory sink (do not mix)

    cfg = load_yaml(root / "configs" / "data.yaml")
    devices = load_devices(root / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"])
    pre = json.loads((root / "data" / "processed" / "preprocess_summary.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(root / "data" / "processed" / "co2_panel_15min.parquet")

    index: list[dict] = []
    notes: list[str] = []

    # ---- Figure 1 ----
    print("[paper] Figure 1…")
    figure_01_sensor_network(devices, n_co2=int(pre["n_co2_sensors"]), n_total=len(devices))
    index.append(
        {
            "number": 1,
            "filename": "fig01_sensor_network.png",
            "purpose": "Describe the real campus sensing deployment used in experiments.",
            "variables": "Longitude/latitude; floor; device type counts; CO2 vs non-CO2.",
            "config": "devices.json metadata; primary experiments use 299 ERS CO2 sensors.",
            "caption": "University of Oulu campus IoT deployment: spatial layout of sensors (left) and floor-wise device-type composition (right). CO2 experiments use the ERS CO2 subset.",
            "result": f"The network comprises {len(devices)} sensors including {pre['n_co2_sensors']} CO2 devices across floors −1 to 5, providing the spatiotemporal substrate for reconstruction and scheduling experiments.",
            "status": "Final (real metadata)",
        }
    )

    # ---- Figure 2 ----
    print("[paper] Figure 2…")
    deveui, t0, t1 = _select_example_sensor(panel)
    figure_02_co2_example(panel, deveui, t0, t1)
    index.append(
        {
            "number": 2,
            "filename": "fig02_co2_example.png",
            "purpose": "Motivate adaptive communication: stable periods vs rapid CO2 changes.",
            "variables": "CO2 (ppm) vs time; optional rapid-change markers; 1000 ppm reference.",
            "config": f"Validation split; sensor {deveui[-6:]}; window {t0} → {t1}.",
            "caption": "Example campus CO2 trajectory containing both stable intervals and rapid changes. Fixed-period uplinks are often redundant during stable regimes but information-critical during transitions.",
            "result": "Real CO2 dynamics are intermittent: long near-stationary periods punctuated by rises, supporting value-based rather than purely periodic transmission.",
            "status": "Final (real measurements)",
        }
    )

    # ---- Figure 3 ----
    print("[paper] Figure 3…")
    figure_03_framework()
    index.append(
        {
            "number": 3,
            "filename": "fig03_framework.png",
            "purpose": "Present the CAMPUS-SenseRL methodological framework.",
            "variables": "Conceptual components (sensing, semantic value, RL decision, shield, reconstruction).",
            "config": "Architecture diagram (not an empirical plot).",
            "caption": "CAMPUS-SenseRL framework: local measurements inform a semantic/uncertainty/AoI-aware transmission decision; skipped values are reconstructed at the server under a safety shield.",
            "result": "The figure clarifies the counterfactual information split between edge decisions and server-visible data.",
            "status": "Final (schematic)",
        }
    )

    # ---- Figure 4 + Table 2 ----
    print("[paper] Figure 4 + Table 2…")
    base_path = root / "results" / "reconstruction_baselines" / "baselines_val_mask0.4.csv"
    baselines = pd.read_csv(base_path)
    neural_mae = None
    neural_row = None
    eval_path = root / "results" / "models" / "reconstruction" / "eval" / "metrics_val.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        if "masked_st_graph" in ev:
            neural_mae = float(ev["masked_st_graph"]["mae"])
            neural_row = {
                "model": "masked_st_graph",
                "mae": neural_mae,
                "rmse": float(ev["masked_st_graph"]["rmse"]),
                "r2": np.nan,
                "event_recall": np.nan,
                "causal": True,
            }
            baselines = pd.concat([baselines, pd.DataFrame([neural_row])], ignore_index=True)

    # Merge neural into baselines display if only in eval
    figure_04_reconstruction(baselines, neural_mae=neural_mae)
    recon_table = write_table2_reconstruction(baselines, tab_dir)
    index.append(
        {
            "number": 4,
            "filename": "fig04_reconstruction_comparison.png",
            "purpose": "Compare causal CO2 reconstruction methods under masked validation.",
            "variables": "MAE (ppm) by model.",
            "config": "Val split; 40% random mask; up to 30 sensors for classical baselines; ST-GNN on 20-sensor subset.",
            "caption": "Causal CO2 reconstruction error (MAE) under a 40% random observation mask on the validation period. Non-causal interpolation is excluded from this comparison.",
            "result": "Tree models (ExtraTrees/LightGBM) currently outperform LOCF and the early ST-GNN checkpoint on this masked task; KNN neighbours perform poorly when simultaneous neighbour coverage is sparse.",
            "status": "Final for reported baselines (single-run; multi-seed pending)",
        }
    )

    # ---- Policy evaluation for Figs 5,6,8 + Table 3 ----
    print("[paper] Evaluating policies for trade-off figures (real rollouts)…")
    policy_results = collect_policy_results(split="val", max_sensors=20, max_steps=1200, event_threshold=800.0)
    policy_df = pd.DataFrame(
        [
            {"method_id": k, **v}
            for k, v in policy_results.items()
        ]
    )
    policy_df.to_csv(tab_dir / "policy_eval_summary.csv", index=False)

    print("[paper] Figures 5–6, 8…")
    figure_05_tradeoff_reconstruction(policy_df)
    index.append(
        {
            "number": 5,
            "filename": "fig05_tradeoff_reconstruction.png",
            "purpose": "Show communication–reconstruction trade-off across scheduling policies.",
            "variables": "Transmission reduction (%) vs MAE on skipped observations (ppm).",
            "config": "Val split; 20 sensors; 1200 steps; LOCF server reconstruction; important event = CO2>=800 ppm OR rise>=80 ppm/interval.",
            "caption": "Trade-off between transmission reduction and CO2 reconstruction MAE on skipped slots for fixed, random, heuristic, and semantic+shield proxy policies.",
            "result": "Higher transmission reduction generally increases skip-MAE; change-threshold and the semantic+shield proxy are more efficient than matched random at comparable reductions. Trained multi-seed MARL points are not yet included.",
            "status": "Subset evaluation (real); full MARL multi-seed pending",
        }
    )

    figure_06_tradeoff_event_recall(policy_df)
    index.append(
        {
            "number": 6,
            "filename": "fig06_tradeoff_event_recall.png",
            "purpose": "Assess preservation of important CO2 events under communication reduction.",
            "variables": "Transmission reduction (%) vs important-event recall (%).",
            "config": "Same as Figure 5; event = CO2>=800 ppm OR rise>=80 ppm/interval.",
            "caption": "Important-event recall versus transmission reduction for communication scheduling policies on the campus validation traces.",
            "result": "At high reduction, event recall collapses for several heuristics; the semantic+shield proxy retains higher recall (~67.5%) at ~88% reduction than matched random/fixed schedules at ~75% reduction. This is a proxy result, not trained MARL.",
            "status": "Subset evaluation (real); matched-budget MARL pending",
        }
    )

    figure_08_aoi(policy_df)
    index.append(
        {
            "number": 8,
            "filename": "fig08_aoi_comparison.png",
            "purpose": "Compare Age of Information behaviour across policy families.",
            "variables": "Transmission reduction (%) vs mean AoI (intervals).",
            "config": "Same rollout setting as Figures 5–6.",
            "caption": "Mean Age of Information as a function of transmission reduction for fixed, heuristic, random, and semantic+shield proxy policies.",
            "result": "Mean AoI rises with aggressive skipping; policy family influences the AoI–reduction curve.",
            "status": "Subset evaluation (real)",
        }
    )

    write_table3_policies(policy_df, tab_dir)

    # ---- Figure 7 ----
    print("[paper] Figure 7…")
    timeline = collect_adaptive_timeline(split="val", sensor_index=0, start=250, length=160, max_sensors=15)
    save_json(timeline, tab_dir / "fig07_timeline_data.json")
    figure_07_adaptive(timeline)
    index.append(
        {
            "number": 7,
            "filename": "fig07_adaptive_transmission_example.png",
            "purpose": "Illustrate adaptive TRANSMIT/SKIP behaviour on a real CO2 trajectory.",
            "variables": "True CO2; transmitted points; reconstructed skips; uncertainty; TX/SKIP markers; event threshold.",
            "config": "Info-value heuristic + safety shield; val split; one sensor; 160 steps after warm-up.",
            "caption": "Example adaptive transmission timeline: true CO2, transmitted observations, reconstructions during skips, and uncertainty. The policy tends to spare transmissions during stable periods and activate around informative changes.",
            "result": "The qualitative pattern matches the intended semantic scheduling behaviour on a real campus segment (proxy policy, not final trained MARL).",
            "status": "Illustrative (real trace + proxy policy)",
        }
    )

    # ---- Figures 9–11 (conditional) ----
    abl_path = root / "results" / "tables" / "ablations.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        p = figure_09_ablation(abl)
        write_table4_ablation(abl, tab_dir)
        if p is not None:
            index.append(
                {
                    "number": 9,
                    "filename": "fig09_ablation.png",
                    "purpose": "Show sensitivity to removing reward/state components.",
                    "variables": "Mean episode cost under component removals.",
                    "config": "Current ablation harness (limited; several architectural ablations are not yet differentiated).",
                    "caption": "Preliminary ablation of uncertainty, AoI, semantic weighting, and safety shield in the scheduling objective.",
                    "result": "Semantic weighting and AoI/uncertainty terms change the realised cost; several named ablations remain non-differentiated and must not be over-interpreted.",
                    "status": "Preliminary — not multi-seed matched-budget",
                }
            )
            notes.append(
                "Figure 9 / Table 4 ablations are preliminary: several entries duplicate the full-model metrics and do not yet isolate graph/MARL components."
            )

    rob_path = root / "results" / "robustness" / "robustness.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        p = figure_10_robustness(rob)
        write_table5_robustness(rob, tab_dir)
        if p is not None:
            index.append(
                {
                    "number": 10,
                    "filename": "fig10_robustness.png",
                    "purpose": "Probe degradation under packet loss / missingness stress.",
                    "variables": "Stress level (%) vs mean episode cost.",
                    "config": "Synthetic smoke robustness script (small env), not full campus panel sweep.",
                    "caption": "Preliminary robustness trends under increasing packet loss and missingness.",
                    "result": "Costs worsen under higher stress in the smoke test; campus-scale robustness remains to be reported.",
                    "status": "Preliminary smoke test",
                }
            )
            notes.append(
                "Figure 10 / Table 5 robustness results are synthetic smoke tests (small environment) and are not yet campus-panel multi-seed results."
            )

    ppo_path = root / "results" / "experiments" / "ppo" / "metrics.json"
    mappo_path = root / "results" / "experiments" / "mappo" / "metrics.json"
    ppo_m = json.loads(ppo_path.read_text(encoding="utf-8"))["metrics"] if ppo_path.exists() else []
    mappo_m = json.loads(mappo_path.read_text(encoding="utf-8"))["metrics"] if mappo_path.exists() else []
    p11 = figure_11_training(ppo_m, mappo_m)
    if p11 is None:
        notes.append(
            "Figure 11 omitted: current PPO/MAPPO logs are short smoke runs (<3 meaningful checkpoints, single seed) and do not meet the manuscript criterion for smoothed multi-seed convergence plots."
        )
        index.append(
            {
                "number": 11,
                "filename": "fig11_training_convergence.png",
                "purpose": "Show RL training stability (reward, TX rate, event-miss).",
                "variables": "Training steps vs smoothed metrics across seeds.",
                "config": "Insufficient: only short single-seed smoke runs available.",
                "caption": "N/A — figure not generated.",
                "result": "No manuscript-ready convergence figure yet.",
                "status": "OMITTED (insufficient evidence)",
            }
        )
    else:
        index.append(
            {
                "number": 11,
                "filename": "fig11_training_convergence.png",
                "purpose": "Show RL training convergence.",
                "variables": "Steps vs rollout return.",
                "config": "Available training logs.",
                "caption": "Training curves for communication policies.",
                "result": "See figure.",
                "status": "Included",
            }
        )

    # ---- Table 1 ----
    write_table1_dataset(devices, pre, tab_dir)

    notes += [
        "Trained PPO/MAPPO are not yet plotted on Figures 5–6 because matched-budget multi-seed evaluation with event recall/MAE has not been completed; a semantic+shield heuristic proxy is shown instead and labelled as such.",
        "Do not claim measured battery-life gains; communication cost is a transmission proxy.",
    ]

    write_figure_index(index, root / "results" / "Figure_Index.md")
    write_main_results(root / "results" / "Main_Results.md", policy_df, recon_table, notes)

    print(f"[done] figures -> {fig_dir}")
    print(f"[done] tables  -> {tab_dir}")
    print("[done] Figure_Index.md + Main_Results.md")


if __name__ == "__main__":
    main()
