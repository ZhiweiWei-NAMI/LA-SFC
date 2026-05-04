from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "intra_region_only": "Intra-Region Only",
    "cross_region_auction": "Cross-Region Auction",
    "pure_semantic_greedy_no_exchange": "Semantic-Only Greedy\n(No Exchange)",
    "local_semantic_runtime_greedy": "Local Semantic\n+ Runtime Greedy",
    "nsga2_semantic_qos": "NSGA-II\nSemantic-QoS",
    "mappo_ctde": "MAPPO-CTDE",
    "iql_offline": "IQL\n(Offline)",
    "utility_prior_with_exchange": "Utility Prior\n+ Exchange",
    "topology_greedy": "Topology Greedy\n+ Exchange",
    "marl_no_semantic": "MARL-noSem",
    "marl_semantic_no_topology": "MARL-sem",
    "marl_topology_no_semantic": "MARL-topo",
    "proposed_semantic_topology_marl": "LASDM-ST-MARL",
    "centralized_planner": "LASDM-Centralized\nUpper Bound",
    "centralized_oracle": "LASDM-Centralized\nUpper Bound",
}
METHOD_ORDER = [
    "intra_region_only",
    "cross_region_auction",
    "pure_semantic_greedy_no_exchange",
    "local_semantic_runtime_greedy",
    "nsga2_semantic_qos",
    "mappo_ctde",
    "iql_offline",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_topology_no_semantic",
    "proposed_semantic_topology_marl",
]
SCENARIO_LABELS = {
    "distributed_service_discovery": "Distributed\nservice discovery",
    "semantic_ambiguity": "Semantic\nambiguity",
    "spatio_temporal_mobility": "Spatio-temporal\nmobility",
    "high_contention_partial_observability": "High contention\npartial observability",
    "semantic_runtime_contention_stress": "Contention",
    "semantic_runtime_mobility_staleness_stress": "Mobility/stale",
}
SCENARIO_ORDER = [
    "semantic_runtime_contention_stress",
    "semantic_runtime_mobility_staleness_stress",
    "distributed_service_discovery",
    "semantic_ambiguity",
    "spatio_temporal_mobility",
    "high_contention_partial_observability",
]

OKABE_ITO = {
    "intra_region_only": "#E69F00",
    "cross_region_auction": "#D55E00",
    "pure_semantic_greedy_no_exchange": "#e377c2",
    "local_semantic_runtime_greedy": "#bcbd22",
    "nsga2_semantic_qos": "#17becf",
    "mappo_ctde": "#e41a1c",
    "iql_offline": "#4daf4a",
    "utility_prior_with_exchange": "#CC79A7",
    "topology_greedy": "#0072B2",
    "marl_no_semantic": "#999999",
    "marl_semantic_no_topology": "#CC79A7",
    "marl_topology_no_semantic": "#56B4E9",
    "proposed_semantic_topology_marl": "#009E73",
    "centralized_planner": "#000000",
    "centralized_oracle": "#000000",
}
METHOD_MARKERS = {
    "intra_region_only": "o",
    "cross_region_auction": "s",
    "pure_semantic_greedy_no_exchange": "p",
    "local_semantic_runtime_greedy": "h",
    "nsga2_semantic_qos": "D",
    "mappo_ctde": "x",
    "iql_offline": "+",
    "utility_prior_with_exchange": "D",
    "topology_greedy": "^",
    "marl_no_semantic": "v",
    "marl_semantic_no_topology": "D",
    "marl_topology_no_semantic": "P",
    "proposed_semantic_topology_marl": "*",
    "centralized_planner": "X",
    "centralized_oracle": "X",
}
SCENARIO_MARKERS = {
    "semantic_runtime_contention_stress": "o",
    "semantic_runtime_mobility_staleness_stress": "s",
    "distributed_service_discovery": "^",
    "semantic_ambiguity": "D",
    "spatio_temporal_mobility": "P",
    "high_contention_partial_observability": "X",
}
PAPER_METHOD_ORDER = [
    "intra_region_only",
    "cross_region_auction",
    "pure_semantic_greedy_no_exchange",
    "local_semantic_runtime_greedy",
    "nsga2_semantic_qos",
    "mappo_ctde",
    "iql_offline",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_semantic_no_topology",
    "marl_topology_no_semantic",
    "proposed_semantic_topology_marl",
]
REFERENCE_METHOD_ORDER = ["centralized_planner", "centralized_oracle"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate status-aware, publication-readable Semantic-Topology MARL figures."
    )
    parser.add_argument(
        "--input-dir",
        default="experiment_artifacts/raw_data/complete_runtime_scheduler_full/semantic_runtime_eval",
        help="Raw semantic eval root, compact export root, or analysis/semantic_topology directory.",
    )
    parser.add_argument(
        "--ttl-input-dir",
        default=None,
        help=(
            "Optional raw TTL/radius sweep root. When provided, F5 and "
            "figure_tables/discovery_ttl_radius.csv are generated from this root "
            "while the other paper figures remain based on --input-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/semantic_topology_paper/figures",
    )
    parser.add_argument(
        "--figure-table-dir",
        default=None,
        help="Directory for normalized figure tables. Defaults to <output-dir>/../figure_tables.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="pdf",
        help="Output format. Use both for quick visual QA.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(args.figure_table_dir) if args.figure_table_dir else output_dir.parent / "figure_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib()
    summary = load_summary(input_dir)
    aggregate = load_aggregate(input_dir)
    raw_root = find_raw_run_root(input_dir)
    if raw_root is None:
        raise FileNotFoundError(
            f"{input_dir} is not a semantic runtime raw root. Expected summary.csv plus semantic_candidate_detail_trace.csv."
        )

    tables = prepare_figure_tables(input_dir, raw_root, summary, table_dir)
    ttl_input_dir = Path(args.ttl_input_dir) if args.ttl_input_dir else None
    if ttl_input_dir is not None:
        ttl_raw_root = find_raw_run_root(ttl_input_dir)
        if ttl_raw_root is None:
            raise FileNotFoundError(f"{ttl_input_dir} is not a semantic TTL/radius raw sweep root")
        ttl_discovery = build_discovery_ttl_radius(ttl_raw_root)
        if not ttl_discovery.empty:
            tables["discovery_ttl_radius"] = ttl_discovery
            ttl_discovery.to_csv(table_dir / "discovery_ttl_radius.csv", index=False)
    manifest: List[Dict[str, Any]] = []
    figure_specs = [
        ("F1_architecture", plot_f1_architecture, ()),
        ("F2_success_deadline_by_method_scenario", plot_f2_success_deadline, (tables["ablation_summary"],)),
        ("F3_e2e_latency_cdf_p95_p99", plot_f3_latency_cdf, (tables["latency_samples"],)),
        ("F4_semantic_candidate_quality", plot_f4_candidate_quality, (tables["candidate_quality"],)),
        ("F5_remote_discovery_stale_ttl_radius", plot_f5_discovery_ttl_radius, (tables["discovery_ttl_radius"],)),
        ("F6_topology_churn_reward_qos", plot_f6_topology_churn, (tables["topology_churn_reward"],)),
        ("F7_ippo_reward_curve", plot_f7_reward_curve, (tables["reward_curve"],)),
        ("F8_message_overhead_performance_pareto", plot_f8_overhead_pareto, (tables["message_pareto"],)),
    ]
    for stem, func, fargs in figure_specs:
        manifest.extend(
            save_semantic_figure(
                func,
                output_dir,
                stem,
                args.format,
                args.dpi,
                figure_quality(stem, tables, input_dir, ttl_input_dir=ttl_input_dir),
                *fargs,
            )
        )

    data_quality = semantic_data_quality(summary, aggregate, raw_root)
    (output_dir / "semantic_figure_manifest.json").write_text(
        json.dumps(
            {
                "figures": manifest,
                "data_quality": data_quality,
                "ttl_input_dir": str(ttl_input_dir) if ttl_input_dir is not None else "",
                "figure_tables": {name: str(table_dir / f"{name}.csv") for name in tables},
                "expected_axes_and_legends": expected_figure_specs(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "figures": len(manifest), "data_quality": data_quality}, indent=2))


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def load_summary(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "summary.csv",
        input_dir / "semantic_runtime_eval" / "summary.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return normalize_summary(df)
    raise FileNotFoundError(f"Missing semantic topology summary table under {input_dir}")


def load_aggregate(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "aggregate.csv",
        input_dir / "semantic_runtime_aggregate.csv",
        input_dir / "analysis" / "semantic_topology" / "aggregate.csv",
        input_dir / "raw" / "semantic_runtime_aggregate.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"baseline", "scenario", "seed", "status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Summary table missing required columns: {missing}")
    for col in [
        "submitted",
        "succeeded",
        "failed",
        "timed_out",
        "success_ratio",
        "qos_hit_ratio",
        "task_success_ratio",
        "avg_graph_finish_time",
        "p95_graph_finish_time",
        "simulation_time_end",
        "avg_decision_time_ms",
        "p95_decision_time_ms",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_raw_run_root(input_dir: Path) -> Optional[Path]:
    if (input_dir / "summary.csv").exists() and any(input_dir.rglob("semantic_candidate_detail_trace.csv")):
        return input_dir
    nested = [
        input_dir / "semantic_runtime_eval",
    ]
    for path in nested:
        if path.exists() and (path / "summary.csv").exists() and any(path.rglob("semantic_candidate_detail_trace.csv")):
            return path
    return None


def find_training_root(input_dir: Path, raw_root: Path) -> Optional[Path]:
    candidates = [
        input_dir.parent / "semantic_runtime_train",
        raw_root.parent / "semantic_runtime_train",
        input_dir / "semantic_runtime_train",
        raw_root / "semantic_runtime_train",
    ]
    for path in candidates:
        if path.exists() and any(path.rglob("reward_curve.csv")):
            return path
    return None


def semantic_data_quality(summary: pd.DataFrame, aggregate: pd.DataFrame, raw_root: Optional[Path]) -> Dict[str, Any]:
    quality: Dict[str, Any] = {
        "summary_rows": int(len(summary)),
        "aggregate_rows": int(len(aggregate)),
        "raw_trace_root_found": raw_root is not None,
    }
    if not summary.empty:
        quality.update(
            {
                "run_count": int(len(summary)),
                "all_success_zero": bool(_numeric(summary, "success_ratio").fillna(0).max() == 0),
                "all_qos_zero": bool(_numeric(summary, "qos_hit_ratio").fillna(0).max() == 0),
                "status_counts": {str(k): int(v) for k, v in summary["status"].value_counts(dropna=False).items()},
            }
        )
    return quality


def plot_runtime_status_matrix(summary: pd.DataFrame, aggregate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if summary.empty:
        return notice_figure("Runtime status matrix unavailable", ["No summary rows were loaded."])
    table = summary.pivot_table(index="baseline", columns="scenario", values="timed_out", aggfunc="mean")
    submitted = summary.pivot_table(index="baseline", columns="scenario", values="submitted", aggfunc="mean")
    rows = ordered_values(table.index, METHOD_ORDER)
    cols = ordered_values(table.columns, SCENARIO_ORDER)
    table = table.reindex(index=rows, columns=cols).fillna(0)
    submitted = submitted.reindex(index=rows, columns=cols).fillna(0)
    ratio = table.divide(submitted.replace(0, np.nan)).fillna(0)
    im = ax.imshow(ratio.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_title("Semantic-Topology runtime status: timeout ratio")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.set_xticks(range(len(cols)), [scenario_label(c) for c in cols], rotation=0)
    ax.set_yticks(range(len(rows)), [method_label(r) for r in rows])
    for i, method in enumerate(rows):
        for j, scenario in enumerate(cols):
            t = table.loc[method, scenario]
            s = submitted.loc[method, scenario]
            ax.text(j, i, f"{t:.0f}/{s:.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("timeout/submitted")
    annotate_caveat(ax, "Every cell is timeout/submitted. This is a runtime configuration failure signature, not a performance figure.")
    return fig


def plot_ablation_success_rate(summary: pd.DataFrame, aggregate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    if summary.empty:
        return notice_figure("Success rate unavailable", ["No summary rows were loaded."])
    grouped = summary.groupby("baseline", as_index=True)["success_ratio"].agg(["mean", "std", "count"])
    ordered = ordered_values(grouped.index, METHOD_ORDER)
    grouped = grouped.reindex(ordered)
    y = np.arange(len(grouped))
    ax.barh(y, grouped["mean"].fillna(0).to_numpy(), xerr=grouped["std"].fillna(0).to_numpy())
    ax.set_yticks(y, [method_label(m) for m in grouped.index])
    ax.set_xlim(0, max(1.0, float(grouped["mean"].fillna(0).max()) * 1.15))
    ax.set_xlabel("SFC success ratio")
    ax.set_title("Ablation outcome by method")
    for yi, value in zip(y, grouped["mean"].fillna(0)):
        ax.text(min(0.98, value + 0.02), yi, f"{value:.2f}", va="center", fontsize=8)
    if grouped["mean"].fillna(0).max() == 0:
        annotate_caveat(ax, "All mean success ratios are 0. Keep this figure as a failure diagnostic only.")
    return fig


def plot_deadline_hit_by_scenario(summary: pd.DataFrame, aggregate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if summary.empty or "qos_hit_ratio" not in summary.columns:
        return notice_figure("Deadline/QoS hit matrix unavailable", ["No qos_hit_ratio column was found."])
    table = summary.pivot_table(index="baseline", columns="scenario", values="qos_hit_ratio", aggfunc="mean")
    rows = ordered_values(table.index, METHOD_ORDER)
    cols = ordered_values(table.columns, SCENARIO_ORDER)
    table = table.reindex(index=rows, columns=cols).fillna(0)
    im = ax.imshow(table.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_title("Deadline satisfaction by method and scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.set_xticks(range(len(cols)), [scenario_label(c) for c in cols])
    ax.set_yticks(range(len(rows)), [method_label(r) for r in rows])
    for i, method in enumerate(rows):
        for j, scenario in enumerate(cols):
            ax.text(j, i, f"{table.loc[method, scenario]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("QoS hit ratio")
    if table.to_numpy().max() == 0:
        annotate_caveat(ax, "All QoS hit ratios are 0. Do not use this as superiority evidence.")
    return fig


def plot_latency_cdf(summary: pd.DataFrame, aggregate: pd.DataFrame, raw_root: Optional[Path]) -> plt.Figure:
    raise RuntimeError(
        "Legacy summary-derived latency plotting is disabled. "
        "Use prepare_figure_tables() and plot_f3_latency_cdf() with raw function_execution_trace.csv e2e_delay only."
    )


def plot_decision_time(summary: pd.DataFrame, aggregate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    if summary.empty or "avg_decision_time_ms" not in summary.columns:
        return notice_figure("Decision-time figure unavailable", ["No avg_decision_time_ms column was found."])
    grouped = summary.groupby("baseline", as_index=True)["avg_decision_time_ms"].agg(["mean", "std", "count"])
    ordered = ordered_values(grouped.index, METHOD_ORDER)
    grouped = grouped.reindex(ordered)
    y = np.arange(len(grouped))
    ax.barh(y, grouped["mean"].to_numpy(), xerr=grouped["std"].fillna(0).to_numpy())
    ax.set_yticks(y, [method_label(m) for m in grouped.index])
    ax.set_xlabel("Decision time (ms)")
    ax.set_title("Scheduler/MARL decision overhead")
    for yi, value in zip(y, grouped["mean"]):
        ax.text(value * 1.002, yi, f"{value:.0f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.0, float(grouped["mean"].max()) * 1.12))
    annotate_caveat(ax, "This is a valid engineering metric even when SFC success is zero.")
    return fig


def plot_reward_curve(raw_root: Optional[Path]) -> plt.Figure:
    if raw_root is None:
        return notice_figure(
            "Reward curve unavailable in compact package",
            ["No reward_curve.csv files were found.", "Use the full raw archive or training output root."],
        )
    rows = list(iter_csv(raw_root, "reward_curve.csv"))
    if not rows:
        return notice_figure("Reward curve unavailable", ["No reward_curve.csv rows were found."])
    df = pd.DataFrame(rows)
    for col in ["episode", "step", "mean_reward", "reward"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mean_reward" not in df.columns and "reward" in df.columns:
        df["mean_reward"] = df["reward"]
    if "mean_reward" not in df.columns:
        return notice_figure("Reward curve unavailable", ["reward_curve.csv does not contain mean_reward or reward."])
    df["method"] = df["__path"].map(method_from_path)
    df["seed"] = df["__path"].map(seed_from_path)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    candidates = ["proposed_semantic_topology_marl", "marl_topology_no_semantic", "marl_semantic_no_topology", "marl_no_semantic"]
    plotted = 0
    for method in candidates:
        sub = df[df["method"].eq(method)].copy()
        if sub.empty:
            continue
        sub["x"] = sub.groupby(["method", "seed"]).cumcount()
        mean = sub.groupby("x")["mean_reward"].mean()
        std = sub.groupby("x")["mean_reward"].std().fillna(0)
        ax.plot(mean.index.to_numpy(), mean.to_numpy(), label=method_label(method).replace("\n", " "))
        if len(mean) > 1:
            ax.fill_between(mean.index.to_numpy(), (mean - std).to_numpy(), (mean + std).to_numpy(), alpha=0.12)
        plotted += 1
    if plotted == 0:
        return notice_figure("Reward curve unavailable", ["No MARL reward rows matched the expected method labels."])
    ax.set_xlabel("Training/evaluation step")
    ax.set_ylabel("Mean reward")
    ax.set_title("Reward trajectory (mean over available seeds)")
    ax.legend(frameon=False)
    return fig


def plot_semantic_similarity_distribution(raw_root: Optional[Path]) -> plt.Figure:
    if raw_root is None:
        return notice_figure(
            "Semantic similarity distribution unavailable in compact package",
            ["No semantic_candidate_trace.csv files were found.", "Use the full raw archive to regenerate this figure."],
        )
    rows = list(iter_csv(raw_root, "semantic_candidate_trace.csv"))
    if not rows:
        return notice_figure("Semantic similarity distribution unavailable", ["No semantic_candidate_trace.csv rows were found."])
    df = pd.DataFrame(rows)
    score_col = first_existing(df, ["top_score", "top1_score", "semantic_score", "max_similarity"])
    if score_col is None:
        return notice_figure("Semantic similarity distribution unavailable", ["No top semantic score column was found."])
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df[score_col])]
    if df.empty:
        return notice_figure("Semantic similarity distribution unavailable", ["Top semantic scores are empty or non-numeric."])
    df["method"] = df["__path"].map(method_from_path)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    selected = ["pure_semantic_greedy_no_exchange", "local_semantic_runtime_greedy", "utility_prior_with_exchange", "proposed_semantic_topology_marl", "centralized_planner"]
    for method in selected:
        values = df.loc[df["method"].eq(method), score_col].dropna().to_numpy()
        if len(values):
            ax.hist(values, bins=24, density=True, histtype="step", linewidth=1.5, label=method_label(method).replace("\n", " "))
    if not ax.has_data():
        ax.hist(df[score_col].dropna().to_numpy(), bins=24, density=True, histtype="step")
    ax.set_xlabel("Top-1 semantic similarity")
    ax.set_ylabel("Density")
    ax.set_title("Semantic candidate quality distribution")
    if len(ax.get_legend_handles_labels()[0]):
        ax.legend(frameon=False)
    return fig


def plot_remote_candidate_discovery_rate(raw_root: Optional[Path]) -> plt.Figure:
    if raw_root is None:
        return notice_figure(
            "Remote discovery rate unavailable in compact package",
            ["No semantic_candidate_trace.csv files were found.", "Use the full raw archive to regenerate this figure."],
        )
    rows = list(iter_csv(raw_root, "semantic_candidate_trace.csv"))
    if not rows:
        return notice_figure("Remote discovery rate unavailable", ["No semantic_candidate_trace.csv rows were found."])
    df = pd.DataFrame(rows)
    col = first_existing(df, ["remote_candidate_count", "remote_candidates", "remote_count"])
    if col is None:
        return notice_figure("Remote discovery rate unavailable", ["No remote-candidate count column was found."])
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["method"] = df["__path"].map(method_from_path)
    grouped = df.groupby("method")[col].agg(lambda x: float((x > 0).mean()))
    ordered = ordered_values(grouped.index, METHOD_ORDER)
    grouped = grouped.reindex(ordered)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(grouped))
    ax.barh(y, grouped.to_numpy())
    ax.set_yticks(y, [method_label(m) for m in grouped.index])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of decisions with at least one remote candidate")
    ax.set_title("Remote candidate discovery rate")
    for yi, value in zip(y, grouped):
        ax.text(min(0.98, value + 0.02), yi, f"{value:.2f}", va="center", fontsize=8)
    return fig


def plot_message_overhead_vs_performance(summary: pd.DataFrame, aggregate: pd.DataFrame, raw_root: Optional[Path]) -> plt.Figure:
    raise RuntimeError(
        "Legacy message-overhead plotting is disabled. "
        "Use prepare_figure_tables() and plot_f8_overhead_pareto() with raw message_overhead.csv only."
    )


def plot_topology_change_vs_reward(raw_root: Optional[Path]) -> plt.Figure:
    if raw_root is None:
        return notice_figure(
            "Topology-change/reward figure unavailable in compact package",
            ["No topology_trace.jsonl or reward_curve.csv files were found.", "Use the full raw archive to regenerate this figure."],
        )
    points = []
    for path in raw_root.rglob("topology_trace.jsonl"):
        edge_counts = []
        for row in read_jsonl(path):
            if isinstance(row.get("edges"), list):
                edge_counts.append(len(row.get("edges", [])))
            elif isinstance(row.get("edge_count"), (int, float)):
                edge_counts.append(float(row.get("edge_count")))
        if len(edge_counts) < 2:
            continue
        churn = float(np.mean(np.abs(np.diff(np.asarray(edge_counts, dtype=float)))))
        reward_path = path.parent / "reward_curve.csv"
        rewards = []
        if reward_path.exists():
            for row in read_csv_dicts(reward_path):
                value = row.get("mean_reward") or row.get("reward")
                try:
                    rewards.append(float(value))
                except Exception:
                    pass
        if rewards:
            points.append((method_from_path(str(path)), churn, float(np.mean(rewards))))
    if not points:
        return notice_figure(
            "Topology-change/reward figure unavailable",
            ["Raw topology traces or reward samples were not sufficient for a scatter plot."],
        )
    df = pd.DataFrame(points, columns=["method", "edge_churn", "reward"])
    grouped = df.groupby("method", as_index=False).agg(edge_churn=("edge_churn", "mean"), reward=("reward", "mean"))
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.scatter(grouped["edge_churn"], grouped["reward"])
    for _, row in grouped.iterrows():
        ax.annotate(method_label(row["method"]).replace("\n", " "), (row["edge_churn"], row["reward"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Mean edge-count churn")
    ax.set_ylabel("Mean reward")
    ax.set_title("Topology churn vs reward")
    return fig


def scatter_metric_vs_performance(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, caveat: str) -> plt.Figure:
    if x.empty or y.empty:
        return notice_figure(title + " unavailable", ["Required metric columns were missing."])
    idx = ordered_values(set(x.index).intersection(set(y.index)), METHOD_ORDER)
    if not idx:
        return notice_figure(title + " unavailable", ["No overlapping methods were found."])
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    xs = x.reindex(idx).astype(float)
    ys = y.reindex(idx).astype(float)
    ax.scatter(xs, ys)
    for method, xv, yv in zip(idx, xs, ys):
        ax.annotate(method_label(method).replace("\n", " "), (xv, yv), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ys.max() == 0 or "unavailable" in caveat:
        annotate_caveat(ax, caveat)
    return fig


def positive_latency_samples(summary: pd.DataFrame, raw_root: Optional[Path]) -> List[Tuple[str, float]]:
    raise RuntimeError("Summary-derived latency samples are disabled for paper figures.")


def overhead_by_method(raw_root: Optional[Path]) -> Dict[str, float]:
    raise RuntimeError("Legacy message-overhead aggregation is disabled for paper figures.")


def save_plot(func, output_dir: Path, stem: str, fmt: str, dpi: int, *args) -> List[Dict[str, Any]]:
    fig = func(*args)
    outputs: List[Path] = []
    formats = ["pdf", "png"] if fmt == "both" else [fmt]
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return [{"name": stem, "path": str(path), "bytes": path.stat().st_size} for path in outputs]


def write_notice(stem_path: Path, fmt: str, dpi: int, title: str, lines: Sequence[str]) -> None:
    fig = notice_figure(title, lines)
    for ext in (["pdf", "png"] if fmt == "both" else [fmt]):
        fig.savefig(stem_path.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


def notice_figure(title: str, lines: Sequence[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.axis("off")
    ax.text(0.02, 0.86, title, fontsize=13, fontweight="bold", transform=ax.transAxes)
    y = 0.68
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=92):
            ax.text(0.04, y, wrapped, fontsize=9.5, transform=ax.transAxes)
            y -= 0.10
        y -= 0.03
    return fig


def annotate_caveat(ax: plt.Axes, text: str) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=82))
    ax.text(
        0.01,
        -0.23,
        wrapped,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "alpha": 0.08},
    )


def ordered_values(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    values_list = [str(v) for v in values]
    present = set(values_list)
    ordered = [v for v in preferred if v in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), "\n".join(textwrap.wrap(str(method), width=18)))


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(str(scenario), "\n".join(textwrap.wrap(str(scenario), width=18)))


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def iter_csv(root: Path, filename: str) -> Iterable[Dict[str, Any]]:
    for path in root.rglob(filename):
        for row in read_csv_dicts(path):
            row["__path"] = str(path)
            yield row


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def method_from_path(path: str) -> str:
    parts = Path(path).parts
    for part in parts:
        if "__seed_" in part:
            return part.split("__seed_")[0].split("__")[0]
    for part in reversed(parts):
        for method in METHOD_ORDER:
            if method in part:
                return method
    return Path(path).parent.name or "unknown"


def seed_from_path(path: str) -> str:
    parts = Path(path).parts
    for part in parts:
        if "__seed_" in part:
            return part.rsplit("__seed_", 1)[-1]
    return "unknown"


def prepare_figure_tables(input_dir: Path, raw_root: Optional[Path], summary: pd.DataFrame, table_dir: Path) -> Dict[str, pd.DataFrame]:
    if raw_root is None:
        raise FileNotFoundError("paper figure generation requires a raw semantic runtime root")
    root = raw_root
    training_root = find_training_root(input_dir, root)
    tables = {
        "ablation_summary": normalize_table_metadata(summary.copy()) if not summary.empty else pd.DataFrame(),
        "latency_samples": build_latency_samples(root, summary),
        "candidate_quality": build_candidate_quality(root),
        "discovery_ttl_radius": build_discovery_ttl_radius(root),
        "topology_churn_reward": build_topology_churn_reward(root, summary),
        "reward_curve": build_reward_curve_table(training_root or root),
        "message_pareto": build_message_pareto(root, summary),
    }
    for name, df in tables.items():
        df.to_csv(table_dir / f"{name}.csv", index=False)
    return tables


def normalize_table_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "method" not in df.columns:
        df["method"] = df.get("baseline", "unknown")
    for col in ["method", "baseline", "scenario", "seed"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ["success_ratio", "qos_hit_ratio", "avg_graph_finish_time", "p95_graph_finish_time", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype("Int64").astype(str)
    return df


def build_latency_samples(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in trace_paths(root, "function_execution_trace.csv"):
        for row in read_csv_dicts(path):
            value = _coerce_float(row.get("e2e_delay"))
            if value is None or value <= 0:
                continue
            rows.append(
                {
                    "method": row.get("baseline") or method_from_path(str(path)),
                    "baseline": row.get("baseline") or method_from_path(str(path)),
                    "scenario": row.get("scenario") or scenario_from_path(str(path)),
                    "seed": row.get("seed") or seed_from_path(str(path)),
                    "sfc_id": row.get("sfc_id", ""),
                    "function_id": row.get("function_id", ""),
                    "latency_s": value,
                    "status": row.get("status", ""),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_quality(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    detail_paths = trace_paths(root, "semantic_candidate_detail_trace.csv")
    if detail_paths:
        for path in detail_paths:
            for row in read_csv_dicts(path):
                score = _coerce_float(row.get("semantic_score") or row.get("top_score"))
                if score is None:
                    continue
                method = row.get("baseline") or row.get("method") or method_from_path(str(path))
                rows.append(
                    {
                        "method": method,
                        "baseline": row.get("baseline") or method,
                        "scenario": row.get("scenario") or scenario_from_path(str(path)),
                        "seed": row.get("seed") or seed_from_path(str(path)),
                        "sfc_id": row.get("sfc_id", ""),
                        "function_id": row.get("function_id") or row.get("sfc_node_id", ""),
                        "candidate_id": row.get("candidate_id") or row.get("instance_id", ""),
                        "rank": _coerce_float(row.get("rank")) or 1,
                        "semantic_score": score,
                        "topology_score": _coerce_float(row.get("topology_score")),
                        "topology_risk": _coerce_float(row.get("topology_risk")),
                        "stale": _truthy_string(row.get("stale")),
                        "local_or_remote": row.get("local_or_remote") or ("remote" if _truthy_string(row.get("is_remote")) else "local"),
                        "selected": _truthy_string(row.get("selected")),
                        "exchange_ttl_s": _coerce_float(row.get("exchange_ttl_s")),
                        "exchange_radius_hops": _coerce_float(row.get("exchange_radius_hops")),
                    }
                )
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_discovery_ttl_radius(root: Path) -> pd.DataFrame:
    candidate = build_candidate_quality(root)
    if candidate.empty:
        return pd.DataFrame()
    candidate["exchange_ttl_s"] = pd.to_numeric(candidate["exchange_ttl_s"], errors="coerce")
    candidate["exchange_radius_hops"] = pd.to_numeric(candidate["exchange_radius_hops"], errors="coerce")
    candidate = candidate.dropna(subset=["exchange_ttl_s", "exchange_radius_hops"])
    if candidate.empty:
        return pd.DataFrame()
    grouped = candidate.groupby(["method", "scenario", "seed", "exchange_ttl_s", "exchange_radius_hops"], dropna=False)
    rows = []
    for keys, group in grouped:
        method, scenario, seed, ttl, radius = keys
        remote_rate = float((group["local_or_remote"].astype(str).str.lower().eq("remote")).mean())
        stale_rate = float(group["stale"].astype(bool).mean()) if "stale" in group.columns else 0.0
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "exchange_ttl_s": ttl,
                "exchange_radius_hops": radius,
                "remote_discovery_rate": remote_rate,
                "stale_ratio": stale_rate,
                "candidate_count": len(group),
            }
        )
    overhead = build_message_overhead_table(root)
    if not overhead.empty:
        overhead_grouped = overhead.groupby(["method", "scenario", "seed", "exchange_ttl_s", "exchange_radius_hops"], dropna=False)["payload_bytes"].sum().reset_index()
        return pd.DataFrame(rows).merge(
            overhead_grouped,
            on=["method", "scenario", "seed", "exchange_ttl_s", "exchange_radius_hops"],
            how="left",
        )
    return pd.DataFrame(rows)


def build_message_overhead_table(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in trace_paths(root, "message_overhead.csv"):
        for row in read_csv_dicts(path):
            method = row.get("baseline") or method_from_path(str(path))
            payload_bytes = _coerce_float(row.get("payload_bytes"))
            ttl = _coerce_float(row.get("exchange_ttl_s"))
            radius = _coerce_float(row.get("exchange_radius_hops"))
            hop_count = _coerce_float(row.get("hop_count"))
            if payload_bytes is None or ttl is None or radius is None or hop_count is None:
                continue
            rows.append(
                {
                    "method": method,
                    "baseline": row.get("baseline") or method,
                    "scenario": row.get("scenario") or scenario_from_path(str(path)),
                    "seed": row.get("seed") or seed_from_path(str(path)),
                    "payload_bytes": payload_bytes,
                    "message_count": 1,
                    "exchange_ttl_s": ttl,
                    "exchange_radius_hops": radius,
                    "hop_count": hop_count,
                }
            )
    return pd.DataFrame(rows)


def build_reward_curve_table(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in trace_paths(root, "reward_curve.csv", prefer_top_level=False):
        for row in read_csv_dicts(path):
            reward = _coerce_float(row.get("mean_reward") or row.get("reward"))
            if reward is None:
                continue
            method = row.get("baseline") or row.get("method") or method_from_path(str(path))
            if method.startswith("ippo_seed") or "semantic_runtime_train" in str(path):
                method = "proposed_semantic_topology_marl"
            rows.append(
                {
                    "method": method,
                    "baseline": row.get("baseline") or method,
                    "scenario": row.get("scenario") or scenario_from_path(str(path)),
                    "seed": row.get("seed") or seed_from_path(str(path)),
                    "episode": _coerce_float(row.get("episode")) or 0,
                    "step": _coerce_float(row.get("step")) or len(rows),
                    "mean_reward": reward,
                }
            )
    return pd.DataFrame(rows)


def build_topology_churn_reward(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    reward = build_reward_curve_table(root)
    reward_by_run = pd.DataFrame()
    if not reward.empty:
        reward_by_run = reward.groupby(["method", "scenario", "seed"], as_index=False)["mean_reward"].mean()
    perf = normalize_table_metadata(summary.copy()) if not summary.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for path in trace_paths(root, "topology_trace.jsonl"):
        records = list(read_jsonl(path))
        if not records:
            continue
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in records:
            method = str(row.get("baseline") or method_from_path(str(path)))
            scenario = str(row.get("scenario") or scenario_from_path(str(path)))
            seed = str(row.get("seed") or seed_from_path(str(path)))
            groups[(method, scenario, seed)].append(row)
        for (method, scenario, seed), items in groups.items():
            items.sort(key=lambda item: _coerce_float(item.get("time_s")) or 0.0)
            churn_values = []
            previous: Optional[set[str]] = None
            for item in items:
                edge_set = topology_edge_set(item)
                if previous is not None:
                    union = len(previous | edge_set)
                    inter = len(previous & edge_set)
                    churn_values.append(1.0 - (inter / union if union else 1.0))
                previous = edge_set
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "topology_churn": float(np.mean(churn_values)) if churn_values else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if not reward_by_run.empty:
        out = out.merge(reward_by_run, on=["method", "scenario", "seed"], how="left")
    if not perf.empty:
        cols = [col for col in ["method", "scenario", "seed", "qos_hit_ratio", "success_ratio"] if col in perf.columns]
        out = out.merge(perf[cols], on=["method", "scenario", "seed"], how="left")
    return out


def build_message_pareto(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    overhead = build_message_overhead_table(root)
    perf = normalize_table_metadata(summary.copy()) if not summary.empty else pd.DataFrame()
    if perf.empty:
        return pd.DataFrame()
    if overhead.empty:
        return pd.DataFrame()
    grouped = overhead.groupby(["method", "scenario", "seed"], as_index=False).agg(
        payload_bytes=("payload_bytes", "sum"),
        message_count=("message_count", "sum"),
        mean_hop_count=("hop_count", "mean"),
    )
    cols = [col for col in ["method", "scenario", "seed", "success_ratio", "qos_hit_ratio", "avg_graph_finish_time"] if col in perf.columns]
    out = perf[cols].merge(grouped, on=["method", "scenario", "seed"], how="left")
    out = out.dropna(subset=["payload_bytes", "message_count", "mean_hop_count"])
    return out


def topology_edge_set(row: Mapping[str, Any]) -> set[str]:
    edges = row.get("edges", [])
    result: set[str] = set()
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, Mapping):
                result.add(f"{edge.get('src')}->{edge.get('dst')}:{edge.get('link_type')}")
            else:
                result.add(str(edge))
    return result


def trace_paths(root: Path, filename: str, prefer_top_level: bool = True) -> List[Path]:
    top_level = root / filename
    if prefer_top_level and top_level.exists():
        return [top_level]
    return sorted(root.rglob(filename))


def method_color(method: str) -> str:
    return OKABE_ITO.get(str(method), "#666666")


def method_linestyle(method: str) -> str:
    if method in {"centralized_planner", "centralized_oracle"}:
        return "--"
    if method in {"pure_semantic_greedy_no_exchange", "local_semantic_runtime_greedy"}:
        return "--"
    if method == "nsga2_semantic_qos":
        return "-."
    if method in {"mappo_ctde", "iql_offline", "utility_prior_with_exchange", "marl_semantic_no_topology"}:
        return ":"
    return "-"


def ci95(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def draw_pareto_front(ax: plt.Axes, x_values: Sequence[float], y_values: Sequence[float]) -> None:
    points = sorted(
        [(float(x), float(y)) for x, y in zip(x_values, y_values) if math.isfinite(float(x)) and math.isfinite(float(y))],
        key=lambda item: (item[0], -item[1]),
    )
    frontier: List[Tuple[float, float]] = []
    best = -math.inf
    for x, y in points:
        if y > best + 1e-12:
            frontier.append((x, y))
            best = y
    if len(frontier) >= 2:
        xs, ys = zip(*frontier)
        ax.plot(xs, ys, color="#333333", linewidth=0.9, linestyle="--", alpha=0.8, label="Pareto frontier")


def plot_f1_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.axis("off")
    boxes = [
        ("SFC\nrequests", 0.03, 0.56, "#e8eef7"),
        ("Semantic\ncatalog", 0.23, 0.72, "#e8f4ea"),
        ("Distributed\nexchange", 0.23, 0.38, "#e8f4ea"),
        ("Topology +\ntemporal state", 0.45, 0.56, "#fff4df"),
        ("MASAC\npolicy", 0.66, 0.56, "#f6e8ef"),
        ("AirFogSim\nscheduler", 0.83, 0.56, "#e8eef7"),
        ("Wireless / compute /\nenergy queues", 0.83, 0.23, "#e8eef7"),
    ]
    for label, x, y, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.13, 0.15, facecolor=color, edgecolor="#333333", linewidth=0.8))
        ax.text(x + 0.065, y + 0.075, label, ha="center", va="center", fontsize=7.4, weight="bold")
    arrows = [
        ((0.17, 0.66), (0.25, 0.78)),
        ((0.17, 0.66), (0.25, 0.50)),
        ((0.38, 0.80), (0.48, 0.66)),
        ((0.38, 0.50), (0.48, 0.62)),
        ((0.61, 0.66), (0.68, 0.66)),
        ((0.81, 0.66), (0.84, 0.66)),
        ((0.905, 0.58), (0.905, 0.40)),
        ((0.84, 0.30), (0.75, 0.58)),
    ]
    for src, dst in arrows:
        ax.annotate("", xy=dst, xytext=src, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"})
    ax.text(
        0.50,
        0.09,
        "Closed loop: runtime traces update service availability, topology, reward, and future decisions.",
        ha="center",
        fontsize=7.5,
    )
    ax.set_title("Semantic-topology MARL runtime loop", loc="left", fontsize=9, weight="bold")
    return fig


def plot_f2_success_deadline(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "success_ratio" not in df.columns:
        return notice_figure("F2 unavailable", ["ablation_summary.csv/summary.csv with success_ratio is required."])
    df = normalize_table_metadata(df)
    scenarios = ordered_values(df["scenario"].dropna().unique(), SCENARIO_ORDER)
    methods = ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER)
    metrics = [("success_ratio", "SFC success ratio"), ("qos_hit_ratio", "Deadline hit ratio")]
    fig, axes = plt.subplots(
        len(metrics),
        max(1, len(scenarios)),
        figsize=(7.2, 4.6),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            sub = df[df["scenario"].eq(scenario)]
            x = np.arange(len(methods))
            means = []
            cis = []
            for method in methods:
                values = pd.to_numeric(sub.loc[sub["method"].eq(method), metric], errors="coerce").dropna().to_numpy(dtype=float)
                means.append(float(values.mean()) if len(values) else np.nan)
                cis.append(ci95(values))
            colors = [method_color(method) for method in methods]
            ax.bar(x, means, yerr=cis, color=colors, edgecolor="#333333", linewidth=0.45, capsize=2.0)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x, [method_label(m) for m in methods], rotation=35, ha="right")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(scenario_label(scenario))
            ax.axhline(1.0, color="#555555", linewidth=0.5, linestyle=":")
            add_panel_label(ax, chr(ord("A") + row_idx * max(1, len(scenarios)) + col_idx))
    return fig


def plot_f3_latency_cdf(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "latency_s" not in df.columns:
        return notice_figure("F3 latency CDF unavailable", ["No positive E2E latency samples found in function_execution_trace.csv or summary.csv."])
    df = df.copy()
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    df = df[df["latency_s"] > 0]
    if df.empty:
        return notice_figure("F3 latency CDF unavailable", ["Latency samples are empty or non-positive."])
    scenarios = ordered_values(df["scenario"].dropna().unique(), SCENARIO_ORDER)
    methods = ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER)
    fig, axes = plt.subplots(1, max(1, len(scenarios)), figsize=(7.2, 2.9), sharey=True, constrained_layout=True, squeeze=False)
    for idx, scenario in enumerate(scenarios):
        ax = axes[0, idx]
        sub_s = df[df["scenario"].eq(scenario)]
        for method in methods:
            values = np.sort(sub_s.loc[sub_s["method"].eq(method), "latency_s"].dropna().to_numpy(dtype=float))
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, y, where="post", color=method_color(method), linestyle=method_linestyle(method), label=method_label(method))
        proposed = sub_s.loc[sub_s["method"].eq("proposed_semantic_topology_marl"), "latency_s"].dropna().to_numpy(dtype=float)
        if len(proposed):
            for q, style in [(95, "--"), (99, ":")]:
                value = float(np.percentile(proposed, q))
                ax.axvline(value, linestyle=style, color=method_color("proposed_semantic_topology_marl"), linewidth=0.8, alpha=0.85)
                ax.text(value, 0.05, f"P{q}", rotation=90, va="bottom", ha="right", fontsize=6.5)
        ax.set_title(scenario_label(scenario))
        ax.set_xlabel("E2E latency (s)")
        if idx == 0:
            ax.set_ylabel("Empirical CDF")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        add_panel_label(ax, chr(ord("A") + idx))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.09))
    return fig


def plot_f4_candidate_quality(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "semantic_score" not in df.columns:
        return notice_figure("F4 semantic quality unavailable", ["semantic_candidate_detail_trace.csv with semantic_score/rank is required."])
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(1).astype(int)
    df["semantic_score"] = pd.to_numeric(df["semantic_score"], errors="coerce")
    df = df[np.isfinite(df["semantic_score"])]
    if df.empty:
        return notice_figure("F4 semantic quality unavailable", ["Semantic scores are not numeric."])
    methods = ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    top_k = df[df["rank"].between(1, 8)]
    for method in methods:
        sub = top_k[top_k["method"].eq(method)]
        if sub.empty:
            continue
        stats = sub.groupby("rank")["semantic_score"].agg(["mean", "count", "std"]).reset_index()
        ci = 1.96 * stats["std"].fillna(0) / np.sqrt(stats["count"].clip(lower=1))
        axes[0].plot(stats["rank"], stats["mean"], marker=METHOD_MARKERS.get(method, "o"), color=method_color(method), linestyle=method_linestyle(method), label=method_label(method))
        axes[0].fill_between(stats["rank"], stats["mean"] - ci, stats["mean"] + ci, color=method_color(method), alpha=0.08, linewidth=0)
    selected = df[df["selected"].map(_truthy_string)] if "selected" in df.columns else df[df["rank"].eq(1)]
    box_values = [selected.loc[selected["method"].eq(method), "semantic_score"].dropna().to_numpy(dtype=float) for method in methods]
    positions = np.arange(len(methods))
    bp = axes[1].boxplot(
        box_values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 0.8},
        boxprops={"linewidth": 0.65},
        whiskerprops={"linewidth": 0.65},
        capprops={"linewidth": 0.65},
    )
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(method_color(method))
        patch.set_alpha(0.75)
    axes[0].set_xlabel("Candidate rank")
    axes[0].set_ylabel("Mean semantic score")
    axes[0].set_title("Top-k candidates")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Semantic score")
    axes[1].set_xticks(positions, [method_label(m) for m in methods], rotation=35, ha="right")
    axes[1].set_title("Selected candidates")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=6.5)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    return fig


def plot_f5_discovery_ttl_radius(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "remote_discovery_rate" not in df.columns:
        return notice_figure("F5 discovery/staleness unavailable", ["TTL/radius discovery table is empty. Run semantic exchange sweep first."])
    df = df.copy()
    for col in ["exchange_ttl_s", "exchange_radius_hops", "remote_discovery_rate", "stale_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    methods = ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey="row", constrained_layout=True)
    metrics = [("remote_discovery_rate", "Remote discovery rate"), ("stale_ratio", "Stale candidate ratio")]
    for row_idx, (metric, ylabel) in enumerate(metrics):
        by_ttl = df.groupby(["method", "exchange_ttl_s"], as_index=False)[metric].mean()
        by_radius = df.groupby(["method", "exchange_radius_hops"], as_index=False)[metric].mean()
        for method in methods:
            sub = by_ttl[by_ttl["method"].eq(method)].sort_values("exchange_ttl_s")
            if not sub.empty:
                axes[row_idx, 0].plot(sub["exchange_ttl_s"], sub[metric], marker=METHOD_MARKERS.get(method, "o"), color=method_color(method), linestyle=method_linestyle(method), label=method_label(method))
            sub = by_radius[by_radius["method"].eq(method)].sort_values("exchange_radius_hops")
            if not sub.empty:
                axes[row_idx, 1].plot(sub["exchange_radius_hops"], sub[metric], marker=METHOD_MARKERS.get(method, "o"), color=method_color(method), linestyle=method_linestyle(method), label=method_label(method))
        axes[row_idx, 0].set_ylabel(ylabel)
        axes[row_idx, 0].set_xlabel("Exchange TTL (s)")
        axes[row_idx, 1].set_xlabel("Exchange radius (hops)")
        axes[row_idx, 0].set_ylim(0, 1.02)
        axes[row_idx, 1].set_ylim(0, 1.02)
    axes[0, 0].set_title("vs TTL")
    axes[0, 1].set_title("vs radius")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.06))
    for idx, ax in enumerate(axes.ravel()):
        add_panel_label(ax, chr(ord("A") + idx))
    return fig


def plot_f6_topology_churn(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "topology_churn" not in df.columns:
        return notice_figure("F6 topology churn unavailable", ["topology_trace.jsonl and reward/summary data are required."])
    df = df.copy()
    df["topology_churn"] = pd.to_numeric(df["topology_churn"], errors="coerce").fillna(0.0)
    methods = ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    jitter_map = {method: (idx - max(1, len(methods)) / 2.0) * 0.0018 for idx, method in enumerate(methods)}
    for metric, ax, ylabel in [("mean_reward", axes[0], "Mean reward"), ("qos_hit_ratio", axes[1], "QoS hit ratio")]:
        if metric not in df.columns:
            ax.axis("off")
            ax.text(0.1, 0.5, f"{metric} unavailable", transform=ax.transAxes)
            continue
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        grouped = df.groupby(["method", "scenario"], as_index=False).agg(topology_churn=("topology_churn", "mean"), metric=(metric, "mean"))
        for method in methods:
            sub = grouped[grouped["method"].eq(method)]
            if sub.empty:
                continue
            x_values = sub["topology_churn"] + jitter_map.get(method, 0.0)
            ax.scatter(
                x_values,
                sub["metric"],
                marker=METHOD_MARKERS.get(method, "o"),
                color=method_color(method),
                s=34 if method != "proposed_semantic_topology_marl" else 58,
                edgecolor="#222222",
                linewidth=0.35,
                label=method_label(method),
            )
        ax.set_xlabel("Topology churn score")
        ax.set_ylabel(ylabel)
        if "ratio" in metric:
            ax.set_ylim(0, 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", ncol=1, bbox_to_anchor=(1.14, 0.5))
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    return fig


def plot_f7_reward_curve(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "mean_reward" not in df.columns:
        return notice_figure("F7 MASAC reward unavailable", ["reward_curve.csv from stress training/evaluation is required."])
    df = df.copy()
    df["mean_reward"] = pd.to_numeric(df["mean_reward"], errors="coerce")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").fillna(0)
    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0)
    df = df[df["method"].eq("proposed_semantic_topology_marl")].copy() if "method" in df.columns else df
    df["x"] = df["episode"] + df["step"] / max(1.0, float(df["step"].max() or 1.0))
    fig, ax = plt.subplots(figsize=(3.55, 2.55), constrained_layout=True)
    for method in ordered_values(df["method"].dropna().unique(), PAPER_METHOD_ORDER):
        sub = df[df["method"].eq(method)]
        if sub.empty:
            continue
        grouped = sub.groupby("episode")["mean_reward"].agg(["mean", "count", "std"]).reset_index()
        ci = 1.96 * grouped["std"].fillna(0) / np.sqrt(grouped["count"].clip(lower=1))
        y = grouped["mean"].rolling(window=3, min_periods=1).mean()
        ax.plot(grouped["episode"], y, color=method_color(method), label=method_label(method))
        ax.fill_between(grouped["episode"], y - ci, y + ci, color=method_color(method), alpha=0.15, linewidth=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward")
    ax.set_title("MASAC training on stress scenarios")
    return fig


def plot_f8_overhead_pareto(df: pd.DataFrame) -> plt.Figure:
    if df.empty or "payload_bytes" not in df.columns:
        return notice_figure("F8 overhead/performance unavailable", ["message_overhead.csv and summary performance rows are required."])
    df = df.copy()
    for col in ["payload_bytes", "message_count", "success_ratio", "qos_hit_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    perf_col = "qos_hit_ratio" if "qos_hit_ratio" in df.columns else "success_ratio"
    if perf_col not in df.columns:
        return notice_figure("F8 overhead/performance unavailable", ["No success_ratio or qos_hit_ratio column found for Pareto y-axis."])
    grouped = df.groupby(["method", "scenario"], as_index=False).agg(payload_bytes=("payload_bytes", "mean"), performance=(perf_col, "mean"))
    grouped["payload_kb"] = grouped["payload_bytes"] / 1024.0
    methods = ordered_values(grouped["method"].dropna().unique(), PAPER_METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for method in methods:
        sub = grouped[grouped["method"].eq(method)]
        ax.scatter(
            sub["payload_kb"],
            sub["performance"],
            label=method_label(method),
            color=method_color(method),
            marker=METHOD_MARKERS.get(method, "o"),
            s=34 if method != "proposed_semantic_topology_marl" else 62,
            edgecolor="#222222",
            linewidth=0.35,
        )
    draw_pareto_front(ax, grouped["payload_kb"].to_numpy(dtype=float), grouped["performance"].to_numpy(dtype=float))
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Semantic exchange overhead (KB/run, symlog)")
    ax.set_ylabel("QoS hit ratio" if perf_col == "qos_hit_ratio" else "SFC success ratio")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="center right", bbox_to_anchor=(1.28, 0.5))
    return fig


def save_semantic_figure(func, output_dir: Path, stem: str, fmt: str, dpi: int, quality: Mapping[str, Any], *args) -> List[Dict[str, Any]]:
    if not bool(quality.get("paper_ready", False)):
        raise RuntimeError(f"{stem} is not paper-ready: {quality.get('risk', 'unknown risk')}")
    fig = func(*args)
    outputs: List[Path] = []
    formats = ["pdf", "png"] if fmt == "both" else [fmt]
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi)
        outputs.append(path)
    plt.close(fig)
    return [
        {
            "figure": stem.split("_", 1)[0],
            "name": stem,
            "path": str(path),
            "bytes": path.stat().st_size,
            **dict(quality),
        }
        for path in outputs
    ]


def figure_quality(
    stem: str,
    tables: Mapping[str, pd.DataFrame],
    input_dir: Path,
    ttl_input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if stem.startswith("F1"):
        return {
            "paper_ready": True,
            "risk": "",
            "source": "manual_matplotlib_architecture",
            "source_role": "manual",
            "source_input_dir": "",
            "method_count": 0,
            "scenario_count": 0,
        }
    table_key = {
        "F2": "ablation_summary",
        "F3": "latency_samples",
        "F4": "candidate_quality",
        "F5": "discovery_ttl_radius",
        "F6": "topology_churn_reward",
        "F7": "reward_curve",
        "F8": "message_pareto",
    }.get(stem.split("_", 1)[0], "")
    df = tables.get(table_key, pd.DataFrame())
    risk: List[str] = []
    paper_ready = not df.empty
    if df.empty:
        risk.append("missing_source_table")
    methods = set(df.get("method", pd.Series(dtype=str)).astype(str)) if not df.empty else set()
    scenarios = {str(item) for item in df.get("scenario", pd.Series(dtype=str)).astype(str) if "calibration" not in str(item)} if not df.empty else set()
    figure_key = stem.split("_", 1)[0]
    source_role = "ttl_radius_sweep" if figure_key == "F5" and ttl_input_dir is not None else "main_matrix"
    source_input = ttl_input_dir if source_role == "ttl_radius_sweep" else input_dir
    if not _is_runtime_raw_source(source_input):
        risk.append("source_not_runtime_raw_data")
        paper_ready = False
    if figure_key != "F7" and not df.empty:
        min_cell_seeds = _min_method_scenario_seed_count(df)
        required_seeds = 3 if figure_key == "F5" else 10
        if min_cell_seeds < required_seeds:
            risk.append(f"insufficient_seed_count_min_{min_cell_seeds}")
            paper_ready = False
    if stem.startswith("F2") and not df.empty:
        values = pd.to_numeric(df.get("success_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(values) and (values.max() == 0 or values.min() == 1):
            risk.append("all_zero_or_all_one_success")
            paper_ready = False
        missing_methods = [method for method in METHOD_ORDER if method not in methods]
        if missing_methods:
            risk.append("missing_main_matrix_methods")
            paper_ready = False
        if len(scenarios) < 2:
            risk.append("missing_two_stress_scenarios")
            paper_ready = False
        if not _proposed_exceeds_topology_on_ratio(df):
            risk.append("proposed_not_above_topology_success_or_deadline")
            paper_ready = False
    if stem.startswith("F3") and not df.empty:
        positive = df[pd.to_numeric(df.get("latency_s", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0]
        positive_methods = set(positive.get("method", pd.Series(dtype=str)).astype(str))
        if "proposed_semantic_topology_marl" not in positive_methods or len(positive_methods) < 3:
            risk.append("positive_latency_missing_for_proposed_or_comparators")
            paper_ready = False
        if not _proposed_improves_tail_latency(positive):
            risk.append("proposed_tail_latency_not_better_than_topology")
            paper_ready = False
    if stem.startswith("F4") and not df.empty:
        if "rank" not in df.columns or pd.to_numeric(df["rank"], errors="coerce").max() <= 1:
            risk.append("top_k_detail_missing_or_degenerate")
            paper_ready = False
    if stem.startswith("F5") and not df.empty:
        ttl_n = pd.to_numeric(df.get("exchange_ttl_s", pd.Series(dtype=float)), errors="coerce").nunique()
        radius_n = pd.to_numeric(df.get("exchange_radius_hops", pd.Series(dtype=float)), errors="coerce").nunique()
        if ttl_n <= 1 or radius_n <= 1:
            risk.append("ttl_radius_sweep_missing")
            paper_ready = False
        remote_span = pd.to_numeric(df.get("remote_discovery_rate", pd.Series(dtype=float)), errors="coerce").max() - pd.to_numeric(df.get("remote_discovery_rate", pd.Series(dtype=float)), errors="coerce").min()
        stale_span = pd.to_numeric(df.get("stale_ratio", pd.Series(dtype=float)), errors="coerce").max() - pd.to_numeric(df.get("stale_ratio", pd.Series(dtype=float)), errors="coerce").min()
        if not (float(remote_span or 0.0) > 0.01 or float(stale_span or 0.0) > 0.01):
            risk.append("ttl_radius_metrics_constant")
            paper_ready = False
    if stem.startswith("F7") and not df.empty:
        rewards = pd.to_numeric(df.get("mean_reward", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(rewards) and rewards.nunique() <= 1:
            risk.append("reward_curve_constant")
            paper_ready = False
        episodes = pd.to_numeric(df.get("episode", pd.Series(dtype=float)), errors="coerce").dropna()
        if "paper_matrix" in str(input_dir) and (episodes.empty or episodes.max() <= 0):
            risk.append("training_curve_missing")
            paper_ready = False
    if stem.startswith("F8") and not df.empty:
        perf = pd.to_numeric(
            df.get("qos_hit_ratio", df.get("success_ratio", pd.Series(dtype=float))),
            errors="coerce",
        ).dropna()
        missing_methods = [method for method in METHOD_ORDER if method not in methods]
        if missing_methods:
            risk.append("missing_main_matrix_methods")
            paper_ready = False
        if len(scenarios) < 2:
            risk.append("missing_two_stress_scenarios")
            paper_ready = False
        if len(perf) and (perf.max() == 0 or perf.min() == 1):
            risk.append("all_zero_or_all_one_performance")
            paper_ready = False
    return {
        "paper_ready": bool(paper_ready),
        "source": f"figure_tables/{table_key}.csv" if table_key else "",
        "source_role": source_role,
        "source_input_dir": str(source_input),
        "method_count": int(len(methods)),
        "scenario_count": int(len(scenarios)),
        "rows": int(len(df)),
        "risk": ",".join(risk),
    }


def _is_runtime_raw_source(path: Optional[Path]) -> bool:
    text = str((path or Path("")).resolve())
    if "analysis/figures_req" in text or "plans/fig_ref" in text:
        return False
    return "experiment_artifacts/raw_data" in text


def _min_method_scenario_seed_count(df: pd.DataFrame) -> int:
    required = {"method", "scenario", "seed"}
    if df.empty or not required.issubset(df.columns):
        return 0
    working = df.copy()
    working["method"] = working["method"].astype(str)
    working["scenario"] = working["scenario"].astype(str)
    working["seed"] = working["seed"].astype(str)
    grouped = working.drop_duplicates(["method", "scenario", "seed"]).groupby(["method", "scenario"])["seed"].nunique()
    return int(grouped.min()) if len(grouped) else 0


def _proposed_exceeds_topology_on_ratio(df: pd.DataFrame) -> bool:
    if df.empty or "method" not in df.columns:
        return False
    method_col = df["method"].astype(str)
    proposed = df[method_col == "proposed_semantic_topology_marl"]
    topology = df[method_col.isin(["topology_greedy", "marl_topology_no_semantic"])]
    if proposed.empty or topology.empty:
        return False
    for metric in ("success_ratio", "deadline_hit_ratio", "qos_hit_ratio"):
        if metric not in df.columns:
            continue
        p = pd.to_numeric(proposed[metric], errors="coerce").dropna()
        t = pd.to_numeric(topology[metric], errors="coerce").dropna()
        if not p.empty and not t.empty and float(p.mean()) >= float(t.mean()) + 0.03:
            return True
    return False


def _proposed_improves_tail_latency(df: pd.DataFrame) -> bool:
    if df.empty or "method" not in df.columns or "latency_s" not in df.columns:
        return False
    method_col = df["method"].astype(str)
    proposed = pd.to_numeric(df.loc[method_col == "proposed_semantic_topology_marl", "latency_s"], errors="coerce").dropna()
    topology = pd.to_numeric(df.loc[method_col.isin(["topology_greedy", "marl_topology_no_semantic"]), "latency_s"], errors="coerce").dropna()
    if proposed.empty or topology.empty:
        return False
    return float(proposed.quantile(0.95)) <= float(topology.quantile(0.95)) - 0.50


def expected_figure_specs() -> Dict[str, Dict[str, str]]:
    return {
        "F1": {"x": "none", "y": "none", "legend": "module colors", "expected": "closed-loop semantic/topology/MARL/runtime architecture"},
        "F2": {"x": "method", "y": "success/deadline ratio", "legend": "method colors; scenario columns", "expected": "proposed close to centralized full-info planner and above ablations"},
        "F3": {"x": "E2E latency seconds", "y": "empirical CDF", "legend": "method", "expected": "proposed CDF left of weaker baselines"},
        "F4": {"x": "candidate rank / method", "y": "mean or selected semantic score", "legend": "method", "expected": "semantic-aware methods show higher top-k quality"},
        "F5": {"x": "TTL seconds or exchange radius hops", "y": "remote discovery rate / stale ratio", "legend": "method", "expected": "larger TTL/radius increases discovery and stale risk"},
        "F6": {"x": "topology churn score", "y": "reward / QoS hit ratio", "legend": "method", "expected": "topology-aware methods degrade less under churn"},
        "F7": {"x": "training episode", "y": "mean reward", "legend": "method/policy", "expected": "MASAC stress reward is non-constant and improves or stabilizes"},
        "F8": {"x": "semantic exchange overhead KB/run", "y": "QoS hit or success ratio", "legend": "method; scenario annotations for planner/proposed", "expected": "proposed near Pareto frontier"},
    }


def scenario_from_path(path: str) -> str:
    parts = Path(path).parts
    for part in parts:
        if "__seed_" in part and "__" in part:
            pieces = part.split("__")
            if len(pieces) >= 2:
                return pieces[1]
    return "default"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _truthy_string(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
