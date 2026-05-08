from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "proposed_semantic_topology_marl": "LASDM-ST-MARL",
    "topology_greedy": "Topology Greedy + Exchange",
    "pure_semantic_greedy_no_exchange": "Semantic-Only Greedy",
    "local_semantic_runtime_greedy": "Local Semantic Runtime",
    "nsga2_semantic_qos": "NSGA-II Semantic-QoS",
    "mappo_ctde": "MAPPO-CTDE",
    "iql_offline": "IQL (Replay)",
    "cross_region_auction": "Cross-Region Auction",
    "intra_region_only": "Intra-Region Only",
    "marl_no_semantic": "w/o Semantic",
    "marl_semantic_no_topology": "w/o Topology",
    "marl_no_exchange": "w/o Exchange",
    "marl_no_temporal": "w/o Temporal",
    "marl_no_cross_region": "w/o Cross-Region",
    "centralized_planner": "LASDM-Centralized",
}

COLORS = {
    "proposed_semantic_topology_marl": "#009E73",
    "topology_greedy": "#0072B2",
    "pure_semantic_greedy_no_exchange": "#E69F00",
    "local_semantic_runtime_greedy": "#56B4E9",
    "nsga2_semantic_qos": "#17becf",
    "mappo_ctde": "#e41a1c",
    "iql_offline": "#4daf4a",
    "cross_region_auction": "#D55E00",
    "intra_region_only": "#999999",
    "marl_no_semantic": "#E69F00",
    "marl_semantic_no_topology": "#56B4E9",
    "marl_no_exchange": "#D55E00",
    "marl_no_temporal": "#0072B2",
    "marl_no_cross_region": "#000000",
    "centralized_planner": "#000000",
}

MARKERS = {
    "proposed_semantic_topology_marl": "o",
    "topology_greedy": "^",
    "pure_semantic_greedy_no_exchange": "v",
    "local_semantic_runtime_greedy": "P",
    "nsga2_semantic_qos": "D",
    "mappo_ctde": "x",
    "iql_offline": "+",
    "cross_region_auction": "s",
    "intra_region_only": "X",
    "marl_no_semantic": "v",
    "marl_semantic_no_topology": "P",
    "marl_no_exchange": "s",
    "marl_no_temporal": "^",
    "marl_no_cross_region": "X",
    "centralized_planner": "*",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate figures required by plans/figures_req.md from real outputs.")
    parser.add_argument("--root", default="experiment_artifacts/raw_data/stage1_v11_full_20260502")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--config", default="methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    root = Path(args.root)
    eval_dir = Path(args.eval_dir) if args.eval_dir else root / "semantic_runtime_eval"
    train_dir = Path(args.train_dir) if args.train_dir else root / "semantic_runtime_train"
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures_req"
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib()
    data = FigureData(eval_dir=eval_dir, train_dir=train_dir, config_path=Path(args.config))
    manifest: list[dict[str, Any]] = []
    specs: list[tuple[str, Callable[[FigureData], tuple[plt.Figure, dict[str, Any]]]]] = [
        ("fig1a_service_nodes_success", plot_fig1a_service_nodes_success),
        ("fig1b_service_nodes_deadline", plot_fig1b_service_nodes_deadline),
        ("fig2a_task_nodes_success", plot_fig2a_task_nodes_success),
        ("fig2b_task_nodes_finish_time", plot_fig2b_task_nodes_finish_time),
        ("fig3a_skew_success", plot_fig3a_skew_success),
        ("fig3b_skew_fairness", plot_fig3b_skew_fairness),
        ("fig4a_cross_region_flow", plot_fig4a_cross_region_flow),
        ("fig4b_region_queue_delay", plot_fig4b_region_queue_delay),
        ("fig5_overhead_pareto", plot_fig5_overhead_pareto),
        ("fig6_ttl_tradeoff", plot_fig6_ttl_tradeoff),
        ("fig7_latency_cdf", plot_fig7_latency_cdf),
        ("fig8a_semantic_embedding", plot_fig8a_semantic_embedding),
        ("fig8b_precision_at_k", plot_fig8b_precision_at_k),
        ("fig9_ablation", plot_fig9_ablation),
        ("fig10_training_curve", plot_fig10_training_curve),
        ("fig11_uav_count", plot_fig11_uav_count),
        ("fig12_sfc_length", plot_fig12_sfc_length),
        ("fig13_speed", plot_fig13_speed),
        ("fig14_centralized_decentralized", plot_fig14_centralized_decentralized),
    ]
    for name, fn in specs:
        fig, meta = fn(data)
        paths = save_figure(fig, output_dir, name, args.format, args.dpi)
        manifest.append({"figure": name, "paths": paths, **meta})
        plt.close(fig)
    (output_dir / "figures_req_manifest.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "eval_dir": str(eval_dir),
                "train_dir": str(train_dir),
                "figures": manifest,
                "missing_or_risky": [item for item in manifest if item.get("status") != "ok"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "figures": len(manifest)}, indent=2))
    return 0


class FigureData:
    def __init__(self, eval_dir: Path, train_dir: Path, config_path: Path):
        self.eval_dir = eval_dir
        self.train_dir = train_dir
        self.config_path = config_path
        self.summary = self._summary()
        self.function_trace = read_csv(eval_dir / "function_execution_trace.csv")
        self.candidate_detail = read_csv(eval_dir / "semantic_candidate_detail_trace.csv")
        self.message_overhead = read_csv(eval_dir / "message_overhead.csv")
        self.lifecycle = read_csv(eval_dir / "runtime_task_lifecycle_trace.csv")

    def _summary(self) -> pd.DataFrame:
        summary = read_csv(self.eval_dir / "summary.csv")
        if summary.empty:
            return summary
        for col in [
            "success_ratio",
            "qos_hit_ratio",
            "avg_graph_finish_time",
            "p95_graph_finish_time",
            "submitted",
            "succeeded",
            "failed",
            "timed_out",
            "task_success_ratio",
            "queue_wait_time_mean",
        ]:
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce")
        return add_scenario_variables(summary)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
        }
    )


def add_scenario_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    names = df.get("scenario", pd.Series([""] * len(df))).astype(str)
    patterns = {
        "service_nodes": r"(?:service|svc)[_\-]?nodes[_\-]?(\d+)",
        "task_node_count": r"(?:task|source)[_\-]?nodes[_\-]?(\d+)",
        "skew_ratio": r"skew[_\-]?(\d+(?:p\d+)?|\d+(?:\.\d+)?)",
        "ttl_s": r"ttl[_\-]?(\d+(?:p\d+)?|\d+(?:\.\d+)?)",
        "uav_count": r"uav[_\-]?(\d+)",
        "speed_kmh": r"speed[_\-]?(\d+)",
        "sfc_length": r"(?:sfc|chain)[_\-]?(?:len|length|stage)[_\-]?(\d+)|(\d+)[_\-]?stage",
    }
    for col, pattern in patterns.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        values = []
        rx = re.compile(pattern, re.I)
        for name in names:
            match = rx.search(name)
            if not match:
                values.append(np.nan)
                continue
            token = next((item for item in match.groups() if item), "")
            values.append(float(str(token).replace("p", ".")))
        df[col] = values
    if "service_node_counts" in df.columns and df["service_nodes"].isna().all():
        df["service_nodes"] = df["service_node_counts"].map(lambda item: _sum_json_counts(item))
    if "task_node_count" in df.columns:
        df["task_node_count"] = pd.to_numeric(df["task_node_count"], errors="coerce")
    return df


def plot_fig1a_service_nodes_success(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "service_nodes",
        "success_ratio",
        ["proposed_semantic_topology_marl", "topology_greedy", "nsga2_semantic_qos", "cross_region_auction", "intra_region_only"],
        "Number of service nodes",
        "SFC success ratio",
        "Fig-1a Service nodes vs success",
    )


def plot_fig1b_service_nodes_deadline(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "service_nodes",
        "qos_hit_ratio",
        ["proposed_semantic_topology_marl", "topology_greedy", "nsga2_semantic_qos", "cross_region_auction", "intra_region_only"],
        "Number of service nodes",
        "Deadline hit ratio",
        "Fig-1b Service nodes vs deadline hit",
    )


def plot_fig2a_task_nodes_success(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "task_node_count",
        "success_ratio",
        ["proposed_semantic_topology_marl", "mappo_ctde", "iql_offline", "topology_greedy", "intra_region_only"],
        "Number of task nodes",
        "SFC success ratio",
        "Fig-2a Task-node load vs success",
    )


def plot_fig2b_task_nodes_finish_time(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "task_node_count",
        "avg_graph_finish_time",
        ["proposed_semantic_topology_marl", "mappo_ctde", "iql_offline", "topology_greedy", "intra_region_only"],
        "Number of task nodes",
        "Average SFC finish time (s)",
        "Fig-2b Task-node load vs finish time",
    )


def plot_fig3a_skew_success(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "skew_ratio",
        "success_ratio",
        ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction", "intra_region_only"],
        "Regional skew ratio",
        "SFC success ratio",
        "Fig-3a Regional skew vs success",
    )


def plot_fig3b_skew_fairness(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = data.summary
    if "jain_fairness" in df.columns:
        return line_metric(
            df,
            "skew_ratio",
            "jain_fairness",
            ["proposed_semantic_topology_marl", "cross_region_auction", "intra_region_only"],
            "Regional skew ratio",
            "Jain fairness index",
            "Fig-3b Regional skew vs fairness",
        )
    return notice("Fig-3b fairness unavailable", ["Summary lacks per-region success or jain_fairness. Run fairness-enabled sweep."])


def plot_fig4a_cross_region_flow(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = selected_candidates(data.candidate_detail)
    needed = {"agent_id", "region_id", "baseline"}
    if df.empty or not needed.issubset(df.columns):
        return notice("Fig-4a flow unavailable", ["semantic_candidate_detail_trace.csv with selected rows is required."])
    return flow_heatmaps(df, "Fig-4a Cross-region selected-candidate flow")


def plot_fig4b_region_queue_delay(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = selected_candidates(data.candidate_detail)
    metric = "expected_rb_wait_s" if "expected_rb_wait_s" in df.columns else "expected_runtime_penalty_s"
    if df.empty or not {"baseline", "region_id", metric}.issubset(df.columns):
        return notice("Fig-4b queue delay unavailable", ["Selected candidate detail needs region_id and queue/runtime penalty."])
    methods = ["intra_region_only", "cross_region_auction", "proposed_semantic_topology_marl"]
    table = (
        df[df["baseline"].isin(methods)]
        .assign(**{metric: pd.to_numeric(df[metric], errors="coerce")})
        .pivot_table(index="baseline", columns="region_id", values=metric, aggfunc="mean")
        .reindex(methods)
    )
    if table.dropna(how="all").empty:
        return notice("Fig-4b queue delay unavailable", ["No queue/runtime penalty values for selected candidates."])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    image = ax.imshow(table.fillna(0.0).to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), [label(m) for m in table.index])
    ax.set_title("Fig-4b Regional queue/runtime pressure")
    ax.set_xlabel("Region")
    ax.set_ylabel("Method")
    fig.colorbar(image, ax=ax, label=metric)
    return fig, {"status": "ok", "source": "semantic_candidate_detail_trace.csv", "metric": metric}


def plot_fig5_overhead_pareto(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    summary = data.summary
    if summary.empty or data.message_overhead.empty:
        return notice("Fig-5 overhead unavailable", ["summary.csv and message_overhead.csv are required."])
    overhead = data.message_overhead.copy()
    if "baseline" not in overhead.columns:
        return notice("Fig-5 overhead unavailable", ["message_overhead.csv must include baseline metadata."])
    overhead["payload_bytes"] = pd.to_numeric(overhead.get("payload_bytes"), errors="coerce").fillna(0.0)
    by_method = overhead.groupby("baseline", as_index=False)["payload_bytes"].sum()
    perf = summary.groupby("baseline", as_index=False).agg(success_ratio=("success_ratio", "mean"), submitted=("submitted", "sum"))
    merged = perf.merge(by_method, on="baseline", how="left").fillna({"payload_bytes": 0.0})
    merged["kb_per_sfc"] = merged["payload_bytes"] / 1024.0 / merged["submitted"].clip(lower=1)
    methods = [
        "intra_region_only",
        "cross_region_auction",
        "pure_semantic_greedy_no_exchange",
        "local_semantic_runtime_greedy",
        "nsga2_semantic_qos",
        "topology_greedy",
        "proposed_semantic_topology_marl",
    ]
    merged = merged[merged["baseline"].isin(methods)]
    if merged.empty:
        return notice("Fig-5 overhead unavailable", ["No requested methods found in summary/message overhead."])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for _, row in merged.iterrows():
        method = str(row["baseline"])
        ax.scatter(row["kb_per_sfc"], row["success_ratio"], color=color(method), marker=marker(method), s=42)
        ax.text(row["kb_per_sfc"], row["success_ratio"], label(method), fontsize=6, va="bottom")
    ax.set_xlabel("Message overhead (KB/SFC)")
    ax.set_ylabel("SFC success ratio")
    ax.set_title("Fig-5 Message overhead vs success")
    return fig, {"status": "ok", "source": "summary.csv,message_overhead.csv", "rows": int(len(merged))}


def plot_fig6_ttl_tradeoff(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    summary = data.summary
    cand = data.candidate_detail
    if summary.empty or cand.empty:
        return notice("Fig-6 TTL unavailable", ["summary.csv and semantic_candidate_detail_trace.csv are required."])
    if "exchange_ttl_s" not in cand.columns:
        return notice("Fig-6 TTL unavailable", ["candidate detail lacks exchange_ttl_s."])
    cand = cand.copy()
    cand["exchange_ttl_s"] = pd.to_numeric(cand["exchange_ttl_s"], errors="coerce")
    cand["is_remote"] = cand.get("is_remote", False).map(to_bool)
    cand["stale"] = cand.get("stale", False).map(to_bool)
    method = "proposed_semantic_topology_marl"
    csub = cand[cand.get("baseline", method).eq(method)] if "baseline" in cand.columns else cand
    ttl_candidate = csub.groupby("exchange_ttl_s", as_index=False).agg(
        remote_discovery_rate=("is_remote", "mean"),
        stale_ratio=("stale", "mean"),
    )
    ssub = summary[summary["baseline"].eq(method)].copy()
    if ssub["ttl_s"].isna().all() and "exchange_ttl_s" in ssub.columns:
        ssub["ttl_s"] = pd.to_numeric(ssub["exchange_ttl_s"], errors="coerce")
    ttl_success = ssub.dropna(subset=["ttl_s"]).groupby("ttl_s", as_index=False)["success_ratio"].mean()
    if ttl_candidate["exchange_ttl_s"].nunique() < 2 and ttl_success["ttl_s"].nunique() < 2:
        return notice("Fig-6 TTL unavailable", ["TTL sweep data has fewer than 2 TTL values."])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=False)
    axes[0].plot(ttl_candidate["exchange_ttl_s"], ttl_candidate["remote_discovery_rate"], marker="o", color=COLORS[method])
    axes[1].plot(ttl_candidate["exchange_ttl_s"], ttl_candidate["stale_ratio"], marker="o", color="#D55E00")
    axes[2].plot(ttl_success["ttl_s"], ttl_success["success_ratio"], marker="o", color=COLORS[method])
    for ax, title, ylabel in zip(
        axes,
        ["Remote discovery", "Selected staleness", "Success"],
        ["Rate", "Ratio", "SFC success ratio"],
    ):
        ax.set_title(title)
        ax.set_xlabel("TTL (s)")
        ax.set_ylabel(ylabel)
    fig.suptitle("Fig-6 TTL trade-off", y=1.05)
    return fig, {"status": "ok", "source": "summary.csv,semantic_candidate_detail_trace.csv"}


def plot_fig7_latency_cdf(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = data.function_trace
    if df.empty or "e2e_delay" not in df.columns:
        return notice("Fig-7 latency CDF unavailable", ["function_execution_trace.csv with e2e_delay is required."])
    methods = ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction", "intra_region_only"]
    df = df[df["baseline"].isin(methods)].copy()
    df["e2e_delay"] = pd.to_numeric(df["e2e_delay"], errors="coerce")
    df = df[df["e2e_delay"] > 0]
    if df.empty:
        return notice("Fig-7 latency CDF unavailable", ["No positive e2e_delay samples."])
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for method in methods:
        vals = np.sort(df.loc[df["baseline"].eq(method), "e2e_delay"].dropna().to_numpy(dtype=float))
        if vals.size == 0:
            continue
        ax.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=label(method), color=color(method), marker=None)
    ax.set_xlabel("SFC/function finish time (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Fig-7 Latency distribution")
    ax.legend()
    return fig, {"status": "ok", "source": "function_execution_trace.csv", "samples": int(len(df))}


def plot_fig8a_semantic_embedding(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    rows = semantic_embedding_rows(data.config_path)
    if not rows:
        return notice("Fig-8a embedding unavailable", ["Could not load service texts from config."])
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, backend in zip(axes, ["hash", "sbert"]):
        sub = frame[frame["backend"].eq(backend)]
        if sub.empty:
            ax.text(0.5, 0.5, f"{backend} unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        for service_id, group in sub.groupby("service_id"):
            ax.scatter(group["x"], group["y"], label=service_id, s=28)
        ax.set_title(backend.upper())
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
    axes[1].legend(fontsize=6, loc="best")
    fig.suptitle("Fig-8a Hash vs SBERT semantic embedding", y=1.04)
    return fig, {"status": "ok", "source": "SemanticEncoder.encode", "rows": int(len(frame))}


def plot_fig8b_precision_at_k(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    precision = semantic_precision_rows(data.config_path)
    if precision.empty:
        return notice("Fig-8b precision unavailable", ["Could not compute encoder ranking precision from config."])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    services = list(precision["service_id"].drop_duplicates())
    width = 0.11
    offsets = np.linspace(-2.5 * width, 2.5 * width, 6)
    labels_order = [("hash", 1), ("sbert", 1), ("hash", 3), ("sbert", 3), ("hash", 5), ("sbert", 5)]
    x = np.arange(len(services))
    for idx, (backend, k) in enumerate(labels_order):
        vals = [
            float(precision.loc[(precision["service_id"].eq(svc)) & (precision["backend"].eq(backend)) & (precision["k"].eq(k)), "precision"].mean())
            for svc in services
        ]
        ax.bar(x + offsets[idx], vals, width=width, label=f"{backend.upper()}@{k}")
    ax.set_xticks(x, services, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision@K")
    ax.set_title("Fig-8b Candidate ranking precision")
    ax.legend(ncol=3, fontsize=6)
    return fig, {"status": "ok", "source": "SemanticEncoder.similarity", "rows": int(len(precision))}


def plot_fig9_ablation(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = data.summary
    methods = [
        "proposed_semantic_topology_marl",
        "marl_no_semantic",
        "marl_semantic_no_topology",
        "marl_no_exchange",
        "marl_no_temporal",
        "marl_no_cross_region",
    ]
    sub = df[df["baseline"].isin(methods)].copy()
    if sub.empty:
        return notice("Fig-9 ablation unavailable", ["Stage2 summary lacks ablation baselines."])
    grouped = sub.groupby("baseline", as_index=False).agg(success_ratio=("success_ratio", "mean"), qos_hit_ratio=("qos_hit_ratio", "mean"))
    grouped["baseline"] = pd.Categorical(grouped["baseline"], categories=methods, ordered=True)
    grouped = grouped.sort_values("baseline")
    x = np.arange(len(grouped))
    fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, grouped["success_ratio"], width=0.34, color="#0072B2", label="Success")
    ax2.bar(x + 0.18, grouped["qos_hit_ratio"], width=0.34, color="#E69F00", label="Deadline")
    ax1.set_xticks(x, [label(str(v)) for v in grouped["baseline"]], rotation=25, ha="right")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.set_ylabel("SFC success ratio")
    ax2.set_ylabel("Deadline hit ratio")
    ax1.set_title("Fig-9 Module ablation")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    return fig, {"status": "ok", "source": "summary.csv", "rows": int(len(sub))}


def plot_fig10_training_curve(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    if not data.train_dir.exists():
        return notice("Fig-10 training unavailable", ["Training root missing."])
    progress = []
    checkpoints = []
    for path in data.train_dir.glob("**/ippo_seed_*/train_progress.csv"):
        variant = path.parent.parent.name if path.parent.parent.name != "semantic_runtime_train" else "proposed_semantic_topology_marl"
        if variant != "proposed_semantic_topology_marl":
            continue
        frame = read_csv(path)
        if not frame.empty:
            frame["variant"] = variant
            progress.append(frame)
    for path in data.train_dir.glob("**/ippo_seed_*/checkpoint_selection.csv"):
        variant = path.parent.parent.name if path.parent.parent.name != "semantic_runtime_train" else "proposed_semantic_topology_marl"
        if variant != "proposed_semantic_topology_marl":
            continue
        frame = read_csv(path)
        if not frame.empty:
            frame["variant"] = variant
            checkpoints.append(frame)
    if not progress:
        return notice("Fig-10 training unavailable", ["No train_progress.csv for proposed method."])
    prog = pd.concat(progress, ignore_index=True)
    prog["episode"] = pd.to_numeric(prog["episode"], errors="coerce")
    prog["total_reward"] = pd.to_numeric(prog.get("total_reward", prog.get("mean_reward")), errors="coerce")
    reward = prog.groupby("episode", as_index=False)["total_reward"].mean().sort_values("episode")
    reward["smooth"] = reward["total_reward"].rolling(10, min_periods=1).mean()
    fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
    ax1.plot(reward["episode"], reward["smooth"], color="#0072B2", label="Smoothed reward")
    ax1.axvline(12, linestyle="--", color="#666666", linewidth=1.0)
    ax1.text(12, ax1.get_ylim()[1], "Progress guard", ha="left", va="top", fontsize=7)
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Smoothed reward")
    ax1.set_title("Fig-10 Training curve")
    if checkpoints:
        ckpt = pd.concat(checkpoints, ignore_index=True)
        ckpt["episode"] = pd.to_numeric(ckpt["episode"], errors="coerce")
        ckpt["success_ratio"] = pd.to_numeric(ckpt["success_ratio"], errors="coerce")
        success = ckpt.groupby("episode", as_index=False)["success_ratio"].mean().sort_values("episode")
        ax2 = ax1.twinx()
        ax2.plot(success["episode"], success["success_ratio"], color="#009E73", marker="o", label="Validation success")
        ax2.set_ylabel("Validation success ratio")
        ax2.set_ylim(0, 1.05)
    return fig, {"status": "ok", "source": "train_progress.csv,checkpoint_selection.csv", "rows": int(len(prog))}


def plot_fig11_uav_count(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "uav_count",
        "success_ratio",
        ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction"],
        "Number of service UAVs",
        "SFC success ratio",
        "Fig-11 UAV count sweep",
    )


def plot_fig12_sfc_length(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = data.summary
    if df.empty or df["sfc_length"].dropna().nunique() < 2 or df["task_node_count"].dropna().nunique() < 2:
        return notice("Fig-12 SFC length unavailable", ["Need scenarios containing sfc_length and task-node sweeps."])
    methods = ["proposed_semantic_topology_marl", "topology_greedy"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for method in methods:
        for length, linestyle in [(3, "-"), (4, "--")]:
            sub = df[(df["baseline"].eq(method)) & (df["sfc_length"].eq(length))]
            if sub.empty:
                continue
            grouped = sub.groupby("task_node_count", as_index=False)["success_ratio"].mean().sort_values("task_node_count")
            ax.plot(
                grouped["task_node_count"],
                grouped["success_ratio"],
                label=f"{label(method)}-{length}stage",
                color=color(method),
                marker=marker(method),
                linestyle=linestyle,
            )
    ax.set_xlabel("Number of task nodes")
    ax.set_ylabel("SFC success ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig-12 SFC length impact")
    ax.legend(fontsize=6)
    return fig, {"status": "ok", "source": "summary.csv"}


def plot_fig13_speed(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    return line_metric(
        data.summary,
        "speed_kmh",
        "success_ratio",
        ["proposed_semantic_topology_marl", "marl_no_temporal", "topology_greedy"],
        "Average vehicle speed (km/h)",
        "SFC success ratio",
        "Fig-13 Vehicle speed sweep",
    )


def plot_fig14_centralized_decentralized(data: FigureData) -> tuple[plt.Figure, dict[str, Any]]:
    df = data.summary
    methods = ["centralized_planner", "proposed_semantic_topology_marl", "topology_greedy"]
    sub = df[df["baseline"].isin(methods)].copy()
    if sub.empty:
        return notice("Fig-14 centralized unavailable", ["Summary lacks centralized/proposed/topology rows."])
    scenario_mask = sub["scenario"].astype(str).str.contains("contention|mobility", case=False, regex=True)
    if scenario_mask.any():
        sub = sub[scenario_mask]
    grouped = sub.groupby(["scenario", "baseline"], as_index=False)["success_ratio"].mean()
    scenarios = list(grouped["scenario"].drop_duplicates())[:2]
    if not scenarios:
        return notice("Fig-14 centralized unavailable", ["No contention/mobility scenarios available."])
    x = np.arange(len(scenarios))
    width = 0.23
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for idx, method in enumerate(methods):
        vals = [
            float(grouped.loc[(grouped["scenario"].eq(s)) & (grouped["baseline"].eq(method)), "success_ratio"].mean())
            for s in scenarios
        ]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=label(method), color=color(method))
    ax.set_xticks(x, [short_scenario(s) for s in scenarios])
    ax.set_ylabel("SFC success ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig-14 Centralized vs decentralized")
    ax.legend(fontsize=6)
    return fig, {"status": "ok", "source": "summary.csv", "scenarios": scenarios}


def line_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    methods: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return notice(f"{title} unavailable", [f"summary.csv must contain {x_col} and {y_col}."])
    sub = df[df["baseline"].isin(methods)].dropna(subset=[x_col, y_col]).copy()
    if sub[x_col].nunique() < 2:
        return notice(f"{title} unavailable", [f"Need at least two {x_col} values; found {sub[x_col].nunique()}."])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for method in methods:
        frame = sub[sub["baseline"].eq(method)]
        if frame.empty:
            continue
        grouped = frame.groupby(x_col, as_index=False)[y_col].agg(["mean", "sem"]).reset_index().sort_values(x_col)
        ax.errorbar(
            grouped[x_col],
            grouped["mean"],
            yerr=grouped["sem"].fillna(0.0),
            label=label(method),
            color=color(method),
            marker=marker(method),
            capsize=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if "ratio" in y_col:
        ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=6)
    return fig, {"status": "ok", "source": "summary.csv", "x_values": int(sub[x_col].nunique()), "rows": int(len(sub))}


def flow_heatmaps(df: pd.DataFrame, title: str) -> tuple[plt.Figure, dict[str, Any]]:
    methods = ["proposed_semantic_topology_marl", "intra_region_only"]
    regions = sorted(set(df["agent_id"].dropna().astype(str)) | set(df["region_id"].dropna().astype(str)))
    regions = [item for item in regions if item.startswith("RSU_")][:4] or ["RSU_0", "RSU_1", "RSU_2", "RSU_3"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, method in zip(axes, methods):
        sub = df[df["baseline"].eq(method)] if "baseline" in df.columns else df
        matrix = np.zeros((len(regions), len(regions)), dtype=float)
        for i, src in enumerate(regions):
            src_rows = sub[sub["agent_id"].astype(str).eq(src)]
            denom = max(1, len(src_rows))
            for j, dst in enumerate(regions):
                matrix[i, j] = float(src_rows["region_id"].astype(str).eq(dst).sum()) / denom
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(regions)), regions, rotation=30, ha="right")
        ax.set_yticks(range(len(regions)), regions)
        ax.set_title(label(method))
        ax.set_xlabel("Destination region")
    axes[0].set_ylabel("Source/agent region")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Placement fraction")
    fig.suptitle(title, y=1.04)
    return fig, {"status": "ok", "source": "semantic_candidate_detail_trace.csv", "rows": int(len(df))}


def selected_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "selected" not in df.columns:
        return pd.DataFrame()
    out = df[df["selected"].map(to_bool)].copy()
    for col in ["expected_rb_wait_s", "expected_runtime_penalty_s", "exchange_ttl_s"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def semantic_embedding_rows(config_path: Path) -> list[dict[str, Any]]:
    texts = service_texts_from_config(config_path)
    if not texts:
        return []
    rows: list[dict[str, Any]] = []
    for backend in ["hash", "sbert"]:
        vectors = encode_texts(texts, backend)
        if vectors is None:
            continue
        coords = project_2d(vectors)
        for item, xy in zip(texts, coords):
            rows.append({"backend": backend, "service_id": item["service_id"], "x": float(xy[0]), "y": float(xy[1])})
    return rows


def semantic_precision_rows(config_path: Path) -> pd.DataFrame:
    texts = service_texts_from_config(config_path)
    if not texts:
        return pd.DataFrame()
    rows = []
    service_ids = sorted({item["service_id"] for item in texts})
    for backend in ["hash", "sbert"]:
        vectors = encode_texts(texts, backend)
        if vectors is None:
            continue
        norm = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
        for service_id in service_ids:
            query_items = [idx for idx, item in enumerate(texts) if item["service_id"] == service_id]
            if not query_items:
                continue
            hits_at_k = {1: [], 3: [], 5: []}
            for q_idx in query_items:
                scores = norm @ norm[q_idx]
                order = [idx for idx in np.argsort(scores)[::-1] if idx != q_idx]
                for k in hits_at_k:
                    top = order[:k]
                    hits_at_k[k].append(float(any(texts[idx]["service_id"] == service_id for idx in top)))
            for k, vals in hits_at_k.items():
                rows.append({"backend": backend, "service_id": service_id, "k": k, "precision": float(np.mean(vals))})
    return pd.DataFrame(rows)


def service_texts_from_config(config_path: Path) -> list[dict[str, str]]:
    try:
        import yaml
    except Exception:
        return []
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    base = cfg
    if "schema_version" in base and "service_chains" not in base:
        base_path = config_path.parent / "semantic_topology_marl.yaml"
        try:
            base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    rows = []
    for chain in base.get("service_chains", []) or []:
        for node in chain.get("nodes", []) or []:
            service_id = str(node.get("service_type") or node.get("service_id") or node.get("node_id", ""))
            text = " ".join(
                [
                    service_id,
                    str(node.get("input_semantic", "")),
                    str(node.get("output_semantic", "")),
                    " ".join(str(item) for item in node.get("required_capabilities", []) or []),
                ]
            )
            for suffix in ["rsu reliable", "uav mobile", "vehicle edge", "cloud high capacity"]:
                rows.append({"service_id": service_id, "text": f"{text} {suffix}"})
    return rows


def encode_texts(items: Sequence[Mapping[str, str]], backend: str) -> np.ndarray | None:
    try:
        sys.path.insert(0, str(Path("AirFogSim").resolve()))
        from airfogsim.lasdm.semantic_encoder import SemanticEncoder

        encoder = SemanticEncoder(backend=backend, device="cuda" if backend == "sbert" else None)
        return np.asarray(encoder.encode([str(item["text"]) for item in items]), dtype=np.float32)
    except Exception:
        if backend == "hash":
            return None
        return None


def project_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 2), dtype=float)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except Exception:
        coords = centered[:, :2]
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords[:, :2]


def notice(title: str, lines: Sequence[str]) -> tuple[plt.Figure, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.set_axis_off()
    ax.text(0.03, 0.82, title, fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.62, "\n".join(lines), fontsize=8, va="top", transform=ax.transAxes)
    return fig, {"status": "missing_data", "reason": "; ".join(lines)}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_figure(fig: plt.Figure, output_dir: Path, name: str, fmt: str, dpi: int) -> list[str]:
    formats = ["png", "pdf"] if fmt == "both" else [fmt]
    paths = []
    for item in formats:
        path = output_dir / f"{name}.{item}"
        fig.savefig(path, dpi=dpi)
        paths.append(str(path))
    return paths


def label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def color(method: str) -> str:
    return COLORS.get(str(method), "#666666")


def marker(method: str) -> str:
    return MARKERS.get(str(method), "o")


def short_scenario(name: str) -> str:
    text = str(name)
    text = text.replace("semantic_runtime_", "").replace("_calibrated", "").replace("_stress", "")
    return text.replace("_", "\n")


def _sum_json_counts(value: Any) -> float:
    try:
        if isinstance(value, str):
            payload = json.loads(value)
        elif isinstance(value, Mapping):
            payload = value
        else:
            payload = ast.literal_eval(str(value))
        return float(sum(float(v) for v in dict(payload).values()))
    except Exception:
        return math.nan


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
