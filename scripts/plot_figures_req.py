from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "intra_region_only": "Intra-Region Only",
    "cross_region_auction": "Cross-Region Auction",
    "nsga2_semantic_qos": "NSGA-II Semantic-QoS",
    "topology_greedy": "Topology Greedy + Exchange",
    "mappo_ctde": "MAPPO-CTDE",
    "iql_offline": "IQL (Replay)",
    "marl_semantic_no_topology": "MARL-sem (w/o Topology)",
    "marl_no_semantic": "MARL-topo (w/o Semantic)",
    "marl_no_exchange": "MARL-no-exchange",
    "marl_no_temporal": "MARL-no-temporal",
    "marl_no_cross_region": "MARL-no-cross-region",
    "proposed_semantic_topology_marl": "LASDM-ST-MARL (Ours)",
    "centralized_planner": "LASDM-Centralized (Upper Bound)",
}

METHOD_STYLE = {
    "intra_region_only": {"color": "#E69F00", "linewidth": 1.5, "marker": "o", "markersize": 6, "linestyle": "--"},
    "cross_region_auction": {"color": "#D55E00", "linewidth": 1.5, "marker": "s", "markersize": 6, "linestyle": "-"},
    "nsga2_semantic_qos": {"color": "#17becf", "linewidth": 1.5, "marker": "D", "markersize": 7, "linestyle": "-."},
    "topology_greedy": {"color": "#0072B2", "linewidth": 1.5, "marker": "^", "markersize": 6, "linestyle": "-"},
    "mappo_ctde": {"color": "#e41a1c", "linewidth": 1.5, "marker": "x", "markersize": 7, "linestyle": ":"},
    "iql_offline": {"color": "#4daf4a", "linewidth": 1.5, "marker": "+", "markersize": 8, "linestyle": ":"},
    "marl_semantic_no_topology": {"color": "#CC79A7", "linewidth": 1.5, "marker": "D", "markersize": 7, "linestyle": ":"},
    "marl_no_semantic": {"color": "#56B4E9", "linewidth": 1.5, "marker": "P", "markersize": 7, "linestyle": ":"},
    "marl_no_exchange": {"color": "#F0E442", "linewidth": 1.5, "marker": "v", "markersize": 6, "linestyle": ":"},
    "marl_no_temporal": {"color": "#D55E00", "linewidth": 1.5, "marker": "<", "markersize": 6, "linestyle": ":"},
    "marl_no_cross_region": {"color": "#999999", "linewidth": 1.5, "marker": ">", "markersize": 6, "linestyle": ":"},
    "proposed_semantic_topology_marl": {"color": "#009E73", "linewidth": 2.5, "marker": "*", "markersize": 10, "linestyle": "-"},
    "centralized_planner": {"color": "#000000", "linewidth": 2.0, "marker": "X", "markersize": 10, "linestyle": ":"},
}

ABLATION_HATCH = {
    "proposed_semantic_topology_marl": "",
    "marl_no_semantic": "xx",
    "marl_semantic_no_topology": "..",
    "marl_no_exchange": "//",
    "marl_no_temporal": "\\\\",
    "marl_no_cross_region": "oo",
}

SUITE_METHODS = {
    "service_nodes": [
        "proposed_semantic_topology_marl",
        "topology_greedy",
        "nsga2_semantic_qos",
        "cross_region_auction",
        "intra_region_only",
    ],
    "task_nodes": ["proposed_semantic_topology_marl", "mappo_ctde", "iql_offline", "topology_greedy", "intra_region_only"],
    "ablation": [
        "proposed_semantic_topology_marl",
        "marl_no_semantic",
        "marl_semantic_no_topology",
        "marl_no_exchange",
        "marl_no_temporal",
        "marl_no_cross_region",
    ],
    "skew": ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction", "intra_region_only"],
    "ttl": ["proposed_semantic_topology_marl"],
    "uav": ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction"],
    "speed": ["proposed_semantic_topology_marl", "marl_no_temporal", "topology_greedy"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot figures required by plans/figures_req.md from real sweep summaries.")
    parser.add_argument("--eval-root", required=True, help="Output root produced by scripts/run_figures_req_eval.py.")
    parser.add_argument("--train-root", default=None, help="semantic_runtime_train root containing reward_curve.csv files for F7.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--reward-window", type=int, default=20, help="Moving-average window for F7 episode total reward.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    suites = {name: _load_suite_summary(eval_root, name) for name in SUITE_METHODS}
    manifest: List[Dict[str, Any]] = []
    specs = [
        ("F1_service_nodes_sweep", lambda: plot_f1(suites["service_nodes"]), "service_nodes"),
        ("F2_task_nodes_sweep", lambda: plot_f2(suites["task_nodes"]), "task_nodes"),
        ("F3_ablation_comparison", lambda: plot_f3(suites["ablation"]), "ablation"),
        ("F4_regional_skew_sweep", lambda: plot_dual_metric_sweep(suites["skew"], "skew", "Regional skew ratio"), "skew"),
        ("F5_ttl_sweep", lambda: plot_f5(suites["ttl"]), "ttl"),
        ("F6_uav_sweep", lambda: plot_dual_metric_sweep(suites["uav"], "uav", "UAV count"), "uav"),
        ("F7_training_reward_curve", lambda: plot_f7(Path(args.train_root) if args.train_root else None, args.reward_window), "training"),
        ("F7_2_training_sfc_completion_curve", lambda: plot_f7_2(Path(args.train_root) if args.train_root else None, args.reward_window), "training"),
        ("F8_speed_sweep", lambda: plot_dual_metric_sweep(suites["speed"], "speed", "Speed (km/h)"), "speed"),
    ]
    for stem, plotter, suite in specs:
        fig = plotter()
        paths = _save_figure(fig, output_dir, stem, args.format, args.dpi)
        plt.close(fig)
        df = suites.get(suite, pd.DataFrame())
        quality = _figure_quality(df, suite, eval_root, args.train_root)
        manifest.append(
            {
                "figure": _figure_id(stem),
                "name": stem,
                "outputs": {path.suffix.lstrip("."): str(path) for path in paths},
                "bytes": {path.suffix.lstrip("."): path.stat().st_size for path in paths},
                **quality,
            }
        )

    payload = {
        "eval_root": str(eval_root),
        "train_root": str(args.train_root or ""),
        "figures": manifest,
        "suite_rows": {name: int(len(df)) for name, df in suites.items()},
    }
    (output_dir / "figure_data_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "figures": len(manifest), "suite_rows": payload["suite_rows"]}, indent=2))
    return 0


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#cccccc",
            "legend.framealpha": 0.85,
            "legend.edgecolor": "#cccccc",
        }
    )


def _figure_id(stem: str) -> str:
    match = re.match(r"^(F\d+(?:_\d+)?)_", stem)
    return match.group(1) if match else stem.split("_", 1)[0]


def _load_suite_summary(eval_root: Path, suite: str) -> pd.DataFrame:
    candidates = [
        eval_root / "suites" / suite / "semantic_runtime_eval" / "summary.csv",
        eval_root / suite / "semantic_runtime_eval" / "summary.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["figure_suite"] = suite
            return _normalize_summary(df)
    combined = eval_root / "semantic_runtime_eval" / "summary.csv"
    if combined.exists():
        df = pd.read_csv(combined)
        if "figure_suite" in df.columns:
            return _normalize_summary(df[df["figure_suite"].astype(str).eq(suite)].copy())
    return pd.DataFrame()


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "seed",
        "submitted",
        "succeeded",
        "failed",
        "timed_out",
        "success_ratio",
        "task_success_ratio",
        "avg_graph_finish_time",
        "remote_candidate_ratio",
        "stale_selected_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_f1(df: pd.DataFrame) -> plt.Figure:
    return _dual_line_figure(
        df,
        suite="service_nodes",
        x_label="Service node count",
        title_left="SFC success vs service nodes",
        title_right="Task success vs service nodes",
        figsize=(7.0, 3.5),
    )


def plot_f2(df: pd.DataFrame) -> plt.Figure:
    if df.empty:
        return _notice("F2 unavailable", ["suite=task_nodes summary.csv is missing."])
    work = _with_x(df, "task_nodes")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True)
    _plot_metric_lines(axes[0], work, "success_ratio", SUITE_METHODS["task_nodes"], ylabel="SFC success ratio", ylim=(0, 1))
    _plot_metric_lines(axes[1], work, "task_success_ratio", SUITE_METHODS["task_nodes"], ylabel="Task success ratio", ylim=(0, 1))
    _plot_metric_lines(axes[2], work, "avg_graph_finish_time", SUITE_METHODS["task_nodes"], ylabel="Avg finish time (s)")
    axes[0].set_title("Task-node stress sweep")
    axes[2].set_xlabel("Task node count")
    _legend_below(fig, axes)
    return fig


def plot_f3(df: pd.DataFrame) -> plt.Figure:
    if df.empty:
        return _notice("F3 unavailable", ["suite=ablation summary.csv is missing."])
    methods = SUITE_METHODS["ablation"]
    grouped = df.groupby("baseline", as_index=True).agg(
        success_ratio=("success_ratio", "mean"),
        task_success_ratio=("task_success_ratio", "mean"),
    ).reindex(methods)
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for offset, metric, label in [(-width / 2, "success_ratio", "SFC success"), (width / 2, "task_success_ratio", "Task success")]:
        bars = ax.bar(x + offset, grouped[metric].fillna(0), width=width, label=label, edgecolor="black", linewidth=0.8)
        for bar, method in zip(bars, methods):
            bar.set_facecolor(METHOD_STYLE[method]["color"])
            bar.set_hatch(ABLATION_HATCH[method])
    ax.set_xticks(x, [METHOD_LABELS[m].replace(" (", "\n(") for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation comparison")
    ax.legend(loc="upper right")
    return fig


def plot_f5(df: pd.DataFrame) -> plt.Figure:
    if df.empty:
        return _notice("F5 unavailable", ["suite=ttl summary.csv is missing."])
    work = _with_x(df, "ttl")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    _plot_metric_lines(ax, work, "remote_candidate_ratio", SUITE_METHODS["ttl"], ylabel="Remote candidate ratio", ylim=(0, 1))
    stale = pd.to_numeric(work.get("stale_selected_ratio", pd.Series(dtype=float)), errors="coerce")
    if stale.dropna().abs().sum() > 0:
        ax2 = ax.twinx()
        _plot_metric_lines(ax2, work, "stale_selected_ratio", SUITE_METHODS["ttl"], ylabel="Stale selected ratio", ylim=(0, 1), legend=False)
    else:
        ax.text(0.01, 0.02, "stale_selected_ratio is all zero; omitted", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Exchange TTL (s)")
    ax.set_title("TTL exchange trade-off")
    _legend_below(fig, [ax])
    return fig


def plot_dual_metric_sweep(df: pd.DataFrame, suite: str, x_label: str) -> plt.Figure:
    return _dual_line_figure(
        df,
        suite=suite,
        x_label=x_label,
        title_left="SFC success ratio",
        title_right="Task success ratio",
        figsize=(7.0, 3.5),
    )


def _dual_line_figure(df: pd.DataFrame, suite: str, x_label: str, title_left: str, title_right: str, figsize: tuple[float, float]) -> plt.Figure:
    if df.empty:
        return _notice(f"{suite} unavailable", [f"suite={suite} summary.csv is missing."])
    work = _with_x(df, suite)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    _plot_metric_lines(axes[0], work, "success_ratio", SUITE_METHODS[suite], ylabel="SFC success ratio", ylim=(0, 1))
    _plot_metric_lines(axes[1], work, "task_success_ratio", SUITE_METHODS[suite], ylabel="Task success ratio", ylim=(0, 1))
    axes[0].set_title(title_left)
    axes[1].set_title(title_right)
    axes[0].set_xlabel(x_label)
    axes[1].set_xlabel(x_label)
    _legend_below(fig, axes)
    return fig


def plot_f7(train_root: Optional[Path], reward_window: int = 20) -> plt.Figure:
    data = _load_training_reward_data(train_root)
    if data.empty:
        return _notice("F7 unavailable", [f"No reward_curve.csv under {train_root}."])
    if "total_reward" not in data.columns and "reward" in data.columns:
        data["total_reward"] = data["reward"]
    methods = ["proposed_semantic_topology_marl", "mappo_ctde", "iql_offline", "marl_no_semantic", "marl_semantic_no_topology"]
    window = max(1, int(reward_window))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for method in methods:
        sub = data[data["method"].eq(method)].dropna(subset=["episode", "total_reward", "seed"])
        if sub.empty:
            continue
        sort_cols = ["seed", "episode"] + (["step"] if "step" in sub.columns else [])
        episode_totals = (
            sub.sort_values(sort_cols)
            .groupby(["seed", "episode"], as_index=False)
            .tail(1)[["seed", "episode", "total_reward"]]
            .sort_values(["seed", "episode"])
        )
        episode_totals["reward_ma"] = episode_totals.groupby("seed")["total_reward"].transform(
            lambda series: series.rolling(window=window, min_periods=1).mean()
        )
        grouped = episode_totals.groupby("episode")["reward_ma"].agg(["mean", "std", "count"]).reset_index()
        style = METHOD_STYLE.get(method, {})
        ax.plot(grouped["episode"], grouped["mean"], label=METHOD_LABELS.get(method, method), **_reward_line_kwargs(style))
        if grouped["count"].min() >= 3:
            ax.fill_between(grouped["episode"], grouped["mean"] - 1.96 * grouped["std"], grouped["mean"] + 1.96 * grouped["std"], color=style.get("color"), alpha=0.12)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Episode total reward (MA{window})")
    ax.set_title("Training reward curve")
    _legend_below(fig, [ax])
    return fig


def plot_f7_2(train_root: Optional[Path], reward_window: int = 20) -> plt.Figure:
    data = _load_training_reward_data(train_root)
    required = {"episode", "seed", "succeeded", "failed", "timed_out", "active_graphs"}
    if data.empty or not required.issubset(data.columns):
        return _notice("F7_2 unavailable", ["Training reward_curve.csv must contain succeeded/failed/timed_out/active_graphs."])
    methods = ["proposed_semantic_topology_marl", "mappo_ctde", "iql_offline", "marl_no_semantic", "marl_semantic_no_topology"]
    window = max(1, int(reward_window))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for method in methods:
        sub = data[data["method"].eq(method)].dropna(subset=["episode", "seed", "succeeded", "failed", "timed_out", "active_graphs"])
        if sub.empty:
            continue
        sort_cols = ["seed", "episode"] + (["step"] if "step" in sub.columns else [])
        episode_rows = (
            sub.sort_values(sort_cols)
            .groupby(["seed", "episode"], as_index=False)
            .tail(1)[["seed", "episode", "succeeded", "failed", "timed_out", "active_graphs"]]
            .sort_values(["seed", "episode"])
        )
        denominator = episode_rows[["succeeded", "failed", "timed_out", "active_graphs"]].sum(axis=1).clip(lower=1.0)
        episode_rows["sfc_completion_rate"] = episode_rows["succeeded"] / denominator
        episode_rows["sfc_completion_ma"] = episode_rows.groupby("seed")["sfc_completion_rate"].transform(
            lambda series: series.rolling(window=window, min_periods=1).mean()
        )
        grouped = episode_rows.groupby("episode")["sfc_completion_ma"].agg(["mean", "std", "count"]).reset_index()
        style = METHOD_STYLE.get(method, {})
        ax.plot(grouped["episode"], grouped["mean"], label=METHOD_LABELS.get(method, method), **_reward_line_kwargs(style))
        if grouped["count"].min() >= 3:
            ax.fill_between(grouped["episode"], grouped["mean"] - 1.96 * grouped["std"], grouped["mean"] + 1.96 * grouped["std"], color=style.get("color"), alpha=0.12)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"SFC completion rate (MA{window})")
    ax.set_ylim(0, 1.02)
    ax.set_title("Training SFC completion curve")
    _legend_below(fig, [ax])
    return fig


def _load_training_reward_data(train_root: Optional[Path]) -> pd.DataFrame:
    if train_root is None or not train_root.exists():
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for path in train_root.rglob("reward_curve.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method"] = _method_from_reward_path(path, train_root)
        df["seed"] = _seed_from_reward_path(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "episode",
        "step",
        "total_reward",
        "mean_reward",
        "reward",
        "seed",
        "succeeded",
        "failed",
        "timed_out",
        "active_graphs",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def _with_x(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    work = df.copy()
    work["x_value"] = work["scenario"].astype(str).map(lambda value: _parse_x(value, suite))
    return work.dropna(subset=["x_value"]).sort_values("x_value")


def _parse_x(scenario: str, suite: str) -> Optional[float]:
    patterns = {
        "service_nodes": r"service_nodes_(\d+)",
        "task_nodes": r"task_nodes_(\d+)",
        "skew": r"skew_(\d+(?:p\d+)?)",
        "ttl": r"ttl_(\d+)",
        "uav": r"uav_(\d+)",
        "speed": r"speed_(\d+)",
    }
    match = re.search(patterns[suite], scenario)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def _plot_metric_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    ylabel: str,
    ylim: Optional[tuple[float, float]] = None,
    legend: bool = True,
) -> None:
    if metric not in df.columns:
        ax.text(0.5, 0.5, f"Missing {metric}", ha="center", va="center", transform=ax.transAxes)
        return
    for method in methods:
        sub = df[df["baseline"].astype(str).eq(method)]
        if sub.empty:
            continue
        grouped = sub.groupby("x_value")[metric].agg(["mean", "std", "count"]).reset_index()
        style = METHOD_STYLE[method]
        yerr = grouped["std"].fillna(0) if grouped["count"].min() >= 3 else None
        ax.errorbar(
            grouped["x_value"],
            grouped["mean"],
            yerr=yerr,
            capsize=3,
            alpha=0.9,
            label=METHOD_LABELS[method] if legend else None,
            **_line_kwargs(style),
        )
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _line_kwargs(style: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "color": style.get("color", "#333333"),
        "linewidth": style.get("linewidth", 1.5),
        "marker": style.get("marker", "o"),
        "markersize": style.get("markersize", 6),
        "linestyle": style.get("linestyle", "-"),
    }


def _reward_line_kwargs(style: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "color": style.get("color", "#333333"),
        "linewidth": min(2.2, float(style.get("linewidth", 1.5) or 1.5)),
        "linestyle": style.get("linestyle", "-"),
    }


def _method_from_reward_path(path: Path, train_root: Path) -> str:
    rel = path.relative_to(train_root)
    if rel.parts[0].startswith("ippo_seed_"):
        return "proposed_semantic_topology_marl"
    return rel.parts[0]


def _seed_from_reward_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def _legend_below(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    handles: List[Any] = []
    labels: List[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(labels)), framealpha=0.85)
        fig.subplots_adjust(bottom=0.28)


def _notice(title: str, lines: Sequence[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, "\n".join(lines), ha="center", va="center", fontsize=10)
    return fig


def _figure_quality(df: pd.DataFrame, suite: str, eval_root: Path, train_root: Optional[str]) -> Dict[str, Any]:
    risk: List[str] = []
    if suite == "training":
        return _training_quality(train_root)
    if df.empty:
        risk.append("missing_suite")
        return {"input_root": str(eval_root), "method_count": 0, "scenario_count": 0, "seed_count": 0, "risk": risk}
    seed_count = _min_seed_count(df)
    if seed_count < 3:
        risk.append(f"insufficient_seed_count_{seed_count}")
    for metric in ["success_ratio", "task_success_ratio", "avg_graph_finish_time", "remote_candidate_ratio", "stale_selected_ratio"]:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(values) and values.abs().sum() == 0:
                risk.append(f"all_zero_{metric}")
    return {
        "input_root": str(eval_root),
        "method_count": int(df["baseline"].nunique()) if "baseline" in df else 0,
        "scenario_count": int(df["scenario"].nunique()) if "scenario" in df else 0,
        "seed_count": int(seed_count),
        "risk": risk,
    }


def _training_quality(train_root: Optional[str]) -> Dict[str, Any]:
    risk: List[str] = []
    if not train_root:
        risk.append("missing_train_root")
        return {"input_root": "", "method_count": 0, "scenario_count": 0, "seed_count": 0, "risk": risk}
    root = Path(train_root)
    if not root.exists():
        risk.append("missing_train_root")
        return {"input_root": str(root), "method_count": 0, "scenario_count": 0, "seed_count": 0, "risk": risk}
    methods = set()
    seeds = set()
    curve_count = 0
    for path in root.rglob("reward_curve.csv"):
        curve_count += 1
        methods.add(_method_from_reward_path(path, root))
        match = re.search(r"seed_(\d+)", str(path))
        if match:
            seeds.add(int(match.group(1)))
    if curve_count == 0:
        risk.append("missing_reward_curve")
    if len(seeds) < 3:
        risk.append(f"insufficient_seed_count_{len(seeds)}")
    return {
        "input_root": str(root),
        "method_count": len(methods),
        "scenario_count": 1 if curve_count else 0,
        "seed_count": len(seeds),
        "risk": risk,
    }


def _min_seed_count(df: pd.DataFrame) -> int:
    if not {"baseline", "scenario", "seed"}.issubset(df.columns):
        return 0
    grouped = df.drop_duplicates(["baseline", "scenario", "seed"]).groupby(["baseline", "scenario"])["seed"].nunique()
    return int(grouped.min()) if len(grouped) else 0


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, fmt: str, dpi: int) -> List[Path]:
    formats = ["png", "pdf"] if fmt == "both" else [fmt]
    paths: List[Path] = []
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None)
        paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
