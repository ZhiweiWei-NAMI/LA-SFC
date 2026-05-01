from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LASDM paper figures from analysis CSV artifacts.")
    parser.add_argument("--analysis-dir", "--input-root", default="analysis")
    parser.add_argument("--output-dir", "--output-root", default=None)
    args = parser.parse_args()

    analysis_dir = os.path.abspath(args.analysis_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(analysis_dir, "figures"))
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_manifest(output_dir, {"status": "matplotlib_unavailable", "error": str(exc)})
        return

    generated = []
    generated.append(plot_success_rate(plt, analysis_dir, output_dir))
    generated.append(plot_delay_cdf(plt, analysis_dir, output_dir))
    generated.append(plot_failure_breakdown(plt, analysis_dir, output_dir))
    generated.append(plot_resource_heatmap(plt, analysis_dir, output_dir))
    generated.append(plot_sensitivity_curves(plt, analysis_dir, output_dir))
    generated.append(plot_continuity_curve(plt, analysis_dir, output_dir))
    write_manifest(output_dir, {"status": "ok", "figures": [item for item in generated if item]})


def plot_success_rate(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = read_csv(os.path.join(analysis_dir, "sensitivity_summary.csv"))
    grouped = defaultdict(list)
    for row in rows:
        if row.get("param_type") == "baseline":
            grouped[row.get("param_value", "")].append(to_float(row.get("success_rate")))
    if not grouped:
        trace = read_csv(os.path.join(analysis_dir, "function_execution_trace.csv"))
        counts = Counter(row.get("baseline", "") for row in trace if row.get("status") in {"succeeded", "completed", ""})
        grouped = {key: [1.0 if value else 0.0] for key, value in counts.items()}
    labels = sorted(grouped)
    values = [avg(grouped[label]) for label in labels]
    return bar_plot(plt, labels, values, "Success Rate", "success_rate_vs_baseline.png", output_dir)


def plot_delay_cdf(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = read_csv(os.path.join(analysis_dir, "function_execution_trace.csv"))
    values = sorted(to_float(row.get("e2e_delay", row.get("compute_delay"))) for row in rows)
    values = [value for value in values if value >= 0]
    if not values:
        return empty_plot(plt, "E2E Delay CDF", "e2e_delay_cdf.png", output_dir)
    y = [(index + 1) / len(values) for index in range(len(values))]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(values, y)
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("CDF")
    ax.set_title("E2E Delay CDF")
    return save(fig, "e2e_delay_cdf.png", output_dir)


def plot_failure_breakdown(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = load_failure_rows(os.path.join(analysis_dir, "failure_trace.json"))
    counts = Counter(str(row.get("lasdm_reason", "unknown")) for row in rows)
    return bar_plot(plt, sorted(counts), [counts[key] for key in sorted(counts)], "Failure Reason Breakdown", "failure_reason_breakdown.png", output_dir)


def plot_resource_heatmap(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = read_csv(os.path.join(analysis_dir, "resource_usage_timeseries.csv"))
    node_values = defaultdict(list)
    for row in rows:
        node_values[row.get("node_id", "")].append(to_float(row.get("cpu_utilization")))
    labels = sorted(label for label in node_values if label)
    values = [[avg(node_values[label]) for label in labels]]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.55), 2.6))
    if labels:
        image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks([0], ["CPU"])
        fig.colorbar(image, ax=ax)
    ax.set_title("Resource Utilization Heatmap")
    return save(fig, "resource_utilization_heatmap.png", output_dir)


def plot_sensitivity_curves(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = read_csv(os.path.join(analysis_dir, "sensitivity_summary.csv"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("param_type", "")].append((row.get("param_value", ""), to_float(row.get("success_rate"))))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for param_type, points in sorted(grouped.items()):
        labels = [item[0] for item in points]
        values = [item[1] for item in points]
        ax.plot(labels, values, marker="o", label=param_type)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Sensitivity Curves")
    if grouped:
        ax.legend()
    return save(fig, "sensitivity_curves.png", output_dir)


def plot_continuity_curve(plt: Any, analysis_dir: str, output_dir: str) -> str:
    rows = read_csv(os.path.join(analysis_dir, "continuity_metrics.csv"))
    labels = [row.get("scenario", "") for row in rows]
    values = [to_float(row.get("reschedule_success_rate")) for row in rows]
    return bar_plot(plt, labels, values, "Continuity Recovery", "continuity_recovery_curve.png", output_dir)


def bar_plot(plt: Any, labels: Sequence[str], values: Sequence[float], title: str, name: str, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.6), 3.2))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(0, max([1.0, *values]) if values else 1.0)
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, name, output_dir)


def empty_plot(plt: Any, title: str, name: str, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()
    return save(fig, name, output_dir)


def save(fig: Any, name: str, output_dir: str) -> str:
    path = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.clear()
    return path


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_failure_rows(path: str) -> List[Mapping[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, list) else []


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_manifest(output_dir: str, payload: Mapping[str, Any]) -> None:
    with open(os.path.join(output_dir, "figure_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
