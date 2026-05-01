from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


SIGNIFICANCE_FIELDS = ["metric", "baseline_a", "baseline_b", "p_value", "significant"]
DEFAULT_COMPARISONS = (("proposed", "nearest_edge"), ("proposed", "centralized_greedy"))
RUN_METRICS = (
    "graph_completion_ratio",
    "deadline_satisfaction_ratio",
    "task_success_ratio",
    "avg_graph_finish_time",
)
TRACE_METRICS = (
    "avg_function_e2e_delay",
    "avg_tx_delay",
    "avg_compute_delay",
    "function_success_ratio",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run significance tests for LASDM runtime benchmark artifacts.")
    parser.add_argument("--input", default=None, help="Path to function_execution_trace.csv.")
    parser.add_argument("--input-root", default="analysis/runtime_final", help="Analysis root containing function trace.")
    parser.add_argument("--run-root", action="append", default=[], help="Optional raw benchmark root with per-run JSON files.")
    parser.add_argument("--output", default=None, help="Output significance_summary.csv path.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    input_path = args.input or os.path.join(args.input_root, "function_execution_trace.csv")
    output_path = args.output or os.path.join(os.path.dirname(input_path), "significance_summary.csv")
    rows = significance_rows(
        trace_rows=read_csv(input_path),
        run_records=load_run_records(args.run_root),
        alpha=args.alpha,
    )
    write_csv(output_path, SIGNIFICANCE_FIELDS, rows)


def significance_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]] = (),
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    metric_values: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for metric, values in run_metric_values(run_records).items():
        metric_values[metric].update(values)
    for metric, values in trace_metric_values(trace_rows).items():
        metric_values[metric].update(values)

    rows: List[Dict[str, Any]] = []
    for metric in [*RUN_METRICS, *TRACE_METRICS]:
        values = metric_values.get(metric, {})
        if not values:
            continue
        for baseline_a, baseline_b in DEFAULT_COMPARISONS:
            paired_a, paired_b = paired_values(values, baseline_a, baseline_b)
            if not paired_a or not paired_b:
                continue
            p_value = welch_p_value(paired_a, paired_b)
            rows.append(
                {
                    "metric": metric,
                    "baseline_a": baseline_a,
                    "baseline_b": baseline_b,
                    "p_value": f"{p_value:.6g}",
                    "significant": str(bool(p_value < alpha)),
                }
            )
    return rows


def run_metric_values(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    values: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for record in records:
        key = metric_key(record)
        if not all(key):
            continue
        for metric in RUN_METRICS:
            parsed = penalized_finish_time(record) if metric == "avg_graph_finish_time" else to_float_or_none(record.get(metric))
            if parsed is not None:
                values[metric][key] = parsed
    return values


def trace_metric_values(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = metric_key(row)
        if all(key):
            grouped[key].append(row)

    values: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for key, group in grouped.items():
        e2e = numeric_cells(group, "e2e_delay")
        tx = numeric_cells(group, "tx_delay")
        compute = numeric_cells(group, "compute_delay")
        statuses = [str(row.get("status", "")).lower() for row in group]
        if e2e:
            values["avg_function_e2e_delay"][key] = mean(e2e)
        if tx:
            values["avg_tx_delay"][key] = mean(tx)
        if compute:
            values["avg_compute_delay"][key] = mean(compute)
        if statuses:
            success_count = sum(1 for status in statuses if status in {"", "succeeded", "completed", "success"})
            values["function_success_ratio"][key] = success_count / len(statuses)
    return values


def paired_values(
    values: Mapping[Tuple[str, str, str], float],
    baseline_a: str,
    baseline_b: str,
) -> Tuple[List[float], List[float]]:
    index: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (baseline, scenario, seed), value in values.items():
        index[(scenario, seed)][baseline] = float(value)
    left: List[float] = []
    right: List[float] = []
    for pair in sorted(index):
        item = index[pair]
        if baseline_a in item and baseline_b in item:
            left.append(item[baseline_a])
            right.append(item[baseline_b])
    return left, right


def welch_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    left = [float(value) for value in left]
    right = [float(value) for value in right]
    if not left or not right:
        return 1.0
    if left == right:
        return 1.0
    if sample_variance(left) == 0.0 and sample_variance(right) == 0.0:
        return 0.0 if mean(left) != mean(right) else 1.0
    try:
        from scipy import stats  # type: ignore

        result = stats.ttest_ind(left, right, equal_var=False)
        p_value = float(result.pvalue)
        if math.isfinite(p_value):
            return max(0.0, min(1.0, p_value))
    except Exception:
        pass
    return normal_approx_p_value(left, right)


def normal_approx_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    var_left = sample_variance(left)
    var_right = sample_variance(right)
    denom = math.sqrt(var_left / max(1, len(left)) + var_right / max(1, len(right)))
    if denom <= 0.0:
        return 0.0 if mean(left) != mean(right) else 1.0
    z_score = abs(mean(left) - mean(right)) / denom
    return max(0.0, min(1.0, math.erfc(z_score / math.sqrt(2.0))))


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def metric_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(row.get("baseline", "")), str(row.get("scenario", "")), str(row.get("seed", "")))


def numeric_cells(rows: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = to_float_or_none(row.get(field))
        if value is not None:
            values.append(value)
    return values


def to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def penalized_finish_time(record: Mapping[str, Any]) -> float | None:
    finish_time = to_float_or_none(record.get("avg_graph_finish_time"))
    if finish_time is None:
        return None
    completion = to_float_or_none(record.get("graph_completion_ratio")) or 0.0
    if completion >= 1.0:
        return finish_time
    penalty = record_deadline_s(record)
    if penalty is None:
        raw = record.get("raw_metrics") if isinstance(record.get("raw_metrics"), Mapping) else {}
        penalty = to_float_or_none(raw.get("current_time")) or to_float_or_none(record.get("simulation_time_end")) or finish_time
    if finish_time <= 0.0:
        return penalty
    return completion * finish_time + (1.0 - completion) * penalty


def record_deadline_s(record: Mapping[str, Any]) -> float | None:
    config = record.get("scenario_config") if isinstance(record.get("scenario_config"), Mapping) else {}
    return to_float_or_none(config.get("deadline_s")) or to_float_or_none(config.get("deadline"))


def load_run_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    try:
        from generate_paper_validation_artifacts import load_run_records as load_records

        return load_records(paths)
    except Exception:
        return []


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
