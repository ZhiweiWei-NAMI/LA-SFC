from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)


FUNCTION_TRACE_FIELDS = [
    "baseline",
    "seed",
    "scenario",
    "sfc_id",
    "function_id",
    "selected_node",
    "route",
    "tx_delay",
    "compute_delay",
    "queue_delay",
    "task_id",
    "finish_time",
    "e2e_delay",
    "status",
]

SENSITIVITY_FIELDS = ["param_type", "param_value", "success_rate", "avg_delay", "failure_rate", "run_count"]

BASELINE_MATRIX_FIELDS = [
    "family",
    "baseline",
    "registered",
    "candidate_scope",
    "candidate_filters",
    "resource_capability",
    "path_capability",
    "composition_capability",
    "region_capability",
    "deployment_capability",
    "intentionally_limited",
    "limitation_axis",
    "fairness_treatment",
    "source_refs",
]

OVERHEAD_FIELDS = ["baseline", "avg_decision_time_ms", "p95_decision_time_ms", "run_count"]

RESOURCE_FIELDS = [
    "baseline",
    "seed",
    "scenario",
    "time_s",
    "node_id",
    "instance_id",
    "cpu_utilization",
    "rb_utilization",
    "backhaul_usage_mb",
    "energy_consumption",
    "computing_task_count",
]

CONTINUITY_FIELDS = [
    "baseline",
    "seed",
    "scenario",
    "event_type",
    "recovery_time",
    "reschedule_success_rate",
    "evidence_status",
]

COMPARISON_FIELDS = [
    "scenario",
    "seed",
    "metric",
    "lasdm_proposed",
    "agentic_proposed",
    "delta_lasdm_minus_agentic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LASDM paper validation artifacts from benchmark outputs.")
    parser.add_argument("input_dirs", nargs="*", help="LASDM benchmark output directories or parent roots.")
    parser.add_argument("--input-root", action="append", default=[], help="Alias for positional LASDM input dirs.")
    parser.add_argument("--agentic-dir", action="append", default=[], help="Optional Agentic benchmark output directory.")
    parser.add_argument("--output-dir", "--output-root", default=os.path.join(WORKSPACE_ROOT, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lasdm_runs = load_run_records([*args.input_dirs, *args.input_root])
    agentic_runs = load_run_records(args.agentic_dir)

    write_csv(os.path.join(args.output_dir, "function_execution_trace.csv"), FUNCTION_TRACE_FIELDS, function_trace_rows(lasdm_runs))
    write_json(os.path.join(args.output_dir, "failure_trace.json"), failure_trace_rows(lasdm_runs))
    write_csv(os.path.join(args.output_dir, "sensitivity_summary.csv"), SENSITIVITY_FIELDS, sensitivity_rows(lasdm_runs))
    write_csv(os.path.join(args.output_dir, "baseline_capability_matrix.csv"), BASELINE_MATRIX_FIELDS, baseline_matrix_rows())
    write_csv(os.path.join(args.output_dir, "runtime_overhead.csv"), OVERHEAD_FIELDS, overhead_rows(lasdm_runs))
    write_csv(os.path.join(args.output_dir, "resource_usage_timeseries.csv"), RESOURCE_FIELDS, resource_rows(lasdm_runs))
    write_csv(os.path.join(args.output_dir, "continuity_metrics.csv"), CONTINUITY_FIELDS, continuity_rows(lasdm_runs))
    write_csv(
        os.path.join(args.output_dir, "comparison_lasdm_vs_agentic.csv"),
        COMPARISON_FIELDS,
        comparison_rows(lasdm_runs, agentic_runs),
    )


def load_run_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path:
            continue
        if os.path.isfile(path) and path.endswith(".json"):
            maybe_add_record(records, path)
            continue
        if not os.path.isdir(path):
            continue
        for root, _, files in os.walk(path):
            for name in files:
                if name.endswith(".json") and name != "manifest.json":
                    maybe_add_record(records, os.path.join(root, name))
    return records


def maybe_add_record(records: List[Dict[str, Any]], path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return
    if "baseline" in payload and ("raw_metrics" in payload or "graph_completion_ratio" in payload):
        payload["_source_path"] = path
        records.append(payload)


def function_trace_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        raw = raw_metrics(record)
        seen: set[tuple[str, str]] = set()
        for item in raw.get("function_execution_trace", []) or []:
            row = {field: item.get(field, "") for field in FUNCTION_TRACE_FIELDS}
            row["baseline"] = row["baseline"] or record.get("baseline", "")
            row["seed"] = row["seed"] if row["seed"] != "" else record.get("seed", "")
            row["scenario"] = row["scenario"] or record.get("scenario", "")
            seen.add((str(row.get("sfc_id", "")), str(row.get("function_id", ""))))
            rows.append(row)
        for decision in record.get("decisions", []) or []:
            rejected_reason = decision.get("rejected_reason")
            if raw.get("function_execution_trace") and not rejected_reason:
                continue
            for function_id, selected_node in (decision.get("node_mapping", {}) or {}).items():
                sfc_id = str(decision.get("sfc_id", ""))
                if (sfc_id, str(function_id)) in seen:
                    continue
                rows.append(
                    {
                        "baseline": record.get("baseline", ""),
                        "seed": record.get("seed", ""),
                        "scenario": record.get("scenario", ""),
                        "sfc_id": sfc_id,
                        "function_id": function_id,
                        "selected_node": selected_node,
                        "route": "->".join(str(x) for x in (decision.get("routes", {}) or {}).get(function_id, [])),
                        "tx_delay": "",
                        "compute_delay": "",
                        "queue_delay": "",
                        "task_id": "",
                        "finish_time": "",
                        "e2e_delay": record.get("avg_graph_finish_time", ""),
                        "status": rejected_reason or record.get("status", ""),
                    }
                )
    return rows


def failure_trace_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        raw = raw_metrics(record)
        explicit_rows = [dict(item) for item in raw.get("failure_trace", []) or []]
        rows.extend(explicit_rows)
        if explicit_rows:
            continue
        failure_reason = record.get("failure_reason")
        if failure_reason and failure_reason != "none":
            for sfc_id in _failed_sfc_ids(record):
                rows.append(
                    {
                        "task_id": "",
                        "sfc_id": sfc_id,
                        "airfogsim_reason": record.get("failure_reason", ""),
                        "lasdm_reason": failure_reason,
                        "timestamp": raw.get("current_time", record.get("simulation_time_end", "")),
                        "mapping_source": "sfc_terminal_failure",
                        "baseline": record.get("baseline", ""),
                        "seed": record.get("seed", ""),
                        "scenario": record.get("scenario", ""),
                    }
                )
    return rows


def sensitivity_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_config = dict(record.get("scenario_config", {}) or {})
        deadline_value = scenario_config.get("deadline_class") or deadline_bucket(record)
        network_value = scenario_config.get("network_fault") or "stable"
        load_value = load_bucket(float(scenario_config.get("load_multiplier", 1.0) or 1.0))
        grouped[("deadline", str(deadline_value))].append(record)
        grouped[("network", str(network_value))].append(record)
        grouped[("load", str(load_value))].append(record)
        grouped[("baseline", str(record.get("baseline", "")))].append(record)

    rows = []
    for (param_type, param_value), group in sorted(grouped.items()):
        rows.append(
            {
                "param_type": param_type,
                "param_value": param_value,
                "success_rate": mean(float(item.get("graph_completion_ratio", item.get("success_ratio", 0.0)) or 0.0) for item in group),
                "avg_delay": mean(penalized_graph_delay(item) for item in group),
                "failure_rate": mean(1.0 - float(item.get("graph_completion_ratio", item.get("success_ratio", 0.0)) or 0.0) for item in group),
                "run_count": len(group),
            }
        )
    return rows


def baseline_matrix_rows() -> List[Dict[str, Any]]:
    scripts_dir = os.path.join(WORKSPACE_ROOT, "analysis", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from generate_baseline_capability_matrix import MATRIX_ROWS

    return [dict(row) for row in MATRIX_ROWS]


def overhead_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        baseline = str(record.get("baseline", ""))
        raw = raw_metrics(record)
        samples = [
            float(item.get("decision_time_ms", 0.0) or 0.0)
            for item in raw.get("runtime_overhead_records", []) or []
        ]
        if samples:
            grouped[baseline].extend(samples)
            continue
        summary = raw.get("runtime_overhead") or record.get("runtime_overhead") or {}
        if summary and int(float(summary.get("decision_samples", 0) or 0)) > 0:
            grouped[baseline].append(float(summary.get("avg_decision_time_ms", 0.0) or 0.0))
    return [
        {
            "baseline": baseline,
            "avg_decision_time_ms": mean(values) if values else 0.0,
            "p95_decision_time_ms": percentile(values, 0.95),
            "run_count": len(values),
        }
        for baseline, values in sorted(grouped.items())
    ]


def resource_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        raw = raw_metrics(record)
        for item in raw.get("resource_usage_timeseries", []) or []:
            rows.extend(expand_resource_item(record, item))
    return rows


def expand_resource_item(record: Mapping[str, Any], item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    metadata = {
        "baseline": record.get("baseline", ""),
        "seed": record.get("seed", ""),
        "scenario": record.get("scenario", ""),
        "time_s": item.get("time_s", ""),
    }
    if item.get("node_id") or item.get("instance_id"):
        row = {field: item.get(field, "") for field in RESOURCE_FIELDS}
        row.update(metadata)
        return [row]

    cpu_by_node = parse_mapping_cell(item.get("cpu_utilization_by_node"))
    computing_by_node = parse_mapping_cell(item.get("computing_tasks_by_node"))
    energy_by_uav = parse_mapping_cell(item.get("energy_by_uav"))
    node_ids = sorted({str(node_id) for node_id in [*cpu_by_node.keys(), *computing_by_node.keys(), *energy_by_uav.keys()]})
    if not node_ids:
        return [
            {
                **metadata,
                "node_id": "",
                "instance_id": "",
                "cpu_utilization": item.get("avg_cpu_utilization", ""),
                "rb_utilization": item.get("rb_utilization", ""),
                "backhaul_usage_mb": bytes_to_mb(item.get("backhaul_usage_bytes", 0.0)),
                "energy_consumption": item.get("energy_consumed_step", ""),
                "computing_task_count": item.get("computing_tasks", ""),
            }
        ]

    rows: List[Dict[str, Any]] = []
    for node_id in node_ids:
        rows.append(
            {
                **metadata,
                "node_id": node_id,
                "instance_id": "",
                "cpu_utilization": cpu_by_node.get(node_id, ""),
                "rb_utilization": item.get("rb_utilization", ""),
                "backhaul_usage_mb": bytes_to_mb(item.get("backhaul_usage_bytes", 0.0)),
                "energy_consumption": item.get("energy_consumed_step", "") if node_id in energy_by_uav else "",
                "computing_task_count": computing_by_node.get(node_id, 0),
            }
        )
    return rows


def continuity_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        scenario = str(record.get("scenario", ""))
        if scenario not in {"high_mobility", "link_fault", "airspace_event"}:
            continue
        rows.append(
            {
                "baseline": record.get("baseline", ""),
                "seed": record.get("seed", ""),
                "scenario": scenario,
                "event_type": scenario,
                "recovery_time": record.get("avg_graph_finish_time", ""),
                "reschedule_success_rate": record.get("graph_completion_ratio", ""),
                "evidence_status": "runtime_summary_proxy",
            }
        )
    return rows


def comparison_rows(lasdm_records: Sequence[Mapping[str, Any]], agentic_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lasdm = index_proposed(lasdm_records)
    agentic = index_proposed(agentic_records)
    metrics = ["graph_completion_ratio", "deadline_satisfaction_ratio", "task_success_ratio", "avg_graph_finish_time"]
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(lasdm) | set(agentic)):
        for metric in metrics:
            lv = value_or_blank(lasdm.get(key), metric)
            av = value_or_blank(agentic.get(key), metric)
            rows.append(
                {
                    "scenario": key[0],
                    "seed": key[1],
                    "metric": metric,
                    "lasdm_proposed": lv,
                    "agentic_proposed": av,
                    "delta_lasdm_minus_agentic": numeric_delta(lv, av),
                }
            )
    return rows


def index_proposed(records: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], Mapping[str, Any]]:
    indexed = {}
    for record in records:
        if record.get("baseline") in {"proposed", "lasdm_greedy", "agentic_proposed"}:
            indexed[(str(record.get("scenario", "")), str(record.get("seed", "")))] = record
    return indexed


def raw_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = record.get("raw_metrics")
    return raw if isinstance(raw, Mapping) else {}


def deadline_bucket(record: Mapping[str, Any]) -> str:
    config = dict(record.get("scenario_config", {}) or {})
    if "deadline_s" in config:
        deadline = float(config["deadline_s"])
    else:
        deadline = float(config.get("deadline", 0.0) or 0.0)
    if deadline and deadline <= 3.0:
        return "tight"
    if deadline and deadline >= 10.0:
        return "loose"
    return "medium"


def load_bucket(value: float) -> str:
    if value <= 1.0:
        return "low"
    if value < 1.8:
        return "medium"
    return "high"


def _first_decision_sfc(record: Mapping[str, Any]) -> str:
    decisions = record.get("decisions") or []
    if decisions:
        return str(decisions[0].get("sfc_id", ""))
    return ""


def _failed_sfc_ids(record: Mapping[str, Any]) -> List[str]:
    decisions = list(record.get("decisions") or [])
    rejected = [str(decision.get("sfc_id", "")) for decision in decisions if decision.get("rejected_reason")]
    if rejected:
        return [sfc_id for sfc_id in rejected if sfc_id]

    raw = raw_metrics(record)
    deadline = _record_deadline_s(record)
    if deadline is not None:
        by_sfc: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for item in raw.get("function_execution_trace", []) or []:
            sfc_id = str(item.get("sfc_id", ""))
            if sfc_id:
                by_sfc[sfc_id].append(item)
        failed: List[str] = []
        for sfc_id, items in by_sfc.items():
            delays = [float(value) for value in (_maybe_float(item.get("e2e_delay")) for item in items) if value is not None]
            statuses = {str(item.get("status", "")).lower() for item in items}
            if any(status not in {"", "succeeded", "completed", "success"} for status in statuses):
                failed.append(sfc_id)
            elif delays and max(delays) > deadline:
                failed.append(sfc_id)
        if failed:
            return sorted(failed)

    return []


def _record_deadline_s(record: Mapping[str, Any]) -> float | None:
    config = dict(record.get("scenario_config", {}) or {})
    value = config.get("deadline_s")
    if value is None:
        value = config.get("deadline")
    return _maybe_float(value)


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def penalized_graph_delay(record: Mapping[str, Any]) -> float:
    completion = _maybe_float(record.get("graph_completion_ratio", record.get("success_ratio", 0.0))) or 0.0
    finish_time = _maybe_float(record.get("avg_graph_finish_time")) or 0.0
    if completion >= 1.0:
        return finish_time
    penalty = _record_deadline_s(record)
    if penalty is None:
        raw = raw_metrics(record)
        penalty = _maybe_float(raw.get("current_time", record.get("simulation_time_end"))) or finish_time
    if finish_time <= 0.0:
        return penalty
    return completion * finish_time + (1.0 - completion) * penalty


def value_or_blank(record: Mapping[str, Any] | None, metric: str) -> Any:
    if record is None:
        return ""
    return record.get(metric, "")


def parse_mapping_cell(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return {}


def bytes_to_mb(value: Any) -> float:
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def numeric_delta(left: Any, right: Any) -> Any:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return ""


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    index = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[index]


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
