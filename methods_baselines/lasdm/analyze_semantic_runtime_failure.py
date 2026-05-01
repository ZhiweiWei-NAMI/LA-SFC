from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose all-timeout Semantic-Topology MARL runtime runs.")
    parser.add_argument("--input-dir", required=True, help="semantic_runtime_eval raw root or compact export root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-baseline", default="centralized_planner")
    parser.add_argument("--oracle-baseline", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--success-threshold", type=float, default=0.50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(input_dir)
    run_root = find_run_root(input_dir)
    run_rows = inspect_runs(run_root) if run_root else []
    planner_baseline = args.oracle_baseline or args.planner_baseline
    summary_diag = summarize_table(summary, planner_baseline, args.success_threshold)

    if run_rows:
        write_csv(output_dir / "semantic_runtime_run_diagnostics.csv", run_rows)
    (output_dir / "semantic_runtime_failure_diagnostics.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "run_root": str(run_root) if run_root else None,
                "summary_diagnostics": summary_diag,
                "raw_run_count": len(run_rows),
                "likely_root_causes": infer_root_causes(summary_diag, run_rows),
                "next_checks": [
                    "Check whether the centralized full-information planner succeeds in a calibration scenario with relaxed deadlines and infrastructure-only service placement.",
                    "Check whether runtime_step_count or simulation_time_end advances after each MARL action.",
                    "Check candidate coverage for every required SFC function before runtime stepping.",
                    "Check whether LASDMRuntimeBridge maps selected service instance node IDs to actual AirFogSim task/offloading node IDs.",
                    "Check deadline units and whether manager.step() is called before AirFogSim has a chance to execute tasks.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary_rows": len(summary), "raw_run_rows": len(run_rows)}, indent=2))


def load_summary(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "ablation_summary.csv",
        input_dir / "summary.csv",
        input_dir / "semantic_runtime_summary.csv",
        input_dir / "analysis" / "semantic_topology" / "ablation_summary.csv",
        input_dir / "analysis" / "semantic_topology" / "summary.csv",
        input_dir / "raw" / "semantic_runtime_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            for col in ["submitted", "succeeded", "failed", "timed_out", "success_ratio", "qos_hit_ratio", "runtime_step_count", "simulation_time_end"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    return pd.DataFrame()


def find_run_root(input_dir: Path) -> Optional[Path]:
    if any(input_dir.rglob("marl_transition_trace.jsonl")) or any(input_dir.rglob("semantic_candidate_trace.csv")):
        return input_dir
    for path in [
        input_dir / "semantic_runtime_eval",
        input_dir / "raw_data" / "complete_runtime_scheduler_full" / "semantic_runtime_eval",
        input_dir.parent / "semantic_runtime_eval",
    ]:
        if path.exists() and (any(path.rglob("marl_transition_trace.jsonl")) or any(path.rglob("semantic_candidate_trace.csv"))):
            return path
    return None


def summarize_table(df: pd.DataFrame, planner_baseline: str, success_threshold: float) -> Dict[str, Any]:
    if df.empty:
        return {"summary_found": False}
    diag: Dict[str, Any] = {"summary_found": True, "rows": int(len(df))}
    if "status" in df.columns:
        diag["status_counts"] = {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()}
    for col in ["submitted", "succeeded", "failed", "timed_out", "success_ratio", "qos_hit_ratio", "runtime_step_count", "simulation_time_end"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0)
            diag[col] = {"min": float(series.min()), "mean": float(series.mean()), "max": float(series.max()), "sum": float(series.sum())}
    if {"baseline", "success_ratio"}.issubset(df.columns):
        planner = df[df["baseline"].astype(str).eq(planner_baseline)]
        planner_mean = float(pd.to_numeric(planner["success_ratio"], errors="coerce").fillna(0).mean()) if not planner.empty else None
        diag["centralized_planner_success_mean"] = planner_mean
        diag["centralized_planner_guard_passed"] = bool(planner_mean is not None and planner_mean >= success_threshold)
    return diag


def inspect_runs(run_root: Path) -> List[Dict[str, Any]]:
    run_dirs = sorted({p.parent for p in run_root.rglob("semantic_candidate_trace.csv")} | {p.parent for p in run_root.rglob("marl_transition_trace.jsonl")})
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.append(inspect_run(run_dir))
    return rows


def inspect_run(run_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = parse_run_id(run_dir.name)
    row["run_dir"] = str(run_dir)
    candidate_rows = read_csv_dicts(run_dir / "semantic_candidate_trace.csv")
    exchange_rows = read_csv_dicts(run_dir / "message_overhead.csv") or read_csv_dicts(run_dir / "distributed_exchange_trace.csv")
    transition_rows = read_jsonl(run_dir / "marl_transition_trace.jsonl")
    topology_rows = read_jsonl(run_dir / "topology_trace.jsonl")
    lifecycle_rows = read_csv_dicts(run_dir / "runtime_task_lifecycle_trace.csv")

    row["candidate_trace_rows"] = len(candidate_rows)
    row["transition_rows"] = len(transition_rows)
    row["topology_rows"] = len(topology_rows)
    row["exchange_trace_rows"] = len(exchange_rows)
    row["lifecycle_trace_rows"] = len(lifecycle_rows)
    row["lifecycle_waiting_to_offload"] = count_lifecycle_state(lifecycle_rows, "waiting_to_offload")
    row["lifecycle_offloading"] = count_lifecycle_state(lifecycle_rows, "offloading")
    row["lifecycle_computing"] = count_lifecycle_state(lifecycle_rows, "computing")
    row["lifecycle_done"] = count_lifecycle_state(lifecycle_rows, "done")
    row["lifecycle_failed"] = count_lifecycle_state(lifecycle_rows, "failed")
    row["missing_runtime_node_rows"] = count_lifecycle_reason(lifecycle_rows, "missing_runtime_node")
    row["offloading_no_rb_rows"] = count_lifecycle_state(lifecycle_rows, "offloading_no_rb", field="substate")
    row["mean_remote_candidates"] = mean_numeric(candidate_rows, ["remote_candidate_count", "remote_candidates", "remote_count"])
    row["mean_local_candidates"] = mean_numeric(candidate_rows, ["local_candidate_count", "local_candidates", "local_count"])
    row["mean_top_score"] = mean_numeric(candidate_rows, ["top_score", "top1_score", "semantic_score", "max_similarity"])
    row["payload_bytes"] = sum_numeric(exchange_rows, ["payload_bytes", "bytes", "payload_tx_bytes"])
    row["first_time_s"] = first_transition_value(transition_rows, "time_s")
    row["last_time_s"] = last_transition_value(transition_rows, "time_s")
    row["max_step"] = max_transition_value(transition_rows, "step")
    row["accepted_decisions"] = count_decisions(transition_rows, accepted=True)
    row["rejected_decisions"] = count_decisions(transition_rows, accepted=False)
    row["runtime_reports_with_tasks"] = count_runtime_reports_with_tasks(transition_rows)
    return row


def parse_run_id(name: str) -> Dict[str, Any]:
    parts = name.split("__")
    row: Dict[str, Any] = {"run_id": name}
    if parts:
        row["baseline"] = parts[0]
    for part in parts[1:]:
        if part.startswith("seed_"):
            row["seed"] = part.replace("seed_", "")
        elif part in {"full_hybrid", "infrastructure_only", "plus_ground_vehicle_services", "plus_uav_services"}:
            row["service_role_sweep"] = part
        elif "scenario" not in row:
            row["scenario"] = part
    return row


def infer_root_causes(summary_diag: Mapping[str, Any], run_rows: List[Dict[str, Any]]) -> List[str]:
    causes: List[str] = []
    if not summary_diag.get("summary_found"):
        causes.append("Summary table was not found; first repair export/analysis paths.")
        return causes
    if summary_diag.get("status_counts", {}).get("timed_out", 0) == summary_diag.get("rows", -1):
        causes.append("All semantic runtime runs timed out. This indicates an environment/scheduler coupling problem or an over-constrained runtime configuration.")
    if summary_diag.get("success_ratio", {}).get("max", 0) == 0:
        causes.append("No run produced a positive graph success ratio; all performance plots should be treated as diagnostics.")
    if summary_diag.get("runtime_step_count", {}).get("max", 1) == 0 or summary_diag.get("simulation_time_end", {}).get("max", 1) == 0:
        causes.append("Runtime time/step counters did not advance in the summary. Inspect SemanticTopologyMARLEnv._time() and post-action AirFogSim stepping.")
    if summary_diag.get("centralized_planner_guard_passed") is False:
        causes.append("The centralized full-information planner failed the easy-success guard. Repair candidate mapping, deadlines, runtime stepping, or service capacity before comparing MARL baselines.")
    if run_rows:
        if max(float(r.get("candidate_trace_rows") or 0) for r in run_rows) == 0:
            causes.append("Candidate traces are empty. Service discovery may not be invoked, or trace files are not exported from runtime mode.")
        if max(float(r.get("accepted_decisions") or 0) for r in run_rows) == 0:
            causes.append("No accepted decisions were observed in transition traces. Check policy action decoding and candidate-instance IDs.")
        if max(float(r.get("runtime_reports_with_tasks") or 0) for r in run_rows) == 0:
            causes.append("Runtime reports do not show task injection/progress. Check LASDMRuntimeBridge.prepare_airfogsim_step().")
        if max(float(r.get("missing_runtime_node_rows") or 0) for r in run_rows) > 0:
            causes.append("Lifecycle trace shows missing runtime nodes. Synthetic task sources must be rebound to concrete AirFogSim node IDs before task creation.")
        if max(float(r.get("lifecycle_offloading") or 0) for r in run_rows) > 0 and max(float(r.get("lifecycle_computing") or 0) for r in run_rows) == 0:
            causes.append("Tasks reach offloading but never computing. Inspect RB allocation, route node existence, and AirFogSim communication progress.")
    else:
        causes.append("Raw per-run traces are unavailable in the compact package; use the raw archive for root-cause localization.")
    return causes


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def first_existing(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in ("", None, "None"):
            return row[key]
    return None


def mean_numeric(rows: List[Mapping[str, Any]], keys: Iterable[str]) -> float:
    values = []
    for row in rows:
        value = first_existing(row, keys)
        try:
            values.append(float(value))
        except Exception:
            pass
    return float(sum(values) / len(values)) if values else 0.0


def sum_numeric(rows: List[Mapping[str, Any]], keys: Iterable[str]) -> float:
    total = 0.0
    for row in rows:
        value = first_existing(row, keys)
        try:
            total += float(value)
        except Exception:
            pass
    return total


def first_transition_value(rows: List[Mapping[str, Any]], key: str) -> float:
    for row in rows:
        try:
            return float(row.get(key, 0) or 0)
        except Exception:
            pass
    return 0.0


def last_transition_value(rows: List[Mapping[str, Any]], key: str) -> float:
    for row in reversed(rows):
        try:
            return float(row.get(key, 0) or 0)
        except Exception:
            pass
    return 0.0


def max_transition_value(rows: List[Mapping[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0) or 0))
        except Exception:
            pass
    return max(values) if values else 0.0


def count_decisions(rows: List[Mapping[str, Any]], accepted: bool) -> int:
    count = 0
    for row in rows:
        info = row.get("info", {}) if isinstance(row, Mapping) else {}
        decisions = info.get("decisions", []) if isinstance(info, Mapping) else []
        for decision in decisions if isinstance(decisions, list) else []:
            if not isinstance(decision, Mapping):
                continue
            rejected = decision.get("rejected_reason") not in (None, "", "None")
            assignments = decision.get("assignments", {})
            is_accepted = bool(assignments) and not rejected
            if is_accepted == accepted:
                count += 1
    return count


def count_runtime_reports_with_tasks(rows: List[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        info = row.get("info", {}) if isinstance(row, Mapping) else {}
        report = info.get("runtime_report", {}) if isinstance(info, Mapping) else {}
        if not isinstance(report, Mapping):
            continue
        encoded = json.dumps(report, ensure_ascii=False).lower()
        if any(token in encoded for token in ["task", "offload", "compute", "execut"]):
            count += 1
    return count


def count_lifecycle_state(rows: List[Mapping[str, Any]], value: str, field: str = "queue_state") -> int:
    return sum(1 for row in rows if str(row.get(field, "")) == value)


def count_lifecycle_reason(rows: List[Mapping[str, Any]], reason: str) -> int:
    return sum(1 for row in rows if str(row.get("offloading_reason", "")) == reason)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
