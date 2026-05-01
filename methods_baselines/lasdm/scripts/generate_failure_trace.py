from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))


TRACE_SCHEMA_VERSION = "lasdm_failure_trace_v1"

REASON_NO_CANDIDATE = "no_candidate"
REASON_TASK_FAILED = "task_failed"
REASON_DEADLINE_MISSED = "deadline_missed"
REASON_RELIABILITY_VIOLATION = "reliability_violation"
REASON_ACCURACY_VIOLATION = "accuracy_violation"
REASON_ENERGY_VIOLATION = "energy_violation"
REASON_INVALID_GRAPH = "invalid_graph"

SFC_REASON_VALUES = {
    REASON_NO_CANDIDATE,
    REASON_TASK_FAILED,
    REASON_DEADLINE_MISSED,
    REASON_RELIABILITY_VIOLATION,
    REASON_ACCURACY_VIOLATION,
    REASON_ENERGY_VIOLATION,
    REASON_INVALID_GRAPH,
}

NATIVE_CODE_NAMES = {
    "TASK_FAIL_OUT_OF_DDL": 0,
    "TASK_FAIL_OUT_OF_TTI": 1,
    "TASK_FAIL_OUT_OF_NODE": 2,
    "TASK_FAIL_PARENT_FAILED": 3,
    "TASK_FAIL_MALICIOUS_RESULT": 4,
}

NATIVE_CODE_DESCRIPTIONS = {
    0: "Task fails due to out of deadline.",
    1: "Task fails due to transmission timeout.",
    2: "Task fails due to out of node.",
    3: "Task fails due to parent task failed.",
    4: "Task fails due to malicious result detected.",
}

NATIVE_CODE_TO_REASON = {
    0: REASON_DEADLINE_MISSED,
    1: REASON_DEADLINE_MISSED,
    2: REASON_NO_CANDIDATE,
    3: REASON_TASK_FAILED,
    4: REASON_RELIABILITY_VIOLATION,
}


FAILURE_CONTRACT: Dict[str, Dict[str, Any]] = {
    "no_candidate": {
        "canonical_reason": REASON_NO_CANDIDATE,
        "mapping_contract": "No feasible service instance or AirFogSim execution node is available.",
        "native_airfogsim_support": "yes",
        "native_failure_codes": ["TASK_FAIL_OUT_OF_NODE"],
        "evidence_sources": [
            "LASDM decision rejected_reason=no_candidate",
            "AirFogSim task failure_code=TASK_FAIL_OUT_OF_NODE",
            "summary failure_reason_count.no_candidate",
        ],
        "needs_native_runtime_hook": False,
    },
    "link_disconnect": {
        "canonical_reason": REASON_DEADLINE_MISSED,
        "mapping_contract": "A selected route or radio/wired hop disconnects and prevents task delivery before the deadline.",
        "native_airfogsim_support": "partial",
        "native_failure_codes": ["TASK_FAIL_OUT_OF_TTI"],
        "evidence_sources": [
            "AirFogSim task failure_code=TASK_FAIL_OUT_OF_TTI",
            "text signal containing link_down/link_fault/link_disconnect/transmission_timeout",
        ],
        "needs_native_runtime_hook": True,
        "hook_gap": "No explicit native link-disconnect task failure code is present; TASK_FAIL_OUT_OF_TTI only proves transmission timeout.",
    },
    "energy_exhausted": {
        "canonical_reason": REASON_ENERGY_VIOLATION,
        "mapping_contract": "A UAV or mobile executor cannot start or finish the service because remaining energy is exhausted.",
        "native_airfogsim_support": "no",
        "native_failure_codes": [],
        "evidence_sources": [
            "text signal containing energy/battery",
            "future EnergyManager or runtime_bridge hook",
        ],
        "needs_native_runtime_hook": True,
        "hook_gap": "EnergyManager exists, but there is no TASK_FAIL_* energy code or bridge collection hook for energy terminal failure.",
    },
    "deadline_too_small": {
        "canonical_reason": REASON_DEADLINE_MISSED,
        "mapping_contract": "The graph/task deadline is so small that completion after submission must time out.",
        "native_airfogsim_support": "yes",
        "native_failure_codes": ["TASK_FAIL_OUT_OF_DDL", "TASK_FAIL_OUT_OF_TTI"],
        "evidence_sources": [
            "AirFogSim task failure_code=TASK_FAIL_OUT_OF_DDL",
            "AirFogSim task failure_code=TASK_FAIL_OUT_OF_TTI",
            "LASDM graph status=timed_out",
        ],
        "needs_native_runtime_hook": False,
    },
    "node_moved_away": {
        "canonical_reason": REASON_NO_CANDIDATE,
        "mapping_contract": "The selected executor/source/relay leaves coverage or is removed after placement.",
        "native_airfogsim_support": "partial",
        "native_failure_codes": ["TASK_FAIL_OUT_OF_NODE"],
        "evidence_sources": [
            "AirFogSim task failure_code=TASK_FAIL_OUT_OF_NODE",
            "text signal containing node_removed/node_moved/coverage_hole/out_of_node",
        ],
        "needs_native_runtime_hook": True,
        "hook_gap": "TaskManager.removeTasksByNodeId stores removed tasks separately and does not set a failure code consumed by runtime_bridge.",
    },
}


def generate_failure_trace(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    input_path = os.path.abspath(input_path)
    run_paths = _discover_run_json_paths(input_path)
    summary_rows = _read_summary_rows(input_path)
    summary_by_key = {_run_key(row): row for row in summary_rows}

    runs: List[Dict[str, Any]] = []
    reason_histogram: Counter[str] = Counter()
    family_histogram: Counter[str] = Counter()

    for run_path in run_paths:
        run = _read_json(run_path)
        summary_row = summary_by_key.get(_run_key(run), {})
        run_record = _trace_run(run, run_path, summary_row)
        runs.append(run_record)
        reason_histogram.update(run_record["reason_histogram"])
        family_histogram.update(run_record["family_histogram"])

    for row in summary_rows:
        if _run_key(row) in {_run_key(run) for run in runs}:
            continue
        run_record = _trace_run(row, None, row)
        runs.append(run_record)
        reason_histogram.update(run_record["reason_histogram"])
        family_histogram.update(run_record["family_histogram"])

    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_path": input_path,
            "run_count": len(runs),
            "summary_row_count": len(summary_rows),
        },
        "mapping_contract": _contract_records(),
        "reason_histogram": dict(sorted(reason_histogram.items())),
        "family_histogram": dict(sorted(family_histogram.items())),
        "coverage": _coverage_records(family_histogram),
        "runs": runs,
    }
    if output_path is not None:
        _write_json(os.path.abspath(output_path), trace)
    return trace


def _trace_run(run: Mapping[str, Any], run_path: Optional[str], summary_row: Mapping[str, Any]) -> Dict[str, Any]:
    events = _extract_failure_events(run, summary_row)
    reason_histogram: Counter[str] = Counter()
    family_histogram: Counter[str] = Counter()
    normalized_events: List[Dict[str, Any]] = []
    unmapped_signals: List[str] = []

    for event in events:
        signal = str(event.get("signal") or event.get("reason") or event.get("family") or "")
        reason = _canonical_reason(event)
        family = _failure_family(signal, reason)
        count = int(event.get("count", 1))
        if reason == "unmapped":
            unmapped_signals.append(signal)
        reason_histogram[reason] += count
        family_histogram[family] += count
        normalized_events.append(
            {
                "family": family,
                "reason": reason,
                "count": count,
                "evidence_type": event.get("evidence_type", "unknown"),
                "signal": signal,
                "native_airfogsim_support": FAILURE_CONTRACT.get(family, {}).get("native_airfogsim_support", "unknown"),
                "native_failure_code": event.get("native_failure_code"),
                "sfc_id": event.get("sfc_id"),
                "task_id": event.get("task_id"),
                "details": event.get("details", {}),
            }
        )

    return {
        "run_id": _run_id(run),
        "run_output_path": run_path or run.get("run_output_path"),
        "baseline": run.get("baseline", summary_row.get("baseline")),
        "scenario": run.get("scenario", summary_row.get("scenario", "unknown")),
        "seed": _optional_int(run.get("seed", summary_row.get("seed"))),
        "mode": run.get("mode", summary_row.get("mode")),
        "status": run.get("status", summary_row.get("status", _status_from_run(run))),
        "completed": _optional_bool(run.get("completed", summary_row.get("completed"))),
        "exit_code": _optional_int(run.get("exit_code", summary_row.get("exit_code"))),
        "error": run.get("error", summary_row.get("error")),
        "failure_events": normalized_events,
        "reason_histogram": dict(sorted(reason_histogram.items())),
        "family_histogram": dict(sorted(family_histogram.items())),
        "unmapped_signals": sorted(set(item for item in unmapped_signals if item)),
    }


def _extract_failure_events(run: Mapping[str, Any], summary_row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    raw_metrics = run.get("raw_metrics") if isinstance(run.get("raw_metrics"), Mapping) else {}
    metrics = run.get("metrics") if isinstance(run.get("metrics"), Mapping) else {}

    for payload in (raw_metrics, metrics, run, summary_row):
        events.extend(_events_from_failure_reason_count(payload))

    for event in _iter_metric_events(raw_metrics):
        if str(event.get("event_type", "")).lower() != "failed":
            continue
        event_payload = event.get("payload", {}) if isinstance(event.get("payload"), Mapping) else {}
        signal = (
            event_payload.get("airfogsim_failure_reason")
            or event_payload.get("airfogsim_status")
            or event_payload.get("reason")
        )
        events.append(
            {
                "reason": event_payload.get("reason"),
                "signal": signal,
                "count": 1,
                "evidence_type": "metrics.event",
                "native_failure_code": event_payload.get("airfogsim_failure_code"),
                "sfc_id": event.get("entity_id"),
                "task_id": event_payload.get("task_id"),
                "details": dict(event_payload),
            }
        )

    for decision in _iter_decisions(run):
        rejected_reason = decision.get("rejected_reason")
        if rejected_reason is None:
            continue
        events.append(
            {
                "reason": _reason_value(rejected_reason),
                "signal": rejected_reason,
                "count": 1,
                "evidence_type": "decision.rejected_reason",
                "sfc_id": decision.get("sfc_id"),
                "details": {
                    "diagnostics": decision.get("diagnostics", {}),
                    "assignments": decision.get("assignments", {}),
                },
            }
        )

    if run.get("error"):
        events.append(
            {
                "reason": "benchmark_error",
                "signal": run.get("error"),
                "count": 1,
                "evidence_type": "run.error",
            }
        )

    return _dedupe_summary_events(events)


def _events_from_failure_reason_count(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("failure_reason_count")
    counts = _parse_count_mapping(raw)
    if not counts:
        reason = payload.get("failure_reason")
        if reason and str(reason).strip().lower() not in {"", "none", "nan"}:
            counts = {str(reason): 1}
    return [
        {
            "reason": reason,
            "signal": reason,
            "count": count,
            "evidence_type": "failure_reason_count",
        }
        for reason, count in sorted(counts.items())
        if int(count) > 0
    ]


def _canonical_reason(event: Mapping[str, Any]) -> str:
    native_code = event.get("native_failure_code")
    if native_code is not None:
        mapped = _map_native_failure_code(native_code)
        if mapped is not None:
            return mapped
    reason = _reason_value(event.get("reason"))
    if reason in SFC_REASON_VALUES:
        return reason
    mapped = _map_signal_to_reason(event.get("signal"))
    if mapped is not None:
        return mapped
    family = _failure_family(str(event.get("signal") or reason), reason)
    if family in FAILURE_CONTRACT:
        return str(FAILURE_CONTRACT[family]["canonical_reason"])
    if reason == "benchmark_error":
        return reason
    return "unmapped"


def _failure_family(signal: str, reason: str) -> str:
    token = _normalize(signal or reason)
    if reason == REASON_NO_CANDIDATE:
        if any(item in token for item in ("node_moved", "node_removed", "coverage_hole", "mobility", "out_of_coverage")):
            return "node_moved_away"
        return "no_candidate"
    if reason == REASON_ENERGY_VIOLATION or any(item in token for item in ("energy", "battery", "power_exhaust")):
        return "energy_exhausted"
    if any(item in token for item in ("link_down", "link_fault", "link_disconnect", "disconnected", "route_broken", "transmission_timeout", "out_of_tti")):
        return "link_disconnect"
    if reason == REASON_DEADLINE_MISSED or any(item in token for item in ("deadline", "timeout", "out_of_ddl", "timed_out")):
        return "deadline_too_small"
    if any(item in token for item in ("node_moved", "node_removed", "coverage_hole", "out_of_node")):
        return "node_moved_away"
    if reason == "benchmark_error":
        return "benchmark_error"
    return "other"


def _map_native_failure_code(code: Any) -> Optional[str]:
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return _map_signal_to_reason(code)
    return NATIVE_CODE_TO_REASON.get(numeric_code, REASON_TASK_FAILED if numeric_code >= 0 else None)


def _map_signal_to_reason(signal: Any) -> Optional[str]:
    token = _normalize(signal)
    if not token:
        return None
    if token in SFC_REASON_VALUES:
        return token
    if any(item in token for item in ("deadline", "timeout", "out_of_ddl", "out_of_tti")):
        return REASON_DEADLINE_MISSED
    if any(item in token for item in ("no_candidate", "out_of_node", "no_node", "resource_unavailable")):
        return REASON_NO_CANDIDATE
    if any(item in token for item in ("malicious", "reliability", "trust")):
        return REASON_RELIABILITY_VIOLATION
    if "accuracy" in token:
        return REASON_ACCURACY_VIOLATION
    if any(item in token for item in ("energy", "battery")):
        return REASON_ENERGY_VIOLATION
    if "invalid_graph" in token or ("invalid" in token and "graph" in token):
        return REASON_INVALID_GRAPH
    if "parent" in token and "failed" in token:
        return REASON_TASK_FAILED
    if "fail" in token or "error" in token or "exception" in token:
        return REASON_TASK_FAILED
    return None


def _contract_records() -> List[Dict[str, Any]]:
    records = []
    for family, contract in FAILURE_CONTRACT.items():
        native_codes = []
        for name in contract["native_failure_codes"]:
            code = NATIVE_CODE_NAMES[name]
            native_codes.append(
                {
                    "name": name,
                    "code": code,
                    "description": NATIVE_CODE_DESCRIPTIONS.get(code, "Unknown code."),
                    "mapped_reason": _map_native_failure_code(code) or REASON_TASK_FAILED,
                }
            )
        records.append({"family": family, **contract, "native_failure_codes": native_codes})
    return records


def _coverage_records(family_histogram: Mapping[str, int]) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    for family, contract in FAILURE_CONTRACT.items():
        coverage[family] = {
            "observed_count": int(family_histogram.get(family, 0)),
            "native_airfogsim_support": contract["native_airfogsim_support"],
            "needs_native_runtime_hook": bool(contract["needs_native_runtime_hook"]),
            "canonical_reason": contract["canonical_reason"],
        }
    return coverage


def _discover_run_json_paths(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path] if input_path.endswith(".json") and os.path.basename(input_path) != "manifest.json" else []
    candidates: List[str] = []
    runs_dir = os.path.join(input_path, "runs")
    search_root = runs_dir if os.path.isdir(runs_dir) else input_path
    for root, _, files in os.walk(search_root):
        for name in files:
            if not name.endswith(".json") or name in {"manifest.json", "failure_trace.json"}:
                continue
            candidates.append(os.path.join(root, name))
    return sorted(candidates)


def _read_summary_rows(input_path: str) -> List[Dict[str, Any]]:
    summary_path = input_path if os.path.isfile(input_path) and input_path.endswith(".csv") else os.path.join(input_path, "summary.csv")
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, "r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _iter_metric_events(raw_metrics: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    events = raw_metrics.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, Mapping):
                yield event


def _iter_decisions(run: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    decisions = run.get("decisions", [])
    if isinstance(decisions, Mapping):
        decisions = decisions.values()
    if isinstance(decisions, list):
        for decision in decisions:
            if isinstance(decision, Mapping):
                yield decision


def _dedupe_summary_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for event in events:
        key = (
            event.get("evidence_type"),
            event.get("reason"),
            event.get("signal"),
            event.get("native_failure_code"),
            event.get("sfc_id"),
            event.get("task_id"),
        )
        if event.get("evidence_type") == "failure_reason_count" and key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _parse_count_mapping(raw: Any) -> Dict[str, int]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    counts = {}
    for key, value in raw.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        counts[str(key)] = count
    return counts


def _run_key(payload: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(payload.get("baseline", "")),
        str(payload.get("scenario", "unknown")),
        str(payload.get("seed", "")),
    )


def _run_id(run: Mapping[str, Any]) -> str:
    parts = [str(run.get("baseline", "unknown")), str(run.get("scenario", "unknown")), f"seed_{run.get('seed', 'unknown')}"]
    return "__".join(parts)


def _status_from_run(run: Mapping[str, Any]) -> str:
    if run.get("error"):
        return "error"
    if _optional_int(run.get("timed_out")):
        return "timed_out"
    if _optional_int(run.get("failed")):
        return "failed"
    if run.get("completed") is True:
        return "completed"
    return "unknown"


def _reason_value(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_").replace(".", "")


def _normalize(value: Any) -> str:
    return _reason_value(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    return None


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LASDM failure_trace.json from benchmark outputs.")
    parser.add_argument("input_path", help="Benchmark output directory, run JSON, or summary.csv path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <input_path>/failure_trace.json for directories, or stdout if omitted for files.",
    )
    args = parser.parse_args(argv)

    output_path = args.output
    if output_path is None and os.path.isdir(args.input_path):
        output_path = os.path.join(args.input_path, "failure_trace.json")
    trace = generate_failure_trace(args.input_path, output_path=output_path)
    if output_path is None:
        json.dump(trace, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
