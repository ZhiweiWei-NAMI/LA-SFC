from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def collect_resource_usage_snapshot(
    env: Any,
    current_time: Optional[float] = None,
    decision_time_s: Optional[float] = None,
    rb_allocations: Optional[Mapping[str, Sequence[int]]] = None,
    previous_energy_remaining: Optional[float] = None,
) -> Dict[str, Any]:
    """Read AirFogSim runtime resource state without mutating the simulator."""

    now = _resolve_time(env, current_time)
    node_ids = _node_ids(env)
    computing_by_node = _computing_tasks_by_node(env)
    cpu_capacity = {node_id: _node_cpu(env, node_id) for node_id in node_ids}
    cpu_allocated = _allocated_cpu_by_node(env, computing_by_node)
    if not cpu_allocated:
        cpu_allocated = _pending_cpu_by_node(computing_by_node)
    cpu_utilization = {
        node_id: _safe_ratio(cpu_allocated.get(node_id, 0.0), cpu_capacity.get(node_id, 0.0))
        for node_id in sorted(set(cpu_capacity) | set(cpu_allocated))
    }

    rb = _rb_usage(env, rb_allocations)
    backhaul = _backhaul_usage(env)
    energy = _energy_usage(env, previous_energy_remaining)
    storage = _storage_usage(env, node_ids)
    tasks = _task_queue_counts(env)

    return {
        "time_s": now,
        "decision_time_s": decision_time_s,
        "cpu_allocated_by_node": dict(sorted(cpu_allocated.items())),
        "cpu_capacity_by_node": dict(sorted(cpu_capacity.items())),
        "cpu_utilization_by_node": dict(sorted(cpu_utilization.items())),
        "avg_cpu_utilization": _mean(cpu_utilization.values()),
        "max_cpu_utilization": max(cpu_utilization.values()) if cpu_utilization else 0.0,
        "computing_tasks_by_node": {key: len(value) for key, value in sorted(computing_by_node.items())},
        **rb,
        **backhaul,
        **energy,
        **storage,
        **tasks,
    }


def summarize_runtime_overhead(
    decision_samples: Iterable[Any],
    baseline: str = "",
    scenario: str = "",
    seed: Any = "",
    mode: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    samples = [_sample_decision_time(sample) for sample in decision_samples]
    clean = [value for value in samples if value is not None and math.isfinite(value)]
    return {
        "run_id": run_id,
        "baseline": baseline,
        "scenario": scenario,
        "seed": seed,
        "mode": mode,
        "decision_samples": len(clean),
        "avg_decision_time_s": _mean(clean),
        "p95_decision_time_s": _percentile(clean, 95.0),
        "max_decision_time_s": max(clean) if clean else 0.0,
    }


def flatten_resource_usage_row(row: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    energy_remaining = _to_float(row.get("energy_remaining_total"), 0.0)
    storage_capacity = _to_float(row.get("storage_capacity_bytes_total"), 0.0)
    storage_used = _to_float(row.get("storage_used_bytes_total"), 0.0)
    return {
        "run_id": metadata.get("run_id", ""),
        "baseline": metadata.get("baseline", ""),
        "scenario": metadata.get("scenario", ""),
        "seed": metadata.get("seed", ""),
        "mode": metadata.get("mode", ""),
        "time_s": row.get("time_s", ""),
        "decision_time_s": row.get("decision_time_s", ""),
        "avg_cpu_utilization": row.get("avg_cpu_utilization", 0.0),
        "max_cpu_utilization": row.get("max_cpu_utilization", 0.0),
        "cpu_allocated_by_node": _json_cell(row.get("cpu_allocated_by_node", {})),
        "cpu_capacity_by_node": _json_cell(row.get("cpu_capacity_by_node", {})),
        "cpu_utilization_by_node": _json_cell(row.get("cpu_utilization_by_node", {})),
        "computing_tasks_by_node": _json_cell(row.get("computing_tasks_by_node", {})),
        "rb_total": row.get("rb_total", 0),
        "rb_used_unique": row.get("rb_used_unique", 0),
        "rb_task_allocations": row.get("rb_task_allocations", 0),
        "rb_utilization": row.get("rb_utilization", 0.0),
        "wireless_data_size_total": row.get("wireless_data_size_total", 0.0),
        "wireless_v2i_data_size": row.get("wireless_v2i_data_size", 0.0),
        "wireless_v2u_data_size": row.get("wireless_v2u_data_size", 0.0),
        "wireless_u2i_data_size": row.get("wireless_u2i_data_size", 0.0),
        "backhaul_usage_bytes": row.get("backhaul_usage_bytes", 0.0),
        "backhaul_queue_bytes": row.get("backhaul_queue_bytes", 0.0),
        "backhaul_active_flows": row.get("backhaul_active_flows", 0),
        "backhaul_capacity_mbps_total": row.get("backhaul_capacity_mbps_total", 0.0),
        "backhaul_queue_utilization": row.get("backhaul_queue_utilization", 0.0),
        "energy_remaining_total": energy_remaining,
        "energy_consumed_step": row.get("energy_consumed_step", 0.0),
        "energy_by_uav": _json_cell(row.get("energy_by_uav", {})),
        "storage_used_bytes_total": storage_used,
        "storage_capacity_bytes_total": storage_capacity,
        "storage_utilization": _safe_ratio(storage_used, storage_capacity),
        "storage_by_node": _json_cell(row.get("storage_by_node", {})),
        "waiting_to_offload_tasks": row.get("waiting_to_offload_tasks", 0),
        "offloading_tasks": row.get("offloading_tasks", 0),
        "computing_tasks": row.get("computing_tasks", 0),
        "done_tasks": row.get("done_tasks", 0),
        "failed_tasks": row.get("failed_tasks", 0),
    }


RESOURCE_USAGE_FIELDS = [
    "run_id",
    "baseline",
    "scenario",
    "seed",
    "mode",
    "time_s",
    "decision_time_s",
    "avg_cpu_utilization",
    "max_cpu_utilization",
    "cpu_allocated_by_node",
    "cpu_capacity_by_node",
    "cpu_utilization_by_node",
    "computing_tasks_by_node",
    "rb_total",
    "rb_used_unique",
    "rb_task_allocations",
    "rb_utilization",
    "wireless_data_size_total",
    "wireless_v2i_data_size",
    "wireless_v2u_data_size",
    "wireless_u2i_data_size",
    "backhaul_usage_bytes",
    "backhaul_queue_bytes",
    "backhaul_active_flows",
    "backhaul_capacity_mbps_total",
    "backhaul_queue_utilization",
    "energy_remaining_total",
    "energy_consumed_step",
    "energy_by_uav",
    "storage_used_bytes_total",
    "storage_capacity_bytes_total",
    "storage_utilization",
    "storage_by_node",
    "waiting_to_offload_tasks",
    "offloading_tasks",
    "computing_tasks",
    "done_tasks",
    "failed_tasks",
]
RUNTIME_OVERHEAD_FIELDS = [
    "run_id",
    "baseline",
    "scenario",
    "seed",
    "mode",
    "decision_samples",
    "avg_decision_time_s",
    "p95_decision_time_s",
    "max_decision_time_s",
]


def _resolve_time(env: Any, current_time: Optional[float]) -> float:
    if current_time is not None:
        return float(current_time)
    return float(getattr(env, "simulation_time", 0.0))


def _node_ids(env: Any) -> List[str]:
    ids: List[str] = []
    for method_name in ("getVehicleIds", "getUAVIds", "getRSUIds", "getCloudServerIds"):
        method = getattr(env, method_name, None)
        if callable(method):
            ids.extend(str(item) for item in method())
    if ids:
        return sorted(set(ids))
    for attr in ("vehicles", "UAVs", "RSUs", "cloudServers"):
        ids.extend(str(item) for item in getattr(env, attr, {}) or {})
    return sorted(set(ids))


def _node_cpu(env: Any, node_id: str) -> float:
    getter = getattr(env, "_getNodeById", None)
    node = getter(node_id) if callable(getter) else None
    if node is None or not hasattr(node, "getFogProfile"):
        return 0.0
    try:
        return max(0.0, float((node.getFogProfile() or {}).get("cpu", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _computing_tasks_by_node(env: Any) -> Dict[str, List[Any]]:
    manager = getattr(env, "task_manager", None)
    if manager is None or not hasattr(manager, "getComputingTasks"):
        return {}
    return {str(node_id): list(tasks) for node_id, tasks in (manager.getComputingTasks() or {}).items()}


def _allocated_cpu_by_node(env: Any, computing_by_node: Mapping[str, Sequence[Any]]) -> Dict[str, float]:
    callback = getattr(env, "alloc_cpu_callback", None)
    if not callable(callback) or not computing_by_node:
        return {}
    try:
        allocations = callback(
            computing_by_node,
            simulation_interval=getattr(env, "simulation_interval", None),
            current_time=getattr(env, "simulation_time", None),
        )
    except Exception:
        return {}
    by_task = {str(key): _to_float(value, 0.0) for key, value in (allocations or {}).items()}
    by_node: Dict[str, float] = {}
    for node_id, tasks in computing_by_node.items():
        total = 0.0
        for task in tasks:
            task_id = getattr(task, "getTaskId", lambda: "")()
            total += by_task.get(str(task_id), 0.0)
        by_node[str(node_id)] = total
    return by_node


def _pending_cpu_by_node(computing_by_node: Mapping[str, Sequence[Any]]) -> Dict[str, float]:
    by_node: Dict[str, float] = {}
    for node_id, tasks in computing_by_node.items():
        total = 0.0
        for task in tasks:
            required = _call_float(task, "getTaskCPU")
            computed = _call_float(task, "getComputedSize")
            total += max(0.0, required - computed)
        by_node[str(node_id)] = total
    return by_node


def _rb_usage(env: Any, rb_allocations: Optional[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
    allocations = rb_allocations if rb_allocations is not None else getattr(env, "activated_offloading_tasks_with_RB_Nos", {})
    rb_lists = [list(value) for value in (allocations or {}).values()]
    unique_rbs = sorted({int(rb) for rb_list in rb_lists for rb in rb_list})
    channel_manager = getattr(env, "channel_manager", None)
    rb_total = int(getattr(channel_manager, "n_RB", 0) or 0)
    return {
        "rb_total": rb_total,
        "rb_used_unique": len(unique_rbs),
        "rb_task_allocations": sum(len(rb_list) for rb_list in rb_lists),
        "rb_utilization": _safe_ratio(len(unique_rbs), rb_total),
        "wireless_data_size_total": _channel_data_size(env, "channel"),
        "wireless_v2i_data_size": _channel_data_size(env, "V2I_channel"),
        "wireless_v2u_data_size": _channel_data_size(env, "V2U_channel"),
        "wireless_u2i_data_size": _channel_data_size(env, "U2I_channel"),
    }


def _channel_data_size(env: Any, attr: str) -> float:
    value = getattr(env, attr, {}) or {}
    if isinstance(value, Mapping):
        return _to_float(value.get("data_size"), 0.0)
    return 0.0


def _backhaul_usage(env: Any) -> Dict[str, Any]:
    manager = getattr(env, "wired_manager", None)
    if manager is None:
        return {
            "backhaul_usage_bytes": 0.0,
            "backhaul_queue_bytes": 0.0,
            "backhaul_active_flows": 0,
            "backhaul_capacity_mbps_total": 0.0,
            "backhaul_queue_utilization": 0.0,
        }
    queues = dict(getattr(manager, "_queues", {}) or {})
    links = dict(getattr(manager, "_links", {}) or {})
    flows = dict(getattr(manager, "_flows", {}) or {})
    queue_bytes = sum(_to_float(value, 0.0) for value in queues.values())
    capacity_mbps = sum(_to_float(link.get("capacity_mbps"), 0.0) for link in links.values() if isinstance(link, Mapping))
    interval = max(0.0, _to_float(getattr(env, "simulation_interval", 0.0), 0.0))
    capacity_bytes = capacity_mbps * 1e6 / 8.0 * interval
    usage = _to_float(getattr(manager, "last_step_transmitted_bytes", 0.0), 0.0)
    return {
        "backhaul_usage_bytes": usage,
        "backhaul_queue_bytes": queue_bytes,
        "backhaul_active_flows": len(flows),
        "backhaul_capacity_mbps_total": capacity_mbps,
        "backhaul_queue_utilization": _safe_ratio(queue_bytes, capacity_bytes),
    }


def _energy_usage(env: Any, previous_energy_remaining: Optional[float]) -> Dict[str, Any]:
    manager = getattr(env, "energy_manager", None)
    info = dict(getattr(manager, "_UAVs_energy_info", {}) or {}) if manager is not None else {}
    by_uav = {
        str(node_id): _to_float(payload.get("energy"), 0.0)
        for node_id, payload in info.items()
        if isinstance(payload, Mapping)
    }
    total = sum(by_uav.values())
    consumed = 0.0
    if previous_energy_remaining is not None:
        consumed = max(0.0, float(previous_energy_remaining) - total)
    return {
        "energy_remaining_total": total,
        "energy_consumed_step": consumed,
        "energy_by_uav": dict(sorted(by_uav.items())),
    }


def _storage_usage(env: Any, node_ids: Sequence[str]) -> Dict[str, Any]:
    manager = getattr(env, "storage_manager", None)
    by_node: Dict[str, Dict[str, Any]] = {}
    for node_id in node_ids:
        state = None
        if manager is not None and hasattr(manager, "getCacheState"):
            try:
                state = manager.getCacheState(node_id)
            except Exception:
                state = None
        if not isinstance(state, Mapping):
            state = {"capacity": 0, "used": 0, "items": 0, "hit_ratio": 0.0}
        by_node[str(node_id)] = {
            "capacity": _to_float(state.get("capacity"), 0.0),
            "used": _to_float(state.get("used"), 0.0),
            "items": int(_to_float(state.get("items"), 0.0)),
            "hit_ratio": _to_float(state.get("hit_ratio"), 0.0),
        }
    used = sum(item["used"] for item in by_node.values())
    capacity = sum(item["capacity"] for item in by_node.values())
    return {
        "storage_used_bytes_total": used,
        "storage_capacity_bytes_total": capacity,
        "storage_by_node": dict(sorted(by_node.items())),
    }


def _task_queue_counts(env: Any) -> Dict[str, int]:
    manager = getattr(env, "task_manager", None)
    if manager is None:
        return {
            "waiting_to_offload_tasks": 0,
            "offloading_tasks": 0,
            "computing_tasks": 0,
            "done_tasks": 0,
            "failed_tasks": 0,
        }
    return {
        "waiting_to_offload_tasks": _count_mapping_tasks(manager, "getWaitingToOffloadTasks"),
        "offloading_tasks": _count_mapping_tasks(manager, "getOffloadingTasks"),
        "computing_tasks": _count_mapping_tasks(manager, "getComputingTasks"),
        "done_tasks": _count_list_tasks(manager, "getDoneTasks"),
        "failed_tasks": _count_list_tasks(manager, "getOutOfDDLTasks"),
    }


def _count_mapping_tasks(manager: Any, method_name: str) -> int:
    method = getattr(manager, method_name, None)
    if not callable(method):
        return 0
    try:
        return sum(len(tasks) for tasks in (method() or {}).values())
    except Exception:
        return 0


def _count_list_tasks(manager: Any, method_name: str) -> int:
    method = getattr(manager, method_name, None)
    if not callable(method):
        return 0
    try:
        return len(method() or [])
    except Exception:
        return 0


def _sample_decision_time(sample: Any) -> Optional[float]:
    if isinstance(sample, Mapping):
        for key in ("decision_time_s", "schedule_step_time_s", "duration_s"):
            if key in sample:
                return _to_float(sample.get(key), None)
        return None
    return _to_float(sample, None)


def _call_float(obj: Any, method_name: str) -> float:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return 0.0
    try:
        return _to_float(method(), 0.0)
    except Exception:
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = _to_float(denominator, 0.0)
    if denominator <= 0.0:
        return 0.0
    return _to_float(numerator, 0.0) / denominator


def _mean(values: Iterable[float]) -> float:
    clean = [_to_float(value, None) for value in values]
    clean = [value for value in clean if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    clean = sorted(_to_float(value, 0.0) for value in values if math.isfinite(_to_float(value, 0.0)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    fraction = rank - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _to_float(value: Any, default: Any = 0.0) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_cell(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"))
