from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RuntimeProgressConfig:
    ticks_per_marl_step: int = 1
    max_ticks_after_action: int = 25
    stop_when_time_advances: bool = False
    stop_when_report_has_progress: bool = True
    stop_when_task_terminal: bool = True


def current_runtime_time(env: Any) -> float | None:
    if env is None:
        return None
    for name in ("simulation_time", "current_time", "sim_time", "time", "timestamp"):
        if hasattr(env, name):
            value = getattr(env, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            parsed = _as_float(value)
            if parsed is not None:
                return parsed
    for owner_name in ("simulator", "sim_manager", "simulation", "traffic_manager", "env"):
        owner = getattr(env, owner_name, None)
        if owner is None or owner is env:
            continue
        for method_name in ("getSimulationTime", "get_simulation_time", "get_time", "time"):
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    parsed = _as_float(method())
                except Exception:
                    parsed = None
                if parsed is not None:
                    return parsed
        for attr_name in ("simulation_time", "current_time", "sim_time", "time"):
            if hasattr(owner, attr_name):
                parsed = _as_float(getattr(owner, attr_name))
                if parsed is not None:
                    return parsed
    return None


def advance_runtime(env: Any, config: RuntimeProgressConfig, bridge_sync=None, before_step=None) -> Dict[str, Any]:
    if env is None or not hasattr(env, "step"):
        raise RuntimeError("advance_runtime requires an AirFogSim env exposing step()")
    before = current_runtime_time(env)
    before_counts = _task_counts(env)
    reports = []
    ticks = max(1, int(config.ticks_per_marl_step))
    max_ticks = max(ticks, int(config.max_ticks_after_action))
    for tick in range(max_ticks):
        if before_step is not None:
            try:
                before_step()
            except Exception as exc:
                reports.append({"type": "before_step_error", "error": repr(exc), "progress_like": False})
                break
        try:
            result = env.step()
        except TypeError:
            result = env.step(None)
        reports.append(_summarize_step_result(result))
        sync_result = None
        if bridge_sync is not None:
            try:
                sync_result = bridge_sync()
                reports[-1]["bridge_sync"] = _summarize_sync(sync_result)
            except Exception as exc:
                reports[-1]["bridge_sync_error"] = repr(exc)
        after = current_runtime_time(env)
        after_counts = _task_counts(env)
        mandatory_done = tick + 1 >= ticks
        time_advanced = before is not None and after is not None and after > before
        terminal_event = (
            after_counts.get("done", 0) > before_counts.get("done", 0)
            or after_counts.get("failed", 0) > before_counts.get("failed", 0)
            or _sync_terminal_event(sync_result)
        )
        queue_event = after_counts != before_counts
        progress = any(runtime_report_has_progress(report) for report in reports[-2:])
        reports[-1]["task_counts"] = after_counts
        reports[-1]["terminal_event"] = terminal_event
        reports[-1]["queue_event"] = queue_event
        if mandatory_done and (
            (config.stop_when_task_terminal and terminal_event)
            or (config.stop_when_time_advances and time_advanced)
            or (config.stop_when_report_has_progress and progress and terminal_event)
        ):
            break
    after = current_runtime_time(env)
    return {
        "runtime_available": True,
        "ticks": len(reports),
        "time_before_s": before,
        "time_after_s": after,
        "time_advanced": bool(before is not None and after is not None and after > before),
        "task_counts_before": before_counts,
        "task_counts_after": _task_counts(env),
        "step_reports": reports[-5:],
    }


def runtime_report_has_progress(report: Any) -> bool:
    if report is None:
        return False
    encoded = str(report).lower()
    return any(token in encoded for token in ("done", "finish", "complete", "execut", "computed", "offload", "task", "success"))


def _summarize_step_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        keys = sorted(result.keys())[:16]
        return {"type": "dict", "keys": keys, "progress_like": runtime_report_has_progress(result)}
    if isinstance(result, (tuple, list)):
        return {"type": type(result).__name__, "length": len(result), "progress_like": runtime_report_has_progress(result)}
    return {"type": type(result).__name__, "progress_like": runtime_report_has_progress(result)}


def _summarize_sync(sync_result: Any) -> Dict[str, Any]:
    if isinstance(sync_result, dict):
        return {
            "completed_tasks": int(sync_result.get("completed_tasks", 0) or 0),
            "failed_tasks": int(sync_result.get("failed_tasks", 0) or 0),
        }
    return {}


def _sync_terminal_event(sync_result: Any) -> bool:
    if not isinstance(sync_result, dict):
        return False
    return int(sync_result.get("completed_tasks", 0) or 0) > 0 or int(sync_result.get("failed_tasks", 0) or 0) > 0


def _task_counts(env: Any) -> Dict[str, int]:
    manager = getattr(env, "task_manager", None)
    if manager is None:
        return {}
    waiting = getattr(manager, "getWaitingToOffloadTasks", lambda: {})()
    offloading = getattr(manager, "getOffloadingTasks", lambda: {})()
    computing = getattr(manager, "getComputingTasks", lambda: {})()
    waiting_return = getattr(manager, "_waiting_to_return_tasks", {}) or {}
    returning = getattr(manager, "_returning_tasks", {}) or {}
    done = getattr(manager, "getDoneTasks", lambda: [])()
    failed = getattr(manager, "getOutOfDDLTasks", lambda: [])()
    return {
        "waiting_to_offload": _count_task_map(waiting),
        "offloading": _count_task_map(offloading),
        "computing": _count_task_map(computing),
        "waiting_to_return": _count_task_map(waiting_return),
        "returning": _count_task_map(returning),
        "done": len(done),
        "failed": len(failed),
    }


def _count_task_map(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(len(tasks or []) for tasks in value.values())


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
