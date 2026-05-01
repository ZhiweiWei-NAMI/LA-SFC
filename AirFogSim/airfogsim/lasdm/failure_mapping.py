from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..enum_const import EnumerateConstants
from .model import GraphStatus, SFCFailureReason


AIRFOGSIM_FAILURE_CODE_TO_SFC_REASON: Dict[int, SFCFailureReason] = {
    EnumerateConstants.TASK_FAIL_OUT_OF_DDL: SFCFailureReason.DEADLINE_MISSED,
    EnumerateConstants.TASK_FAIL_OUT_OF_TTI: SFCFailureReason.DEADLINE_MISSED,
    EnumerateConstants.TASK_FAIL_OUT_OF_NODE: SFCFailureReason.NO_CANDIDATE,
    EnumerateConstants.TASK_FAIL_PARENT_FAILED: SFCFailureReason.TASK_FAILED,
    EnumerateConstants.TASK_FAIL_MALICIOUS_RESULT: SFCFailureReason.RELIABILITY_VIOLATION,
}

_FAILURE_STATUS_TO_SFC_REASON: Dict[str, SFCFailureReason] = {
    "accuracy_violation": SFCFailureReason.ACCURACY_VIOLATION,
    "candidate_unavailable": SFCFailureReason.NO_CANDIDATE,
    "deadline_missed": SFCFailureReason.DEADLINE_MISSED,
    "energy_violation": SFCFailureReason.ENERGY_VIOLATION,
    "failed": SFCFailureReason.TASK_FAILED,
    "invalid_graph": SFCFailureReason.INVALID_GRAPH,
    "malicious_result": SFCFailureReason.RELIABILITY_VIOLATION,
    "no_candidate": SFCFailureReason.NO_CANDIDATE,
    "no_node": SFCFailureReason.NO_CANDIDATE,
    "out_of_ddl": SFCFailureReason.DEADLINE_MISSED,
    "out_of_deadline": SFCFailureReason.DEADLINE_MISSED,
    "out_of_node": SFCFailureReason.NO_CANDIDATE,
    "out_of_tti": SFCFailureReason.DEADLINE_MISSED,
    "parent_failed": SFCFailureReason.TASK_FAILED,
    "reliability_violation": SFCFailureReason.RELIABILITY_VIOLATION,
    "resource_unavailable": SFCFailureReason.NO_CANDIDATE,
    "task_failed": SFCFailureReason.TASK_FAILED,
    "task_fails_due_to_malicious_result_detected": SFCFailureReason.RELIABILITY_VIOLATION,
    "task_fails_due_to_out_of_deadline": SFCFailureReason.DEADLINE_MISSED,
    "task_fails_due_to_out_of_node": SFCFailureReason.NO_CANDIDATE,
    "task_fails_due_to_parent_task_failed": SFCFailureReason.TASK_FAILED,
    "task_fails_due_to_transmission_timeout": SFCFailureReason.DEADLINE_MISSED,
    "timed_out": SFCFailureReason.DEADLINE_MISSED,
    "timeout": SFCFailureReason.DEADLINE_MISSED,
    "transmission_timeout": SFCFailureReason.DEADLINE_MISSED,
}

_NON_FAILURE_STATUSES = {
    "computed",
    "computing",
    "done",
    "finished",
    "generated",
    "offloading",
    "pending",
    "queued",
    "returning",
    "running",
    "success",
    "succeeded",
    "to_generate",
    "to_offload",
    "to_return",
    "waiting",
    "waiting_to_offload",
    "waiting_to_return",
}


def map_airfogsim_task_failure(
    task: Any = None,
    failure_code: Any = None,
    status: Any = None,
    error: Any = None,
    default: Optional[SFCFailureReason] = SFCFailureReason.TASK_FAILED,
) -> Optional[SFCFailureReason]:
    """Map AirFogSim task failure/status/error signals to an LASDM SFC reason."""

    if isinstance(status, GraphStatus):
        if status == GraphStatus.TIMED_OUT:
            return SFCFailureReason.DEADLINE_MISSED
        if status == GraphStatus.FAILED:
            return default
        if status != GraphStatus.RUNNING:
            return None
    if isinstance(status, SFCFailureReason):
        return status

    task_failure_code = failure_code if failure_code is not None else _task_value(task, "failure_reason_code", "_failure_reason_code")
    code_reason = _reason_from_failure_code(task_failure_code)
    if code_reason is not None:
        return code_reason

    for signal in (status, error, _task_failure_reason_text(task), _task_value(task, "task_lifecycle_state")):
        reason = _reason_from_signal(signal)
        if reason is not None:
            return reason

    if _has_failure_evidence(task_failure_code, status, error):
        return default
    return None


def build_failure_event_payload(
    task: Any = None,
    reason: Optional[SFCFailureReason] = None,
    failure_code: Any = None,
    status: Any = None,
    error: Any = None,
    sfc_node_id: Optional[str] = None,
    service_type: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the normalized payload used by LASDM failure events."""

    mapped_reason = reason or map_airfogsim_task_failure(
        task=task,
        failure_code=failure_code,
        status=status,
        error=error,
        default=SFCFailureReason.TASK_FAILED,
    )
    if mapped_reason is None:
        mapped_reason = SFCFailureReason.TASK_FAILED

    task_failure_code = failure_code if failure_code is not None else _task_value(task, "failure_reason_code", "_failure_reason_code")
    payload: Dict[str, Any] = {
        "reason": mapped_reason.value,
        "status": _graph_status_for_reason(mapped_reason, status).value,
    }
    _add_if_present(payload, "sfc_node_id", sfc_node_id)
    _add_if_present(payload, "service_type", service_type)
    _add_if_present(payload, "task_id", _task_value(task, "task_id", "_task_id", "getTaskId"))
    _add_if_present(payload, "task_node_id", _task_value(task, "task_node_id", "_task_node_id", "getTaskNodeId"))
    _add_if_present(payload, "current_node_id", _task_value(task, "current_node_id", "getCurrentNodeId"))
    _add_if_present(payload, "airfogsim_status", _serializable_signal(status))
    _add_if_present(payload, "airfogsim_failure_code", task_failure_code)
    _add_if_present(payload, "airfogsim_failure_reason", _task_failure_reason_text(task))

    if error is not None:
        payload["error_type"] = error.__class__.__name__
        payload["error_message"] = str(error)
    if details:
        payload["details"] = dict(details)
    return payload


def build_failure_event_record(
    sfc_id: str,
    time_s: float,
    task: Any = None,
    reason: Optional[SFCFailureReason] = None,
    failure_code: Any = None,
    status: Any = None,
    error: Any = None,
    sfc_node_id: Optional[str] = None,
    service_type: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable event record compatible with LASDMEvent fields."""

    return {
        "event_type": "failed",
        "entity_type": "sfc",
        "entity_id": sfc_id,
        "time_s": float(time_s),
        "payload": build_failure_event_payload(
            task=task,
            reason=reason,
            failure_code=failure_code,
            status=status,
            error=error,
            sfc_node_id=sfc_node_id,
            service_type=service_type,
            details=details,
        ),
    }


def _reason_from_failure_code(code: Any) -> Optional[SFCFailureReason]:
    if code is None:
        return None
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return _reason_from_signal(code)
    if numeric_code < 0:
        return None
    return AIRFOGSIM_FAILURE_CODE_TO_SFC_REASON.get(numeric_code, SFCFailureReason.TASK_FAILED)


def _reason_from_signal(signal: Any) -> Optional[SFCFailureReason]:
    if signal is None:
        return None
    if isinstance(signal, SFCFailureReason):
        return signal
    if isinstance(signal, GraphStatus):
        if signal == GraphStatus.TIMED_OUT:
            return SFCFailureReason.DEADLINE_MISSED
        if signal == GraphStatus.FAILED:
            return SFCFailureReason.TASK_FAILED
        return None

    token = _normalize_signal(signal)
    if not token or token in _NON_FAILURE_STATUSES:
        return None
    if token in _FAILURE_STATUS_TO_SFC_REASON:
        return _FAILURE_STATUS_TO_SFC_REASON[token]
    if any(item in token for item in ("deadline", "timeout", "out_of_ddl", "out_of_tti")):
        return SFCFailureReason.DEADLINE_MISSED
    if any(item in token for item in ("no_candidate", "out_of_node", "no_node", "resource_unavailable")):
        return SFCFailureReason.NO_CANDIDATE
    if any(item in token for item in ("malicious", "reliability", "trust")):
        return SFCFailureReason.RELIABILITY_VIOLATION
    if "accuracy" in token:
        return SFCFailureReason.ACCURACY_VIOLATION
    if "energy" in token:
        return SFCFailureReason.ENERGY_VIOLATION
    if "invalid_graph" in token or ("invalid" in token and "graph" in token):
        return SFCFailureReason.INVALID_GRAPH
    if "parent" in token and "failed" in token:
        return SFCFailureReason.TASK_FAILED
    if "fail" in token or "error" in token or "exception" in token:
        return SFCFailureReason.TASK_FAILED
    return None


def _normalize_signal(signal: Any) -> str:
    value = getattr(signal, "value", signal)
    return str(value).strip().lower().replace("-", "_").replace(" ", "_").replace(".", "")


def _has_failure_evidence(failure_code: Any, status: Any, error: Any) -> bool:
    if error is not None:
        return True
    if failure_code is not None:
        try:
            return int(failure_code) >= 0
        except (TypeError, ValueError):
            return bool(str(failure_code).strip())
    token = _normalize_signal(status) if status is not None else ""
    return bool(token and token not in _NON_FAILURE_STATUSES)


def _graph_status_for_reason(reason: SFCFailureReason, status: Any) -> GraphStatus:
    if isinstance(status, GraphStatus):
        return GraphStatus.TIMED_OUT if status == GraphStatus.TIMED_OUT else GraphStatus.FAILED
    if reason == SFCFailureReason.DEADLINE_MISSED:
        token = _normalize_signal(status) if status is not None else ""
        if token in {"timed_out", "timeout", "out_of_ddl", "deadline_missed"}:
            return GraphStatus.TIMED_OUT
    return GraphStatus.FAILED


def _task_failure_reason_text(task: Any) -> Optional[str]:
    value = _task_value(task, "failure_reason", "getTaskFailureReason")
    if value == "Unknown code.":
        return None
    return value


def _task_value(task: Any, *names: str) -> Any:
    if task is None:
        return None
    for name in names:
        if isinstance(task, Mapping) and name in task:
            return task[name]
        value = getattr(task, name, None)
        if callable(value):
            try:
                return value()
            except TypeError:
                continue
        if value is not None:
            return value
    return None


def _serializable_signal(signal: Any) -> Any:
    if isinstance(signal, (GraphStatus, SFCFailureReason)):
        return signal.value
    return signal


def _add_if_present(payload: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value
