from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


LASDM_EVENT_SCHEMA_VERSION = "lasdm.event.v1"
SFC_ENTITY_TYPE = "sfc"
FUNCTION_ENTITY_TYPE = "function_node"


@dataclass(frozen=True)
class LASDMSFCMetrics:
    sfc_id: str
    status: Optional[str] = None
    latency_s: Optional[float] = None
    qos_hit: Optional[bool] = None
    deadline_s: Optional[float] = None
    payload_mb: Optional[float] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(frozen=True)
class LASDMFunctionMetrics:
    sfc_id: str
    function_id: str
    status: Optional[str] = None
    service_type: Optional[str] = None
    instance_id: Optional[str] = None
    node_id: Optional[str] = None
    latency_s: Optional[float] = None
    queue_delay_s: Optional[float] = None
    processing_time_s: Optional[float] = None
    cpu_mb: Optional[float] = None
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(frozen=True)
class LASDMEventRecord:
    event_type: str
    entity_type: str
    entity_id: str
    time_s: float
    schema_version: str = LASDM_EVENT_SCHEMA_VERSION
    sfc_id: Optional[str] = None
    function_id: Optional[str] = None
    status: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "time_s": float(self.time_s),
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "sfc_id": self.sfc_id,
            "function_id": self.function_id,
            "status": self.status,
            "metrics": _json_safe_dict(self.metrics),
            "payload": _json_safe_dict(self.payload),
        }


class LASDMEventLogger:
    """Collects standardized LASDM event records and writes JSONL output."""

    def __init__(self):
        self.records: List[LASDMEventRecord] = []

    def record_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        time_s: float,
        payload: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Union[Mapping[str, Any], LASDMSFCMetrics, LASDMFunctionMetrics]] = None,
        sfc_id: Optional[str] = None,
        function_id: Optional[str] = None,
        status: Optional[Any] = None,
    ) -> LASDMEventRecord:
        record = build_event_record(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            time_s=time_s,
            payload=payload,
            metrics=metrics,
            sfc_id=sfc_id,
            function_id=function_id,
            status=status,
        )
        self.records.append(record)
        return record

    def record_sfc_event(
        self,
        sfc_id: str,
        event_type: str,
        time_s: float,
        status: Optional[Any] = None,
        metrics: Optional[Union[Mapping[str, Any], LASDMSFCMetrics]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> LASDMEventRecord:
        return self.record_event(
            event_type=event_type,
            entity_type=SFC_ENTITY_TYPE,
            entity_id=sfc_id,
            time_s=time_s,
            payload=payload,
            metrics=metrics,
            sfc_id=sfc_id,
            status=status,
        )

    def record_function_event(
        self,
        sfc_id: str,
        function_id: str,
        event_type: str,
        time_s: float,
        status: Optional[Any] = None,
        metrics: Optional[Union[Mapping[str, Any], LASDMFunctionMetrics]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> LASDMEventRecord:
        return self.record_event(
            event_type=event_type,
            entity_type=FUNCTION_ENTITY_TYPE,
            entity_id=function_id,
            time_s=time_s,
            payload=payload,
            metrics=metrics,
            sfc_id=sfc_id,
            function_id=function_id,
            status=status,
        )

    def record_sfc_metrics(
        self,
        metrics: LASDMSFCMetrics,
        time_s: float,
        event_type: str = "sfc_metrics",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> LASDMEventRecord:
        return self.record_sfc_event(
            sfc_id=metrics.sfc_id,
            event_type=event_type,
            time_s=time_s,
            status=metrics.status,
            metrics=metrics,
            payload=payload,
        )

    def record_function_metrics(
        self,
        metrics: LASDMFunctionMetrics,
        time_s: float,
        event_type: str = "function_metrics",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> LASDMEventRecord:
        return self.record_function_event(
            sfc_id=metrics.sfc_id,
            function_id=metrics.function_id,
            event_type=event_type,
            time_s=time_s,
            status=metrics.status,
            metrics=metrics,
            payload=payload,
        )

    def record_standardized_event(self, event: Any) -> LASDMEventRecord:
        record = standardize_event(event)
        self.records.append(record)
        return record

    def extend_standardized_events(self, events: Iterable[Any]) -> None:
        for event in events:
            self.record_standardized_event(event)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def write_jsonl(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            for record in self.records:
                file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def write_events_jsonl(self, path: Union[str, Path]) -> None:
        self.write_jsonl(path)


def build_event_record(
    event_type: str,
    entity_type: str,
    entity_id: str,
    time_s: float,
    payload: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Union[Mapping[str, Any], LASDMSFCMetrics, LASDMFunctionMetrics]] = None,
    sfc_id: Optional[str] = None,
    function_id: Optional[str] = None,
    status: Optional[Any] = None,
) -> LASDMEventRecord:
    payload_dict = _json_safe_dict(dict(payload or {}))
    metrics_dict = _metrics_to_dict(metrics)
    status_value = _enum_value(status) if status is not None else _enum_value(payload_dict.get("status"))
    inferred_sfc_id = sfc_id or (entity_id if entity_type == SFC_ENTITY_TYPE else None)
    inferred_function_id = function_id or (entity_id if entity_type == FUNCTION_ENTITY_TYPE else None)

    if entity_type == SFC_ENTITY_TYPE:
        metrics_dict = _merge_metrics_from_payload(metrics_dict, payload_dict, ("latency_s", "qos_hit", "deadline_s", "payload_mb"))
        if "reason" in payload_dict and "failure_reason" not in metrics_dict:
            metrics_dict["failure_reason"] = payload_dict["reason"]
    elif entity_type == FUNCTION_ENTITY_TYPE:
        metrics_dict = _merge_metrics_from_payload(
            metrics_dict,
            payload_dict,
            ("latency_s", "queue_delay_s", "processing_time_s", "cpu_mb", "memory_mb"),
        )

    return LASDMEventRecord(
        event_type=str(event_type),
        entity_type=str(entity_type),
        entity_id=str(entity_id),
        time_s=float(time_s),
        sfc_id=inferred_sfc_id,
        function_id=inferred_function_id,
        status=status_value,
        metrics=metrics_dict,
        payload=payload_dict,
    )


def standardize_event(event: Any) -> LASDMEventRecord:
    payload = dict(getattr(event, "payload", {}) or {})
    return build_event_record(
        event_type=getattr(event, "event_type"),
        entity_type=getattr(event, "entity_type"),
        entity_id=getattr(event, "entity_id"),
        time_s=getattr(event, "time_s"),
        payload=payload,
        status=payload.get("status"),
    )


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _metrics_to_dict(metrics: Optional[Union[Mapping[str, Any], LASDMSFCMetrics, LASDMFunctionMetrics]]) -> Dict[str, Any]:
    if metrics is None:
        return {}
    if isinstance(metrics, (LASDMSFCMetrics, LASDMFunctionMetrics)):
        return _json_safe_dict(metrics.to_dict())
    return _json_safe_dict(dict(metrics))


def _merge_metrics_from_payload(metrics: Dict[str, Any], payload: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    merged = dict(metrics)
    for key in keys:
        if key in payload and key not in merged:
            merged[key] = payload[key]
    return merged


def _drop_none(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in raw.items() if value is not None}


def _json_safe_dict(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in raw.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
