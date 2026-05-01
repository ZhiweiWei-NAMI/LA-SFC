from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .event_logger import LASDMEventLogger, standardize_event
from .model import GraphStatus, SFCFailureReason


@dataclass(frozen=True)
class LASDMEvent:
    event_type: str
    entity_type: str
    entity_id: str
    time_s: float
    payload: Dict[str, Any] = field(default_factory=dict)


class LASDMMetrics:
    def __init__(self):
        self.events: List[LASDMEvent] = []
        self.submitted = 0
        self.succeeded = 0
        self.failed = 0
        self.timed_out = 0
        self.qos_hits = 0
        self.failure_reason_count: Dict[str, int] = {}
        self.latencies_s: List[float] = []

    def record_event(self, event: LASDMEvent) -> None:
        self.events.append(event)

    def record_submit(self, sfc_id: str, time_s: float) -> None:
        self.submitted += 1
        self.record_event(LASDMEvent("submitted", "sfc", sfc_id, time_s))

    def record_success(self, sfc_id: str, submit_time: float, finish_time: float, qos_hit: bool) -> None:
        self.succeeded += 1
        self.latencies_s.append(max(0.0, finish_time - submit_time))
        if qos_hit:
            self.qos_hits += 1
        self.record_event(
            LASDMEvent(
                "succeeded",
                "sfc",
                sfc_id,
                finish_time,
                {"latency_s": max(0.0, finish_time - submit_time), "qos_hit": qos_hit},
            )
        )

    def record_failure(
        self,
        sfc_id: str,
        time_s: float,
        reason: SFCFailureReason,
        status: GraphStatus = GraphStatus.FAILED,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if status == GraphStatus.TIMED_OUT:
            self.timed_out += 1
        else:
            self.failed += 1
        self.failure_reason_count[reason.value] = self.failure_reason_count.get(reason.value, 0) + 1
        payload = {"reason": reason.value, "status": status.value}
        if details:
            payload.update(details)
        self.record_event(LASDMEvent("failed", "sfc", sfc_id, time_s, payload))

    def summary(self) -> Dict[str, Any]:
        completed = self.succeeded + self.failed + self.timed_out
        avg_latency = sum(self.latencies_s) / len(self.latencies_s) if self.latencies_s else 0.0
        p95_latency = _percentile(self.latencies_s, 95.0)
        p99_latency = _percentile(self.latencies_s, 99.0)
        return {
            "submitted": self.submitted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "completed_terminal": completed,
            "success_ratio": self.succeeded / max(1, self.submitted),
            "terminal_ratio": completed / max(1, self.submitted),
            "qos_hit_ratio": self.qos_hits / max(1, self.succeeded),
            "avg_latency_s": avg_latency,
            "avg_graph_finish_time": avg_latency,
            "p95_graph_finish_time": p95_latency,
            "p99_graph_finish_time": p99_latency,
            "failure_reason_count": dict(sorted(self.failure_reason_count.items())),
        }

    def event_records(self) -> List[Dict[str, Any]]:
        return [standardize_event(event).to_dict() for event in self.events]

    def write_events_jsonl(self, path: str) -> None:
        logger = LASDMEventLogger()
        logger.extend_standardized_events(self.events)
        logger.write_jsonl(path)

    def write_summary_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.summary(), file, indent=2, sort_keys=True)


def _percentile(values: List[float], percentile: float) -> float:
    clean = sorted(max(0.0, float(value)) for value in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    fraction = rank - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction
