from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceInstanceStatus
from .metrics import LASDMEvent


class InstanceLifecyclePhase(str, Enum):
    COLD_START = "cold_start"
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DRAINING = "draining"
    MIGRATING = "migrating"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class InstanceLifecycleUpdate:
    phase: InstanceLifecyclePhase
    instance_id: Optional[str] = None
    health_score: Optional[float] = None
    reliability_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    trust_score: Optional[float] = None
    cold_start_s: Optional[float] = None
    current_load: Optional[int] = None
    max_concurrency: Optional[int] = None
    used: Optional[Dict[str, float]] = None
    capacity: Optional[Dict[str, float]] = None
    reason: Optional[str] = None
    migration_target_node_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LASDMInstanceLifecycle:
    """Applies runtime lifecycle updates to LASDM service instances."""

    def __init__(self, directory: Optional[ServiceInstanceDirectory] = None):
        self.directory = directory

    def apply(
        self,
        instance: ServiceInstance,
        update: InstanceLifecycleUpdate,
        current_time: float,
    ) -> LASDMEvent:
        return apply_instance_lifecycle_update(instance, update, current_time)

    def apply_by_id(self, instance_id: str, update: InstanceLifecycleUpdate, current_time: float) -> LASDMEvent:
        if self.directory is None:
            raise ValueError("apply_by_id requires a ServiceInstanceDirectory")
        return self.apply(self.directory.get(instance_id), update, current_time)

    def apply_step_updates(
        self,
        updates: Union[
            Dict[str, InstanceLifecycleUpdate],
            Iterable[Union[InstanceLifecycleUpdate, Tuple[str, InstanceLifecycleUpdate]]],
        ],
        current_time: float,
    ) -> List[LASDMEvent]:
        if self.directory is None:
            raise ValueError("apply_step_updates requires a ServiceInstanceDirectory")
        events: List[LASDMEvent] = []
        items = updates.items() if isinstance(updates, dict) else updates
        for item in items:
            if isinstance(item, tuple):
                instance_id, update = item
            else:
                update = item
                if update.instance_id is None:
                    raise ValueError("step updates must provide an instance_id")
                instance_id = update.instance_id
            events.append(self.apply_by_id(str(instance_id), update, current_time))
        return events

    def cold_start(
        self,
        instance: ServiceInstance,
        current_time: float,
        cold_start_s: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.COLD_START,
                cold_start_s=cold_start_s,
                reason=reason,
            ),
            current_time,
        )

    def start(
        self,
        instance: ServiceInstance,
        current_time: float,
        cold_start_s: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.STARTING,
                cold_start_s=cold_start_s,
                reason=reason,
            ),
            current_time,
        )

    def activate(
        self,
        instance: ServiceInstance,
        current_time: float,
        health_score: Optional[float] = None,
        reliability_score: Optional[float] = None,
        accuracy_score: Optional[float] = None,
        trust_score: Optional[float] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.ACTIVE,
                health_score=health_score,
                reliability_score=reliability_score,
                accuracy_score=accuracy_score,
                trust_score=trust_score,
                cold_start_s=0.0,
            ),
            current_time,
        )

    def recover(
        self,
        instance: ServiceInstance,
        current_time: float,
        health_score: Optional[float] = None,
        reliability_score: Optional[float] = None,
        accuracy_score: Optional[float] = None,
        trust_score: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.RECOVERED,
                health_score=health_score,
                reliability_score=reliability_score,
                accuracy_score=accuracy_score,
                trust_score=trust_score,
                cold_start_s=0.0,
                reason=reason,
            ),
            current_time,
        )

    def degrade(
        self,
        instance: ServiceInstance,
        current_time: float,
        health_score: Optional[float] = None,
        reliability_score: Optional[float] = None,
        accuracy_score: Optional[float] = None,
        trust_score: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.DEGRADED,
                health_score=health_score,
                reliability_score=reliability_score,
                accuracy_score=accuracy_score,
                trust_score=trust_score,
                reason=reason,
            ),
            current_time,
        )

    def drain(self, instance: ServiceInstance, current_time: float, reason: Optional[str] = None) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(phase=InstanceLifecyclePhase.DRAINING, reason=reason),
            current_time,
        )

    def migrate(
        self,
        instance: ServiceInstance,
        current_time: float,
        target_node_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(
                phase=InstanceLifecyclePhase.MIGRATING,
                migration_target_node_id=target_node_id,
                reason=reason,
            ),
            current_time,
        )

    def fail(self, instance: ServiceInstance, current_time: float, reason: Optional[str] = None) -> LASDMEvent:
        return self.apply(
            instance,
            InstanceLifecycleUpdate(phase=InstanceLifecyclePhase.FAILED, health_score=0.0, reason=reason),
            current_time,
        )


def apply_instance_lifecycle_update(
    instance: ServiceInstance,
    update: InstanceLifecycleUpdate,
    current_time: float,
) -> LASDMEvent:
    previous_status = instance.status
    previous_phase = instance.metadata.get("lifecycle_phase", instance.status.value)
    phase = _coerce_phase(update.phase)

    instance.status = _status_for_phase(phase)
    instance.last_heartbeat = current_time
    instance.metadata["lifecycle_phase"] = phase.value
    if update.reason is not None:
        instance.metadata["lifecycle_reason"] = update.reason
    if update.migration_target_node_id is not None:
        instance.metadata["migration_target_node_id"] = update.migration_target_node_id
    elif phase != InstanceLifecyclePhase.MIGRATING:
        instance.metadata.pop("migration_target_node_id", None)
    if update.metadata:
        instance.metadata.update(update.metadata)

    if update.health_score is not None:
        instance.health_score = _clamp_score(update.health_score)
    if update.reliability_score is not None:
        instance.reliability_score = _clamp_score(update.reliability_score)
    if update.accuracy_score is not None:
        instance.accuracy_score = _clamp_score(update.accuracy_score)
    if update.trust_score is not None:
        instance.trust_score = _clamp_score(update.trust_score)
    if update.cold_start_s is not None:
        instance.cold_start_s = max(0.0, float(update.cold_start_s))
    instance.update_runtime_load(
        current_load=update.current_load,
        used=update.used,
        capacity=update.capacity,
        max_concurrency=update.max_concurrency,
    )

    return LASDMEvent(
        "instance_lifecycle_updated",
        "service_instance",
        instance.instance_id,
        current_time,
        lifecycle_event_payload(
            instance,
            previous_status=previous_status,
            previous_phase=str(previous_phase),
            reason=update.reason,
        ),
    )


def lifecycle_event_payload(
    instance: ServiceInstance,
    previous_status: Optional[ServiceInstanceStatus] = None,
    previous_phase: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "instance_id": instance.instance_id,
        "service_id": instance.service_id,
        "node_id": instance.node_id,
        "status": instance.status.value,
        "lifecycle_phase": instance.metadata.get("lifecycle_phase", instance.status.value),
        "health_score": instance.health_score,
        "reliability_score": instance.reliability_score,
        "accuracy_score": instance.accuracy_score,
        "trust_score": instance.trust_score,
        "cold_start_s": instance.cold_start_s,
        "current_load": instance.current_load,
        "max_concurrency": instance.max_concurrency,
        "load_ratio": instance.load_ratio(),
        "selectable": instance.can_reserve({}),
    }
    if previous_status is not None:
        payload["previous_status"] = previous_status.value
    if previous_phase is not None:
        payload["previous_lifecycle_phase"] = previous_phase
    if reason is not None:
        payload["reason"] = reason
    if "migration_target_node_id" in instance.metadata:
        payload["migration_target_node_id"] = instance.metadata["migration_target_node_id"]
    return payload


def instance_qos_metadata(
    instance: ServiceInstance,
    resource_request: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    resource_request = resource_request or {}
    selectable = instance.can_reserve(resource_request)
    blocked_reason = None if selectable else _blocked_reason(instance, resource_request)
    metadata = {
        "instance_id": instance.instance_id,
        "service_id": instance.service_id,
        "node_id": instance.node_id,
        "status": instance.status.value,
        "lifecycle_phase": instance.metadata.get("lifecycle_phase", instance.status.value),
        "selectable": selectable,
        "blocked_reason": blocked_reason,
        "health_score": instance.health_score,
        "reliability_score": instance.reliability_score,
        "accuracy_score": instance.accuracy_score,
        "trust_score": instance.trust_score,
        "cold_start_s": instance.cold_start_s,
        "current_load": instance.current_load,
        "max_concurrency": instance.max_concurrency,
        "load_ratio": instance.load_ratio(),
    }
    if "migration_target_node_id" in instance.metadata:
        metadata["migration_target_node_id"] = instance.metadata["migration_target_node_id"]
    return metadata


def _status_for_phase(phase: InstanceLifecyclePhase) -> ServiceInstanceStatus:
    if phase == InstanceLifecyclePhase.COLD_START:
        return ServiceInstanceStatus.STARTING
    if phase == InstanceLifecyclePhase.MIGRATING:
        return ServiceInstanceStatus.DRAINING
    if phase == InstanceLifecyclePhase.RECOVERED:
        return ServiceInstanceStatus.ACTIVE
    return ServiceInstanceStatus(phase.value)


def _blocked_reason(instance: ServiceInstance, resource_request: Dict[str, float]) -> str:
    phase = instance.metadata.get("lifecycle_phase", instance.status.value)
    if instance.status not in {ServiceInstanceStatus.ACTIVE, ServiceInstanceStatus.DEGRADED}:
        return f"status:{phase}"
    if instance.current_load >= instance.max_concurrency:
        return "capacity:concurrency"
    for key, value in resource_request.items():
        if instance.available(key) + 1e-9 < float(value):
            return f"capacity:{key}"
    return "unknown"


def _coerce_phase(value: InstanceLifecyclePhase) -> InstanceLifecyclePhase:
    if isinstance(value, InstanceLifecyclePhase):
        return value
    return InstanceLifecyclePhase(str(value))


def _clamp_score(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
