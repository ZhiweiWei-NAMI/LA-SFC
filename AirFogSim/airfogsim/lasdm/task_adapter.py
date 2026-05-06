from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..entities.task import Task
from ..enum_const import EnumerateConstants
from .model import LASDMServiceChain, LASDMSFCNode


LASDM_TO_TASK_FIELD_MAPPING = {
    "LASDMServiceChain.source_node_id": "Task.task_node_id",
    "LASDMServiceChain.payload_mb": "Task.task_size",
    "LASDMSFCNode.metadata.cpu_per_mb * input_payload_mb": "Task.task_cpu",
    "LASDMSFCNode.cpu_mb": "Task.task_cpu",
    "LASDMQoS.deadline_s - elapsed_s": "Task.task_deadline",
    "LASDMQoS.priority": "Task.task_priority",
    "LASDMSFCNode.output_payload_mb": "Task.required_returned_size",
    "LASDMServiceChain.sink_node_id": "Task.to_return_node_id",
}


@dataclass(frozen=True)
class LASDMTaskMapping:
    sfc_id: str
    sfc_node_id: str
    service_type: str
    task_id: str
    task_node_id: str
    task_cpu: float
    task_size: float
    task_deadline: float
    task_priority: float
    required_returned_size: float
    to_return_node_id: Optional[str]
    is_root: bool
    is_child: bool
    is_sink: bool
    predecessors: Sequence[str]
    successors: Sequence[str]
    input_semantic: str
    output_semantic: str
    input_payload_mb: float
    output_payload_mb: float

    def to_metadata(self, parent_task_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        role = "root_sink" if self.is_root and self.is_sink else "root" if self.is_root else "sink" if self.is_sink else "child"
        return {
            "sfc_id": self.sfc_id,
            "sfc_node_id": self.sfc_node_id,
            "service_type": self.service_type,
            "task_id": self.task_id,
            "field_mapping": dict(LASDM_TO_TASK_FIELD_MAPPING),
            "task_fields": {
                "task_id": self.task_id,
                "task_node_id": self.task_node_id,
                "task_cpu": self.task_cpu,
                "task_size": self.task_size,
                "task_deadline": self.task_deadline,
                "task_priority": self.task_priority,
                "required_returned_size": self.required_returned_size,
                "to_return_node_id": self.to_return_node_id,
            },
            "lifecycle": {
                "role": role,
                "state": "ready",
                "is_root": self.is_root,
                "is_child": self.is_child,
                "is_sink": self.is_sink,
            },
            "data_flow": {
                "predecessors": list(self.predecessors),
                "successors": list(self.successors),
                "parent_task_ids": list(parent_task_ids or ()),
                "input_semantic": self.input_semantic,
                "output_semantic": self.output_semantic,
                "input_payload_mb": self.input_payload_mb,
                "output_payload_mb": self.output_payload_mb,
                "required_returned_size": self.required_returned_size,
                "to_return_node_id": self.to_return_node_id,
            },
        }


class LASDMTaskAdapter:
    """Create AirFogSim Task objects for LASDM SFC service nodes."""

    def build_mapping(
        self,
        chain: LASDMServiceChain,
        sfc_node_id: str,
        current_time: float,
        input_payload_mb: Optional[float] = None,
        origin_node_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> LASDMTaskMapping:
        if sfc_node_id not in chain.nodes:
            raise KeyError(f"SFC {chain.sfc_id} has no service node {sfc_node_id}")

        sfc_node = chain.nodes[sfc_node_id]
        predecessors = chain.predecessors(sfc_node_id)
        successors = chain.successors(sfc_node_id)
        is_root = not predecessors
        is_sink = not successors
        task_size = float(chain.payload_mb if input_payload_mb is None else input_payload_mb)
        output_payload_mb = self._output_payload_mb(sfc_node, task_size)
        qos = sfc_node.effective_qos(chain.qos)
        task_deadline = self._task_deadline(chain, sfc_node_id, current_time, qos)
        task_node_id = origin_node_id or chain.source_node_id
        required_returned_size = self._required_returned_size(sfc_node, output_payload_mb) if is_sink else 0.0

        return LASDMTaskMapping(
            sfc_id=chain.sfc_id,
            sfc_node_id=sfc_node_id,
            service_type=sfc_node.service_type,
            task_id=task_id or self.task_id(chain.sfc_id, sfc_node_id),
            task_node_id=task_node_id,
            task_cpu=self._task_cpu(sfc_node, task_size),
            task_size=task_size,
            task_deadline=task_deadline,
            task_priority=float(qos.priority),
            required_returned_size=required_returned_size,
            to_return_node_id=chain.sink_node_id if is_sink else None,
            is_root=is_root,
            is_child=not is_root,
            is_sink=is_sink,
            predecessors=tuple(predecessors),
            successors=tuple(successors),
            input_semantic=chain.payload_semantic if is_root else sfc_node.input_semantic,
            output_semantic=sfc_node.output_semantic,
            input_payload_mb=task_size,
            output_payload_mb=output_payload_mb,
        )

    def build_task(
        self,
        chain: LASDMServiceChain,
        sfc_node_id: str,
        current_time: float,
        input_payload_mb: Optional[float] = None,
        origin_node_id: Optional[str] = None,
        task_id: Optional[str] = None,
        parent_task_ids: Optional[Sequence[str]] = None,
    ) -> Task:
        mapping = self.build_mapping(
            chain=chain,
            sfc_node_id=sfc_node_id,
            current_time=current_time,
            input_payload_mb=input_payload_mb,
            origin_node_id=origin_node_id,
            task_id=task_id,
        )
        task = Task(
            task_id=mapping.task_id,
            task_node_id=mapping.task_node_id,
            task_cpu=mapping.task_cpu,
            task_size=mapping.task_size,
            task_deadline=mapping.task_deadline,
            task_priority=mapping.task_priority,
            task_arrival_time=current_time,
            required_returned_size=mapping.required_returned_size,
            to_return_node_id=mapping.to_return_node_id,
            return_lazy_set=False,
        )
        self.attach_metadata(task, mapping, parent_task_ids)
        return task

    def attach_metadata(
        self,
        task: Task,
        mapping: LASDMTaskMapping,
        parent_task_ids: Optional[Sequence[str]] = None,
    ) -> Task:
        metadata = mapping.to_metadata(parent_task_ids)
        task.setAttribute("_lasdm", metadata)
        task.setAttribute("_lasdm_sfc_id", mapping.sfc_id)
        task.setAttribute("_lasdm_sfc_node_id", mapping.sfc_node_id)
        task.setAttribute("_lasdm_service_type", mapping.service_type)
        task.setAttribute("_lasdm_parent_task_ids", list(parent_task_ids or ()))
        task.setAttribute("_lasdm_parent_sfc_node_ids", list(mapping.predecessors))
        task.setAttribute("_lasdm_child_sfc_node_ids", list(mapping.successors))
        task.setAttribute("_lasdm_task_fields", metadata["task_fields"])
        task.setAttribute("_lasdm_input_semantic", mapping.input_semantic)
        task.setAttribute("_lasdm_input_payload_mb", mapping.input_payload_mb)
        task.setAttribute("_lasdm_output_payload_mb", mapping.output_payload_mb)
        task.setAttribute("_lasdm_output_semantic", mapping.output_semantic)
        task.setAttribute("_lasdm_required_returned_size", mapping.required_returned_size)
        task.setAttribute("_lasdm_to_return_node_id", mapping.to_return_node_id)
        task.setAttribute("_lasdm_is_root", mapping.is_root)
        task.setAttribute("_lasdm_is_child", mapping.is_child)
        task.setAttribute("_lasdm_is_sink", mapping.is_sink)
        task.setAttribute("_is_lasdm_task", True)
        return task

    def task_id(self, sfc_id: str, sfc_node_id: str, sequence: Optional[int] = None) -> str:
        base = f"{sfc_id}::{sfc_node_id}"
        return base if sequence is None else f"{base}::{sequence}"

    def _task_cpu(self, sfc_node: LASDMSFCNode, input_payload_mb: float) -> float:
        metadata = sfc_node.metadata
        if "task_cpu" in metadata:
            return float(metadata["task_cpu"])
        if "cpu" in metadata:
            return float(metadata["cpu"])
        if "cpu_mb" in metadata:
            return float(metadata["cpu_mb"])
        if "cpu_per_mb" in metadata:
            return input_payload_mb * float(metadata["cpu_per_mb"])
        return float(sfc_node.cpu_mb)

    def _output_payload_mb(self, sfc_node: LASDMSFCNode, input_payload_mb: float) -> float:
        if "output_payload_mb" in sfc_node.metadata:
            return float(sfc_node.metadata["output_payload_mb"])
        if "output_ratio" in sfc_node.metadata:
            return input_payload_mb * float(sfc_node.metadata["output_ratio"])
        return input_payload_mb

    def _required_returned_size(self, sfc_node: LASDMSFCNode, output_payload_mb: float) -> float:
        if "required_returned_size" in sfc_node.metadata:
            return float(sfc_node.metadata["required_returned_size"])
        if "return_size_mb" in sfc_node.metadata:
            return float(sfc_node.metadata["return_size_mb"])
        return output_payload_mb

    def _task_deadline(
        self,
        chain: LASDMServiceChain,
        sfc_node_id: str,
        current_time: float,
        qos: Any,
    ) -> float:
        graph_deadline = max(0.0, float(qos.deadline_s))
        submit_time = float(chain.submit_time) if chain.submit_time is not None else float(current_time)
        elapsed_s = max(0.0, float(current_time) - submit_time)
        return max(1e-6, graph_deadline - elapsed_s)


def build_lasdm_task(
    chain: LASDMServiceChain,
    sfc_node_id: str,
    current_time: float,
    input_payload_mb: Optional[float] = None,
    origin_node_id: Optional[str] = None,
    task_id: Optional[str] = None,
    parent_task_ids: Optional[Sequence[str]] = None,
) -> Task:
    return LASDMTaskAdapter().build_task(
        chain=chain,
        sfc_node_id=sfc_node_id,
        current_time=current_time,
        input_payload_mb=input_payload_mb,
        origin_node_id=origin_node_id,
        task_id=task_id,
        parent_task_ids=parent_task_ids,
    )


def propagate_parent_failure(
    tasks: Union[Iterable[Task], Mapping[str, Task]],
    failed_task: Union[Task, str],
    failure_code: int = EnumerateConstants.TASK_FAIL_PARENT_FAILED,
) -> List[Task]:
    task_list = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
    failed_task_id = failed_task if isinstance(failed_task, str) else failed_task.getTaskId()
    failed_task_ids = {failed_task_id}
    failed_sfc_node_ids = set()
    for task in task_list:
        metadata = _lasdm_metadata(task)
        if task.getTaskId() == failed_task_id:
            failed_sfc_node_ids.add(metadata.get("sfc_node_id"))
        if metadata.get("sfc_node_id") == failed_task_id:
            failed_sfc_node_ids.add(failed_task_id)
    failed_sfc_node_ids.discard(None)
    if isinstance(failed_task, str) and not failed_sfc_node_ids:
        failed_sfc_node_ids.add(failed_task)

    affected: List[Task] = []
    changed = True
    while changed:
        changed = False
        for task in task_list:
            if task.getTaskId() in failed_task_ids:
                continue
            metadata = _lasdm_metadata(task)
            data_flow = metadata.get("data_flow", {})
            parent_task_ids = set(data_flow.get("parent_task_ids", ()))
            parent_sfc_node_ids = set(data_flow.get("predecessors", ()))
            if not (parent_task_ids & failed_task_ids or parent_sfc_node_ids & failed_sfc_node_ids):
                continue

            task.setTaskFailueCode(failure_code)
            lifecycle = dict(metadata.get("lifecycle", {}))
            lifecycle["state"] = "blocked_parent_failed"
            metadata["lifecycle"] = lifecycle
            metadata["failure"] = {
                "reason": "parent_failed",
                "failed_parent_task_ids": sorted(parent_task_ids & failed_task_ids),
                "failed_parent_sfc_node_ids": sorted(parent_sfc_node_ids & failed_sfc_node_ids),
            }
            task.setAttribute("_lasdm", metadata)
            task.setAttribute("_lasdm_parent_failed", True)
            task.setAttribute("_lasdm_failure_reason", "parent_failed")
            task.setAttribute("_lasdm_failed_parent_task_ids", sorted(parent_task_ids & failed_task_ids))
            task.setAttribute("_lasdm_failed_parent_sfc_node_ids", sorted(parent_sfc_node_ids & failed_sfc_node_ids))
            affected.append(task)
            failed_task_ids.add(task.getTaskId())
            node_id = metadata.get("sfc_node_id")
            if node_id is not None:
                failed_sfc_node_ids.add(node_id)
            changed = True

    return affected


def _lasdm_metadata(task: Task) -> Dict[str, Any]:
    return dict(getattr(task, "_lasdm", {}) or {})
