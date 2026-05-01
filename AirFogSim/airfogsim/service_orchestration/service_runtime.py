from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from airfogsim.entities.task import Task

from .metrics import OrchestrationMetrics
from .service_spec import MicroserviceSpec, ServiceGraphSpec


@dataclass
class OutputPacket:
    node_id: str
    payload_mb: float
    semantic_type: str
    finish_time: float


@dataclass
class RunningServiceGraph:
    graph: ServiceGraphSpec
    spawned_ms: Set[str] = field(default_factory=set)
    completed_ms: Set[str] = field(default_factory=set)
    outputs: Dict[str, OutputPacket] = field(default_factory=dict)
    task_to_ms: Dict[str, str] = field(default_factory=dict)
    processed_task_ids: Set[str] = field(default_factory=set)
    completion_recorded: bool = False


class AirFogServiceRuntime:
    """Incrementally map service graphs onto AirFogSim tasks."""

    def __init__(self, metrics: Optional[OrchestrationMetrics] = None):
        self.running_graphs: Dict[str, RunningServiceGraph] = {}
        self.metrics = metrics

    def submit(self, env, graph: ServiceGraphSpec) -> None:
        state = RunningServiceGraph(graph=graph)
        self.running_graphs[graph.graph_id] = state
        if self.metrics is not None:
            self.metrics.record_graph_submit(graph.graph_id, env.simulation_time)
            self.metrics.record_composition(graph.intent, graph)
        self._spawn_ready_microservices(env, state)

    def _make_task(
        self,
        env,
        state: RunningServiceGraph,
        ms: MicroserviceSpec,
        origin_node_id: str,
        input_payload_mb: float,
        terminal: bool,
    ) -> Task:
        env.task_manager._task_id += 1
        task_id = f"{state.graph.graph_id}::{ms.ms_id}::{env.task_manager._task_id}"
        output_mb = input_payload_mb * ms.output_ratio
        cpu = input_payload_mb * ms.cpu_per_mb
        sink_node_id = state.graph.intent.sink_node_id or state.graph.intent.source_node_id
        required_returned_size = output_mb if terminal else 0.0
        to_return_node_id = sink_node_id if terminal else None

        task = Task(
            task_id=task_id,
            task_node_id=origin_node_id,
            task_cpu=cpu,
            task_size=input_payload_mb,
            task_deadline=state.graph.intent.qos.deadline_s,
            task_priority=state.graph.intent.qos.priority,
            task_arrival_time=env.simulation_time,
            required_returned_size=required_returned_size,
            to_return_node_id=to_return_node_id,
            return_lazy_set=False,
        )
        task.setAttribute("_service_graph_id", state.graph.graph_id)
        task.setAttribute("_microservice_id", ms.ms_id)
        task.setAttribute("_input_payload_mb", input_payload_mb)
        task.setAttribute("_output_payload_mb", output_mb)
        task.setAttribute("_output_semantic", ms.output_semantic)
        task.setAttribute("_is_service_task", True)
        env.task_scheduler.registerGeneratedTask(env, task)
        state.spawned_ms.add(ms.ms_id)
        state.task_to_ms[task_id] = ms.ms_id
        if self.metrics is not None:
            self.metrics.record_payload_tx(input_payload_mb)
        return task

    def _get_ready_input(self, state: RunningServiceGraph, ms_id: str) -> Optional[OutputPacket]:
        preds = state.graph.predecessors(ms_id)
        if not preds:
            intent = state.graph.intent
            return OutputPacket(
                node_id=intent.source_node_id,
                payload_mb=intent.payload.input_mb,
                semantic_type=intent.payload.semantic_type,
                finish_time=0.0,
            )
        if not all(parent in state.outputs for parent in preds):
            return None

        parent_outputs = [state.outputs[parent] for parent in preds]
        carrier = max(parent_outputs, key=lambda packet: packet.payload_mb)
        return OutputPacket(
            node_id=carrier.node_id,
            payload_mb=sum(packet.payload_mb for packet in parent_outputs),
            semantic_type=carrier.semantic_type,
            finish_time=max(packet.finish_time for packet in parent_outputs),
        )

    def _spawn_ready_microservices(self, env, state: RunningServiceGraph) -> None:
        for ms_id, ms in state.graph.nodes.items():
            if ms_id in state.spawned_ms:
                continue
            ready_input = self._get_ready_input(state, ms_id)
            if ready_input is None:
                continue
            self._make_task(
                env=env,
                state=state,
                ms=ms,
                origin_node_id=ready_input.node_id,
                input_payload_mb=ready_input.payload_mb,
                terminal=state.graph.is_terminal_ms(ms_id),
            )

    def on_airfogsim_step_finished(self, env) -> None:
        done_tasks = env.task_manager.getDoneTasks()
        for task in done_tasks:
            task_id = task.getTaskId()
            graph_id = getattr(task, "_service_graph_id", None)
            ms_id = getattr(task, "_microservice_id", None)
            if graph_id is None or ms_id is None:
                continue
            if graph_id not in self.running_graphs:
                continue

            state = self.running_graphs[graph_id]
            if task_id in state.processed_task_ids:
                continue
            assigned_node = task.getAssignedTo() or task.getCurrentNodeId()
            state.outputs[ms_id] = OutputPacket(
                node_id=assigned_node,
                payload_mb=getattr(task, "_output_payload_mb", 0.0),
                semantic_type=getattr(task, "_output_semantic", "any"),
                finish_time=env.simulation_time,
            )
            state.completed_ms.add(ms_id)
            state.processed_task_ids.add(task_id)
            self._spawn_ready_microservices(env, state)
            if self.is_graph_finished(graph_id) and self.metrics is not None and not state.completion_recorded:
                self.metrics.record_graph_completion(graph_id, env.simulation_time)
                state.completion_recorded = True

    def is_graph_finished(self, graph_id: str) -> bool:
        state = self.running_graphs[graph_id]
        terminal_nodes = [ms_id for ms_id in state.graph.nodes if state.graph.is_terminal_ms(ms_id)]
        return all(ms_id in state.completed_ms for ms_id in terminal_nodes)
