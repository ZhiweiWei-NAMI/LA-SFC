from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from airfogsim.entities.task import Task

from .failure_mapping import build_failure_event_payload, map_airfogsim_task_failure
from .env_adapter import LASDMEnvAdapter
from .manager import LASDMManager
from .model import GraphStatus, LASDMServiceChain, SFCFailureReason
from .orchestrator import LASDMDecision
from .task_adapter import LASDMTaskAdapter


class LASDMRuntimeBridge:
    """Minimal deterministic bridge between LASDM SFCs and AirFogSim tasks."""

    def __init__(self, manager: Optional[LASDMManager] = None, env_adapter: Optional[LASDMEnvAdapter] = None):
        self.manager = manager or LASDMManager()
        self.env_adapter = env_adapter or LASDMEnvAdapter(directory=self.manager.directory)
        self.task_adapter = LASDMTaskAdapter()
        self.task_to_sfc: Dict[str, str] = {}
        self.task_to_sfc_node: Dict[str, str] = {}
        self.tasks: Dict[str, Task] = {}
        self.sfc_node_to_task: Dict[Tuple[str, str], str] = {}
        self.completed_nodes: Dict[str, Set[str]] = {}
        self.processed_done_tasks: Set[str] = set()
        self.processed_failed_tasks: Set[str] = set()
        self.node_outputs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.function_execution_trace: List[Dict[str, Any]] = []
        self.failure_trace: List[Dict[str, Any]] = []
        self.runtime_task_lifecycle_trace: List[Dict[str, Any]] = []
        self._task_counter = 0

    def submit_service_chain(self, chain: LASDMServiceChain, current_time: float = 0.0) -> LASDMServiceChain:
        """Submit an LASDM service chain to the manager and initialize bridge state."""
        submitted = self.manager.submit(chain, current_time=current_time)
        self.completed_nodes.setdefault(chain.sfc_id, set())
        return submitted

    def create_ready_function_tasks(
        self,
        env: Any,
        sfc_id: Optional[str] = None,
        current_time: Optional[float] = None,
        register: bool = True,
    ) -> List[Task]:
        """Create AirFogSim tasks for ready, unspawned LASDM function nodes."""
        now = self._resolve_time(env, current_time)
        self._ensure_decisions(now)
        created: List[Task] = []

        for chain in self._iter_running_chains(sfc_id):
            decision = self.manager.decisions.get(chain.sfc_id)
            if decision is None or not decision.accepted:
                continue
            for node_id in chain.topological_order():
                if (chain.sfc_id, node_id) in self.sfc_node_to_task:
                    continue
                if node_id not in decision.assignments:
                    continue
                if not self._predecessors_completed(chain, node_id):
                    continue
                task = self._make_task(env, chain, node_id, now)
                self._remember_task(task, chain.sfc_id, node_id)
                if register:
                    self._register_task(env, task)
                created.append(task)
        return created

    def apply_orchestration_decision(
        self,
        decision: LASDMDecision,
        env: Any = None,
        tasks: Optional[Iterable[Task]] = None,
        current_time: float = 0.0,
    ) -> Dict[str, str]:
        """Apply an accepted LASDM placement decision to generated AirFogSim tasks."""
        if not decision.accepted:
            return {}
        applied: Dict[str, str] = {}
        task_iter = list(tasks) if tasks is not None else self._tasks_for_sfc(decision.sfc_id)
        for task in task_iter:
            if self._task_sfc_id(task) != decision.sfc_id:
                continue
            if task.getAssignedTo() is not None:
                continue
            node_id = self._task_node_id(task)
            if node_id is None or node_id not in decision.node_mapping:
                continue
            target_node_id = decision.node_mapping[node_id]
            route = self.env_adapter.build_route(
                env,
                task.getCurrentNodeId(),
                target_node_id,
                decision.routes.get(node_id, []),
            )
            if env is not None:
                self._apply_selected_candidate_runtime_cost(task, decision, node_id, target_node_id, env)
            if env is not None:
                scheduled = self.env_adapter.schedule_task_offloading(
                    env,
                    task,
                    target_node_id,
                    route,
                    current_time=current_time,
                )
                if not scheduled:
                    self._record_offloading_rejection(task, current_time)
                    continue
            else:
                task.offloadTo(target_node_id, route, current_time)
            applied[task.getTaskId()] = target_node_id
        return applied

    def prepare_airfogsim_step(
        self,
        env: Any,
        chains: Optional[Iterable[LASDMServiceChain]] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Prepare LASDM-generated tasks and AirFogSim scheduler decisions before env.step()."""

        now = self._resolve_time(env, current_time)
        self.record_task_lifecycle(env, now, phase="prepare_start")
        sync_report = self.sync_from_airfogsim_tasks(env, current_time=now)
        env_snapshot = self.env_adapter.sync_from_env(env, current_time=now)
        submitted: List[str] = []
        for chain in chains or []:
            if chain.sfc_id not in self.manager.chains:
                self.submit_service_chain(chain, current_time=now)
                submitted.append(chain.sfc_id)
        created = self.create_ready_function_tasks(env, current_time=now, register=True)
        self.record_task_lifecycle(env, now, phase="tasks_created")
        ready_tasks = self._unscheduled_ready_tasks(created)
        applied: Dict[str, str] = {}
        for decision in list(self.manager.decisions.values()):
            applied.update(self.apply_orchestration_decision(decision, env=env, tasks=ready_tasks, current_time=now))
        returning = self.env_adapter.schedule_returning(env)
        communication = self.env_adapter.schedule_communication(env)
        self.env_adapter.schedule_computation(env)
        self.record_task_lifecycle(env, now, phase="prepare_end")
        return {
            "time_s": now,
            "submitted_sfc_ids": submitted,
            "created_task_ids": [task.getTaskId() for task in created],
            "ready_task_ids": [task.getTaskId() for task in ready_tasks],
            "applied_offloads": applied,
            "return_routes": returning,
            "wireless_rb": communication,
            "sync": sync_report,
            "env_snapshot": env_snapshot,
            "metrics": self.collect_step_metrics(current_time=now),
        }

    def sync_from_airfogsim_tasks(
        self,
        env: Any = None,
        done_tasks: Optional[Iterable[Task]] = None,
        failed_tasks: Optional[Iterable[Task]] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, int]:
        """Synchronize LASDM function-node state from AirFogSim task lifecycle lists."""
        now = self._resolve_time(env, current_time)
        self.record_task_lifecycle(env, now, phase="sync_start")
        done = self._collect_done_tasks(env, done_tasks)
        failed = self._collect_failed_tasks(env, failed_tasks)
        updated = {"completed_tasks": 0, "failed_tasks": 0}

        for task in done:
            task_id = task.getTaskId()
            sfc_id = self._task_sfc_id(task)
            node_id = self._task_node_id(task)
            if sfc_id is None or node_id is None or task_id in self.processed_done_tasks or task_id in self.processed_failed_tasks:
                continue
            semantic_failure = self._semantic_quality_failure(task)
            if semantic_failure is not None and sfc_id in self.manager.chains:
                reason = SFCFailureReason.ACCURACY_VIOLATION
                status = GraphStatus.FAILED
                failure_payload = build_failure_event_payload(
                    task=task,
                    reason=reason,
                    sfc_node_id=node_id,
                    service_type=getattr(task, "_lasdm_service_type", None),
                    status=status,
                )
                failure_payload.update(semantic_failure)
                self.manager.fail(
                    sfc_id,
                    reason,
                    now,
                    status=status,
                    details=failure_payload,
                )
                self.failure_trace.append(self._failure_trace_record(task, now, reason, failure_payload))
                self.function_execution_trace.append(self._task_trace_record(task, now, status=status.value))
                self.processed_failed_tasks.add(task_id)
                updated["failed_tasks"] += 1
                continue
            self.completed_nodes.setdefault(sfc_id, set()).add(node_id)
            self.node_outputs[(sfc_id, node_id)] = self._task_output(task, now)
            self.function_execution_trace.append(self._task_trace_record(task, now, status="succeeded"))
            self.processed_done_tasks.add(task_id)
            updated["completed_tasks"] += 1

        for task in failed:
            task_id = task.getTaskId()
            sfc_id = self._task_sfc_id(task)
            if sfc_id is None or task_id in self.processed_failed_tasks or task_id in self.processed_done_tasks:
                continue
            if sfc_id in self.manager.chains:
                reason = map_airfogsim_task_failure(task=task) or SFCFailureReason.TASK_FAILED
                status = GraphStatus.TIMED_OUT if reason == SFCFailureReason.DEADLINE_MISSED else GraphStatus.FAILED
                failure_payload = build_failure_event_payload(
                    task=task,
                    reason=reason,
                    sfc_node_id=self._task_node_id(task),
                    service_type=getattr(task, "_lasdm_service_type", None),
                    status=status,
                )
                self.manager.fail(
                    sfc_id,
                    reason,
                    now,
                    status=status,
                    details=failure_payload,
                )
                self.failure_trace.append(self._failure_trace_record(task, now, reason, failure_payload))
                self.function_execution_trace.append(self._task_trace_record(task, now, status=status.value))
            self.processed_failed_tasks.add(task_id)
            updated["failed_tasks"] += 1

        self.update_sfc_status(current_time=now)
        self.record_task_lifecycle(env, now, phase="sync_end")
        return updated

    def _semantic_quality_failure(self, task: Task) -> Optional[Dict[str, Any]]:
        min_score = float(getattr(task, "_lasdm_semantic_min_score", 0.0) or 0.0)
        score = float(getattr(task, "_lasdm_semantic_cumulative_quality", 1.0) or 1.0)
        if not bool(getattr(task, "_lasdm_is_sink", False)):
            return None
        if min_score <= 0.0 or score >= min_score:
            return None
        return {
            "airfogsim_failure_reason": "semantic_accuracy_violation",
            "semantic_score": float(getattr(task, "_lasdm_semantic_score", score) or score),
            "semantic_cumulative_quality": score,
            "semantic_min_score": min_score,
            "semantic_shortfall": max(0.0, min_score - score),
        }

    def update_sfc_status(self, current_time: float, sfc_id: Optional[str] = None) -> Dict[str, str]:
        """Complete SFCs whose terminal function nodes have all finished."""
        self.manager.step(current_time=current_time)
        statuses: Dict[str, str] = {}
        for chain in self._iter_chains(sfc_id):
            if chain.status == GraphStatus.RUNNING and self._terminal_nodes_completed(chain):
                self.manager.complete(chain.sfc_id, current_time=current_time)
            statuses[chain.sfc_id] = chain.status.value
        return statuses

    def collect_step_metrics(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Return manager summary plus bridge-local runtime counters."""
        summary = self.manager.summary()
        chains = {}
        for sfc_id, chain in sorted(self.manager.chains.items()):
            order = list(chain.topological_order())
            completed = set(self.completed_nodes.get(sfc_id, set()) or set())
            chains[sfc_id] = {
                "status": chain.status.value,
                "completed_nodes": sorted(completed),
                "spawned_tasks": len([key for key in self.sfc_node_to_task if key[0] == sfc_id]),
                "stage_count": len(order),
                "chain_progress_ratio": len(completed.intersection(order)) / max(1, len(order)),
            }
        progress_values = [float(item["chain_progress_ratio"]) for item in chains.values()]
        summary.update(
            {
                "current_time": current_time,
                "chain_progress_ratio_mean": sum(progress_values) / len(progress_values) if progress_values else 0.0,
                "runtime_bridge": {
                    "spawned_tasks": len(self.task_to_sfc),
                    "completed_function_tasks": len(self.processed_done_tasks),
                    "failed_function_tasks": len(self.processed_failed_tasks),
                    "lifecycle_trace_rows": len(self.runtime_task_lifecycle_trace),
                    "chains": chains,
                },
                "function_execution_trace": list(self.function_execution_trace),
                "failure_trace": list(self.failure_trace),
                "runtime_task_lifecycle_trace": list(self.runtime_task_lifecycle_trace),
            }
        )
        return summary

    def record_task_lifecycle(self, env: Any, current_time: Optional[float] = None, phase: str = "step") -> List[Dict[str, Any]]:
        """Record LASDM task state across AirFogSim queues for runtime debugging."""

        if env is None or getattr(env, "task_manager", None) is None:
            return []
        now = self._resolve_time(env, current_time)
        rows: List[Dict[str, Any]] = []
        manager = env.task_manager
        queue_specs = [
            ("waiting_to_offload", getattr(manager, "getWaitingToOffloadTasks", lambda: {})()),
            ("offloading", getattr(manager, "getOffloadingTasks", lambda: {})()),
            ("computing", getattr(manager, "getComputingTasks", lambda: {})()),
            ("waiting_to_return", getattr(manager, "_waiting_to_return_tasks", {}) or {}),
            ("returning", getattr(manager, "_returning_tasks", {}) or {}),
            ("done", _tasks_by_node(getattr(manager, "getDoneTasks", lambda: [])())),
            ("failed", _tasks_by_node(getattr(manager, "getOutOfDDLTasks", lambda: [])())),
        ]
        for queue_name, task_map in queue_specs:
            for owner_node_id, tasks in dict(task_map or {}).items():
                for task in list(tasks or []):
                    if task.getTaskId() not in self.task_to_sfc and not getattr(task, "_is_lasdm_task", False):
                        continue
                    rows.append(self._task_lifecycle_record(env, task, queue_name, str(owner_node_id), now, phase))
        self.runtime_task_lifecycle_trace.extend(rows)
        return rows

    def _record_offloading_rejection(self, task: Task, current_time: float) -> None:
        diagnostic = dict(getattr(self.env_adapter, "last_offloading_diagnostics", {}).get(task.getTaskId(), {}) or {})
        self.runtime_task_lifecycle_trace.append(
            {
                "time_s": current_time,
                "phase": "offloading_rejected",
                "queue_state": "waiting_to_offload",
                "task_id": task.getTaskId(),
                "sfc_id": self._task_sfc_id(task) or "",
                "function_id": self._task_node_id(task) or "",
                "task_node_id": task.getTaskNodeId(),
                "current_node_id": task.getCurrentNodeId(),
                "assigned_to": task.getAssignedTo() or "",
                "route": "->".join(list(getattr(task, "_to_offload_route", []) or [])),
                "offloading_scheduled": False,
                "offloading_reason": diagnostic.get("reason", "schedule_task_offloading_failed"),
                "missing_nodes": ",".join(diagnostic.get("missing_nodes", []) or []),
            }
        )

    def _task_lifecycle_record(
        self,
        env: Any,
        task: Task,
        queue_name: str,
        owner_node_id: str,
        current_time: float,
        phase: str,
    ) -> Dict[str, Any]:
        task_id = task.getTaskId()
        route = list(getattr(task, "_to_offload_route", []) or [])
        current_node_id = str(task.getCurrentNodeId())
        next_hop = str(route[0]) if route else ""
        assigned_to = task.getAssignedTo() or ""
        rb = list(getattr(self.env_adapter, "last_wireless_allocations", {}).get(task_id, []) or [])
        offload_diag = dict(getattr(self.env_adapter, "last_offloading_diagnostics", {}).get(task_id, {}) or {})
        wireless_diag = dict(getattr(self.env_adapter, "last_wireless_diagnostics", {}).get(task_id, {}) or {})
        if queue_name == "offloading" and not rb and wireless_diag.get("needs_wireless_rb"):
            substate = "offloading_no_rb"
        elif queue_name == "offloading" and (not route or not next_hop):
            substate = "offloading_no_route"
        elif queue_name == "computing" and float(task.getComputedSize()) <= 0:
            substate = "computing_cpu_no_progress"
        else:
            substate = queue_name
        return {
            "time_s": current_time,
            "phase": phase,
            "queue_state": queue_name,
            "substate": substate,
            "owner_node_id": owner_node_id,
            "task_id": task_id,
            "sfc_id": self._task_sfc_id(task) or "",
            "function_id": self._task_node_id(task) or "",
            "service_type": getattr(task, "_lasdm_service_type", ""),
            "task_node_id": task.getTaskNodeId(),
            "current_node_id": current_node_id,
            "current_node_exists": self.env_adapter.runtime_node_exists(env, current_node_id),
            "assigned_to": assigned_to,
            "assigned_node_exists": self.env_adapter.runtime_node_exists(env, assigned_to) if assigned_to else "",
            "next_hop": next_hop,
            "next_hop_exists": self.env_adapter.runtime_node_exists(env, next_hop) if next_hop else "",
            "route": "->".join(route),
            "decided_route": "->".join(list(getattr(task, "_decided_route", []) or [])),
            "allocated_rb": ",".join(str(item) for item in rb),
            "needs_wireless_rb": wireless_diag.get("needs_wireless_rb", ""),
            "offloading_scheduled": offload_diag.get("scheduled", ""),
            "offloading_reason": offload_diag.get("reason", ""),
            "task_size": float(task.getTaskSize()),
            "transmitted_size": float(task.getTransmittedSize()),
            "transmitted_ratio": _safe_ratio(float(task.getTransmittedSize()), float(task.getTaskSize())),
            "task_cpu": float(task.getTaskCPU()),
            "candidate_runtime_cost_s": float(getattr(task, "_lasdm_candidate_runtime_cost_s", 0.0) or 0.0),
            "semantic_mismatch_runtime_cost_s": float(
                getattr(task, "_lasdm_semantic_mismatch_runtime_cost_s", 0.0) or 0.0
            ),
            "computed_size": float(task.getComputedSize()),
            "computed_ratio": _safe_ratio(float(task.getComputedSize()), float(task.getTaskCPU())),
            "deadline_s": float(task.getTaskDeadline()),
            "age_s": max(0.0, current_time - float(task.getTaskArrivalTime())),
            "failure_reason": task.getTaskFailureReason() if queue_name == "failed" else "",
        }

    def _task_trace_record(self, task: Task, current_time: float, status: str) -> Dict[str, Any]:
        sfc_id = self._task_sfc_id(task) or ""
        chain = self.manager.chains.get(sfc_id)
        submit_time = chain.submit_time if chain is not None and chain.submit_time is not None else task.getTaskArrivalTime()
        route_nodes, route_times = self._safe_routed_nodes_with_time(task)
        start_tx = max(0.0, float(getattr(task, "_start_to_transmit_time", -1.0)))
        last_tx = float(getattr(task, "_last_transmission_time", start_tx))
        start_compute = float(getattr(task, "_start_to_compute_time", -1.0))
        last_compute = float(getattr(task, "_last_compute_time", start_compute))
        return {
            "baseline": self._chain_context_value(chain, "baseline"),
            "seed": self._chain_seed(sfc_id),
            "scenario": self._chain_context_value(chain, "scenario"),
            "sfc_id": sfc_id,
            "function_id": self._task_node_id(task) or "",
            "task_id": task.getTaskId(),
            "selected_node": task.getAssignedTo() or "",
            "route": "->".join(route_nodes or list(getattr(task, "_decided_route", []) or [])),
            "tx_delay": max(0.0, last_tx - start_tx) if last_tx >= 0 and start_tx >= 0 else 0.0,
            "compute_delay": max(0.0, last_compute - start_compute) if last_compute >= 0 and start_compute >= 0 else 0.0,
            "queue_delay": max(0.0, start_tx - float(task.getTaskArrivalTime())) if start_tx >= 0 else 0.0,
            "candidate_runtime_cost_s": float(getattr(task, "_lasdm_candidate_runtime_cost_s", 0.0) or 0.0),
            "semantic_mismatch_runtime_cost_s": float(
                getattr(task, "_lasdm_semantic_mismatch_runtime_cost_s", 0.0) or 0.0
            ),
            "semantic_score": float(getattr(task, "_lasdm_semantic_score", 1.0) or 1.0),
            "semantic_cumulative_quality": float(getattr(task, "_lasdm_semantic_cumulative_quality", 1.0) or 1.0),
            "semantic_link_truth_relation": str(getattr(task, "_lasdm_semantic_link_truth_relation", "") or ""),
            "semantic_link_truth_score": float(getattr(task, "_lasdm_semantic_link_truth_score", 0.0) or 0.0),
            "semantic_min_score": float(getattr(task, "_lasdm_semantic_min_score", 0.0) or 0.0),
            "semantic_quality_violation": bool(getattr(task, "_lasdm_semantic_quality_violation", False)),
            "finish_time": current_time,
            "e2e_delay": max(0.0, current_time - float(submit_time)),
            "status": status,
        }

    def _failure_trace_record(
        self,
        task: Task,
        current_time: float,
        reason: SFCFailureReason,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "task_id": task.getTaskId(),
            "sfc_id": self._task_sfc_id(task),
            "function_id": self._task_node_id(task),
            "airfogsim_reason": payload.get("airfogsim_failure_reason") or payload.get("airfogsim_failure_code"),
            "lasdm_reason": reason.value,
            "timestamp": current_time,
            "mapping_source": "airfogsim_task_failure",
        }

    def _safe_routed_nodes_with_time(self, task: Task) -> Tuple[List[str], List[float]]:
        try:
            nodes, times = task.getRoutedNodeIdsWithTime()
            return list(nodes), [float(item) for item in times]
        except Exception:
            return [], []

    def _chain_context_value(self, chain: Optional[LASDMServiceChain], key: str) -> str:
        if chain is None:
            return ""
        return str(chain.context.get(key, ""))

    def _chain_seed(self, sfc_id: str) -> str:
        marker = "__seed_"
        if marker not in sfc_id:
            return ""
        return sfc_id.rsplit(marker, 1)[-1]

    def _make_task(self, env: Any, chain: LASDMServiceChain, node_id: str, current_time: float) -> Task:
        ready_input = self._ready_input(chain, node_id, current_time)
        parent_task_ids = [
            self.sfc_node_to_task[(chain.sfc_id, predecessor)]
            for predecessor in chain.predecessors(node_id)
            if (chain.sfc_id, predecessor) in self.sfc_node_to_task
        ]
        task = self.task_adapter.build_task(
            chain=chain,
            sfc_node_id=node_id,
            current_time=current_time,
            input_payload_mb=float(ready_input["payload_mb"]),
            origin_node_id=str(ready_input["node_id"]),
            task_id=self._next_task_id(env, chain.sfc_id, node_id),
            parent_task_ids=parent_task_ids,
        )
        task.setAttribute("_lasdm_node_id", node_id)
        task.setAttribute("_service_graph_id", chain.sfc_id)
        task.setAttribute("_microservice_id", node_id)
        return task

    def _apply_selected_candidate_runtime_cost(
        self,
        task: Task,
        decision: LASDMDecision,
        sfc_node_id: str,
        target_node_id: str,
        env: Any,
    ) -> None:
        if getattr(task, "_lasdm_candidate_runtime_cost_applied", False):
            return
        diagnostics = dict(getattr(decision, "diagnostics", {}) or {})
        candidate = dict(dict(diagnostics.get("selected_candidates", {}) or {}).get(sfc_node_id, {}) or {})
        metadata = dict(candidate.get("metadata", {}) or {})
        cold_start_s = _float(metadata.get("cold_start_s"), 0.0)
        stale_penalty_s = _float(metadata.get("stale_latency_penalty_s"), 0.0)
        if (
            _float(candidate.get("staleness_s"), 0.0) <= 0.0
            and str(metadata.get("semantic_group", "")) != "stale_clone_exact"
        ):
            stale_penalty_s = 0.0
        semantic_score = _float(metadata.get("link_similarity"), _float(candidate.get("semantic_score"), 1.0))
        semantic_quality_before = _float(metadata.get("semantic_quality_before"), 1.0)
        semantic_cumulative_quality = _float(
            metadata.get("semantic_cumulative_quality_if_selected"),
            max(0.0, min(1.0, semantic_quality_before * max(0.0, min(1.0, semantic_score)))),
        )
        chain = self.manager.chains.get(decision.sfc_id)
        context = dict(getattr(chain, "context", {}) or {}) if chain is not None else {}
        semantic_min_score = max(0.0, _float(context.get("semantic_min_score"), 0.0))
        task.setAttribute("_lasdm_semantic_score", semantic_score)
        task.setAttribute("_lasdm_link_similarity", semantic_score)
        task.setAttribute("_lasdm_semantic_quality_before", semantic_quality_before)
        task.setAttribute("_lasdm_semantic_cumulative_quality", semantic_cumulative_quality)
        task.setAttribute("_lasdm_semantic_min_score", semantic_min_score)
        task.setAttribute(
            "_lasdm_semantic_quality_violation",
            bool(getattr(task, "_lasdm_is_sink", False) and semantic_min_score > 0.0 and semantic_cumulative_quality < semantic_min_score),
        )
        task.setAttribute("_lasdm_semantic_link_truth_relation", str(metadata.get("semantic_link_truth_relation", "")))
        task.setAttribute("_lasdm_semantic_link_truth_score", _float(metadata.get("semantic_link_truth_score"), 0.0))
        if metadata.get("link_source_semantic"):
            task.setAttribute("_lasdm_input_semantic", str(metadata.get("link_source_semantic")))
        selected_output = str(candidate.get("output_semantic", metadata.get("candidate_output_semantic", "")) or "")
        if selected_output:
            task.setAttribute("_lasdm_output_semantic", selected_output)
        resource_allocation = dict(getattr(decision, "resource_allocations", {}).get(sfc_node_id, {}) or {})
        compute_level = _discrete_resource_level(resource_allocation.get("compute_level"), 1.0)
        bandwidth_level = _discrete_resource_level(resource_allocation.get("bandwidth_level"), 1.0)
        task.setAttribute("_lasdm_compute_level", compute_level)
        task.setAttribute("_lasdm_bandwidth_level", bandwidth_level)
        task.setAttribute("_lasdm_resource_allocation", {"compute_level": compute_level, "bandwidth_level": bandwidth_level})
        mismatch_scale_s = _float(context.get("semantic_mismatch_penalty_s_per_unit"), 0.0)
        semantic_mismatch_s = max(0.0, mismatch_scale_s) * max(0.0, 1.0 - semantic_score) ** 2
        extra_seconds = max(0.0, cold_start_s + stale_penalty_s + semantic_mismatch_s)
        if extra_seconds <= 0.0:
            task.setAttribute("_lasdm_candidate_runtime_cost_applied", True)
            return
        cpu = self.env_adapter.node_cpu(env, target_node_id)
        task._task_cpu = float(task.getTaskCPU()) + max(0.0, float(cpu) * extra_seconds)
        task.setAttribute("_lasdm_candidate_runtime_cost_applied", True)
        task.setAttribute("_lasdm_candidate_runtime_cost_s", extra_seconds)
        task.setAttribute("_lasdm_semantic_mismatch_runtime_cost_s", semantic_mismatch_s)

    def _ready_input(self, chain: LASDMServiceChain, node_id: str, current_time: float) -> Dict[str, Any]:
        predecessors = chain.predecessors(node_id)
        if not predecessors:
            return {
                "node_id": chain.source_node_id,
                "payload_mb": chain.payload_mb,
                "semantic": chain.payload_semantic,
                "finish_time": current_time,
            }
        outputs = [self.node_outputs[(chain.sfc_id, pred)] for pred in predecessors]
        carrier = max(outputs, key=lambda item: item["payload_mb"])
        return {
            "node_id": carrier["node_id"],
            "payload_mb": sum(float(item["payload_mb"]) for item in outputs),
            "semantic": carrier["semantic"],
            "finish_time": max(float(item["finish_time"]) for item in outputs),
        }

    def _output_payload_mb(self, chain: LASDMServiceChain, node_id: str, input_payload_mb: float) -> float:
        node = chain.nodes[node_id]
        if "output_payload_mb" in node.metadata:
            return float(node.metadata["output_payload_mb"])
        output_ratio = float(node.metadata.get("output_ratio", 1.0))
        return float(input_payload_mb) * output_ratio

    def _task_output(self, task: Task, current_time: float) -> Dict[str, Any]:
        return {
            "node_id": task.getAssignedTo() or task.getCurrentNodeId(),
            "payload_mb": float(getattr(task, "_lasdm_output_payload_mb", task.getTaskSize())),
            "semantic": getattr(task, "_lasdm_output_semantic", "any"),
            "semantic_cumulative_quality": float(getattr(task, "_lasdm_semantic_cumulative_quality", 1.0) or 1.0),
            "semantic_link_truth_relation": str(getattr(task, "_lasdm_semantic_link_truth_relation", "") or ""),
            "finish_time": current_time,
        }

    def _remember_task(self, task: Task, sfc_id: str, node_id: str) -> None:
        task_id = task.getTaskId()
        self.tasks[task_id] = task
        self.task_to_sfc[task_id] = sfc_id
        self.task_to_sfc_node[task_id] = node_id
        self.sfc_node_to_task[(sfc_id, node_id)] = task_id

    def _register_task(self, env: Any, task: Task) -> Task:
        if env is not None and getattr(env, "task_manager", None) is not None:
            return self.env_adapter.register_task(env, task)
        task.setGenerated()
        return task

    def _next_task_id(self, env: Any, sfc_id: str, node_id: str) -> str:
        task_manager = getattr(env, "task_manager", None)
        if task_manager is not None and hasattr(task_manager, "_task_id"):
            task_manager._task_id += 1
            serial = task_manager._task_id
        else:
            self._task_counter += 1
            serial = self._task_counter
        return f"{sfc_id}::{node_id}::{serial}"

    def _ensure_decisions(self, current_time: float) -> None:
        if any(chain.status == GraphStatus.RUNNING and chain.sfc_id not in self.manager.decisions for chain in self.manager.chains.values()):
            self.manager.step(current_time=current_time)

    def _predecessors_completed(self, chain: LASDMServiceChain, node_id: str) -> bool:
        completed = self.completed_nodes.setdefault(chain.sfc_id, set())
        return all(pred in completed for pred in chain.predecessors(node_id))

    def _terminal_nodes_completed(self, chain: LASDMServiceChain) -> bool:
        terminals = chain.terminal_service_nodes()
        completed = self.completed_nodes.setdefault(chain.sfc_id, set())
        return bool(terminals) and all(node_id in completed for node_id in terminals)

    def _offload_route(self, task: Task, target_node_id: str, decision_route: Sequence[str]) -> List[str]:
        route = list(decision_route) if decision_route else [target_node_id]
        if route and route[0] == task.getTaskNodeId() and len(route) > 1:
            route = route[1:]
        if not route or route[-1] != target_node_id:
            route.append(target_node_id)
        return route

    def _tasks_for_sfc(self, sfc_id: str) -> List[Task]:
        tasks: List[Task] = []
        for (task_sfc_id, _), task_id in self.sfc_node_to_task.items():
            if task_sfc_id != sfc_id:
                continue
            task = self._task_by_id(task_id)
            if task is not None:
                tasks.append(task)
        return tasks

    def _unscheduled_ready_tasks(self, created: Iterable[Task]) -> List[Task]:
        tasks: List[Task] = []
        seen: Set[str] = set()
        for task in list(created) + list(self.tasks.values()):
            task_id = task.getTaskId()
            if task_id in seen or task.getAssignedTo() is not None:
                continue
            seen.add(task_id)
            tasks.append(task)
        return tasks

    def _task_by_id(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def _collect_done_tasks(self, env: Any, done_tasks: Optional[Iterable[Task]]) -> List[Task]:
        if done_tasks is not None:
            return list(done_tasks)
        task_manager = getattr(env, "task_manager", None)
        if task_manager is not None and hasattr(task_manager, "getDoneTasks"):
            return list(task_manager.getDoneTasks())
        return []

    def _collect_failed_tasks(self, env: Any, failed_tasks: Optional[Iterable[Task]]) -> List[Task]:
        if failed_tasks is not None:
            return list(failed_tasks)
        task_manager = getattr(env, "task_manager", None)
        if task_manager is None:
            return []
        tasks: List[Task] = []
        seen: Set[str] = set()
        for method_name in ("getRecentlyFailedTasks", "getOutOfDDLTasks"):
            if not hasattr(task_manager, method_name):
                continue
            for task in getattr(task_manager, method_name)():
                if task.getTaskId() not in seen:
                    tasks.append(task)
                    seen.add(task.getTaskId())
        return tasks

    def _task_sfc_id(self, task: Task) -> Optional[str]:
        value = getattr(task, "_lasdm_sfc_id", None) or self.task_to_sfc.get(task.getTaskId())
        if value:
            return value
        task_id = str(task.getTaskId())
        if "::" in task_id:
            return task_id.split("::", 1)[0]
        return None

    def _task_node_id(self, task: Task) -> Optional[str]:
        value = getattr(
            task,
            "_lasdm_sfc_node_id",
            getattr(task, "_lasdm_node_id", self.task_to_sfc_node.get(task.getTaskId())),
        )
        if value:
            return value
        parts = str(task.getTaskId()).split("::")
        if len(parts) >= 2:
            return parts[1]
        return None

    def _iter_chains(self, sfc_id: Optional[str] = None) -> Iterable[LASDMServiceChain]:
        if sfc_id is not None:
            chain = self.manager.chains.get(sfc_id)
            return [] if chain is None else [chain]
        return list(self.manager.chains.values())

    def _iter_running_chains(self, sfc_id: Optional[str] = None) -> Iterable[LASDMServiceChain]:
        return [chain for chain in self._iter_chains(sfc_id) if chain.status == GraphStatus.RUNNING]

    def _resolve_time(self, env: Any = None, current_time: Optional[float] = None) -> float:
        if current_time is not None:
            return float(current_time)
        if env is not None and hasattr(env, "simulation_time"):
            return float(env.simulation_time)
        return 0.0


def _tasks_by_node(tasks: Iterable[Task]) -> Dict[str, List[Task]]:
    grouped: Dict[str, List[Task]] = {}
    for task in tasks:
        grouped.setdefault(str(task.getTaskNodeId()), []).append(task)
    return grouped


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _discrete_resource_level(value: Any, default: float = 1.0) -> float:
    levels = tuple(round(0.1 * index, 1) for index in range(1, 11))
    numeric = _float(value, default)
    numeric = max(levels[0], min(levels[-1], numeric))
    return min(levels, key=lambda level: abs(level - numeric))
