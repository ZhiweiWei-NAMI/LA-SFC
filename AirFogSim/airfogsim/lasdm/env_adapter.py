from __future__ import annotations

import copy
import os
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from airfogsim.airfogsim_scheduler import AirFogSimScheduler

from .instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceInstanceStatus
from .manager import LASDMManager
from .model import GraphStatus, LASDMServiceChain, SFCFailureReason
from .resource_usage import collect_resource_usage_snapshot, summarize_runtime_overhead


AIRFOGSIM_TYPE_TO_LASDM = {
    "V": "vehicle",
    "U": "uav",
    "I": "rsu",
    "C": "cloud_server",
}


class LASDMEnvAdapter:
    """Adapter between LASDM state and the real AirFogSim scheduler/runtime API."""

    def __init__(
        self,
        directory: Optional[ServiceInstanceDirectory] = None,
        task_scheduler: Optional[Any] = None,
        communication_scheduler: Optional[Any] = None,
        computation_scheduler: Optional[Any] = None,
        traffic_scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        baseline_name: Optional[str] = None,
        seed: Optional[int] = None,
        scenario: Optional[Dict[str, Any]] = None,
        manager: Optional[LASDMManager] = None,
        chains: Optional[Sequence[LASDMServiceChain]] = None,
        policy: Optional[Any] = None,
        env_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or {}
        self.runtime_config = runtime_config or self.config.get("runtime", {}) or {}
        self.baseline_name = baseline_name
        self.seed = seed
        self.scenario = scenario or {}
        self.manager = manager
        self.directory = directory or (manager.directory if manager is not None else None)
        self.chains = list(chains or [])
        self.policy = policy
        self.env_factory = env_factory
        self.node_aliases = {str(key): str(value) for key, value in self.runtime_config.get("node_aliases", {}).items()}
        self.task_scheduler = task_scheduler or AirFogSimScheduler.getTaskScheduler()
        self.communication_scheduler = communication_scheduler or AirFogSimScheduler.getCommunicationScheduler()
        self.computation_scheduler = computation_scheduler or AirFogSimScheduler.getComputationScheduler()
        self.traffic_scheduler = traffic_scheduler or AirFogSimScheduler.getTrafficScheduler()
        self.last_snapshot: Dict[str, Any] = {}
        self.last_offloading_diagnostics: Dict[str, Dict[str, Any]] = {}
        self.last_wireless_diagnostics: Dict[str, Dict[str, Any]] = {}
        self.last_wireless_allocations: Dict[str, List[int]] = {}

    def sync_from_env(self, env: Any, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Synchronize AirFogSim node/task/link state into LASDM service-instance metadata."""

        now = _resolve_time(env, current_time)
        self._prime_auth_state(env)
        nodes = self.collect_node_snapshot(env)
        tasks = self.collect_task_snapshot(env)
        links = self.collect_link_snapshot(env)
        instances = self.sync_service_instances(env, nodes, tasks, now)
        self.last_snapshot = {
            "time_s": now,
            "nodes": nodes,
            "tasks": tasks,
            "links": links,
            "instances": instances,
        }
        return self.last_snapshot

    def run_benchmark(
        self,
        config: Optional[Dict[str, Any]] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        baseline_name: Optional[str] = None,
        seed: Optional[int] = None,
        scenario: Optional[Dict[str, Any]] = None,
        manager: Optional[LASDMManager] = None,
        directory: Optional[ServiceInstanceDirectory] = None,
        chains: Optional[Sequence[LASDMServiceChain]] = None,
        policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run LASDM inside AirFogSim's real scheduleStep/env.step loop.

        The default backend builds an AirFogSimEnv from runtime.airfogsim_config_path.
        Tests can inject env_factory; production runs should use the AirFogSimEnv path.
        """

        config = config or self.config
        runtime_config = runtime_config or self.runtime_config
        baseline_name = baseline_name or self.baseline_name
        seed = int(seed if seed is not None else (self.seed if self.seed is not None else 0))
        scenario = scenario or self.scenario
        manager = manager or self.manager
        if manager is None:
            raise ValueError("LASDMEnvAdapter.run_benchmark requires a LASDMManager")
        if directory is not None:
            self.directory = directory
        elif self.directory is None:
            self.directory = manager.directory
        chains = list(chains if chains is not None else self.chains)
        policy = policy or self.policy
        self.config = config
        self.runtime_config = runtime_config
        self.baseline_name = baseline_name
        self.seed = seed
        self.scenario = scenario
        self.manager = manager
        self.chains = chains
        self.policy = policy
        self.node_aliases = {str(key): str(value) for key, value in runtime_config.get("node_aliases", {}).items()}

        self._set_seed(seed)
        env = None
        reports: List[Dict[str, Any]] = []
        overhead_records: List[Dict[str, Any]] = []
        resource_usage_timeseries: List[Dict[str, Any]] = []
        completed = False
        error: Optional[str] = None
        from .runtime_bridge import LASDMRuntimeBridge
        from .scheduler import LASDMScheduler

        bridge = LASDMRuntimeBridge(manager, env_adapter=self)
        scheduler = LASDMScheduler(
            manager=manager,
            chain_provider=lambda _env: chains,
            runtime_bridge=bridge,
            env_adapter=self,
        )

        try:
            env = self._build_env(runtime_config)
            self._warmup_env(env, int(runtime_config.get("warmup_steps", 0)))
            self.apply_node_aliases_to_directory(env)
            max_steps = int(runtime_config.get("max_steps", runtime_config.get("airfogsim_max_steps", 1000)))
            steps = 0
            while not self._env_is_done(env) and steps < max_steps and not self._all_chains_terminal(manager, chains):
                decision_start = time.perf_counter()
                scheduler.scheduleStep(env)
                decision_time_s = time.perf_counter() - decision_start
                decision_time_ms = decision_time_s * 1000.0
                report = dict(getattr(env, "lasdm_runtime_report", {}) or {})
                report["decision_time_s"] = decision_time_s
                before_resource = collect_resource_usage_snapshot(
                    env,
                    current_time=_resolve_time(env),
                    decision_time_s=decision_time_s,
                    rb_allocations=report.get("wireless_rb"),
                )
                self._advance_env_step(env, bridge)
                bridge.sync_from_airfogsim_tasks(env, current_time=_resolve_time(env))
                report["post_step_time_s"] = _resolve_time(env)
                reports.append(report)
                resource_usage_timeseries.append(
                    collect_resource_usage_snapshot(
                        env,
                        current_time=_resolve_time(env),
                        decision_time_s=decision_time_s,
                        rb_allocations=report.get("wireless_rb"),
                        previous_energy_remaining=before_resource.get("energy_remaining_total"),
                    )
                )
                overhead_records.append(
                    {
                        "baseline": baseline_name or "",
                        "seed": seed,
                        "scenario": scenario.get("name", ""),
                        "step": steps,
                        "time_s": _resolve_time(env),
                        "decision_time_s": decision_time_s,
                        "decision_time_ms": decision_time_ms,
                    }
                )
                steps += 1
            bridge.sync_from_airfogsim_tasks(env, current_time=_resolve_time(env))
            self._mark_unfinished_chains(manager, _resolve_time(env), max_steps, steps)
            completed = True
        except Exception as exc:
            error = str(exc)
            if env is not None:
                try:
                    bridge.sync_from_airfogsim_tasks(env, current_time=_resolve_time(env))
                except Exception:
                    pass
        finally:
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

        raw_metrics = self._runtime_metrics(
            manager,
            bridge,
            env,
            steps=len(reports),
            error=error,
            reports=reports,
            overhead_records=overhead_records,
            resource_usage_timeseries=resource_usage_timeseries,
        )
        overhead = summarize_runtime_overhead(
            overhead_records,
            baseline=baseline_name or "",
            scenario=str((scenario or {}).get("name", "")),
            seed=seed,
            mode="airfogsim",
        )
        overhead["avg_decision_time_ms"] = raw_metrics.get("runtime_overhead", {}).get("avg_decision_time_ms", 0.0)
        overhead["p95_decision_time_ms"] = raw_metrics.get("runtime_overhead", {}).get("p95_decision_time_ms", 0.0)
        raw_metrics["runtime_overhead"] = overhead
        raw_metrics["avg_decision_time_s"] = overhead["avg_decision_time_s"]
        raw_metrics["p95_decision_time_s"] = overhead["p95_decision_time_s"]
        raw_metrics["max_decision_time_s"] = overhead["max_decision_time_s"]
        raw_metrics["decision_samples"] = overhead["decision_samples"]
        return {
            "completed": completed and error is None,
            "error": error,
            "raw_metrics": raw_metrics,
            "decisions": [decision.to_dict() for decision in manager.decisions.values()],
            "runtime_reports": reports,
            "runtime_overhead": raw_metrics["runtime_overhead"],
            "resource_usage_timeseries": resource_usage_timeseries,
        }

    def collect_node_snapshot(self, env: Any) -> Dict[str, Dict[str, Any]]:
        """Return node_id keyed node state from AirFogSim entities."""

        snapshot: Dict[str, Dict[str, Any]] = {}
        for node_id in self._node_ids(env):
            node = self._node_by_id(env, node_id)
            if node is None:
                continue
            raw = node.to_dict() if hasattr(node, "to_dict") else {}
            node_type = self.node_type(env, node_id)
            fog_profile = dict(raw.get("fog_profile", {}) or getattr(node, "_fog_profile", {}) or {})
            position = raw.get("position")
            if position is None and hasattr(node, "getPosition"):
                position = list(node.getPosition())
            snapshot[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "airfogsim_type": self._airfogsim_node_type(env, node_id),
                "position": position,
                "fog_profile": fog_profile,
                "trust_score": self._trust_score(env, node_id),
                "authenticated": self._authenticated(env, node_id),
                "energy_consumption": _node_energy(node),
            }
        return snapshot

    def collect_task_snapshot(self, env: Any) -> Dict[str, Any]:
        """Return queue-level task counts and per-node compute load from TaskManager."""

        manager = getattr(env, "task_manager", None)
        if manager is None:
            return {"computing_by_node": {}, "waiting_to_offload": 0, "offloading": 0, "done": 0, "failed": 0}
        computing = getattr(manager, "getComputingTasks", lambda: {})()
        waiting = getattr(manager, "getWaitingToOffloadTasks", lambda: {})()
        offloading = getattr(manager, "getOffloadingTasks", lambda: {})()
        done = getattr(manager, "getDoneTasks", lambda: [])()
        failed = getattr(manager, "getOutOfDDLTasks", lambda: [])()
        return {
            "computing_by_node": {node_id: len(tasks) for node_id, tasks in computing.items()},
            "waiting_to_offload": sum(len(tasks) for tasks in waiting.values()),
            "offloading": sum(len(tasks) for tasks in offloading.values()),
            "done": len(done),
            "failed": len(failed),
        }

    def collect_link_snapshot(self, env: Any) -> Dict[str, Any]:
        """Return lightweight communication state available without mutating AirFogSim."""

        return {
            "channel": dict(getattr(env, "channel", {}) or {}),
            "v2u": dict(getattr(env, "V2U_channel", {}) or {}),
            "v2i": dict(getattr(env, "V2I_channel", {}) or {}),
            "u2i": dict(getattr(env, "U2I_channel", {}) or {}),
        }

    def sync_service_instances(
        self,
        env: Any,
        nodes: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tasks: Optional[Mapping[str, Any]] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Update known LASDM service instances from AirFogSim node availability and load."""

        if self.directory is None:
            return {}
        nodes = nodes or self.collect_node_snapshot(env)
        tasks = tasks or self.collect_task_snapshot(env)
        computing_by_node = dict(tasks.get("computing_by_node", {}) or {})
        updated: Dict[str, Dict[str, Any]] = {}
        for instance in self.directory.all():
            resolved_node_id = self._resolve_node_alias(env, instance.node_id, nodes)
            if resolved_node_id != instance.node_id and resolved_node_id in nodes:
                instance.metadata.setdefault("declared_node_id", instance.node_id)
                instance.node_id = resolved_node_id
            node = nodes.get(instance.node_id)
            if node is None:
                instance.status = ServiceInstanceStatus.FAILED
                instance.health_score = 0.0
                instance.metadata["blocked_reason"] = "node_missing"
            else:
                fog_profile = dict(node.get("fog_profile", {}) or {})
                if fog_profile:
                    instance.capacity.update({key: float(value) for key, value in fog_profile.items() if _is_number(value)})
                instance.node_type = str(node.get("node_type") or instance.node_type)
                instance.current_load = int(computing_by_node.get(instance.node_id, 0))
                instance.trust_score = float(node.get("trust_score", instance.trust_score))
                if node.get("authenticated") is False:
                    instance.status = ServiceInstanceStatus.DEGRADED
                    instance.metadata["blocked_reason"] = "unauthenticated"
                elif instance.status == ServiceInstanceStatus.FAILED and instance.metadata.get("allow_auto_recover", True):
                    instance.status = ServiceInstanceStatus.ACTIVE
                    instance.health_score = max(instance.health_score, 0.5)
                    instance.metadata.pop("blocked_reason", None)
                elif instance.status in {ServiceInstanceStatus.ACTIVE, ServiceInstanceStatus.DEGRADED}:
                    instance.metadata.pop("blocked_reason", None)
                instance.last_heartbeat = _resolve_time(env, current_time)
            updated[instance.instance_id] = {
                "node_id": instance.node_id,
                "status": instance.status.value,
                "current_load": instance.current_load,
                "max_concurrency": instance.max_concurrency,
                "health_score": instance.health_score,
                "trust_score": instance.trust_score,
                "selectable": instance.can_reserve({}),
            }
        return updated

    def register_task(self, env: Any, task: Any) -> Any:
        """Register a generated LASDM Task through AirFogSim's task scheduler path."""

        scheduler = getattr(env, "task_scheduler", None) or self.task_scheduler
        if hasattr(scheduler, "registerGeneratedTask"):
            return scheduler.registerGeneratedTask(env, task)
        return env.task_manager.registerGeneratedTask(task)

    def schedule_task_offloading(
        self,
        env: Any,
        task: Any,
        target_node_id: str,
        route: Optional[Sequence[str]] = None,
        current_time: Optional[float] = None,
    ) -> bool:
        """Move a generated task into AirFogSim's offloading/computing queues."""

        route = self.build_route(env, task.getCurrentNodeId(), target_node_id, route)
        task_id = str(task.getTaskId())
        task_node_id = str(task.getTaskNodeId())
        diagnostic = {
            "task_id": task_id,
            "task_node_id": task_node_id,
            "current_node_id": str(task.getCurrentNodeId()),
            "target_node_id": str(target_node_id),
            "route": list(route or []),
            "scheduled": False,
            "reason": "",
        }
        missing_nodes = [node_id for node_id in [task.getCurrentNodeId(), target_node_id, *(route or [])] if not self.runtime_node_exists(env, str(node_id))]
        if missing_nodes:
            diagnostic["reason"] = "missing_runtime_node"
            diagnostic["missing_nodes"] = sorted({str(node_id) for node_id in missing_nodes})
            self.last_offloading_diagnostics[task_id] = diagnostic
            return False
        scheduler = getattr(env, "task_scheduler", None) or self.task_scheduler
        try:
            if hasattr(scheduler, "setTaskOffloading"):
                ok = bool(scheduler.setTaskOffloading(
                    env,
                    task_node_id,
                    task_id,
                    target_node_id,
                    route,
                ))
            else:
                ok = bool(
                    env.task_manager.offloadTask(
                        task_node_id,
                        task_id,
                        target_node_id,
                        _resolve_time(env, current_time),
                        route,
                    )
                )
        except Exception as exc:
            diagnostic["reason"] = "scheduler_exception"
            diagnostic["error"] = repr(exc)
            self.last_offloading_diagnostics[task_id] = diagnostic
            return False

        diagnostic["scheduled"] = ok
        if not ok:
            waiting = getattr(env.task_manager, "getWaitingToOffloadTasksByNodeId", lambda _node_id: [])(task_node_id)
            diagnostic["reason"] = "task_scheduler_rejected"
            diagnostic["waiting_queue_task_ids"] = [item.getTaskId() for item in waiting]
            diagnostic["dependency_ok"] = getattr(env.task_manager, "checkTaskDependency", lambda *_args: None)(task_node_id, task_id)
        else:
            diagnostic["reason"] = "scheduled"
        self.last_offloading_diagnostics[task_id] = diagnostic
        return ok

    def bind_chains_to_runtime_nodes(
        self,
        env: Any,
        chains: Sequence[LASDMServiceChain],
        scenario: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Replace scenario-materialized SFC source/sink node IDs with concrete AirFogSim nodes."""

        scenario = scenario or self.scenario or {}
        nodes = self.collect_node_snapshot(env)
        forced_source = (
            scenario.get("forced_source_node_id")
            or scenario.get("runtime_source_node_id")
            or scenario.get("source_node_id")
        )
        forced_sink = scenario.get("forced_sink_node_id") or scenario.get("sink_node_id")
        preferred_source_type = str(scenario.get("forced_source_node_type", scenario.get("source_node_type", "vehicle")))
        bound: Dict[str, Dict[str, str]] = {}
        for chain in chains:
            old_source = str(chain.source_node_id)
            old_sink = str(chain.sink_node_id)
            if forced_source is not None:
                source = self._select_runtime_node(nodes, forced_source, preferred_source_type)
            elif self.runtime_node_exists(env, old_source):
                source = old_source
            else:
                source = self._select_runtime_node(nodes, None, preferred_source_type)
            if source is None:
                source = self._select_runtime_node(nodes, None, "rsu")
            sink = str(forced_sink) if forced_sink and self.runtime_node_exists(env, str(forced_sink)) else source
            if source is None or sink is None:
                continue
            chain.context.setdefault("declared_source_node_id", old_source)
            chain.context.setdefault("declared_sink_node_id", old_sink)
            chain.context["runtime_bound_source_node_id"] = source
            chain.context["runtime_bound_sink_node_id"] = sink
            chain.source_node_id = source
            chain.sink_node_id = sink
            bound[chain.sfc_id] = {"source_node_id": source, "sink_node_id": sink}
        return bound

    def bind_service_instances_to_runtime_nodes(
        self,
        env: Any,
        scenario: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Replace scenario-materialized service-node IDs with concrete AirFogSim runtime nodes."""

        if self.directory is None:
            return {}
        scenario = scenario or self.scenario or {}
        nodes = self.collect_node_snapshot(env)
        assigned_by_type: Dict[str, int] = {}
        bound: Dict[str, Dict[str, str]] = {}
        for instance in self.directory.all():
            declared = str(instance.node_id)
            if self.runtime_node_exists(env, declared):
                continue
            preferred_type = str(instance.node_type or _infer_node_type(declared) or "rsu")
            forced = _forced_service_node_for_instance(instance, scenario)
            node_id = self._select_runtime_node_round_robin(nodes, forced, preferred_type, assigned_by_type)
            if node_id is None:
                instance.status = ServiceInstanceStatus.FAILED
                instance.health_score = 0.0
                instance.metadata["blocked_reason"] = "no_runtime_node_for_service_instance"
                continue
            node = dict(nodes.get(node_id, {}) or {})
            instance.metadata.setdefault("declared_node_id", declared)
            instance.metadata["runtime_bound_node_id"] = node_id
            instance.node_id = node_id
            instance.node_type = str(node.get("node_type") or preferred_type)
            if instance.node_type == "rsu":
                instance.region_id = node_id
            elif instance.node_type == "cloud_server":
                instance.region_id = "cloud"
            else:
                nearest = self.nearest_rsu(env, node_id)
                if nearest:
                    instance.region_id = str(nearest)
            bound[instance.instance_id] = {"declared_node_id": declared, "node_id": node_id}
        return bound

    def runtime_node_exists(self, env: Any, node_id: str) -> bool:
        return self._node_by_id(env, str(node_id)) is not None

    def _select_runtime_node(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        forced_node_id: Optional[Any],
        preferred_type: str,
    ) -> Optional[str]:
        if forced_node_id is not None and str(forced_node_id) in nodes:
            return str(forced_node_id)
        preferred_order = [preferred_type, "vehicle", "uav", "rsu", "cloud_server"]
        seen = set()
        for node_type in preferred_order:
            if node_type in seen:
                continue
            seen.add(node_type)
            for node_id, node in sorted(nodes.items()):
                if str(node.get("node_type")) == node_type:
                    return str(node_id)
        return next(iter(sorted(nodes)), None)

    def _select_runtime_node_round_robin(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        forced_node_id: Optional[Any],
        preferred_type: str,
        assigned_by_type: Dict[str, int],
    ) -> Optional[str]:
        if forced_node_id is not None and str(forced_node_id) in nodes:
            return str(forced_node_id)
        candidates = [
            str(node_id)
            for node_id, node in sorted(nodes.items())
            if str(node.get("node_type")) == str(preferred_type)
        ]
        if not candidates and str(preferred_type) == "cloud_server":
            candidates = [str(node_id) for node_id, node in sorted(nodes.items()) if str(node.get("node_type")) == "cloud_server"]
        if not candidates:
            candidates = [str(node_id) for node_id, node in sorted(nodes.items()) if str(node.get("node_type")) == "rsu"]
        if not candidates:
            return next(iter(sorted(nodes)), None)
        key = str(preferred_type)
        index = assigned_by_type.get(key, 0)
        assigned_by_type[key] = index + 1
        return candidates[index % len(candidates)]

    def schedule_returning(self, env: Any) -> Dict[str, List[str]]:
        """Set return routes for AirFogSim tasks waiting to return."""

        scheduler = getattr(env, "task_scheduler", None) or self.task_scheduler
        waiting = scheduler.getWaitingToReturnTaskInfos(env) if hasattr(scheduler, "getWaitingToReturnTaskInfos") else {}
        scheduled: Dict[str, List[str]] = {}
        for current_node_id, tasks in waiting.items():
            for task in tasks:
                sink = task.getToReturnNodeId()
                if sink is None:
                    continue
                route = self.build_route(env, current_node_id, sink)
                scheduler.setTaskReturnRoute(env, task.getTaskId(), route)
                scheduled[task.getTaskId()] = route
        return scheduled

    def schedule_communication(self, env: Any) -> Dict[str, List[int]]:
        """Allocate wireless RBs for offloading tasks that use wireless hops."""

        scheduler = getattr(env, "communication_scheduler", None) or self.communication_scheduler
        task_scheduler = getattr(env, "task_scheduler", None) or self.task_scheduler
        n_rb = scheduler.getNumberOfRB(env)
        budget = self.scenario.get("wireless_rb_budget") if isinstance(self.scenario, Mapping) else None
        if budget is not None:
            try:
                n_rb = max(1, min(int(n_rb), int(budget)))
            except Exception:
                n_rb = scheduler.getNumberOfRB(env)
        contention = self.scenario.get("wireless_contention_factor") if isinstance(self.scenario, Mapping) else None
        if contention is not None:
            try:
                n_rb = max(1, int(n_rb / max(1.0, float(contention))))
            except Exception:
                pass
        if hasattr(env, "activated_offloading_tasks_with_RB_Nos"):
            env.activated_offloading_tasks_with_RB_Nos = {}
        self.last_wireless_diagnostics = {}
        self.last_wireless_allocations = {}
        offloading = task_scheduler.getAllOffloadingTaskInfos(env) if hasattr(task_scheduler, "getAllOffloadingTaskInfos") else []
        wireless = []
        for task_info in offloading:
            task_id = str(_task_info_value(task_info, "task_id", ""))
            needs_rb = self.needs_wireless_rb(env, task_info)
            self.last_wireless_diagnostics[task_id] = {
                "task_id": task_id,
                "needs_wireless_rb": needs_rb,
                "current_node_id": _task_info_value(task_info, "current_node_id", _task_info_value(task_info, "task_node_id", "")),
                "route": list(_task_info_value(task_info, "to_offload_route", []) or []),
            }
            if needs_rb:
                wireless.append(task_info)
        if not wireless:
            return {}
        active = wireless[:n_rb]
        rb_per_task = max(1, n_rb // max(1, len(active)))
        rb_cursor = 0
        assigned: Dict[str, List[int]] = {}
        for task_info in active:
            task_id = str(_task_info_value(task_info, "task_id", ""))
            rb_list = [(rb_cursor + index) % n_rb for index in range(rb_per_task)]
            rb_cursor = (rb_cursor + rb_per_task) % n_rb
            scheduler.setCommunicationWithRB(env, task_id, rb_list)
            assigned[task_id] = rb_list
            self.last_wireless_allocations[task_id] = rb_list
            self.last_wireless_diagnostics.setdefault(task_id, {})["allocated_rb"] = rb_list
        return assigned

    def schedule_computation(self, env: Any) -> bool:
        """Install the CPU allocation callback required by AirFogSimEnv._updateComputation."""

        scheduler = getattr(env, "computation_scheduler", None) or self.computation_scheduler

        def alloc_cpu_callback(computing_tasks, **kwargs):
            allocation: Dict[str, float] = {}
            for node_id, tasks in computing_tasks.items():
                if not tasks:
                    continue
                cpu = self.node_cpu(env, node_id)
                share = cpu / max(1, len(tasks))
                for task in tasks:
                    allocation[task.getTaskId()] = share
            return allocation

        scheduler.setComputingCallBack(env, alloc_cpu_callback)
        return True

    def build_route(
        self,
        env: Any,
        source_node_id: str,
        target_node_id: str,
        proposed_route: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Build an AirFogSim offload route excluding the source and ending at the target."""

        if source_node_id == target_node_id:
            return [target_node_id]
        proposed = self._normalize_route(source_node_id, target_node_id, proposed_route)
        if env is None:
            return proposed
        src_type = self._airfogsim_node_type(env, source_node_id)
        dst_type = self._airfogsim_node_type(env, target_node_id)
        if src_type in {"V", "U"} and dst_type == "C":
            rsu_id = self.nearest_rsu(env, source_node_id)
            return [rsu_id, target_node_id] if rsu_id and rsu_id != source_node_id else [target_node_id]
        if src_type == "C" and dst_type in {"V", "U", "I"}:
            rsu_id = self.nearest_rsu(env, target_node_id)
            if rsu_id and rsu_id != target_node_id:
                return [rsu_id, target_node_id]
        return proposed

    def needs_wireless_rb(self, env: Any, task_info: Mapping[str, Any]) -> bool:
        if _task_info_value(task_info, "executed_locally", False):
            return False
        route = _task_info_value(task_info, "to_offload_route", []) or []
        if not route:
            return False
        tx_node_id = _task_info_value(task_info, "current_node_id", None) or _task_info_value(task_info, "task_node_id", None)
        rx_node_id = route[0]
        tx_type = self._airfogsim_node_type(env, tx_node_id)
        rx_type = self._airfogsim_node_type(env, rx_node_id)
        return tx_type in {"V", "U", "I"} and rx_type in {"V", "U", "I"}

    def node_cpu(self, env: Any, node_id: str) -> float:
        node = self._node_by_id(env, node_id)
        if node is None or not hasattr(node, "getFogProfile"):
            return 0.1
        profile = node.getFogProfile() or {}
        return max(0.1, float(profile.get("cpu", 0.1)))

    def node_type(self, env: Any, node_id: str) -> str:
        return AIRFOGSIM_TYPE_TO_LASDM.get(self._airfogsim_node_type(env, node_id), "unknown")

    def nearest_rsu(self, env: Any, node_id: str) -> Optional[str]:
        scheduler = getattr(env, "traffic_scheduler", None) or self.traffic_scheduler
        if hasattr(scheduler, "getNearestRSUById"):
            try:
                return scheduler.getNearestRSUById(env, node_id)
            except Exception:
                return None
        return None

    def apply_node_aliases_to_directory(self, env: Any) -> Dict[str, str]:
        """Rewrite configured logical node ids to concrete AirFogSim runtime ids."""

        if self.directory is None or not self.node_aliases:
            return {}
        nodes = self.collect_node_snapshot(env)
        applied: Dict[str, str] = {}
        for instance in self.directory.all():
            resolved = self._resolve_node_alias(env, instance.node_id, nodes)
            if resolved == instance.node_id or resolved not in nodes:
                continue
            declared = instance.node_id
            instance.metadata.setdefault("declared_node_id", declared)
            instance.node_id = resolved
            instance.node_type = str(nodes[resolved].get("node_type") or instance.node_type)
            applied[declared] = resolved
        return applied

    def _build_env(self, runtime_config: Mapping[str, Any]) -> Any:
        if self.env_factory is not None:
            return self.env_factory()
        config_path = runtime_config.get("airfogsim_config_path")
        if not config_path:
            raise ValueError("runtime.airfogsim_config_path is required for LASDM airfogsim mode")
        from airfogsim import AirFogSimEnv

        env_config = self._load_airfogsim_config(str(config_path), runtime_config)
        return AirFogSimEnv(env_config, interactive_mode=runtime_config.get("interactive_mode"))

    def _load_airfogsim_config(self, config_path: str, runtime_config: Mapping[str, Any]) -> Dict[str, Any]:
        resolved_path = self._resolve_repo_or_config_path(config_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"runtime.airfogsim_config_path not found: {resolved_path}")
        with open(resolved_path, "r", encoding="utf-8") as file:
            env_config = yaml.safe_load(file) or {}
        env_config = self._resolve_airfogsim_paths(env_config, os.path.dirname(resolved_path))
        env_config.setdefault("task", {})
        env_config["task"]["task_generation_model"] = "None"
        overrides = runtime_config.get("airfogsim_config_overrides") or {}
        if overrides:
            env_config = _deep_merge(env_config, copy.deepcopy(overrides))
        env_config = self._apply_semantic_scenario_runtime_overrides(env_config)
        return env_config

    def _apply_semantic_scenario_runtime_overrides(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        scenario = self.scenario if isinstance(self.scenario, Mapping) else {}
        mobility = dict(scenario.get("mobility_stress", {}) or {})
        traffic = env_config.setdefault("traffic", {})
        if scenario.get("active_uav_count") is not None:
            traffic["active_uav_count"] = min(20, max(0, int(scenario["active_uav_count"])))
        if scenario.get("uav_count") is not None:
            traffic["active_uav_count"] = min(20, max(0, int(scenario["uav_count"])))
        if scenario.get("speed_scale") is not None:
            traffic["vehicle_speed_scale"] = float(scenario["speed_scale"])
        if scenario.get("vehicle_speed_scale") is not None:
            traffic["vehicle_speed_scale"] = float(scenario["vehicle_speed_scale"])
        if scenario.get("uav_speed_scale") is not None:
            traffic["uav_speed_scale"] = float(scenario["uav_speed_scale"])
        if mobility.get("enabled"):
            if mobility.get("uav_speed_scale") is not None:
                traffic["uav_speed_scale"] = float(mobility["uav_speed_scale"])
            if mobility.get("vehicle_speed_scale") is not None:
                traffic["vehicle_speed_scale"] = float(mobility["vehicle_speed_scale"])
            env_config.setdefault("lasdm_semantic_runtime", {})
            env_config["lasdm_semantic_runtime"]["topology_churn_interval_s"] = mobility.get("topology_churn_interval_s", "")
        return env_config

    def _resolve_repo_or_config_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.abspath(path)
        candidates = []
        config_meta = self.config.get("_meta", {}) if isinstance(self.config, dict) else {}
        config_dir = config_meta.get("benchmark_config_dir")
        if config_dir:
            candidates.append(os.path.abspath(os.path.join(config_dir, path)))
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        candidates.append(os.path.abspath(os.path.join(workspace_root, path)))
        candidates.append(os.path.abspath(os.path.join(workspace_root, "AirFogSim", path)))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _resolve_airfogsim_paths(self, env_config: Dict[str, Any], config_dir: str) -> Dict[str, Any]:
        resolved = copy.deepcopy(env_config)
        path_keys = {
            "sumo": ["sumo_config", "sumo_osm", "sumo_net", "tripinfo_output"],
            "traffic": ["tripinfo", "uav_traffic_file"],
            "visualization": ["icon_path"],
        }
        for section, keys in path_keys.items():
            if section not in resolved:
                continue
            for key in keys:
                value = resolved[section].get(key)
                if not isinstance(value, str) or not value or os.path.isabs(value):
                    continue
                resolved[section][key] = os.path.abspath(os.path.join(config_dir, value))
        return resolved

    def _advance_env_step(self, env: Any, bridge: Any) -> None:
        if hasattr(env, "step"):
            env.step()
            return
        raise RuntimeError("AirFogSim runtime env must expose step()")

    def _warmup_env(self, env: Any, warmup_steps: int) -> None:
        for _ in range(max(0, warmup_steps)):
            self.schedule_returning(env)
            self.schedule_communication(env)
            self.schedule_computation(env)
            if hasattr(env, "step"):
                env.step()
            else:
                raise RuntimeError("AirFogSim warmup requires env.step()")

    def _runtime_metrics(
        self,
        manager: LASDMManager,
        bridge: Any,
        env: Any,
        steps: int,
        error: Optional[str],
        reports: Optional[Sequence[Dict[str, Any]]] = None,
        overhead_records: Optional[Sequence[Dict[str, Any]]] = None,
        resource_usage_timeseries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        metrics = bridge.collect_step_metrics(current_time=_resolve_time(env)) if bridge is not None else manager.summary()
        task_done_num = self._lasdm_task_count(env, bridge, done=True)
        task_fail_num = self._lasdm_task_count(env, bridge, done=False)
        submitted = int(metrics.get("submitted", 0))
        succeeded = int(metrics.get("succeeded", 0))
        failed = int(metrics.get("failed", 0))
        timed_out = int(metrics.get("timed_out", 0))
        metrics.update(
            {
                "graph_submit_count": submitted,
                "graph_submitted_count": submitted,
                "graph_complete_count": succeeded,
                "graph_failed_count": failed,
                "graph_timeout_count": timed_out,
                "graph_completion_ratio": succeeded / max(1, submitted),
                "deadline_satisfaction_ratio": float(metrics.get("qos_hit_ratio", 0.0)),
                "task_done_num": task_done_num,
                "task_fail_num": task_fail_num,
                "task_success_ratio": task_done_num / max(1, task_done_num + task_fail_num),
                "simulation_time_end": _resolve_time(env),
                "runtime_step_count": steps,
                "runtime_overhead": self._overhead_summary(overhead_records or []),
                "runtime_overhead_records": list(overhead_records or []),
                "resource_usage_timeseries": list(resource_usage_timeseries or self._resource_timeseries_from_reports(reports or [])),
            }
        )
        if error is not None:
            metrics["runtime_error"] = error
        return metrics

    def _overhead_summary(self, records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        values = sorted(float(record.get("decision_time_ms", 0.0)) for record in records)
        if not values:
            return {"avg_decision_time_ms": 0.0, "p95_decision_time_ms": 0.0}
        return {
            "avg_decision_time_ms": sum(values) / len(values),
            "p95_decision_time_ms": _percentile(values, 0.95),
        }

    def _resource_timeseries_from_reports(self, reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for report in reports:
            time_s = float(report.get("post_step_time_s", report.get("time_s", 0.0)) or 0.0)
            snapshot = dict(report.get("env_snapshot", {}) or {})
            nodes = dict(snapshot.get("nodes", {}) or {})
            instances = dict(snapshot.get("instances", {}) or {})
            tasks = dict(snapshot.get("tasks", {}) or {})
            links = dict(snapshot.get("links", {}) or {})
            wireless_rb = dict(report.get("wireless_rb", {}) or {})
            rb_used = sum(len(rbs) for rbs in wireless_rb.values())
            for instance_id, instance in instances.items():
                max_concurrency = max(1, int(instance.get("max_concurrency", 1) or 1))
                current_load = int(instance.get("current_load", 0) or 0)
                node_id = str(instance.get("node_id", ""))
                rows.append(
                    {
                        "time_s": time_s,
                        "node_id": node_id,
                        "instance_id": str(instance_id),
                        "cpu_utilization": min(1.0, current_load / float(max_concurrency)),
                        "rb_utilization": rb_used,
                        "backhaul_usage_mb": _link_data_size(links),
                        "energy_consumption": float(dict(nodes.get(node_id, {}) or {}).get("energy_consumption", 0.0) or 0.0),
                        "computing_task_count": int(dict(tasks.get("computing_by_node", {}) or {}).get(node_id, 0)),
                    }
                )
        return rows

    def _lasdm_task_count(self, env: Any, bridge: Any, done: bool) -> int:
        manager = getattr(env, "task_manager", None)
        if manager is None:
            return 0
        getter = getattr(manager, "getDoneTasks" if done else "getOutOfDDLTasks", None)
        if getter is None:
            return 0
        task_ids = set(getattr(bridge, "task_to_sfc", {}).keys())
        return sum(1 for task in getter() if not task_ids or task.getTaskId() in task_ids)

    def _all_chains_terminal(self, manager: LASDMManager, chains: Sequence[LASDMServiceChain]) -> bool:
        if not chains:
            return True
        for chain in chains:
            managed = manager.chains.get(chain.sfc_id)
            if managed is None or not managed.is_terminal():
                return False
        return True

    def _mark_unfinished_chains(self, manager: LASDMManager, current_time: float, max_steps: int, steps: int) -> None:
        if steps < max_steps:
            return
        for chain in manager.chains.values():
            if chain.status == GraphStatus.RUNNING:
                manager.fail(
                    chain.sfc_id,
                    SFCFailureReason.DEADLINE_MISSED,
                    current_time,
                    status=GraphStatus.TIMED_OUT,
                    details={"reason": "runtime_max_steps_reached", "max_steps": max_steps},
                )

    def _env_is_done(self, env: Any) -> bool:
        if hasattr(env, "isDone"):
            return bool(env.isDone())
        max_time = float(self.runtime_config.get("max_simulation_time", float("inf")))
        return float(getattr(env, "simulation_time", 0.0)) >= max_time

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

    def _node_ids(self, env: Any) -> List[str]:
        ids: List[str] = []
        for method_name in ("getVehicleIds", "getUAVIds", "getRSUIds", "getCloudServerIds"):
            if hasattr(env, method_name):
                ids.extend(str(node_id) for node_id in getattr(env, method_name)())
        return ids

    def _node_by_id(self, env: Any, node_id: str) -> Any:
        node_id = self.node_aliases.get(str(node_id), str(node_id))
        if hasattr(env, "_getNodeById"):
            return env._getNodeById(node_id)
        for attr in ("vehicles", "UAVs", "RSUs", "cloudServers"):
            collection = getattr(env, attr, {})
            if node_id in collection:
                return collection[node_id]
        return None

    def _airfogsim_node_type(self, env: Any, node_id: str) -> Optional[str]:
        node_id = self.node_aliases.get(str(node_id), str(node_id))
        if hasattr(env, "_getNodeTypeById"):
            return env._getNodeTypeById(node_id)
        if node_id in getattr(env, "vehicles", {}):
            return "V"
        if node_id in getattr(env, "UAVs", {}):
            return "U"
        if node_id in getattr(env, "RSUs", {}):
            return "I"
        if node_id in getattr(env, "cloudServers", {}):
            return "C"
        return None

    def _resolve_node_alias(
        self,
        env: Any,
        node_id: str,
        nodes: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> str:
        node_id = str(node_id)
        if node_id in self.node_aliases:
            return self.node_aliases[node_id]
        return node_id

    def _trust_score(self, env: Any, node_id: str) -> float:
        auth_manager = getattr(env, "auth_manager", None)
        if auth_manager is not None and node_id not in getattr(auth_manager, "node_auth_status", {}):
            node = self._node_by_id(env, node_id)
            return float(getattr(node, "trust_score", 1.0))
        if hasattr(env, "getNodeTrustScore"):
            try:
                return float(env.getNodeTrustScore(node_id))
            except Exception:
                return 1.0
        return 1.0

    def _authenticated(self, env: Any, node_id: str) -> Optional[bool]:
        auth_manager = getattr(env, "auth_manager", None)
        if auth_manager is not None and node_id not in getattr(auth_manager, "node_auth_status", {}):
            return None
        if hasattr(env, "isNodeAuthenticated"):
            try:
                return bool(env.isNodeAuthenticated(node_id))
            except Exception:
                return None
        return None

    def _prime_auth_state(self, env: Any) -> None:
        if hasattr(env, "_registerNewNodesForAuth"):
            try:
                env._registerNewNodesForAuth()
            except Exception:
                pass

    def _normalize_route(
        self,
        source_node_id: str,
        target_node_id: str,
        route: Optional[Sequence[str]],
    ) -> List[str]:
        normalized = [str(node_id) for node_id in (route or []) if node_id is not None]
        if normalized and normalized[0] == source_node_id:
            normalized = normalized[1:]
        if not normalized:
            normalized = [target_node_id]
        if normalized[-1] != target_node_id:
            normalized.append(target_node_id)
        return normalized


def _resolve_time(env: Any = None, current_time: Optional[float] = None) -> float:
    if current_time is not None:
        return float(current_time)
    if env is not None and hasattr(env, "simulation_time"):
        return float(env.simulation_time)
    return 0.0


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _percentile(values: Sequence[float], percentile: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    index = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * percentile))))
    return clean[index]


def _link_data_size(links: Mapping[str, Any]) -> float:
    total = 0.0
    for value in links.values():
        if isinstance(value, Mapping):
            try:
                total += float(value.get("data_size", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _task_info_value(task_info: Any, key: str, default: Any = None) -> Any:
    if isinstance(task_info, Mapping):
        return task_info.get(key, default)
    attr_name = f"_{key}"
    if hasattr(task_info, attr_name):
        return getattr(task_info, attr_name)
    getter_name = {
        "task_id": "getTaskId",
        "task_node_id": "getTaskNodeId",
        "current_node_id": "getCurrentNodeId",
        "to_offload_route": "getToOffloadRoute",
        "executed_locally": "isExecutedLocally",
    }.get(key)
    getter = getattr(task_info, getter_name, None) if getter_name else None
    if callable(getter):
        try:
            return getter()
        except Exception:
            return default
    return default


def _node_energy(node: Any) -> float:
    if node is None:
        return 0.0
    for attr in ("energy_consumption", "energy", "_energy_consumption", "_energy"):
        if hasattr(node, attr):
            try:
                return float(getattr(node, attr))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _infer_node_type(node_id: str) -> str:
    text = str(node_id).lower()
    if text.startswith("uav"):
        return "uav"
    if text.startswith("vehicle") or text.startswith("car") or text.startswith("veh"):
        return "vehicle"
    if text.startswith("rsu"):
        return "rsu"
    if text.startswith("cloud"):
        return "cloud_server"
    return "rsu"


def _forced_service_node_for_instance(instance: ServiceInstance, scenario: Mapping[str, Any]) -> Optional[str]:
    forced = scenario.get("forced_service_node_id") or scenario.get("force_service_node_id")
    if forced:
        return str(forced)
    by_service = scenario.get("forced_service_node_by_service")
    if isinstance(by_service, Mapping):
        value = by_service.get(instance.service_id)
        if value:
            return str(value)
    return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base
