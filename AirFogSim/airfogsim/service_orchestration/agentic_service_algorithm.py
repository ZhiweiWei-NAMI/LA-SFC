from __future__ import annotations

from typing import Callable, Dict, List, Optional

from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
from airfogsim.airfogsim_scheduler import AirFogSimScheduler

from .deployment_registry import DeploymentRegistry
from .metrics import OrchestrationMetrics
from .regional_agent import RegionalServiceAgent
from .service_catalog import ServiceCatalog
from .service_matcher import RuleBasedServiceMatcher
from .service_orchestrator import ServiceOrchestrator
from .service_runtime import AirFogServiceRuntime
from .service_spec import ServiceIntent


class AgenticServiceAlgorithmModule(BaseAlgorithmModule):
    """Agentic service orchestration module for AirFogSim."""

    def __init__(
        self,
        catalog_path: str,
        intent_generator: Callable[[object], List[ServiceIntent]],
        matcher_policy: str = "adaptive_rule_based",
        serving_policy: str = "proposed",
        lazy_deploy: bool = True,
        regional_agent_enabled: bool = False,
        allowed_node_types: Optional[List[str]] = None,
        strict_node_type_filter: bool = True,
    ):
        super().__init__()
        self.catalog_path = catalog_path
        self.intent_generator = intent_generator
        self.matcher_policy = matcher_policy
        self.serving_policy = serving_policy
        self.lazy_deploy = lazy_deploy
        self.regional_agent_enabled = regional_agent_enabled
        self.allowed_node_types = allowed_node_types
        self.strict_node_type_filter = strict_node_type_filter

        self.catalog = ServiceCatalog.from_yaml(catalog_path)
        self.metrics = OrchestrationMetrics()
        self.matcher = RuleBasedServiceMatcher(self.catalog, policy=matcher_policy)
        self.runtime = AirFogServiceRuntime(metrics=self.metrics)
        self.deployment = DeploymentRegistry(lazy_deploy=lazy_deploy)
        self.orchestrator = ServiceOrchestrator(
            self.deployment,
            serving_policy=serving_policy,
            metrics=self.metrics,
            allowed_node_types=allowed_node_types,
            strict_node_type_filter=strict_node_type_filter,
        )
        self.regional_agents: List[RegionalServiceAgent] = []

        self.task_sched = AirFogSimScheduler.getTaskScheduler()
        self.comm_sched = AirFogSimScheduler.getCommunicationScheduler()
        self.traffic_sched = AirFogSimScheduler.getTrafficScheduler()
        self.region_node_map: Dict[str, List[str]] = {}

    def initialize(self, env, config=None):
        super().initialize(env)
        env.task_scheduler = self.task_sched
        if self.regional_agent_enabled:
            self._initialize_regional_agents(env)

    def _initialize_regional_agents(self, env) -> None:
        self.regional_agents = []
        for rsu_id in env.getRSUIds():
            managed_nodes = [rsu_id]
            self.regional_agents.append(RegionalServiceAgent(region_id=rsu_id, managed_nodes=managed_nodes))
        self._refresh_regional_agents(env)

    def _refresh_regional_agents(self, env) -> None:
        if not self.regional_agents:
            return
        self.region_node_map = {agent.region_id: [agent.region_id] for agent in self.regional_agents}
        rsu_ids = [agent.region_id for agent in self.regional_agents]
        all_nodes = []
        for node_type in ["vehicle", "uav", "rsu", "cloud_server"]:
            all_nodes.extend(self.entityScheduler.getAllNodeInfos(env, [node_type]))

        for info in all_nodes:
            node_id = info["id"]
            if node_id in rsu_ids:
                continue
            if info.get("node_type") == "C" and rsu_ids:
                self.region_node_map[rsu_ids[0]].append(node_id)
                continue
            nearest_rsu = self.traffic_sched.getNearestRSUById(env, node_id)
            if nearest_rsu in self.region_node_map:
                self.region_node_map[nearest_rsu].append(node_id)

        for agent in self.regional_agents:
            agent.managed_nodes = set(self.region_node_map.get(agent.region_id, [agent.region_id]))

    def _sync_regional_ads(self, env) -> None:
        if not self.regional_agents:
            return
        self._refresh_regional_agents(env)
        ads = [agent.observe_local_state(env, self.deployment, self.catalog) for agent in self.regional_agents]
        for agent in self.regional_agents:
            for ad in ads:
                if ad.region_id != agent.region_id:
                    agent.receive_advertisement(ad)

    def _get_region_agent(self, node_id: str) -> Optional[RegionalServiceAgent]:
        for agent in self.regional_agents:
            if node_id in agent.managed_nodes:
                return agent
        return None

    def scheduleStep(self, env):
        self.runtime.on_airfogsim_step_finished(env)
        if self.regional_agents:
            self._sync_regional_ads(env)

        for intent in self.intent_generator(env):
            graph = self.matcher.match(intent)
            self.runtime.submit(env, graph)

        self.scheduleServiceOffloading(env)
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleReturning(env)
        self.scheduleTraffic(env)

    def scheduleServiceOffloading(self, env):
        waiting_tasks = self.task_sched.getAllToOffloadTaskInfos(env, check_dependency=False)
        for task_info in waiting_tasks:
            ms_id = task_info.get("microservice_id")
            if ms_id is None:
                if env.task_manager.checkTaskDependency(task_info["task_node_id"], task_info["task_id"]) is not True:
                    continue
                continue
            ms = self.catalog.get(ms_id)
            decision = None
            source_agent = self._get_region_agent(task_info["task_node_id"]) if self.regional_agents else None
            if source_agent is not None:
                local_candidates = self.region_node_map.get(source_agent.region_id, [])
                decision = self.orchestrator.select_serving_node(
                    env,
                    task_info,
                    ms,
                    candidate_node_ids=local_candidates,
                )
                if decision is None:
                    proposals = source_agent.propose_remote_serving(
                        graph_id=task_info.get("service_graph_id", "unknown"),
                        ms_id=ms.ms_id,
                        expected_payload_mb=float(task_info.get("task_size", 0.0)),
                        deadline_s=float(task_info.get("task_deadline", 0.0)),
                    )
                    for proposal in proposals:
                        remote_candidates = self.region_node_map.get(proposal.dst_region, [])
                        decision = self.orchestrator.select_serving_node(
                            env,
                            task_info,
                            ms,
                            candidate_node_ids=remote_candidates,
                        )
                        if decision is not None:
                            self.metrics.record_cross_region_forward()
                            break
            if decision is None:
                decision = self.orchestrator.select_serving_node(env, task_info, ms)
            if decision is None:
                continue
            target_node_id, route = decision
            if float(task_info.get("required_returned_size", 0.0)) <= 0:
                self.task_sched.setTaskAttribute(
                    env,
                    task_info["task_node_id"],
                    task_info["task_id"],
                    "_to_return_node_id",
                    target_node_id,
                )
            self.task_sched.setTaskOffloading(
                env,
                task_node_id=task_info["task_node_id"],
                task_id=task_info["task_id"],
                target_node_id=target_node_id,
                route=route,
            )

    def scheduleReturning(self, env):
        waiting_to_return = self.task_sched.getWaitingToReturnTaskInfos(env)
        for current_node_id, tasks in waiting_to_return.items():
            for task in tasks:
                sink = task.getToReturnNodeId()
                if sink is None:
                    continue
                route = self.orchestrator._build_route(env, current_node_id, sink)
                self.task_sched.setTaskReturnRoute(env, task.getTaskId(), route)

    def scheduleCommunication(self, env):
        n_rb = self.comm_sched.getNumberOfRB(env)
        env.activated_offloading_tasks_with_RB_Nos = {}
        offloading_tasks = [
            task_info
            for task_info in self.task_sched.getAllOffloadingTaskInfos(env)
            if self._needs_wireless_rb(env, task_info)
        ]
        if not offloading_tasks:
            return
        active = offloading_tasks[:n_rb]
        rb_per_task = max(1, n_rb // max(1, len(active)))
        rb_cursor = 0
        for task_info in active:
            rb_list = [(rb_cursor + i) % n_rb for i in range(rb_per_task)]
            rb_cursor = (rb_cursor + rb_per_task) % n_rb
            self.comm_sched.setCommunicationWithRB(env, task_info["task_id"], rb_list)

    def _needs_wireless_rb(self, env, task_info: Dict) -> bool:
        if task_info.get("executed_locally"):
            return False
        route = task_info.get("to_offload_route") or []
        if not route:
            return False
        tx_node_id = task_info.get("current_node_id") or task_info.get("task_node_id")
        rx_node_id = route[0]
        tx_type = env._getNodeTypeById(tx_node_id)
        rx_type = env._getNodeTypeById(rx_node_id)
        return tx_type in ["V", "U", "I"] and rx_type in ["V", "U", "I"]

    def scheduleComputing(self, env):
        def alloc_cpu_callback(computing_tasks, **kwargs):
            allocation: Dict[str, float] = {}
            for node_id, tasks in computing_tasks.items():
                if not tasks:
                    continue
                node_info = self.entityScheduler.getNodeInfoById(env, node_id)
                cpu = node_info.get("fog_profile", {}).get("cpu", 0.0)
                share = cpu / max(1, len(tasks))
                for task in tasks:
                    allocation[task.getTaskId()] = share
            return allocation

        self.compScheduler.setComputingCallBack(env, alloc_cpu_callback)

    def scheduleTraffic(self, env):
        if not hasattr(env, "mission_manager"):
            return
        super().scheduleTraffic(env)
