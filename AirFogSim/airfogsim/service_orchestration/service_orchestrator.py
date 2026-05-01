from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from airfogsim.airfogsim_scheduler import AirFogSimScheduler

from .deployment_registry import AIRFOGSIM_TYPE_MAP, DeploymentRegistry
from .metrics import OrchestrationMetrics
from .service_spec import MicroserviceSpec


class ServiceOrchestrator:
    def __init__(
        self,
        deployment: DeploymentRegistry,
        serving_policy: str = "proposed",
        metrics: Optional[OrchestrationMetrics] = None,
        allowed_node_types: Optional[List[str]] = None,
        strict_node_type_filter: bool = True,
    ):
        self.deployment = deployment
        self.serving_policy = serving_policy
        self.metrics = metrics
        self.allowed_node_types = set(allowed_node_types or [])
        self.strict_node_type_filter = strict_node_type_filter
        self.entity_sched = AirFogSimScheduler.getEntityScheduler()
        self.comp_sched = AirFogSimScheduler.getComputationScheduler()
        self.comm_sched = AirFogSimScheduler.getCommunicationScheduler()
        self.traffic_sched = AirFogSimScheduler.getTrafficScheduler()

    def _matches_allowed_node_types(self, env, node_id: str) -> bool:
        if not self.allowed_node_types:
            return True
        node_type = AIRFOGSIM_TYPE_MAP.get(env._getNodeTypeById(node_id))
        return node_type in self.allowed_node_types

    def _all_candidate_nodes(self, env, ms: MicroserviceSpec) -> List[str]:
        candidates: List[str] = []
        for node_type in ["vehicle", "uav", "rsu", "cloud_server"]:
            for info in self.entity_sched.getAllNodeInfos(env, [node_type]):
                node_id = info["id"]
                if not self.deployment.can_host(env, node_id, ms):
                    continue
                if self._matches_allowed_node_types(env, node_id):
                    candidates.append(node_id)
        return candidates

    def _build_route(self, env, src: str, dst: str) -> List[str]:
        if src == dst:
            return [dst]
        src_type = env._getNodeTypeById(src)
        dst_type = env._getNodeTypeById(dst)

        if src_type in ["V", "U"] and dst_type in ["V", "U", "I"]:
            return [dst]
        if src_type == "I" and dst_type in ["V", "U"]:
            return [dst]
        if dst_type == "C" and src_type in ["V", "U", "I"]:
            rsu_id = self.traffic_sched.getNearestRSUById(env, src)
            return [dst] if src == rsu_id else [rsu_id, dst]
        if src_type == "C" and dst_type in ["V", "U", "I"]:
            rsu_id = self.traffic_sched.getNearestRSUById(env, dst)
            return [rsu_id, dst]
        return [dst]

    def _estimate_comm_delay(self, env, route: List[str], payload_mb: float, src: str) -> float:
        current = src
        total_delay = 0.0
        for hop in route:
            current_type = env._getNodeTypeById(current)
            hop_type = env._getNodeTypeById(hop)
            if current_type in ["V", "U"] and hop_type in ["V", "U", "I"]:
                try:
                    rate_matrix, wait_delay = self.comm_sched.getEstimatedRateBetweenNodeIds(env, [current], [hop])
                    rate = float(rate_matrix[0][0])
                    if rate <= 1e-9:
                        return 1e9
                    total_delay += payload_mb / rate + float(wait_delay)
                except Exception:
                    return 1e3
            elif current_type == "I" and hop_type in ["V", "U"]:
                try:
                    rate_matrix, wait_delay = self.comm_sched.getEstimatedRateBetweenNodeIds(env, [current], [hop])
                    rate = float(rate_matrix[0][0])
                    if rate <= 1e-9:
                        return 1e9
                    total_delay += payload_mb / rate + float(wait_delay)
                except Exception:
                    return 1e3
            else:
                if hasattr(env, "wired_manager") and env.wired_manager.hasLink(current, hop):
                    total_delay += 0.001 + payload_mb / max(0.1, 100.0)
                else:
                    return 1e9
            current = hop
        return total_delay

    def _score_node(self, env, task_info: Dict, ms: MicroserviceSpec, node_id: str) -> float:
        src = task_info["task_node_id"]
        payload = float(task_info.get("input_payload_mb", task_info.get("task_size", 0.0)))
        route = self._build_route(env, src, node_id)
        comm_delay = self._estimate_comm_delay(env, route, payload, src)
        compute_delay = self.comp_sched.getComputeDelayByNodeId(
            env,
            node_id,
            added_task_cpu=float(task_info.get("task_cpu", 0.0)),
        )
        deploy_penalty = 0.0
        if not self.deployment.is_deployed(node_id, ms.ms_id):
            deploy_penalty = ms.cold_start_s + ms.deployment_cost
        trust_penalty = 0.0
        if hasattr(env, "getNodeTrustScore"):
            trust_penalty = max(0.0, ms.min_trust - env.getNodeTrustScore(node_id))

        if self.serving_policy == "nearest_edge":
            return len(route) + compute_delay
        if self.serving_policy == "cats_style_score":
            return comm_delay + compute_delay
        if self.serving_policy == "centralized_greedy":
            return comm_delay + compute_delay + deploy_penalty + 10.0 * trust_penalty

        payload_bonus = ms.output_ratio * 0.1 * len(route)
        return comm_delay + compute_delay + deploy_penalty + 10.0 * trust_penalty + payload_bonus

    def select_serving_node(
        self,
        env,
        task_info: Dict,
        ms: MicroserviceSpec,
        candidate_node_ids: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, List[str]]]:
        if candidate_node_ids is None:
            candidates = self._all_candidate_nodes(env, ms)
        else:
            candidates = [node_id for node_id in candidate_node_ids if self._matches_allowed_node_types(env, node_id)]
            if not candidates and not self.strict_node_type_filter:
                candidates = list(candidate_node_ids)
        candidates = [node_id for node_id in candidates if self.deployment.can_host(env, node_id, ms)]
        if not candidates:
            return None

        best_node = min(candidates, key=lambda node_id: self._score_node(env, task_info, ms, node_id))
        if not self.deployment.is_deployed(best_node, ms.ms_id):
            self.deployment.deploy(best_node, ms.ms_id)
            if self.metrics is not None:
                self.metrics.record_cold_start(ms.cold_start_s + ms.deployment_cost)
        return best_node, self._build_route(env, task_info["task_node_id"], best_node)
