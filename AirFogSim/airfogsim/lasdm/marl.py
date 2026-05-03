from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .manager import LASDMManager
from .model import GraphStatus, LASDMServiceChain, SFCFailureReason
from .orchestrator import LASDMDecision


class LASDMMARLInterface:
    """Observation/action/reward helpers for Agentic Orchestrator MARL loops."""

    def __init__(
        self,
        env: Optional[object] = None,
        manager: Optional[LASDMManager] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        max_steps: Optional[int] = None,
    ):
        self.env = env
        self.manager = manager
        self.reward_weights = reward_weights or {}
        self.max_steps = max_steps
        self.step_count = 0
        self.previous_summary: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}
        self.last_decisions: List[LASDMDecision] = []

    def reset(
        self,
        env: Optional[object] = None,
        manager: Optional[LASDMManager] = None,
        current_time: Optional[float] = None,
        reset_underlying_env: bool = False,
    ) -> Dict[str, Any]:
        """Reset wrapper-local MARL state and return the initial observation."""

        if env is not None:
            self.env = env
        if manager is not None:
            self.manager = manager
        manager = self._require_manager()

        if reset_underlying_env and self.env is not None and hasattr(self.env, "reset"):
            self.env.reset()
        if current_time is not None and self.env is not None and hasattr(self.env, "simulation_time"):
            setattr(self.env, "simulation_time", float(current_time))

        self.step_count = 0
        self.previous_summary = manager.summary()
        self.last_decisions = []
        self.last_observation = self.build_observation(self.env, manager)
        return self.last_observation

    def step(
        self,
        action: Optional[Mapping[str, Any]] = None,
        current_time: Optional[float] = None,
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an optional MARL action, advance the wrapped env if present, and score the transition."""

        manager = self._require_manager()
        if not self.previous_summary:
            self.previous_summary = manager.summary()

        previous_summary = dict(self.previous_summary)
        time_s = self._current_time(current_time)
        if action is None:
            decisions = manager.step(current_time=time_s)
        else:
            decisions = self._apply_action(action, manager, time_s)

        if self.env is not None and hasattr(self.env, "step"):
            self.env.step()

        self.step_count += 1
        current_summary = manager.summary()
        reward = self.reward(previous_summary, current_summary, self.reward_weights)
        done = self.done(manager)
        observation = self.build_observation(self.env, manager)

        self.previous_summary = current_summary
        self.last_decisions = decisions
        self.last_observation = observation
        info = {
            "time_s": time_s,
            "decisions": [decision.to_dict() for decision in decisions],
            "summary": current_summary,
            "previous_summary": previous_summary,
        }
        return observation, reward, done, info

    def observation(
        self,
        env: Optional[object] = None,
        manager: Optional[LASDMManager] = None,
    ) -> Dict[str, Any]:
        """Return the current LASDM MARL observation."""

        return self.build_observation(env if env is not None else self.env, manager or self._require_manager())

    def done(self, manager: Optional[LASDMManager] = None) -> bool:
        """Return whether all submitted SFCs are terminal or the optional step cap was reached."""

        manager = manager or self._require_manager()
        if self.max_steps is not None and self.step_count >= self.max_steps:
            return True
        summary = manager.summary()
        return bool(summary.get("submitted", 0)) and int(summary.get("active_graphs", 0)) == 0

    def build_observation(self, env: Optional[object], manager: LASDMManager) -> Dict[str, Any]:
        time_s = float(getattr(env, "simulation_time", 0.0)) if env is not None else 0.0
        instances = []
        for instance in manager.directory.all():
            instances.append(
                {
                    "instance_id": instance.instance_id,
                    "service_id": instance.service_id,
                    "node_id": instance.node_id,
                    "node_type": instance.node_type,
                    "region_id": instance.region_id,
                    "load_ratio": instance.load_ratio(),
                    "health": instance.health_score,
                    "reliability": instance.reliability_score,
                    "accuracy": instance.accuracy_score,
                    "trust": instance.trust_score,
                    "status": instance.status.value,
                }
            )
        graphs = [
            {
                "sfc_id": chain.sfc_id,
                "status": chain.status.value,
                "deadline_s": chain.qos.deadline_s,
                "priority": chain.qos.priority,
                "node_count": len(chain.nodes),
                "edge_count": len(chain.edges),
            }
            for chain in manager.chains.values()
        ]
        return {"time_s": time_s, "instances": instances, "graphs": graphs, "metrics": manager.metrics.summary()}

    def decode_action(
        self,
        action: Dict[str, Any],
        chain: LASDMServiceChain,
        manager: LASDMManager,
    ) -> LASDMDecision:
        decision = LASDMDecision(sfc_id=chain.sfc_id)
        for sfc_node_id, action_value in action.items():
            instance_id = _action_instance_id(action_value)
            if sfc_node_id not in chain.nodes:
                decision.rejected_reason = SFCFailureReason.INVALID_GRAPH
                decision.diagnostics["invalid_sfc_node_id"] = sfc_node_id
                return decision
            if not instance_id:
                decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                decision.diagnostics["missing_instance_id"] = instance_id
                return decision
            try:
                instance = manager.directory.get(instance_id)
            except KeyError:
                decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                decision.diagnostics["missing_instance_id"] = instance_id
                return decision
            decision.assignments[sfc_node_id] = instance_id
            decision.node_mapping[sfc_node_id] = instance.node_id
            decision.resource_allocations[sfc_node_id] = {
                "compute_level": _resource_level(_action_resource_value(action_value, "compute_level"), 1.0),
                "bandwidth_level": _resource_level(_action_resource_value(action_value, "bandwidth_level"), 1.0),
            }
            decision.routes[sfc_node_id] = [chain.source_node_id, instance.node_id]
        return decision

    def reward(
        self,
        previous_summary: Dict[str, Any],
        current_summary: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        weights = weights or {}
        success_delta = float(current_summary.get("succeeded", 0)) - float(previous_summary.get("succeeded", 0))
        fail_delta = float(current_summary.get("failed", 0)) - float(previous_summary.get("failed", 0))
        timeout_delta = float(current_summary.get("timed_out", 0)) - float(previous_summary.get("timed_out", 0))
        latency = float(current_summary.get("avg_latency_s", 0.0))
        qos = float(current_summary.get("qos_hit_ratio", 0.0))
        return (
            weights.get("success", 10.0) * success_delta
            + weights.get("qos", 2.0) * qos
            - weights.get("failure", 5.0) * fail_delta
            - weights.get("timeout", 8.0) * timeout_delta
            - weights.get("latency", 0.1) * latency
        )

    def _require_manager(self) -> LASDMManager:
        if self.manager is None:
            raise ValueError("LASDMMARLInterface requires a LASDMManager for reset/step/done")
        return self.manager

    def _current_time(self, current_time: Optional[float]) -> float:
        if current_time is not None:
            return float(current_time)
        if self.env is not None and hasattr(self.env, "simulation_time"):
            return float(getattr(self.env, "simulation_time"))
        return float(self.step_count)

    def _apply_action(
        self,
        action: Mapping[str, Any],
        manager: LASDMManager,
        current_time: float,
    ) -> List[LASDMDecision]:
        decisions: List[LASDMDecision] = []
        for chain, assignments in self._iter_chain_actions(action, manager):
            decision = self.decode_action(assignments, chain, manager)
            manager.decisions[chain.sfc_id] = decision
            decisions.append(decision)
            if decision.rejected_reason is not None:
                manager.fail(chain.sfc_id, decision.rejected_reason, current_time)
        return decisions

    def _iter_chain_actions(
        self,
        action: Mapping[str, Any],
        manager: LASDMManager,
    ) -> List[tuple[LASDMServiceChain, Dict[str, Any]]]:
        if "assignments" in action:
            chain = self._chain_for_action(manager, action.get("sfc_id"))
            return [(chain, self._action_mapping(action["assignments"]))]

        nested_actions = []
        for sfc_id, assignments in action.items():
            if sfc_id in manager.chains and isinstance(assignments, Mapping):
                nested_actions.append((manager.chains[sfc_id], self._action_mapping(assignments)))
        if nested_actions:
            return nested_actions

        chain = self._chain_for_action(manager, None)
        return [(chain, self._action_mapping(action))]

    def _chain_for_action(self, manager: LASDMManager, sfc_id: Optional[Any]) -> LASDMServiceChain:
        if sfc_id is not None:
            return manager.chains[str(sfc_id)]
        candidates = [
            chain
            for chain in manager.chains.values()
            if chain.status == GraphStatus.RUNNING and chain.sfc_id not in manager.decisions
        ]
        if len(candidates) != 1:
            raise ValueError("Action must include sfc_id when there is not exactly one undecided running SFC")
        return candidates[0]

    def _action_mapping(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("LASDM MARL action assignments must be a mapping")
        return {str(key): value for key, value in raw.items()}


class LASDMMARLWrapper(LASDMMARLInterface):
    """Named env-style wrapper alias for callers that prefer wrapper terminology."""


def _action_instance_id(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("instance_id", value.get("service_instance_id", "")) or "")
    return str(value or "")


def _action_resource_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _resource_level(value: Any, default: float = 1.0) -> float:
    levels = tuple(round(0.1 * index, 1) for index in range(1, 11))
    try:
        numeric = float(value if value is not None else default)
    except (TypeError, ValueError):
        numeric = float(default)
    numeric = max(levels[0], min(levels[-1], numeric))
    return min(levels, key=lambda level: abs(level - numeric))


def build_semantic_topology_marl_env(*args, **kwargs):
    """Factory kept in marl.py for backwards-compatible entry points."""
    from .marl_env import SemanticTopologyMARLEnv

    return SemanticTopologyMARLEnv(*args, **kwargs)


from .marl_env import MARLEnvConfig, SemanticTopologyMARLEnv
