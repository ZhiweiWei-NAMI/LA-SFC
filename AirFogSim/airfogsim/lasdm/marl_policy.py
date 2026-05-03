from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .graph_observation import flatten_observation


class BaseMARLPolicy:
    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        raise NotImplementedError


class SemanticGreedyPolicy(BaseMARLPolicy):
    """Selects the semantically highest-ranked visible candidate for each SFC node."""

    def __init__(
        self,
        prefer_local: bool = False,
        remote_penalty: float = 0.0,
        stale_penalty: float = 0.05,
        require_route_available: bool = False,
        prefer_runtime_feasible: bool = False,
    ):
        self.prefer_local = bool(prefer_local)
        self.remote_penalty = float(remote_penalty)
        self.stale_penalty = float(stale_penalty)
        self.require_route_available = bool(require_route_available)
        self.prefer_runtime_feasible = bool(prefer_runtime_feasible)

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        for agent_id, observation in observations.items():
            for candidate_set in observation.get("candidate_sets", []) or []:
                best = self._best_candidate(candidate_set)
                if best is None:
                    continue
                sfc_id = str(candidate_set.get("sfc_id"))
                sfc_node_id = str(candidate_set.get("sfc_node_id"))
                actions.setdefault(str(agent_id), {}).setdefault(sfc_id, {})[sfc_node_id] = best
        return actions

    def _best_candidate(self, candidate_set: Mapping[str, Any]) -> Optional[str]:
        best_id = None
        best_score = -float("inf")
        candidates = list(candidate_set.get("raw_candidates", []) or [])
        if self.require_route_available:
            reachable = [
                candidate
                for candidate in candidates
                if float(dict(candidate.get("metadata", {}) or {}).get("route_available", 0.0) or 0.0) > 0.0
            ]
            if reachable:
                candidates = reachable
        if self.prefer_runtime_feasible:
            feasible = [
                candidate
                for candidate in candidates
                if float(dict(candidate.get("metadata", {}) or {}).get("route_available", 0.0) or 0.0) > 0.0
                and float(dict(candidate.get("metadata", {}) or {}).get("deadline_slack_s", 0.0) or 0.0) >= 0.0
            ]
            if feasible:
                candidates = feasible
        for candidate in candidates:
            score = float(candidate.get("semantic_score", 0.0) or 0.0)
            metadata = dict(candidate.get("metadata", {}) or {})
            if candidate.get("is_remote"):
                score -= self.remote_penalty
            score -= self.stale_penalty * float(candidate.get("staleness_s", 0.0) or 0.0)
            if self.prefer_local and not candidate.get("is_remote"):
                score += 0.1
            if self.prefer_runtime_feasible:
                budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
                deadline_slack = float(metadata.get("deadline_slack_s", 0.0) or 0.0)
                if deadline_slack < 0.0:
                    score -= 0.50 * (1.0 + abs(deadline_slack) / budget)
                score -= 0.15 * float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0) / budget
                score -= 0.10 * float(metadata.get("route_tx_time_s", 0.0) or 0.0)
                score -= 0.50 * float(metadata.get("topology_risk", 0.0) or 0.0)
                score -= 0.75 * float(metadata.get("mobility_risk", 0.0) or 0.0)
                score -= 0.10 * float(metadata.get("wireless_hops", 0.0) or 0.0)
            if score > best_score:
                best_score = score
                best_id = str(candidate.get("instance_id"))
        return best_id


class TopologyGreedyPolicy(SemanticGreedyPolicy):
    """Greedy policy using the same observable runtime costs as the simulator."""

    def __init__(
        self,
        load_penalty: float = 0.5,
        uav_energy_penalty: float = 0.05,
        topology_risk_penalty: float = 0.5,
        mobility_risk_penalty: float = 0.2,
        route_hops_penalty: float = 0.0,
        route_tx_penalty: float = 0.0,
        route_unavailable_penalty: float = 2.0,
        cold_start_penalty: float = 0.0,
        deadline_violation_penalty: float = 0.0,
        runtime_penalty: float = 0.0,
        semantic_mismatch_penalty: float = 0.0,
        semantic_weight: float = 1.0,
        utility_prior_weight: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.load_penalty = float(load_penalty)
        self.uav_energy_penalty = float(uav_energy_penalty)
        self.topology_risk_penalty = float(topology_risk_penalty)
        self.mobility_risk_penalty = float(mobility_risk_penalty)
        self.route_hops_penalty = float(route_hops_penalty)
        self.route_tx_penalty = float(route_tx_penalty)
        self.route_unavailable_penalty = float(route_unavailable_penalty)
        self.cold_start_penalty = float(cold_start_penalty)
        self.deadline_violation_penalty = float(deadline_violation_penalty)
        self.runtime_penalty = float(runtime_penalty)
        self.semantic_mismatch_penalty = float(semantic_mismatch_penalty)
        self.semantic_weight = float(semantic_weight)
        self.utility_prior_weight = float(utility_prior_weight)

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        planned_node_load: Dict[str, int] = {}
        for agent_id, observation in observations.items():
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                ordered_sets = sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0))
                current_source = str(ordered_sets[0].get("source_node_id", "")) if ordered_sets else ""
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(ordered_sets)
                for candidate_set in ordered_sets:
                    best, best_candidate = self._best_candidate_with_context(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    if best is None:
                        break
                    sfc_node_id = str(candidate_set.get("sfc_node_id"))
                    actions.setdefault(str(agent_id), {}).setdefault(str(sfc_id), {})[sfc_node_id] = best
                    if best_candidate is not None:
                        node_id = str(best_candidate.get("node_id", ""))
                        if node_id:
                            current_source = node_id
                            planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                            remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, best_candidate)
                    if not current_source:
                        current_source = str(candidate_set.get("source_node_id", ""))
        return actions

    def _best_candidate(self, candidate_set: Mapping[str, Any]) -> Optional[str]:
        best, _ = self._best_candidate_with_context(
            candidate_set,
            str(candidate_set.get("source_node_id", "")),
            {},
        )
        return best

    def _best_candidate_with_context(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str = "",
        planned_node_load: Optional[Mapping[str, int]] = None,
        remaining_deadline_s: Optional[float] = None,
        total_deadline_s: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        best_id = None
        best_candidate: Optional[Mapping[str, Any]] = None
        best_score = -float("inf")
        source = str(source_node_id or candidate_set.get("source_node_id", ""))
        planned = dict(planned_node_load or {})
        for candidate in candidate_set.get("raw_candidates", []) or []:
            metadata = dict(candidate.get("metadata", {}) or {})
            semantic_score = float(candidate.get("semantic_score", 0.0) or 0.0)
            score = self.semantic_weight * semantic_score
            utility = _source_metric(metadata, "utility_prior", source, metadata.get("utility_prior", None))
            if utility is not None:
                score += self.utility_prior_weight * float(utility)
            route_available = float(_source_metric(metadata, "route_available", source, metadata.get("route_available", 0.0)) or 0.0)
            route_hops = float(_source_metric(metadata, "route_hops", source, metadata.get("route_hops", candidate.get("route_hops", 0.0))) or 0.0)
            route_tx_time = float(_source_metric(metadata, "route_tx_time_s", source, metadata.get("route_tx_time_s", 0.0)) or 0.0)
            deadline_slack = float(_source_metric(metadata, "deadline_slack_s", source, metadata.get("deadline_slack_s", 0.0)) or 0.0)
            expected_runtime_penalty = float(
                _source_metric(
                    metadata,
                    "expected_runtime_penalty_s",
                    source,
                    metadata.get("expected_runtime_penalty_s", 0.0),
                )
                or 0.0
            )
            node_id = str(candidate.get("node_id", ""))
            planned_count = max(0, int(planned.get(node_id, 0) or 0))
            task_cpu = max(0.0, float(metadata.get("task_cpu", 0.0) or 0.0))
            effective_cpu = max(0.1, float(metadata.get("effective_cpu", 0.1) or 0.1))
            max_concurrency = max(1, int(float(metadata.get("max_concurrency", 1) or 1)))
            current_load = max(0, int(float(metadata.get("current_load", 0) or 0)))
            available_slots, available_ratio = _available_slots_after_plan(metadata, node_id, planned)
            if available_slots <= 0:
                continue
            budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
            remaining_for_candidate = _candidate_remaining_deadline(metadata, source, remaining_deadline_s)
            expected_runtime_penalty = float(
                _source_metric(
                    metadata,
                    "expected_runtime_penalty_s",
                    source,
                    metadata.get("expected_runtime_penalty_s", 0.0),
                )
                or 0.0
            )
            if remaining_for_candidate is not None and expected_runtime_penalty > max(0.0, remaining_for_candidate) + 1e-9:
                continue
            score -= self.load_penalty * float(metadata.get("load_ratio", 0.0) or 0.0)
            score -= self.topology_risk_penalty * float(metadata.get("topology_risk", 0.0) or 0.0)
            score -= self.mobility_risk_penalty * float(metadata.get("mobility_risk", 0.0) or 0.0)
            score -= self.route_hops_penalty * route_hops
            score -= self.route_tx_penalty * route_tx_time
            if route_available <= 0.0:
                score -= self.route_unavailable_penalty
            score -= self.cold_start_penalty * float(metadata.get("cold_start_s", candidate.get("cold_start_s", 0.0)) or 0.0)
            score += 0.05 * available_ratio
            if deadline_slack < 0.0:
                score -= self.deadline_violation_penalty * (1.0 + abs(deadline_slack) / budget)
            score -= self.runtime_penalty * expected_runtime_penalty / budget
            if self.semantic_weight > 0.0:
                score -= self.semantic_mismatch_penalty * max(0.0, 1.0 - semantic_score) ** 2
            if candidate.get("node_type") == "uav":
                score -= self.uav_energy_penalty
            if candidate.get("is_remote"):
                score -= self.remote_penalty
            score -= self.stale_penalty * float(candidate.get("staleness_s", 0.0) or 0.0)
            if score > best_score:
                best_score = score
                best_id = str(candidate.get("instance_id"))
                best_candidate = candidate
        return best_id, best_candidate


def _source_metric(metadata: Mapping[str, Any], metric_name: str, source_node_id: str, default: Any = None) -> Any:
    by_source = metadata.get(f"{metric_name}_by_source")
    if isinstance(by_source, Mapping):
        source = str(source_node_id)
        if source in by_source:
            return by_source[source]
    return default


def _chain_deadline_budget(candidate_sets: Sequence[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    for candidate_set in candidate_sets:
        for candidate in candidate_set.get("raw_candidates", []) or []:
            metadata = dict(candidate.get("metadata", {}) or {})
            total = float(metadata.get("chain_deadline_s", 0.0) or 0.0)
            remaining = float(metadata.get("chain_remaining_deadline_s", total) or 0.0)
            if total > 0.0:
                return max(0.0, remaining), total
    return None, None


def _consume_deadline_budget(remaining_deadline_s: Optional[float], candidate: Optional[Mapping[str, Any]]) -> Optional[float]:
    if remaining_deadline_s is None or candidate is None:
        return remaining_deadline_s
    metadata = dict(candidate.get("metadata", {}) or {})
    spent = max(0.0, float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0))
    return max(0.0, float(remaining_deadline_s) - spent)


def _candidate_remaining_deadline(
    metadata: Mapping[str, Any],
    source_node_id: str,
    remaining_deadline_s: Optional[float],
) -> Optional[float]:
    if remaining_deadline_s is not None:
        return max(0.0, float(remaining_deadline_s))
    value = _source_metric(metadata, "chain_remaining_deadline_s", source_node_id, metadata.get("chain_remaining_deadline_s"))
    if value in (None, ""):
        return None
    return max(0.0, float(value))


def _available_slots_after_plan(
    metadata: Mapping[str, Any],
    node_id: str,
    planned_node_load: Mapping[str, int],
) -> Tuple[int, float]:
    max_concurrency = max(
        1,
        int(float(metadata.get("resource_capacity_total", metadata.get("max_concurrency", 1)) or 1)),
    )
    base_available_raw = metadata.get("resource_available_slots")
    if base_available_raw in (None, ""):
        current_load = max(0, int(float(metadata.get("current_load", 0) or 0)))
        reserved = max(0, int(float(metadata.get("resource_reserved_count", 0) or 0)))
        base_available = max(0, max_concurrency - current_load - reserved)
    else:
        base_available = max(0, int(float(base_available_raw) or 0))
    planned = max(0, int(planned_node_load.get(str(node_id), 0) or 0))
    available = max(0, base_available - planned)
    return available, max(0.0, min(1.0, float(available) / float(max_concurrency)))


class UtilityPriorPolicy(BaseMARLPolicy):
    """Training expert that follows the runtime-derived candidate utility prior."""

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        planned_node_load: Dict[str, int] = {}
        for agent_id, observation in observations.items():
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                ordered_sets = sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0))
                current_source = str(ordered_sets[0].get("source_node_id", "")) if ordered_sets else ""
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(ordered_sets)
                for candidate_set in ordered_sets:
                    best, best_candidate = self._best_candidate(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    if best is None:
                        break
                    sfc_node_id = str(candidate_set.get("sfc_node_id"))
                    actions.setdefault(str(agent_id), {}).setdefault(sfc_id, {})[sfc_node_id] = best
                    if best_candidate is not None:
                        current_source = str(best_candidate.get("node_id", current_source))
                        planned_node_load[current_source] = planned_node_load.get(current_source, 0) + 1
                        remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, best_candidate)
                    if not current_source:
                        current_source = str(candidate_set.get("source_node_id", ""))
        return actions

    def _best_candidate(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str = "",
        planned_node_load: Optional[Mapping[str, int]] = None,
        remaining_deadline_s: Optional[float] = None,
        total_deadline_s: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        best_id = None
        best_candidate: Optional[Mapping[str, Any]] = None
        best_score = -float("inf")
        source = str(source_node_id or candidate_set.get("source_node_id", ""))
        planned = dict(planned_node_load or {})
        for candidate in candidate_set.get("raw_candidates", []) or []:
            metadata = dict(candidate.get("metadata", {}) or {})
            utility = _source_metric(metadata, "utility_prior", source, metadata.get("utility_prior", None))
            if utility is None:
                continue
            score = float(utility)
            route_available = float(_source_metric(metadata, "route_available", source, metadata.get("route_available", 0.0)) or 0.0)
            node_id = str(candidate.get("node_id", ""))
            available_slots, available_ratio = _available_slots_after_plan(metadata, node_id, planned)
            if available_slots <= 0:
                continue
            budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
            expected_runtime_penalty = float(
                _source_metric(metadata, "expected_runtime_penalty_s", source, metadata.get("expected_runtime_penalty_s", 0.0))
                or 0.0
            )
            remaining_for_candidate = _candidate_remaining_deadline(metadata, source, remaining_deadline_s)
            if remaining_for_candidate is not None and expected_runtime_penalty > max(0.0, remaining_for_candidate) + 1e-9:
                continue
            score += 0.05 * available_ratio
            score += 0.01 * float(candidate.get("semantic_score", 0.0) or 0.0)
            score += 0.01 * route_available
            if score > best_score:
                best_score = score
                best_id = str(candidate.get("instance_id"))
                best_candidate = candidate
        return best_id, best_candidate


class IntraRegionOnlyPolicy(TopologyGreedyPolicy):
    """Single-region orchestration baseline: never selects exchanged candidates."""

    def _best_candidate_with_context(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str = "",
        planned_node_load: Optional[Mapping[str, int]] = None,
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        local_candidates = [
            candidate
            for candidate in candidate_set.get("raw_candidates", []) or []
            if not bool(candidate.get("is_remote"))
        ]
        if not local_candidates:
            return None, None
        scoped = dict(candidate_set)
        scoped["raw_candidates"] = local_candidates
        return super()._best_candidate_with_context(scoped, source_node_id, planned_node_load)


class CrossRegionAuctionPolicy(UtilityPriorPolicy):
    """Distributed auction-style baseline using utility bids plus remote prices."""

    def __init__(self, load_price: float = 0.8, remote_price: float = 0.15, stale_price: float = 0.10):
        self.load_price = float(load_price)
        self.remote_price = float(remote_price)
        self.stale_price = float(stale_price)

    def _best_candidate(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str = "",
        planned_node_load: Optional[Mapping[str, int]] = None,
        remaining_deadline_s: Optional[float] = None,
        total_deadline_s: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
        best_id = None
        best_candidate: Optional[Mapping[str, Any]] = None
        best_bid = -float("inf")
        source = str(source_node_id or candidate_set.get("source_node_id", ""))
        for candidate in candidate_set.get("raw_candidates", []) or []:
            metadata = dict(candidate.get("metadata", {}) or {})
            utility = _source_metric(metadata, "utility_prior", source, metadata.get("utility_prior", None))
            if utility is None:
                continue
            load_ratio = float(metadata.get("load_ratio", 0.0) or 0.0)
            route_hops = float(_source_metric(metadata, "route_hops", source, metadata.get("route_hops", 0.0)) or 0.0)
            route_available = float(_source_metric(metadata, "route_available", source, metadata.get("route_available", 0.0)) or 0.0)
            available_slots, available_ratio = _available_slots_after_plan(metadata, str(candidate.get("node_id", "")), planned_node_load or {})
            if available_slots <= 0:
                continue
            expected_runtime_penalty = float(
                _source_metric(metadata, "expected_runtime_penalty_s", source, metadata.get("expected_runtime_penalty_s", 0.0))
                or 0.0
            )
            remaining_for_candidate = _candidate_remaining_deadline(metadata, source, remaining_deadline_s)
            if remaining_for_candidate is not None and expected_runtime_penalty > max(0.0, remaining_for_candidate) + 1e-9:
                continue
            bid = float(utility)
            bid += 0.05 * available_ratio
            bid -= self.load_price * load_ratio
            bid -= self.remote_price * route_hops
            bid -= self.stale_price * float(candidate.get("staleness_s", 0.0) or 0.0)
            if route_available <= 0.0:
                bid -= 2.0
            if bid > best_bid:
                best_bid = bid
                best_id = str(candidate.get("instance_id"))
                best_candidate = candidate
        return best_id, best_candidate


class CentralizedPlannerPolicy(UtilityPriorPolicy):
    """Full-information centralized planner baseline.

    This is a deterministic search heuristic over the visible finite candidate
    sets. It is not a mathematical oracle and does not claim global optimality
    over the coupled AirFogSim runtime.
    """

    def __init__(self, branch_width: int = 12, **_: Any):
        super().__init__()
        self.branch_width = max(1, int(branch_width))

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        planned_node_load: Dict[str, int] = {}
        planned_wireless_load = 0.0
        for agent_id, observation in observations.items():
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                ordered_sets = sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0))
                chain_choice = self._best_chain_assignment(ordered_sets, planned_node_load, planned_wireless_load)
                if not chain_choice:
                    continue
                chain_actions = actions.setdefault(str(agent_id), {}).setdefault(sfc_id, {})
                for sfc_node_id, instance_id, node_id, wireless_hops in chain_choice:
                    chain_actions[str(sfc_node_id)] = str(instance_id)
                    if node_id:
                        planned_node_load[str(node_id)] = planned_node_load.get(str(node_id), 0) + 1
                    planned_wireless_load += max(0.0, float(wireless_hops))
        return actions

    def _best_chain_assignment(
        self,
        ordered_sets: Sequence[Mapping[str, Any]],
        planned_node_load: Mapping[str, int],
        planned_wireless_load: float = 0.0,
    ) -> List[Tuple[str, str, str, float]]:
        if not ordered_sets:
            return []
        best_score = -float("inf")
        best_path: List[Tuple[str, str, str, float]] = []
        initial_source = str(ordered_sets[0].get("source_node_id", ""))
        initial_planned = dict(planned_node_load or {})
        initial_remaining_deadline_s, _total_deadline_s = _chain_deadline_budget(ordered_sets)

        def search(
            index: int,
            source: str,
            planned: Dict[str, int],
            wireless_load: float,
            remaining_deadline_s: Optional[float],
            score: float,
            path: List[Tuple[str, str, str, float]],
        ) -> None:
            nonlocal best_score, best_path
            if index >= len(ordered_sets):
                if score > best_score:
                    best_score = score
                    best_path = list(path)
                return
            candidate_set = ordered_sets[index]
            contextual = self._contextual_candidate_set_for_planner(
                candidate_set,
                source,
                planned,
                wireless_load,
                remaining_deadline_s=remaining_deadline_s,
            )
            sfc_node_id = str(contextual.get("sfc_node_id", ""))
            candidates = list(contextual.get("raw_candidates", []) or [])
            if not candidates:
                return
            expanded = []
            for candidate in candidates:
                metadata = dict(candidate.get("metadata", {}) or {})
                if float(metadata.get("route_available", 0.0) or 0.0) <= 0.0:
                    continue
                utility = metadata.get("utility_prior")
                if utility is None:
                    continue
                deadline_slack = float(metadata.get("deadline_slack_s", 0.0) or 0.0)
                expected_penalty = float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0)
                wireless_time = float(metadata.get("expected_wireless_tx_time_s", metadata.get("route_tx_time_s", 0.0)) or 0.0)
                budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
                candidate_score = float(utility) - expected_penalty / budget - wireless_time / budget
                if deadline_slack < 0.0:
                    candidate_score -= 1.0 + abs(deadline_slack) / budget
                expanded.append((candidate_score, candidate))
            expanded.sort(key=lambda item: item[0], reverse=True)
            for utility, candidate in expanded[: self.branch_width]:
                node_id = str(candidate.get("node_id", ""))
                instance_id = str(candidate.get("instance_id", ""))
                metadata = dict(candidate.get("metadata", {}) or {})
                wireless_hops = max(0.0, float(metadata.get("wireless_hops", 0.0) or 0.0))
                next_remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, candidate)
                next_planned = dict(planned)
                if node_id:
                    next_planned[node_id] = next_planned.get(node_id, 0) + 1
                search(
                    index + 1,
                    node_id or source,
                    next_planned,
                    wireless_load + wireless_hops,
                    next_remaining_deadline_s,
                    score + utility,
                    path + [(sfc_node_id, instance_id, node_id, wireless_hops)],
                )

        search(0, initial_source, initial_planned, float(planned_wireless_load), initial_remaining_deadline_s, 0.0, [])
        return best_path

    def _contextual_candidate_set_for_planner(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str,
        planned_node_load: Mapping[str, int],
        planned_wireless_load: float = 0.0,
        remaining_deadline_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        source = str(source_node_id or candidate_set.get("source_node_id", ""))
        planned = dict(planned_node_load or {})
        raw_candidates: List[Dict[str, Any]] = []
        for candidate in [dict(item) for item in candidate_set.get("raw_candidates", []) or []]:
            metadata = dict(candidate.get("metadata", {}) or {})
            route_available = float(_source_metric(metadata, "route_available", source, metadata.get("route_available", 0.0)) or 0.0)
            route_hops = float(_source_metric(metadata, "route_hops", source, metadata.get("route_hops", 4.0)) or 0.0)
            utility = float(_source_metric(metadata, "utility_prior", source, metadata.get("utility_prior", 0.0)) or 0.0)
            deadline_slack = float(_source_metric(metadata, "deadline_slack_s", source, metadata.get("deadline_slack_s", 0.0)) or 0.0)
            expected_penalty = _source_metric(metadata, "expected_runtime_penalty_s", source, metadata.get("expected_runtime_penalty_s", None))
            if expected_penalty is not None:
                metadata["expected_runtime_penalty_s"] = float(expected_penalty)
            node_id = str(candidate.get("node_id", ""))
            wireless_hops = max(0.0, float(_source_metric(metadata, "wireless_hops", source, metadata.get("wireless_hops", 0.0)) or 0.0))
            route_tx_time_s = max(0.0, float(_source_metric(metadata, "route_tx_time_s", source, metadata.get("route_tx_time_s", 0.0)) or 0.0))
            effective_wireless_rb = max(1.0, float(metadata.get("effective_wireless_rb", 1.0) or 1.0))
            max_concurrency = max(1, int(float(metadata.get("max_concurrency", 1) or 1)))
            current_load = max(0, int(float(metadata.get("current_load", 0) or 0)))
            planned_count = max(0, int(planned.get(node_id, 0) or 0))
            available_slots, available_ratio = _available_slots_after_plan(metadata, node_id, planned)
            if available_slots <= 0:
                continue
            estimated_compute_s = max(0.0, float(metadata.get("estimated_compute_s", 0.0) or 0.0))
            budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
            planned_queue_delay_s = (float(planned_count) / float(max_concurrency)) * estimated_compute_s
            wireless_overload = max(0.0, float(planned_wireless_load) + wireless_hops - effective_wireless_rb)
            planned_wireless_delay_s = wireless_overload * max(1.0, route_tx_time_s)
            utility -= planned_queue_delay_s / budget
            utility -= planned_wireless_delay_s / budget
            expected_penalty = metadata.get("expected_runtime_penalty_s")
            if expected_penalty is not None:
                metadata["expected_runtime_penalty_s"] = float(expected_penalty) + planned_queue_delay_s + planned_wireless_delay_s
            remaining_for_candidate = _candidate_remaining_deadline(metadata, source, remaining_deadline_s)
            if (
                remaining_for_candidate is not None
                and float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0) > max(0.0, remaining_for_candidate) + 1e-9
            ):
                continue
            metadata["actual_route_source_node_id"] = source
            metadata["route_available"] = route_available
            metadata["route_hops"] = route_hops
            metadata["route_hops_norm"] = min(1.0, route_hops / 4.0)
            metadata["route_tx_time_s"] = route_tx_time_s
            metadata["wireless_hops"] = wireless_hops
            metadata["utility_prior"] = utility
            metadata["deadline_slack_s"] = deadline_slack
            metadata["planned_queue_delay_s"] = planned_queue_delay_s
            metadata["planned_wireless_delay_s"] = planned_wireless_delay_s
            metadata["load_ratio"] = min(1.0, float(current_load + planned_count) / float(max_concurrency))
            metadata["resource_available_slots"] = available_slots
            metadata["resource_available_ratio"] = available_ratio
            candidate["metadata"] = metadata
            raw_candidates.append(candidate)
        updated = dict(candidate_set)
        updated["raw_candidates"] = raw_candidates
        return updated


CentralizedOraclePolicy = CentralizedPlannerPolicy


class RandomValidPolicy(BaseMARLPolicy):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        for agent_id, observation in observations.items():
            for candidate_set in observation.get("candidate_sets", []) or []:
                ids = list(candidate_set.get("candidate_ids", []) or [])
                if not ids:
                    continue
                chosen = ids[0] if deterministic else self.rng.choice(ids)
                actions.setdefault(str(agent_id), {}).setdefault(str(candidate_set.get("sfc_id")), {})[
                    str(candidate_set.get("sfc_node_id"))
                ] = chosen
        return actions


@dataclass
class PolicyStep:
    actions: Dict[str, Dict[str, Dict[str, str]]]
    log_probs: Dict[str, float]
    values: Dict[str, float]
    log_prob_tensors: List[Any] = field(default_factory=list)
    value_tensors: List[Any] = field(default_factory=list)
    entropy_tensors: List[Any] = field(default_factory=list)


@dataclass
class PolicyEvaluation:
    log_prob_tensor: Any
    value_tensor: Any
    entropy_tensor: Any
    action_count: int


class IPPOPolicy(BaseMARLPolicy):
    """Candidate-choice actor with optional centralized context support.

    Execution remains decentralized: candidate actions are sampled from each
    agent's local observation.  MASAC reuses this actor surface and trains
    centralized Q critics around it.
    """

    def __init__(
        self,
        observation_dim: int,
        max_candidates: int = 16,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        seed: int = 0,
        candidate_feature_dim: int = 31,
        device: Optional[str] = None,
        route_unavailable_penalty: float = 20.0,
        utility_prior_logit_weight: float = 2.5,
        include_semantic_features: bool = True,
        include_topology_features: bool = True,
        include_temporal_features: bool = True,
        centralized_critic: bool = False,
        critic_observation_dim: Optional[int] = None,
        max_critic_agents: int = 4,
        use_region_encoder: bool = True,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 17,
        temporal_feature_dim: int = 8,
        learnable_prior: bool = True,
        prior_l2_coef: float = 1e-3,
        learned_logit_scale: float = 1.0,
        prior_logit_scale: float = 1.0,
        learnable_logit_blend: bool = False,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as exc:  # pragma: no cover - exercised only when torch is absent
            raise RuntimeError("Learned candidate policies require PyTorch. Use SemanticGreedyPolicy for no-torch smoke tests.") from exc

        self.torch = torch
        self.nn = nn
        self.observation_dim = int(observation_dim)
        self.max_candidates = int(max_candidates)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.route_unavailable_penalty = abs(float(route_unavailable_penalty))
        self.utility_prior_logit_weight = float(utility_prior_logit_weight)
        self.include_semantic_features = bool(include_semantic_features)
        self.include_topology_features = bool(include_topology_features)
        self.include_temporal_features = bool(include_temporal_features)
        self.centralized_critic = bool(centralized_critic)
        self.max_critic_agents = max(1, int(max_critic_agents))
        self.critic_observation_dim = int(critic_observation_dim or self.observation_dim)
        self.use_region_encoder = bool(use_region_encoder)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.learnable_prior = bool(learnable_prior)
        self.prior_l2_coef = float(prior_l2_coef)
        self.learned_logit_scale = float(learned_logit_scale)
        self.prior_logit_scale = float(prior_logit_scale)
        self.learnable_logit_blend = bool(learnable_logit_blend)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        prior_init = (
            1.5,
            self.utility_prior_logit_weight,
            -1.5,
            -0.25,
            -0.35,
            -0.80,
            -1.40,
            -1.80,
            -0.35,
        )

        class ActorCritic(nn.Module):
            temporal_keys = (
                "window_size",
                "time_span_s",
                "node_churn_rate",
                "edge_churn_rate",
                "mean_load_delta",
                "mean_speed_mps",
                "active_node_count",
                "active_edge_count",
            )

            def __init__(
                self,
                obs_dim: int,
                critic_dim: int,
                action_dim: int,
                hid: int,
                cand_dim: int,
                node_dim: int,
                edge_dim: int,
                temporal_dim: int,
                max_agents: int,
                prior_init_values: Sequence[float],
                learnable_prior_value: bool,
                learned_logit_scale_value: float,
                prior_logit_scale_value: float,
                learnable_logit_blend_value: bool,
            ):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.actor = nn.Linear(hid, action_dim)
                self.candidate_actor = nn.Sequential(nn.Linear(hid + cand_dim, hid), nn.Tanh(), nn.Linear(hid, 1))
                self.critic_body = nn.Sequential(nn.Linear(critic_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.critic = nn.Linear(hid, 1)
                self.hidden_dim = int(hid)
                self.node_dim = int(node_dim)
                self.edge_dim = int(edge_dim)
                self.temporal_dim = int(temporal_dim)
                self.max_agents = int(max_agents)
                prior_init_tensor = torch.tensor(list(prior_init_values), dtype=torch.float32)
                if bool(learnable_prior_value):
                    self.prior_feature_weights = nn.Parameter(prior_init_tensor.clone())
                else:
                    self.register_buffer("prior_feature_weights", prior_init_tensor.clone())
                self.register_buffer("prior_feature_init", prior_init_tensor.clone())
                learned_scale_init = torch.tensor(max(1e-4, float(learned_logit_scale_value)), dtype=torch.float32)
                prior_scale_init = torch.tensor(max(1e-4, float(prior_logit_scale_value)), dtype=torch.float32)
                learned_scale_raw = torch.log(torch.expm1(learned_scale_init))
                prior_scale_raw = torch.log(torch.expm1(prior_scale_init))
                if bool(learnable_logit_blend_value):
                    self.learned_logit_scale_raw = nn.Parameter(learned_scale_raw.clone())
                    self.prior_logit_scale_raw = nn.Parameter(prior_scale_raw.clone())
                else:
                    self.register_buffer("learned_logit_scale_raw", learned_scale_raw.clone())
                    self.register_buffer("prior_logit_scale_raw", prior_scale_raw.clone())
                self.register_buffer("learned_logit_scale_init", learned_scale_init.clone())
                self.register_buffer("prior_logit_scale_init", prior_scale_init.clone())
                self.node_encoder = nn.Sequential(nn.Linear(node_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.gnn_message_layers = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(hid * 3, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                        for _ in range(2)
                    ]
                )
                self.gnn_update_layers = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(hid * 2, hid), nn.Tanh(), nn.Linear(hid, hid))
                        for _ in range(2)
                    ]
                )
                self.gnn_norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(2)])
                self.remote_attention = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
                self.temporal_encoder = nn.Sequential(nn.Linear(temporal_dim, hid), nn.Tanh())
                self.region_context = nn.Sequential(nn.Linear(hid * 4, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.region_candidate_actor = nn.Sequential(nn.Linear(hid + cand_dim, hid), nn.Tanh(), nn.Linear(hid, 1))
                self.region_critic_body = nn.Sequential(nn.Linear(hid * max_agents, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
                self.region_critic = nn.Linear(hid, 1)

            def forward(self, x):
                return self.actor_logits(x), self.critic_value(x)

            def actor_logits(self, x):
                h = self.body(x)
                return self.actor(h)

            def critic_value(self, critic_x):
                h = self.critic_body(critic_x)
                return self.critic(h).squeeze(-1)

            def candidate_logits(self, x, candidate_features):
                h = self.body(x)
                if candidate_features.ndim == 1:
                    candidate_features = candidate_features.unsqueeze(0)
                h_expanded = h.expand(candidate_features.shape[0], -1)
                return self.candidate_actor(self.torch_cat([h_expanded, candidate_features], dim=-1)).squeeze(-1)

            def region_context_tensor(self, observation, device):
                import torch

                node_features = self._tensor_2d(observation.get("node_features", []), self.node_dim, device)
                node_mask = self._tensor_1d(observation.get("node_mask", []), node_features.shape[0], device)
                edge_features = self._tensor_2d(observation.get("edge_features", []), self.edge_dim, device)
                edge_mask = self._tensor_1d(observation.get("edge_mask", []), edge_features.shape[0], device)
                node_h = self.node_encoder(node_features)
                edge_h = self.edge_encoder(edge_features)
                edge_index = torch.as_tensor(observation.get("edge_index", []), dtype=torch.long, device=device)
                node_h = self._edge_conditioned_message_passing(node_h, edge_h, edge_index, edge_mask)
                if node_h.shape[0] == 0:
                    zero = torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)
                    temporal_h = self.temporal_encoder(self._temporal_tensor(observation, device))
                    return self.region_context(torch.cat([zero, zero, zero, temporal_h], dim=-1))
                local_col = min(node_features.shape[1] - 1, 12)
                local_mask = (node_features[:, local_col] > 0.5) & (node_mask > 0.0)
                active_mask = node_mask > 0.0
                if not bool(local_mask.any()):
                    local_mask = active_mask
                remote_mask = active_mask & (~local_mask)
                local_h = self._masked_mean(node_h, local_mask)
                if bool(remote_mask.any()):
                    query = local_h.view(1, 1, -1)
                    remote_seq = node_h[remote_mask].view(1, -1, self.hidden_dim)
                    remote_h = self.remote_attention(query, remote_seq, remote_seq, need_weights=False)[0].reshape(-1)
                else:
                    remote_h = torch.zeros_like(local_h)
                edge_summary = self._masked_mean(edge_h, edge_mask > 0.0) if edge_h.shape[0] else torch.zeros_like(local_h)
                temporal_h = self.temporal_encoder(self._temporal_tensor(observation, device))
                return self.region_context(torch.cat([local_h, remote_h, edge_summary, temporal_h], dim=-1))

            def _edge_conditioned_message_passing(self, node_h, edge_h, edge_index, edge_mask):
                import torch

                if (
                    node_h.shape[0] == 0
                    or edge_h.shape[0] == 0
                    or edge_index.ndim != 2
                    or edge_index.shape[0] != 2
                    or edge_index.shape[1] == 0
                ):
                    return node_h
                usable = min(edge_index.shape[1], edge_h.shape[0], edge_mask.numel())
                if usable <= 0:
                    return node_h
                valid = edge_mask[:usable] > 0.0
                if not bool(valid.any()):
                    return node_h
                src = edge_index[0, :usable].clamp(0, max(0, node_h.shape[0] - 1))[valid]
                dst = edge_index[1, :usable].clamp(0, max(0, node_h.shape[0] - 1))[valid]
                edge_valid = edge_h[:usable][valid]
                for message_layer, update_layer, norm in zip(self.gnn_message_layers, self.gnn_update_layers, self.gnn_norms):
                    fwd_input = torch.cat([node_h[src], node_h[dst], edge_valid], dim=-1)
                    rev_input = torch.cat([node_h[dst], node_h[src], edge_valid], dim=-1)
                    fwd_msg = message_layer(fwd_input)
                    rev_msg = message_layer(rev_input)
                    agg = torch.zeros_like(node_h)
                    degree = torch.zeros((node_h.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
                    agg.index_add_(0, dst, fwd_msg)
                    agg.index_add_(0, src, rev_msg)
                    ones = torch.ones((fwd_msg.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
                    degree.index_add_(0, dst, ones)
                    degree.index_add_(0, src, ones)
                    neighbor_h = agg / degree.clamp_min(1.0)
                    node_h = norm(node_h + update_layer(torch.cat([node_h, neighbor_h], dim=-1)))
                return node_h

            def region_candidate_logits(self, context, candidate_features):
                if candidate_features.ndim == 1:
                    candidate_features = candidate_features.unsqueeze(0)
                context_expanded = context.view(1, -1).expand(candidate_features.shape[0], -1)
                return self.region_candidate_actor(torch.cat([context_expanded, candidate_features], dim=-1)).squeeze(-1)

            def candidate_prior_logits(self, prior_features):
                return torch.matmul(prior_features, self.prior_feature_weights.to(dtype=prior_features.dtype))

            def blend_candidate_logits(self, learned_scores, prior_scores):
                learned_scale, prior_scale = self.logit_blend_scales()
                return learned_scale.to(dtype=learned_scores.dtype, device=learned_scores.device) * learned_scores + prior_scale.to(
                    dtype=prior_scores.dtype,
                    device=prior_scores.device,
                ) * prior_scores

            def logit_blend_scales(self):
                learned_scale = torch.nn.functional.softplus(self.learned_logit_scale_raw) + 1e-4
                prior_scale = torch.nn.functional.softplus(self.prior_logit_scale_raw) + 1e-4
                return learned_scale, prior_scale

            def prior_regularization_loss(self):
                weights = self.prior_feature_weights
                init = self.prior_feature_init.to(dtype=weights.dtype, device=weights.device)
                prior_loss = (weights - init).pow(2).mean()
                learned_scale, prior_scale = self.logit_blend_scales()
                learned_init = self.learned_logit_scale_init.to(dtype=learned_scale.dtype, device=learned_scale.device)
                prior_init = self.prior_logit_scale_init.to(dtype=prior_scale.dtype, device=prior_scale.device)
                blend_loss = (learned_scale - learned_init).pow(2) + (prior_scale - prior_init).pow(2)
                return prior_loss + 0.1 * blend_loss

            def logit_blend_snapshot(self):
                learned_scale, prior_scale = self.logit_blend_scales()
                return {
                    "learned": float(learned_scale.detach().cpu().item()),
                    "prior": float(prior_scale.detach().cpu().item()),
                    "initial_learned": float(self.learned_logit_scale_init.detach().cpu().item()),
                    "initial_prior": float(self.prior_logit_scale_init.detach().cpu().item()),
                }

            def region_critic_value(self, contexts):
                import torch

                padded = []
                for context in list(contexts)[: self.max_agents]:
                    padded.append(context.reshape(-1))
                while len(padded) < self.max_agents:
                    padded.append(torch.zeros((self.hidden_dim,), dtype=torch.float32, device=next(self.parameters()).device))
                critic_x = torch.cat(padded, dim=-1).unsqueeze(0)
                return self.region_critic(self.region_critic_body(critic_x)).squeeze(-1)

            def _temporal_tensor(self, observation, device):
                import torch

                temporal = observation.get("temporal_features", {}) or {}
                values = [float(temporal.get(key, 0.0) or 0.0) for key in self.temporal_keys]
                if len(values) < self.temporal_dim:
                    values.extend([0.0] * (self.temporal_dim - len(values)))
                return torch.tensor(values[: self.temporal_dim], dtype=torch.float32, device=device)

            def _tensor_2d(self, value, expected_dim, device):
                import torch

                tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
                if tensor.ndim == 0:
                    tensor = tensor.reshape(1, 1)
                if tensor.ndim == 1:
                    tensor = tensor.reshape(1, -1)
                if tensor.shape[-1] < expected_dim:
                    pad = torch.zeros((tensor.shape[0], expected_dim - tensor.shape[-1]), dtype=tensor.dtype, device=device)
                    tensor = torch.cat([tensor, pad], dim=-1)
                elif tensor.shape[-1] > expected_dim:
                    tensor = tensor[:, :expected_dim]
                return tensor

            def _tensor_1d(self, value, expected_len, device):
                import torch

                tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
                if tensor.numel() < expected_len:
                    pad = torch.zeros((expected_len - tensor.numel(),), dtype=tensor.dtype, device=device)
                    tensor = torch.cat([tensor, pad], dim=0)
                elif tensor.numel() > expected_len:
                    tensor = tensor[:expected_len]
                return tensor

            def _masked_mean(self, values, mask):
                import torch

                if values.shape[0] == 0:
                    return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=values.device)
                mask = mask.reshape(-1)
                if mask.numel() != values.shape[0] or not bool(mask.any()):
                    mask = torch.ones((values.shape[0],), dtype=torch.bool, device=values.device)
                weights = mask.to(dtype=values.dtype).unsqueeze(-1)
                return (values * weights).sum(dim=0) / weights.sum().clamp_min(1.0)

            @staticmethod
            def torch_cat(items, dim=-1):
                import torch

                return torch.cat(items, dim=dim)

        critic_dim = self.critic_observation_dim if self.centralized_critic else self.observation_dim
        self.model = ActorCritic(
            int(observation_dim),
            int(critic_dim),
            self.max_candidates,
            int(hidden_dim),
            self.candidate_feature_dim,
            self.node_feature_dim,
            self.edge_feature_dim,
            self.temporal_feature_dim,
            self.max_critic_agents,
            prior_init,
            self.learnable_prior,
            self.learned_logit_scale,
            self.prior_logit_scale,
            self.learnable_logit_blend,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))
        self.last_supervised_loss = 0.0
        self.last_supervised_samples = 0
        self.last_supervised_relabels = 0

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
        return self.act_with_logprobs(observations, deterministic=deterministic).actions

    def act_with_logprobs(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        deterministic: bool = False,
        track_grad: bool = False,
    ) -> PolicyStep:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        log_probs: Dict[str, float] = {}
        values: Dict[str, float] = {}
        log_prob_tensors: List[Any] = []
        value_tensors: List[Any] = []
        entropy_tensors: List[Any] = []
        torch = self.torch
        grad_context = torch.enable_grad() if track_grad else torch.no_grad()
        with grad_context:
            centralized_value = self._centralized_value_tensor(observations) if self.centralized_critic else None
            centralized_action_count = 0
            for agent_id, observation in observations.items():
                if self.use_region_encoder:
                    obs_tensor = self.model.region_context_tensor(observation, self.device)
                    logits = None
                    value = None if self.centralized_critic else self.model.region_critic_value([obs_tensor])
                else:
                    obs_tensor = torch.tensor(
                        self._fit_dim(flatten_observation(observation)),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    if self.centralized_critic:
                        logits = self.model.actor_logits(obs_tensor)
                        value = None
                    else:
                        logits, value = self.model(obs_tensor)
                if value is not None:
                    values[str(agent_id)] = float(value.item())
                agent_action_count = 0
                grouped: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
                for set_idx, candidate_set in enumerate(observation.get("candidate_sets", []) or []):
                    grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append((set_idx, candidate_set))
                for _sfc_id, items in grouped.items():
                    current_source = str(items[0][1].get("source_node_id", "")) if items else ""
                    planned_node_load: Dict[str, int] = {}
                    remaining_deadline_s, total_deadline_s = _chain_deadline_budget([item[1] for item in items])
                    for set_idx, candidate_set in sorted(items, key=lambda item: int(item[1].get("sfc_node_index", 0) or 0)):
                        contextual = self._contextual_candidate_set(
                            candidate_set,
                            current_source,
                            planned_node_load,
                            remaining_deadline_s=remaining_deadline_s,
                            total_deadline_s=total_deadline_s,
                        )
                        ids = list(contextual.get("candidate_ids", []) or [])
                        if not ids:
                            continue
                        mask_len = min(len(ids), self.max_candidates)
                        base_logits = None if logits is None else logits[0, :mask_len]
                        scores = self._candidate_scores(obs_tensor, base_logits, contextual, mask_len)
                        scores = self._apply_route_penalty(scores, contextual, mask_len)
                        if deterministic:
                            action_idx = int(torch.argmax(scores).item())
                            dist = torch.distributions.Categorical(logits=scores)
                        else:
                            dist = torch.distributions.Categorical(logits=scores)
                            action_idx = int(dist.sample().item())
                        log_prob_tensor = dist.log_prob(torch.tensor(action_idx, device=scores.device))
                        entropy_tensor = dist.entropy()
                        lp = float(log_prob_tensor.item())
                        log_prob_tensors.append(log_prob_tensor if track_grad else log_prob_tensor.detach())
                        entropy_tensors.append(entropy_tensor if track_grad else entropy_tensor.detach())
                        chosen = ids[action_idx]
                        actions.setdefault(str(agent_id), {}).setdefault(str(contextual.get("sfc_id")), {})[
                            str(contextual.get("sfc_node_id"))
                        ] = chosen
                        log_probs[f"{agent_id}:{set_idx}"] = lp
                        agent_action_count += 1
                        chosen_candidate = self._candidate_by_id(contextual, str(chosen))
                        if chosen_candidate is not None:
                            node_id = str(chosen_candidate.get("node_id", ""))
                            if node_id:
                                current_source = node_id
                                planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                                remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
                if agent_action_count > 0:
                    centralized_action_count += agent_action_count
                    if value is not None:
                        value_tensors.append(value.squeeze() if track_grad else value.squeeze().detach())
            if centralized_value is not None and centralized_action_count > 0:
                values["__global__"] = float(centralized_value.item())
                value_tensors.append(centralized_value.squeeze() if track_grad else centralized_value.squeeze().detach())
        return PolicyStep(
            actions=actions,
            log_probs=log_probs,
            values=values,
            log_prob_tensors=log_prob_tensors,
            value_tensors=value_tensors,
            entropy_tensors=entropy_tensors,
        )

    def evaluate_actions(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, Any],
        action_filter: Optional[Mapping[str, str]] = None,
    ) -> Optional[PolicyEvaluation]:
        log_prob_tensors: List[Any] = []
        value_tensors: List[Any] = []
        entropy_tensors: List[Any] = []
        action_count = 0
        torch = self.torch
        centralized_value = self._centralized_value_tensor(observations) if self.centralized_critic else None
        filter_agent = str(action_filter.get("agent_id", "")) if action_filter else ""
        filter_sfc = str(action_filter.get("sfc_id", "")) if action_filter else ""
        filter_node = str(action_filter.get("sfc_node_id", "")) if action_filter else ""
        for agent_id, observation in observations.items():
            if filter_agent and str(agent_id) != filter_agent:
                continue
            if self.use_region_encoder:
                obs_tensor = self.model.region_context_tensor(observation, self.device)
                logits = None
                value = None if self.centralized_critic else self.model.region_critic_value([obs_tensor])
            else:
                obs_tensor = torch.tensor(
                    self._fit_dim(flatten_observation(observation)),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                if self.centralized_critic:
                    logits = self.model.actor_logits(obs_tensor)
                    value = None
                else:
                    logits, value = self.model(obs_tensor)
            agent_action_count = 0
            grouped: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
            for set_idx, candidate_set in enumerate(observation.get("candidate_sets", []) or []):
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append((set_idx, candidate_set))
            for sfc_id, items in grouped.items():
                if filter_sfc and str(sfc_id) != filter_sfc:
                    continue
                current_source = str(items[0][1].get("source_node_id", "")) if items else ""
                planned_node_load: Dict[str, int] = {}
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget([item[1] for item in items])
                for _set_idx, candidate_set in sorted(items, key=lambda item: int(item[1].get("sfc_node_index", 0) or 0)):
                    contextual = self._contextual_candidate_set(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    sfc_node_id = str(contextual.get("sfc_node_id", ""))
                    chosen = self._chosen_action(actions, str(agent_id), str(sfc_id), sfc_node_id)
                    if not chosen:
                        continue
                    ids = list(contextual.get("candidate_ids", []) or [])
                    mask_len = min(len(ids), self.max_candidates)
                    if str(chosen) not in ids[:mask_len]:
                        continue
                    if not filter_node or sfc_node_id == filter_node:
                        action_idx = ids[:mask_len].index(str(chosen))
                        base_logits = None if logits is None else logits[0, :mask_len]
                        candidate_logits = self._candidate_scores(obs_tensor, base_logits, contextual, mask_len)
                        candidate_logits = self._apply_route_penalty(candidate_logits, contextual, mask_len)
                        dist = torch.distributions.Categorical(logits=candidate_logits)
                        target = torch.tensor(action_idx, device=candidate_logits.device)
                        log_prob_tensors.append(dist.log_prob(target))
                        entropy_tensors.append(dist.entropy())
                        action_count += 1
                        agent_action_count += 1
                    chosen_candidate = self._candidate_by_id(contextual, str(chosen))
                    if chosen_candidate is not None:
                        node_id = str(chosen_candidate.get("node_id", ""))
                        if node_id:
                            current_source = node_id
                            planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                            remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
            if agent_action_count > 0:
                if value is not None:
                    value_tensors.append(value.squeeze())
        if centralized_value is not None and action_count > 0:
            value_tensors.append(centralized_value.squeeze())
        if not log_prob_tensors or not value_tensors:
            return None
        return PolicyEvaluation(
            log_prob_tensor=sum(item.reshape(()) for item in log_prob_tensors) / float(len(log_prob_tensors)),
            value_tensor=sum(item.reshape(()) for item in value_tensors) / float(len(value_tensors)),
            entropy_tensor=sum(item.reshape(()) for item in entropy_tensors) / float(len(entropy_tensors)),
            action_count=action_count,
        )

    def _candidate_scores(self, obs_tensor: Any, logits: Any, candidate_set: Mapping[str, Any], mask_len: int) -> Any:
        """Return learned candidate logits blended with runtime-derived priors."""

        features = self.torch.tensor(
            np.asarray(candidate_set.get("candidate_features", []), dtype=np.float32)[:mask_len],
            dtype=self.torch.float32 if logits is None else logits.dtype,
            device=logits.device,
        ) if logits is not None else self.torch.tensor(
            np.asarray(candidate_set.get("candidate_features", []), dtype=np.float32)[:mask_len],
            dtype=obs_tensor.dtype,
            device=obs_tensor.device,
        )
        if features.numel() == 0:
            if logits is not None:
                return logits[:mask_len].clone()
            return self.torch.zeros((mask_len,), dtype=obs_tensor.dtype, device=obs_tensor.device)
        if features.shape[-1] < self.candidate_feature_dim:
            pad = self.torch.zeros((features.shape[0], self.candidate_feature_dim - features.shape[-1]), dtype=features.dtype, device=features.device)
            features = self.torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > self.candidate_feature_dim:
            features = features[:, : self.candidate_feature_dim]
        if self.use_region_encoder:
            learned_scores = self.model.region_candidate_logits(obs_tensor, features)
        else:
            learned_scores = self.model.candidate_logits(obs_tensor, features)
        prior_scores = self._candidate_prior_logits(candidate_set, mask_len, learned_scores)
        return self.model.blend_candidate_logits(learned_scores, prior_scores)

    def _candidate_prior_logits(self, candidate_set: Mapping[str, Any], mask_len: int, reference: Any) -> Any:
        rows: List[List[float]] = []
        for candidate in (candidate_set.get("raw_candidates", []) or [])[:mask_len]:
            metadata = dict(candidate.get("metadata", {}) or {})
            budget = max(1.0, float(metadata.get("function_budget_s", 1.0) or 1.0))
            deadline_slack = float(metadata.get("deadline_slack_s", 0.0) or 0.0)
            deadline_violation = max(0.0, -deadline_slack) / budget
            semantic_score = float(candidate.get("semantic_score", metadata.get("semantic_score", 0.0)) or 0.0)
            runtime_utility = float(metadata.get("utility_prior", 0.0) or 0.0) - semantic_score
            rows.append(
                [
                    semantic_score if self.include_semantic_features else 0.0,
                    runtime_utility if self.include_topology_features else 0.0,
                    deadline_violation if self.include_topology_features else 0.0,
                    float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0) / budget
                    if self.include_topology_features
                    else 0.0,
                    float(metadata.get("route_tx_time_s", 0.0) or 0.0) if self.include_topology_features else 0.0,
                    float(metadata.get("route_hops", 0.0) or 0.0) if self.include_topology_features else 0.0,
                    float(metadata.get("topology_risk", 0.0) or 0.0) if self.include_topology_features else 0.0,
                    float(metadata.get("mobility_risk", 0.0) or 0.0) if self.include_temporal_features else 0.0,
                    float(candidate.get("staleness_s", 0.0) or 0.0) if self.include_temporal_features else 0.0,
                ]
            )
        if not rows:
            return self.torch.zeros_like(reference[:mask_len])
        features = self.torch.tensor(rows, dtype=reference.dtype, device=reference.device)
        return self.model.candidate_prior_logits(features)

    def prior_regularization_loss(self) -> Any:
        if not hasattr(self.model, "prior_regularization_loss"):
            return None
        return self.model.prior_regularization_loss()

    def prior_weight_snapshot(self) -> Dict[str, Any]:
        if not hasattr(self.model, "prior_feature_weights"):
            return {}
        names = [
            "semantic",
            "runtime_utility",
            "deadline_violation",
            "runtime_penalty",
            "route_tx_time",
            "route_hops",
            "topology_risk",
            "mobility_risk",
            "staleness",
        ]
        weights = self.model.prior_feature_weights.detach().cpu().tolist()
        init = self.model.prior_feature_init.detach().cpu().tolist()
        snapshot = {
            "names": names,
            "initial": [float(item) for item in init],
            "learned": [float(item) for item in weights],
            "delta": [float(w - i) for w, i in zip(weights, init)],
        }
        if hasattr(self.model, "logit_blend_snapshot"):
            snapshot["logit_blend"] = self.model.logit_blend_snapshot()
        return snapshot

    def supervised_update(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        expert_actions: Mapping[str, Any],
        label_strategy: str = "expert",
        stale_relabel_margin: float = 0.05,
        deadline_relabel_margin: float = 0.05,
    ) -> int:
        """Behavior-clone one batch of expert candidate choices from observations."""

        torch = self.torch
        expert_actions, relabels = self._supervised_action_labels(
            observations,
            expert_actions,
            label_strategy=label_strategy,
            stale_relabel_margin=stale_relabel_margin,
            deadline_relabel_margin=deadline_relabel_margin,
        )
        self.last_supervised_relabels = relabels
        losses = []
        for agent_id, observation in observations.items():
            agent_payload = expert_actions.get(str(agent_id), {}) if isinstance(expert_actions, Mapping) else {}
            if not isinstance(agent_payload, Mapping):
                continue
            if self.use_region_encoder:
                obs_tensor = self.model.region_context_tensor(observation, self.device)
                logits = None
            else:
                obs_tensor = torch.tensor(
                    self._fit_dim(flatten_observation(observation)),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                logits = self.model.actor_logits(obs_tensor)
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                current_source = str(candidate_sets[0].get("source_node_id", "")) if candidate_sets else ""
                planned_node_load: Dict[str, int] = {}
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(candidate_sets)
                for candidate_set in sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0)):
                    candidate_set = self._contextual_candidate_set(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    sfc_node_id = str(candidate_set.get("sfc_node_id", ""))
                    chosen = None
                    if isinstance(agent_payload.get(sfc_id), Mapping):
                        chosen = agent_payload.get(sfc_id, {}).get(sfc_node_id)
                    if not chosen:
                        continue
                    ids = list(candidate_set.get("candidate_ids", []) or [])
                    if str(chosen) not in ids:
                        continue
                    mask_len = min(len(ids), self.max_candidates)
                    target_idx = ids[:mask_len].index(str(chosen)) if str(chosen) in ids[:mask_len] else None
                    if target_idx is None:
                        continue
                    base_logits = None if logits is None else logits[0, :mask_len]
                    candidate_logits = self._candidate_scores(obs_tensor, base_logits, candidate_set, mask_len)
                    candidate_logits = self._apply_route_penalty(candidate_logits, candidate_set, mask_len)
                    target = torch.tensor([target_idx], dtype=torch.long, device=candidate_logits.device)
                    losses.append(torch.nn.functional.cross_entropy(candidate_logits.unsqueeze(0), target))
                    chosen_candidate = self._candidate_by_id(candidate_set, str(chosen))
                    if chosen_candidate is not None:
                        node_id = str(chosen_candidate.get("node_id", ""))
                        if node_id:
                            current_source = node_id
                            planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                            remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
        if not losses:
            self.last_supervised_loss = 0.0
            self.last_supervised_samples = 0
            return 0
        loss = sum(losses) / len(losses)
        prior_l2 = self.prior_regularization_loss()
        if prior_l2 is not None and self.prior_l2_coef > 0.0:
            loss = loss + self.prior_l2_coef * prior_l2
        self.last_supervised_loss = float(loss.detach().cpu().item())
        self.last_supervised_samples = len(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return len(losses)

    def _supervised_action_labels(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        expert_actions: Mapping[str, Any],
        label_strategy: str,
        stale_relabel_margin: float,
        deadline_relabel_margin: float,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], int]:
        strategy = str(label_strategy or "expert")
        labels: Dict[str, Dict[str, Dict[str, str]]] = {}
        for agent_id, payload in dict(expert_actions or {}).items():
            if not isinstance(payload, Mapping):
                continue
            for sfc_id, assignments in payload.items():
                if isinstance(assignments, Mapping):
                    labels.setdefault(str(agent_id), {}).setdefault(str(sfc_id), {}).update(
                        {str(node_id): str(instance_id) for node_id, instance_id in assignments.items()}
                    )
        if strategy in {"expert", "none"}:
            return labels, 0
        relabels = 0
        for agent_id, observation in observations.items():
            agent_payload = labels.get(str(agent_id), {})
            if not isinstance(agent_payload, Mapping):
                continue
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                current_source = str(candidate_sets[0].get("source_node_id", "")) if candidate_sets else ""
                planned_node_load: Dict[str, int] = {}
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(candidate_sets)
                for candidate_set in sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0)):
                    contextual = self._contextual_candidate_set(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    sfc_node_id = str(contextual.get("sfc_node_id", ""))
                    current_label = agent_payload.get(sfc_id, {}).get(sfc_node_id) if isinstance(agent_payload.get(sfc_id), Mapping) else None
                    if current_label:
                        guarded = self._guarded_supervised_label(
                            contextual,
                            str(current_label),
                            strategy=strategy,
                            stale_relabel_margin=float(stale_relabel_margin),
                            deadline_relabel_margin=float(deadline_relabel_margin),
                        )
                        if guarded and guarded != str(current_label):
                            labels.setdefault(str(agent_id), {}).setdefault(str(sfc_id), {})[sfc_node_id] = guarded
                            relabels += 1
                        chosen_candidate = self._candidate_by_id(contextual, guarded or str(current_label))
                        if chosen_candidate is not None:
                            node_id = str(chosen_candidate.get("node_id", ""))
                            if node_id:
                                current_source = node_id
                                planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                                remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
        return labels, relabels

    def _guarded_supervised_label(
        self,
        candidate_set: Mapping[str, Any],
        current_label: str,
        strategy: str,
        stale_relabel_margin: float,
        deadline_relabel_margin: float,
    ) -> str:
        ids = list(candidate_set.get("candidate_ids", []) or [])[: self.max_candidates]
        raw_candidates = list(candidate_set.get("raw_candidates", []) or [])[: self.max_candidates]
        if current_label not in ids or not raw_candidates:
            return current_label
        labeled_candidate = self._candidate_by_id(candidate_set, current_label)
        if labeled_candidate is None:
            return current_label
        if strategy == "utility_best":
            return str(ids[max(range(len(raw_candidates)), key=lambda idx: self._bc_candidate_utility(raw_candidates[idx]))])
        label_utility = self._bc_candidate_utility(labeled_candidate)
        replacement = current_label
        replacement_utility = label_utility
        for candidate_id, candidate in zip(ids, raw_candidates):
            candidate_utility = self._bc_candidate_utility(candidate)
            if self._bc_route_available(candidate) <= 0.0:
                continue
            stale_guard = self._bc_candidate_stale(labeled_candidate) > 0.0 and self._bc_candidate_stale(candidate) <= 0.0
            deadline_guard = (
                self._bc_deadline_feasible(labeled_candidate) <= 0.0 and self._bc_deadline_feasible(candidate) > 0.0
            )
            if stale_guard and candidate_utility + float(stale_relabel_margin) >= label_utility:
                if candidate_utility >= replacement_utility - float(stale_relabel_margin):
                    replacement = str(candidate_id)
                    replacement_utility = candidate_utility
            if deadline_guard and candidate_utility + float(deadline_relabel_margin) >= label_utility:
                if candidate_utility >= replacement_utility - float(deadline_relabel_margin):
                    replacement = str(candidate_id)
                    replacement_utility = candidate_utility
        return replacement

    @staticmethod
    def _bc_candidate_utility(candidate: Mapping[str, Any]) -> float:
        metadata = dict(candidate.get("metadata", {}) or {})
        return float(metadata.get("utility_prior", candidate.get("utility_prior", 0.0)) or 0.0)

    @staticmethod
    def _bc_candidate_stale(candidate: Mapping[str, Any]) -> float:
        metadata = dict(candidate.get("metadata", {}) or {})
        semantic_group = str(metadata.get("semantic_group", candidate.get("semantic_group", "")) or "")
        stale = semantic_group == "stale_remote_candidates" or bool(candidate.get("stale", False))
        stale = stale or float(candidate.get("staleness_s", metadata.get("staleness_s", 0.0)) or 0.0) > 0.0
        return 1.0 if stale else 0.0

    @staticmethod
    def _bc_deadline_feasible(candidate: Mapping[str, Any]) -> float:
        metadata = dict(candidate.get("metadata", {}) or {})
        return 1.0 if float(metadata.get("deadline_slack_s", candidate.get("deadline_slack_s", 0.0)) or 0.0) >= 0.0 else 0.0

    @staticmethod
    def _bc_route_available(candidate: Mapping[str, Any]) -> float:
        metadata = dict(candidate.get("metadata", {}) or {})
        return 1.0 if float(metadata.get("route_available", candidate.get("route_available", 1.0)) or 0.0) > 0.0 else 0.0

    def _contextual_candidate_set(
        self,
        candidate_set: Mapping[str, Any],
        source_node_id: str,
        planned_node_load: Mapping[str, int],
        remaining_deadline_s: Optional[float] = None,
        total_deadline_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        source = str(source_node_id or candidate_set.get("source_node_id", ""))
        original_candidates = [dict(candidate) for candidate in candidate_set.get("raw_candidates", []) or []]
        original_ids = [str(item) for item in candidate_set.get("candidate_ids", []) or []]
        original_features = np.asarray(candidate_set.get("candidate_features", []), dtype=np.float32)
        if original_features.ndim != 2:
            original_features = np.zeros((self.max_candidates, self.candidate_feature_dim), dtype=np.float32)
        feature_dim = max(int(original_features.shape[1]), self.candidate_feature_dim)
        features = np.zeros((self.max_candidates, feature_dim), dtype=np.float32)
        candidate_mask = np.zeros((self.max_candidates,), dtype=np.float32)
        raw_candidates: List[Dict[str, Any]] = []
        candidate_ids: List[str] = []
        pruned_count = 0
        for idx, candidate in enumerate(original_candidates[: self.max_candidates]):
            metadata = dict(candidate.get("metadata", {}) or {})
            route_available = float(_source_metric(metadata, "route_available", source, metadata.get("route_available", 0.0)) or 0.0)
            route_hops = float(_source_metric(metadata, "route_hops", source, metadata.get("route_hops", 4.0)) or 0.0)
            utility = float(_source_metric(metadata, "utility_prior", source, metadata.get("utility_prior", 0.0)) or 0.0)
            deadline_slack = float(_source_metric(metadata, "deadline_slack_s", source, metadata.get("deadline_slack_s", 0.0)) or 0.0)
            expected_penalty = _source_metric(metadata, "expected_runtime_penalty_s", source, metadata.get("expected_runtime_penalty_s", None))
            if expected_penalty is not None:
                metadata["expected_runtime_penalty_s"] = float(expected_penalty)
            route_tx_time = float(_source_metric(metadata, "route_tx_time_s", source, metadata.get("route_tx_time_s", 0.0)) or 0.0)
            expected_rb_wait = float(_source_metric(metadata, "expected_rb_wait_s", source, metadata.get("expected_rb_wait_s", 0.0)) or 0.0)
            wireless_pressure = float(_source_metric(metadata, "wireless_pressure", source, metadata.get("wireless_pressure", 0.0)) or 0.0)
            wireless_hops = float(_source_metric(metadata, "wireless_hops", source, metadata.get("wireless_hops", 0.0)) or 0.0)
            rb_slowdown = float(_source_metric(metadata, "rb_slowdown", source, metadata.get("rb_slowdown", 1.0)) or 1.0)
            node_id = str(candidate.get("node_id", ""))
            available_slots, available_ratio = _available_slots_after_plan(metadata, node_id, planned_node_load)
            remaining_for_candidate = _candidate_remaining_deadline(metadata, source, remaining_deadline_s)
            expected_runtime_penalty = float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0)
            if available_slots <= 0 or (
                remaining_for_candidate is not None
                and expected_runtime_penalty > max(0.0, remaining_for_candidate) + 1e-9
            ):
                pruned_count += 1
                continue
            max_concurrency = max(
                1,
                int(float(metadata.get("resource_capacity_total", metadata.get("max_concurrency", 1)) or 1)),
            )
            load_ratio = min(1.0, 1.0 - available_ratio)
            total_deadline = float(total_deadline_s or metadata.get("chain_deadline_s", 0.0) or 0.0)
            remaining_ratio = (
                max(0.0, min(1.0, float(remaining_for_candidate) / total_deadline))
                if remaining_for_candidate is not None and total_deadline > 0.0
                else max(0.0, min(1.0, float(metadata.get("remaining_deadline_ratio", 1.0) or 0.0)))
            )
            route_hops_norm = min(1.0, route_hops / 4.0)

            metadata["actual_route_source_node_id"] = source
            metadata["route_available"] = route_available
            metadata["route_hops"] = route_hops
            metadata["route_hops_norm"] = route_hops_norm
            metadata["hops_from_prev_function"] = route_hops
            metadata["hops_from_prev_function_norm"] = route_hops_norm
            metadata["utility_prior"] = utility
            metadata["deadline_slack_s"] = deadline_slack
            metadata["route_tx_time_s"] = route_tx_time
            metadata["expected_rb_wait_s"] = expected_rb_wait
            metadata["wireless_pressure"] = wireless_pressure
            metadata["wireless_hops"] = wireless_hops
            metadata["rb_slowdown"] = rb_slowdown
            metadata["load_ratio"] = load_ratio
            metadata["resource_available_slots"] = available_slots
            metadata["resource_available_ratio"] = available_ratio
            metadata["resource_capacity_total"] = max_concurrency
            metadata["remaining_deadline_ratio"] = remaining_ratio
            metadata["chain_remaining_deadline_s"] = (
                float(remaining_for_candidate) if remaining_for_candidate is not None else metadata.get("chain_remaining_deadline_s", 0.0)
            )
            metadata["sequential_deadline_feasible"] = 1.0
            candidate["metadata"] = metadata
            new_idx = len(raw_candidates)
            raw_candidates.append(candidate)
            candidate_ids.append(original_ids[idx] if idx < len(original_ids) else str(candidate.get("instance_id", "")))
            candidate_mask[new_idx] = 1.0

            if idx < original_features.shape[0]:
                width = min(feature_dim, original_features.shape[1])
                features[new_idx, :width] = original_features[idx, :width]
            if features.shape[1] >= self.candidate_feature_dim:
                if self.include_topology_features:
                    feature_utility = utility
                    if not self.include_semantic_features:
                        feature_utility -= float(candidate.get("semantic_score", metadata.get("semantic_score", 0.0)) or 0.0)
                    features[new_idx, 5] = np.float32(load_ratio)
                    features[new_idx, 14] = np.float32(metadata["route_hops_norm"])
                    features[new_idx, 15] = np.float32(route_available)
                    features[new_idx, 18] = np.float32(max(-1.0, min(1.0, feature_utility)))
                    features[new_idx, 19] = np.float32(max(-1.0, min(1.0, deadline_slack / 20.0)))
                    features[new_idx, 20] = np.float32(min(1.0, route_tx_time / 20.0))
                    features[new_idx, 21] = np.float32(
                        min(
                            1.0,
                            float(
                                _source_metric(
                                    metadata,
                                    "expected_runtime_penalty_s",
                                    source,
                                    metadata.get("expected_runtime_penalty_s", 0.0),
                                )
                                or 0.0
                            )
                            / 40.0,
                        )
                    )
                    features[new_idx, 22] = np.float32(min(1.0, float(metadata.get("estimated_compute_s", 0.0) or 0.0) / 20.0))
                    features[new_idx, 23] = np.float32(1.0 if deadline_slack < 0.0 else 0.0)
                    features[new_idx, 26] = np.float32(min(1.0, wireless_hops / 4.0))
                    features[new_idx, 28] = np.float32(available_ratio)
                    features[new_idx, 29] = np.float32(route_hops_norm)
                    features[new_idx, 30] = np.float32(remaining_ratio)
                if self.include_temporal_features:
                    features[new_idx, 12] = np.float32(max(-1.0, min(1.0, float(metadata.get("mobility_risk", 0.0) or 0.0))))
                    features[new_idx, 24] = np.float32(min(1.0, expected_rb_wait / 20.0))
                    features[new_idx, 25] = np.float32(min(1.0, wireless_pressure / 8.0))
                    features[new_idx, 27] = np.float32(min(1.0, rb_slowdown / 8.0))
        updated = dict(candidate_set)
        updated["raw_candidates"] = raw_candidates
        updated["candidate_ids"] = candidate_ids
        updated["candidate_features"] = features
        updated["candidate_mask"] = candidate_mask
        updated["sequential_pruned_count"] = pruned_count
        return updated

    def _candidate_by_id(self, candidate_set: Mapping[str, Any], instance_id: str) -> Optional[Mapping[str, Any]]:
        for candidate in candidate_set.get("raw_candidates", []) or []:
            if str(candidate.get("instance_id", "")) == str(instance_id):
                return candidate
        return None

    def _apply_route_penalty(self, scores: Any, candidate_set: Mapping[str, Any], mask_len: int) -> Any:
        penalties = []
        for candidate in (candidate_set.get("raw_candidates", []) or [])[:mask_len]:
            metadata = dict(candidate.get("metadata", {}) or {})
            route_available = float(metadata.get("route_available", 0.0) or 0.0)
            penalties.append(self.route_unavailable_penalty if route_available <= 0.0 else 0.0)
        if not penalties:
            return scores
        penalty_tensor = self.torch.tensor(penalties, dtype=scores.dtype, device=scores.device)
        return scores - penalty_tensor

    @staticmethod
    def _chosen_action(actions: Mapping[str, Any], agent_id: str, sfc_id: str, sfc_node_id: str) -> Optional[str]:
        agent_payload = actions.get(agent_id, {}) if isinstance(actions, Mapping) else {}
        if not isinstance(agent_payload, Mapping):
            return None
        chain_payload = agent_payload.get(sfc_id, {})
        if not isinstance(chain_payload, Mapping):
            return None
        chosen = chain_payload.get(sfc_node_id)
        return str(chosen) if chosen else None

    def _centralized_value_tensor(self, observations: Mapping[str, Mapping[str, Any]]) -> Any:
        if self.use_region_encoder:
            contexts = []
            for agent_id in sorted(str(item) for item in observations):
                contexts.append(self.model.region_context_tensor(observations.get(agent_id, {}), self.device))
                if len(contexts) >= self.max_critic_agents:
                    break
            return self.model.region_critic_value(contexts).squeeze()
        torch_tensor = self.torch.tensor(
            self._fit_critic_dim(self._flatten_global_state(observations)),
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self.model.critic_value(torch_tensor).squeeze()

    def _flatten_global_state(self, observations: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for raw_agent_id, observation in sorted(dict(observations or {}).items(), key=lambda item: str(item[0])):
            agent_id = str(raw_agent_id)
            chunks.append(self._fit_dim(flatten_observation(observation)))
            if len(chunks) >= self.max_critic_agents:
                break
        while len(chunks) < self.max_critic_agents:
            chunks.append(np.zeros(self.observation_dim, dtype=np.float32))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _fit_critic_dim(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size == self.critic_observation_dim:
            return vector
        if vector.size > self.critic_observation_dim:
            return vector[: self.critic_observation_dim]
        padded = np.zeros(self.critic_observation_dim, dtype=np.float32)
        padded[: vector.size] = vector
        return padded

    def _fit_dim(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size == self.observation_dim:
            return vector
        if vector.size > self.observation_dim:
            return vector[: self.observation_dim]
        padded = np.zeros(self.observation_dim, dtype=np.float32)
        padded[: vector.size] = vector
        return padded


class MASACPolicy(IPPOPolicy):
    """Discrete-action MASAC policy for CTDE service-function placement.

    The actor is inherited from ``IPPOPolicy`` because the decentralized
    execution surface is already correct: each region chooses among its local
    candidate IDs.  Training is changed to off-policy discrete SAC with twin
    centralized Q critics over global region context, local region context, and
    candidate features.
    """

    def __init__(
        self,
        *args: Any,
        q_lr: Optional[float] = None,
        alpha: float = 0.05,
        auto_alpha: bool = False,
        alpha_lr: Optional[float] = None,
        target_entropy: Optional[float] = None,
        target_entropy_scale: float = 0.90,
        alpha_min: float = 0.005,
        alpha_max: float = 0.25,
        tau: float = 0.005,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)
        self.auto_alpha = bool(auto_alpha)
        self.alpha_min = max(1e-8, float(alpha_min))
        self.alpha_max = max(self.alpha_min, float(alpha_max))
        self.fixed_alpha = min(max(float(alpha), self.alpha_min), self.alpha_max)
        self.alpha_lr = float(alpha_lr if alpha_lr is not None else kwargs.get("lr", 3e-4))
        self.target_entropy = None if target_entropy is None else float(target_entropy)
        self.target_entropy_scale = max(0.0, float(target_entropy_scale))
        self.log_alpha = self.nn.Parameter(
            self.torch.tensor(math.log(self.fixed_alpha), dtype=self.torch.float32, device=self.device)
        )
        hidden_dim = int(getattr(self.model, "hidden_dim", 128))
        q_input_dim = hidden_dim * self.max_critic_agents + hidden_dim + self.candidate_feature_dim
        self.q1 = self._build_q_network(q_input_dim, hidden_dim).to(self.device)
        self.q2 = self._build_q_network(q_input_dim, hidden_dim).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)
        for module in (self.target_q1, self.target_q2):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.q_optimizer = self.torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(q_lr if q_lr is not None else kwargs.get("lr", 3e-4)),
        )
        self.alpha_optimizer = (
            self.torch.optim.Adam([self.log_alpha], lr=float(self.alpha_lr)) if self.auto_alpha else None
        )

    def _build_q_network(self, input_dim: int, hidden_dim: int) -> Any:
        return self.nn.Sequential(
            self.nn.Linear(int(input_dim), int(hidden_dim)),
            self.nn.ReLU(),
            self.nn.Linear(int(hidden_dim), int(hidden_dim)),
            self.nn.ReLU(),
            self.nn.Linear(int(hidden_dim), 1),
        )

    @property
    def alpha_tensor(self) -> Any:
        if self.auto_alpha:
            return self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)
        return self.torch.tensor(float(self.fixed_alpha), dtype=self.torch.float32, device=self.device)

    def _alpha_log_bounds(self) -> Tuple[float, float]:
        return math.log(self.alpha_min), math.log(self.alpha_max)

    def _target_entropy_for_count(self, candidate_count: int) -> float:
        if self.target_entropy is not None:
            return float(self.target_entropy)
        return float(self.target_entropy_scale) * math.log(max(2, int(candidate_count)))

    def update_alpha(self, entropy: Any, target_entropy: Any) -> Any:
        zero = self.torch.tensor(0.0, dtype=self.torch.float32, device=self.device)
        if not self.auto_alpha or self.alpha_optimizer is None:
            return zero
        alpha_loss = self.log_alpha * (entropy.detach() - target_entropy.detach())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with self.torch.no_grad():
            lower, upper = self._alpha_log_bounds()
            self.log_alpha.data.clamp_(lower, upper)
        return alpha_loss.detach()

    def sac_state_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "masac_discrete_ctde",
            "actor": self.model.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "alpha": float(self.alpha_tensor.detach().cpu().item()),
            "fixed_alpha": float(self.fixed_alpha),
            "auto_alpha": bool(self.auto_alpha),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "alpha_lr": float(self.alpha_lr),
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "target_entropy": self.target_entropy,
            "target_entropy_scale": float(self.target_entropy_scale),
            "tau": float(self.tau),
        }

    def load_sac_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        actor_state = state.get("actor", state) if isinstance(state, Mapping) else state
        self.model.load_state_dict(actor_state, strict=strict)
        if isinstance(state, Mapping) and "q1" in state:
            self.q1.load_state_dict(state["q1"], strict=True)
            self.q2.load_state_dict(state["q2"], strict=True)
            self.target_q1.load_state_dict(state.get("target_q1", state["q1"]), strict=True)
            self.target_q2.load_state_dict(state.get("target_q2", state["q2"]), strict=True)
            self.auto_alpha = bool(state.get("auto_alpha", self.auto_alpha))
            self.alpha_min = max(1e-8, float(state.get("alpha_min", self.alpha_min) or self.alpha_min))
            self.alpha_max = max(self.alpha_min, float(state.get("alpha_max", self.alpha_max) or self.alpha_max))
            self.alpha_lr = float(state.get("alpha_lr", self.alpha_lr) or self.alpha_lr)
            self.target_entropy = state.get("target_entropy", self.target_entropy)
            if self.target_entropy is not None:
                self.target_entropy = float(self.target_entropy)
            self.target_entropy_scale = float(
                state.get("target_entropy_scale", self.target_entropy_scale) or self.target_entropy_scale
            )
            self.fixed_alpha = min(
                max(float(state.get("fixed_alpha", state.get("alpha", self.fixed_alpha)) or self.fixed_alpha), self.alpha_min),
                self.alpha_max,
            )
            log_alpha = state.get("log_alpha", math.log(float(state.get("alpha", self.fixed_alpha) or self.fixed_alpha)))
            with self.torch.no_grad():
                self.log_alpha.data.fill_(float(log_alpha))
                lower, upper = self._alpha_log_bounds()
                self.log_alpha.data.clamp_(lower, upper)
            if self.auto_alpha and self.alpha_optimizer is None:
                self.alpha_optimizer = self.torch.optim.Adam([self.log_alpha], lr=float(self.alpha_lr))
            self.tau = float(state.get("tau", self.tau) or self.tau)

    def soft_update_targets(self, tau: Optional[float] = None) -> None:
        value = self.tau if tau is None else float(tau)
        with self.torch.no_grad():
            for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.data.mul_(1.0 - value).add_(source_param.data, alpha=value)

    def masac_selected_q_values(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, Any],
        detach_encoder: bool = True,
    ) -> Optional[Dict[str, Any]]:
        q1_values: List[Any] = []
        q2_values: List[Any] = []
        bundle, global_context = self._masac_context_bundle(observations)
        for agent_id, observation in observations.items():
            agent_key = str(agent_id)
            if agent_key not in bundle:
                continue
            actor_input, base_logits, local_context = bundle[agent_key]
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                current_source = str(candidate_sets[0].get("source_node_id", "")) if candidate_sets else ""
                planned_node_load: Dict[str, int] = {}
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(candidate_sets)
                for candidate_set in sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0)):
                    contextual = self._contextual_candidate_set(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    sfc_node_id = str(contextual.get("sfc_node_id", ""))
                    chosen = self._chosen_action(actions, agent_key, str(sfc_id), sfc_node_id)
                    ids = list(contextual.get("candidate_ids", []) or [])
                    mask_len = min(len(ids), self.max_candidates)
                    if not chosen or str(chosen) not in ids[:mask_len] or mask_len <= 0:
                        continue
                    action_idx = ids[:mask_len].index(str(chosen))
                    q1, q2 = self._masac_q_values(
                        global_context,
                        local_context,
                        contextual,
                        mask_len,
                        target=False,
                        detach_encoder=detach_encoder,
                    )
                    q1_values.append(q1[action_idx])
                    q2_values.append(q2[action_idx])
                    chosen_candidate = self._candidate_by_id(contextual, str(chosen))
                    if chosen_candidate is not None:
                        node_id = str(chosen_candidate.get("node_id", ""))
                        if node_id:
                            current_source = node_id
                            planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                            remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
        if not q1_values:
            return None
        return {
            "q1": self.torch.stack([item.reshape(()) for item in q1_values]).mean(),
            "q2": self.torch.stack([item.reshape(()) for item in q2_values]).mean(),
            "action_count": len(q1_values),
        }

    def masac_soft_state_value(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        target: bool = False,
        detach_encoder: bool = True,
    ) -> Tuple[Any, Any, int]:
        values: List[Any] = []
        entropies: List[Any] = []
        alpha = self.alpha_tensor
        for item in self._masac_candidate_items(observations, target=target, detach_encoder=detach_encoder):
            logits = item["logits"]
            log_probs = self.torch.log_softmax(logits, dim=-1)
            probs = self.torch.softmax(logits, dim=-1)
            q_min = self.torch.minimum(item["q1"], item["q2"])
            values.append((probs * (q_min - alpha * log_probs)).sum())
            entropies.append(-(probs * log_probs).sum())
        if not values:
            zero = self.torch.tensor(0.0, dtype=self.torch.float32, device=self.device)
            return zero, zero, 0
        return (
            self.torch.stack([item.reshape(()) for item in values]).mean(),
            self.torch.stack([item.reshape(()) for item in entropies]).mean(),
            len(values),
        )

    def masac_actor_loss(self, observations: Mapping[str, Mapping[str, Any]]) -> Tuple[Any, Any, Any, int]:
        losses: List[Any] = []
        entropies: List[Any] = []
        target_entropies: List[Any] = []
        alpha = self.alpha_tensor.detach()
        for item in self._masac_candidate_items(observations, target=False, detach_encoder=False):
            logits = item["logits"]
            log_probs = self.torch.log_softmax(logits, dim=-1)
            probs = self.torch.softmax(logits, dim=-1)
            q_min = self.torch.minimum(item["q1"], item["q2"]).detach()
            losses.append((probs * (alpha * log_probs - q_min)).sum())
            entropies.append(-(probs * log_probs).sum())
            target_entropies.append(
                self.torch.tensor(
                    self._target_entropy_for_count(int(logits.numel())),
                    dtype=self.torch.float32,
                    device=self.device,
                )
            )
        if not losses:
            zero = self.torch.tensor(0.0, dtype=self.torch.float32, device=self.device)
            return zero, zero, zero, 0
        return (
            self.torch.stack([item.reshape(()) for item in losses]).mean(),
            self.torch.stack([item.reshape(()) for item in entropies]).mean(),
            self.torch.stack([item.reshape(()) for item in target_entropies]).mean(),
            len(losses),
        )

    def _masac_candidate_items(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        target: bool = False,
        detach_encoder: bool = False,
    ) -> List[Dict[str, Any]]:
        items_out: List[Dict[str, Any]] = []
        bundle, global_context = self._masac_context_bundle(observations)
        for agent_id, observation in observations.items():
            agent_key = str(agent_id)
            if agent_key not in bundle:
                continue
            actor_input, base_logits, local_context = bundle[agent_key]
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append(candidate_set)
            for _sfc_id, candidate_sets in grouped.items():
                current_source = str(candidate_sets[0].get("source_node_id", "")) if candidate_sets else ""
                planned_node_load: Dict[str, int] = {}
                remaining_deadline_s, total_deadline_s = _chain_deadline_budget(candidate_sets)
                for candidate_set in sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0)):
                    contextual = self._contextual_candidate_set(
                        candidate_set,
                        current_source,
                        planned_node_load,
                        remaining_deadline_s=remaining_deadline_s,
                        total_deadline_s=total_deadline_s,
                    )
                    ids = list(contextual.get("candidate_ids", []) or [])
                    mask_len = min(len(ids), self.max_candidates)
                    if mask_len <= 0:
                        continue
                    logits_slice = None if base_logits is None else base_logits[0, :mask_len]
                    logits = self._candidate_scores(actor_input, logits_slice, contextual, mask_len)
                    logits = self._apply_route_penalty(logits, contextual, mask_len)
                    q1, q2 = self._masac_q_values(
                        global_context,
                        local_context,
                        contextual,
                        mask_len,
                        target=target,
                        detach_encoder=detach_encoder,
                    )
                    items_out.append({"logits": logits, "q1": q1, "q2": q2, "contextual": contextual})
                    chosen_idx = int(self.torch.argmax(logits.detach()).item())
                    chosen_candidate = self._candidate_by_id(contextual, ids[chosen_idx])
                    if chosen_candidate is not None:
                        node_id = str(chosen_candidate.get("node_id", ""))
                        if node_id:
                            current_source = node_id
                            planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
                            remaining_deadline_s = _consume_deadline_budget(remaining_deadline_s, chosen_candidate)
        return items_out

    def _masac_context_bundle(self, observations: Mapping[str, Mapping[str, Any]]) -> Tuple[Dict[str, Tuple[Any, Any, Any]], Any]:
        bundle: Dict[str, Tuple[Any, Any, Any]] = {}
        local_contexts: List[Any] = []
        for agent_id in sorted(str(item) for item in observations):
            observation = observations.get(agent_id, {})
            if self.use_region_encoder:
                actor_input = self.model.region_context_tensor(observation, self.device)
                base_logits = None
                local_context = actor_input
            else:
                actor_input = self.torch.tensor(
                    self._fit_dim(flatten_observation(observation)),
                    dtype=self.torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                base_logits = self.model.actor_logits(actor_input)
                local_context = self.model.body(actor_input).squeeze(0)
            bundle[agent_id] = (actor_input, base_logits, local_context)
            if len(local_contexts) < self.max_critic_agents:
                local_contexts.append(local_context.reshape(-1))
        hidden_dim = int(getattr(self.model, "hidden_dim", 128))
        while len(local_contexts) < self.max_critic_agents:
            local_contexts.append(self.torch.zeros((hidden_dim,), dtype=self.torch.float32, device=self.device))
        return bundle, self.torch.cat(local_contexts[: self.max_critic_agents], dim=-1)

    def _masac_q_values(
        self,
        global_context: Any,
        local_context: Any,
        candidate_set: Mapping[str, Any],
        mask_len: int,
        target: bool = False,
        detach_encoder: bool = False,
    ) -> Tuple[Any, Any]:
        features = self._masac_candidate_features(candidate_set, mask_len, global_context)
        if features.shape[0] <= 0:
            zero = self.torch.zeros((0,), dtype=global_context.dtype, device=global_context.device)
            return zero, zero
        global_expanded = global_context.reshape(1, -1).expand(features.shape[0], -1)
        local_expanded = local_context.reshape(1, -1).expand(features.shape[0], -1)
        q_input = self.torch.cat([global_expanded, local_expanded, features], dim=-1)
        if detach_encoder:
            q_input = q_input.detach()
        q1_net, q2_net = (self.target_q1, self.target_q2) if target else (self.q1, self.q2)
        return q1_net(q_input).squeeze(-1), q2_net(q_input).squeeze(-1)

    def _masac_candidate_features(self, candidate_set: Mapping[str, Any], mask_len: int, reference: Any) -> Any:
        array = np.asarray(candidate_set.get("candidate_features", []), dtype=np.float32)[:mask_len]
        features = self.torch.tensor(array, dtype=reference.dtype, device=reference.device)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.numel() == 0:
            return self.torch.zeros((0, self.candidate_feature_dim), dtype=reference.dtype, device=reference.device)
        if features.shape[-1] < self.candidate_feature_dim:
            pad = self.torch.zeros(
                (features.shape[0], self.candidate_feature_dim - features.shape[-1]),
                dtype=features.dtype,
                device=features.device,
            )
            features = self.torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > self.candidate_feature_dim:
            features = features[:, : self.candidate_feature_dim]
        return features


def policy_from_name(name: str, **kwargs: Any) -> BaseMARLPolicy:
    name = str(name)
    if name in {"semantic_greedy_with_exchange", "utility_prior_with_exchange"}:
        return UtilityPriorPolicy()
    if name == "intra_region_only":
        return IntraRegionOnlyPolicy(
            **_policy_kwargs(
                kwargs,
                "prefer_local",
                "remote_penalty",
                "stale_penalty",
                "load_penalty",
                "uav_energy_penalty",
                "topology_risk_penalty",
                "mobility_risk_penalty",
                "route_hops_penalty",
                "route_tx_penalty",
                "route_unavailable_penalty",
                "cold_start_penalty",
                "deadline_violation_penalty",
                "runtime_penalty",
                "semantic_mismatch_penalty",
            )
        )
    if name == "cross_region_auction":
        return CrossRegionAuctionPolicy(**_policy_kwargs(kwargs, "load_price", "remote_price", "stale_price"))
    if name == "semantic_greedy_no_exchange":
        return SemanticGreedyPolicy(
            require_route_available=True,
            prefer_runtime_feasible=True,
            **_policy_kwargs(kwargs, "prefer_local", "remote_penalty", "stale_penalty"),
        )
    if name == "semantic_greedy":
        return SemanticGreedyPolicy(**_policy_kwargs(kwargs, "prefer_local", "remote_penalty", "stale_penalty"))
    if name == "marl_semantic_no_topology":
        return SemanticGreedyPolicy(
            require_route_available=True,
            prefer_runtime_feasible=True,
            **_policy_kwargs(kwargs, "prefer_local", "remote_penalty", "stale_penalty"),
        )
    if name == "proposed_semantic_topology_marl":
        raise ValueError("proposed_semantic_topology_marl requires an explicit MASAC checkpoint loader; heuristic policy is not allowed")
    if name == "utility_prior":
        return UtilityPriorPolicy()
    if name in {"centralized_planner", "centralized_oracle"}:
        return CentralizedPlannerPolicy(
            **_policy_kwargs(
                kwargs,
                "branch_width",
                "prefer_local",
                "remote_penalty",
                "stale_penalty",
                "uav_energy_penalty",
                "load_penalty",
                "topology_risk_penalty",
                "mobility_risk_penalty",
                "route_hops_penalty",
                "route_tx_penalty",
                "route_unavailable_penalty",
                "cold_start_penalty",
                "deadline_violation_penalty",
                "runtime_penalty",
                "semantic_mismatch_penalty",
            )
        )
    if name == "topology_greedy":
        return TopologyGreedyPolicy(
            **_policy_kwargs(
                kwargs,
                "prefer_local",
                "remote_penalty",
                "stale_penalty",
                "load_penalty",
                "uav_energy_penalty",
                "topology_risk_penalty",
                "mobility_risk_penalty",
                "route_hops_penalty",
                "route_tx_penalty",
                "route_unavailable_penalty",
                "cold_start_penalty",
                "deadline_violation_penalty",
                "runtime_penalty",
                "semantic_mismatch_penalty",
                "utility_prior_weight",
            )
        )
    if name == "marl_topology_no_semantic":
        return TopologyGreedyPolicy(
            semantic_weight=0.0,
            **_policy_kwargs(
                kwargs,
                "prefer_local",
                "remote_penalty",
                "stale_penalty",
                "load_penalty",
                "uav_energy_penalty",
                "topology_risk_penalty",
                "mobility_risk_penalty",
                "route_hops_penalty",
                "route_tx_penalty",
                "route_unavailable_penalty",
                "cold_start_penalty",
                "deadline_violation_penalty",
                "runtime_penalty",
                "semantic_mismatch_penalty",
                "utility_prior_weight",
            )
        )
    if name == "marl_no_semantic":
        return TopologyGreedyPolicy(
            semantic_weight=0.0,
            **_policy_kwargs(
                kwargs,
                "prefer_local",
                "remote_penalty",
                "stale_penalty",
                "load_penalty",
                "uav_energy_penalty",
                "topology_risk_penalty",
                "mobility_risk_penalty",
                "route_hops_penalty",
                "route_tx_penalty",
                "route_unavailable_penalty",
                "cold_start_penalty",
                "deadline_violation_penalty",
                "runtime_penalty",
                "semantic_mismatch_penalty",
                "utility_prior_weight",
            )
        )
    if name == "random_valid":
        return RandomValidPolicy(seed=int(kwargs.get("seed", 0)))
    raise ValueError(f"Unknown policy name: {name}")


def _policy_kwargs(raw: Mapping[str, Any], *allowed: str) -> Dict[str, Any]:
    return {key: raw[key] for key in allowed if key in raw}
