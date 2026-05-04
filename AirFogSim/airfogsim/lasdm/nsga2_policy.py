from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .marl_policy import BaseMARLPolicy, RESOURCE_LEVEL_VALUES, _resource_action_payload


@dataclass(frozen=True)
class Genome:
    candidate_indices: Tuple[int, ...]
    compute_bins: Tuple[int, ...]
    bandwidth_bins: Tuple[int, ...]


class NSGA2SemanticQoSPolicy(BaseMARLPolicy):
    """Per-chain NSGA-II semantic/QoS planner baseline."""

    def __init__(
        self,
        pop_size: int = 48,
        generations: int = 24,
        crossover_p: float = 0.90,
        mutation_p: float = 0.10,
        seed: int = 0,
    ):
        self.pop_size = max(8, int(pop_size))
        self.generations = max(1, int(generations))
        self.crossover_p = float(crossover_p)
        self.mutation_p = float(mutation_p)
        self.rng = random.Random(int(seed))

    def act(self, observations: Mapping[str, Mapping[str, Any]], deterministic: bool = True) -> Dict[str, Dict[str, Dict[str, str]]]:
        actions: Dict[str, Dict[str, Dict[str, str]]] = {}
        for agent_id, observation in observations.items():
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                ordered_sets = sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0))
                if not ordered_sets or any(not item.get("raw_candidates") for item in ordered_sets):
                    continue
                genome = self._optimize(ordered_sets)
                for node_idx, candidate_set in enumerate(ordered_sets):
                    candidates = list(candidate_set.get("raw_candidates", []) or [])
                    if not candidates:
                        continue
                    candidate = candidates[min(genome.candidate_indices[node_idx], len(candidates) - 1)]
                    compute = RESOURCE_LEVEL_VALUES[genome.compute_bins[node_idx]]
                    bandwidth = RESOURCE_LEVEL_VALUES[genome.bandwidth_bins[node_idx]]
                    actions.setdefault(str(agent_id), {}).setdefault(str(sfc_id), {})[
                        str(candidate_set.get("sfc_node_id"))
                    ] = _resource_action_payload(candidate.get("instance_id", ""), compute, bandwidth)
        return actions

    def _optimize(self, ordered_sets: Sequence[Mapping[str, Any]]) -> Genome:
        population = [self._random_genome(ordered_sets) for _ in range(self.pop_size)]
        for _ in range(self.generations):
            fitness = [self._evaluate(genome, ordered_sets) for genome in population]
            fronts = self._fast_non_dominated_sort(fitness)
            parents = [population[self._tournament(fitness, fronts)] for _ in range(self.pop_size)]
            offspring: List[Genome] = []
            for idx in range(0, len(parents), 2):
                first = parents[idx]
                second = parents[(idx + 1) % len(parents)]
                child_a, child_b = self._crossover(first, second, ordered_sets)
                offspring.append(self._mutate(child_a, ordered_sets))
                offspring.append(self._mutate(child_b, ordered_sets))
            combined = population + offspring
            combined_fitness = [self._evaluate(genome, ordered_sets) for genome in combined]
            fronts = self._fast_non_dominated_sort(combined_fitness)
            next_population: List[Genome] = []
            for front in fronts:
                if len(next_population) + len(front) <= self.pop_size:
                    next_population.extend(combined[index] for index in front)
                    continue
                distances = self._crowding_distance(front, combined_fitness)
                ranked = sorted(front, key=lambda index: distances.get(index, 0.0), reverse=True)
                next_population.extend(combined[index] for index in ranked[: self.pop_size - len(next_population)])
                break
            population = next_population[: self.pop_size]
        final_fitness = [self._evaluate(genome, ordered_sets) for genome in population]
        return population[min(range(len(population)), key=lambda index: self._reference_score(final_fitness[index]))]

    def _random_genome(self, ordered_sets: Sequence[Mapping[str, Any]]) -> Genome:
        candidates = []
        compute = []
        bandwidth = []
        for candidate_set in ordered_sets:
            count = max(1, len(candidate_set.get("raw_candidates", []) or []))
            candidates.append(self.rng.randrange(count))
            compute.append(self.rng.randrange(len(RESOURCE_LEVEL_VALUES)))
            bandwidth.append(self.rng.randrange(len(RESOURCE_LEVEL_VALUES)))
        return Genome(tuple(candidates), tuple(compute), tuple(bandwidth))

    def _evaluate(self, genome: Genome, ordered_sets: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        semantic_quality = 1.0
        latency = 0.0
        cost = 0.0
        reliability = 1.0
        violations = 0.0
        for node_idx, candidate_set in enumerate(ordered_sets):
            candidates = list(candidate_set.get("raw_candidates", []) or [])
            candidate = candidates[min(genome.candidate_indices[node_idx], len(candidates) - 1)]
            metadata = dict(candidate.get("metadata", {}) or {})
            semantic_quality *= max(0.0, min(1.0, float(candidate.get("semantic_score", 0.0) or 0.0)))
            latency += max(0.0, float(metadata.get("estimated_compute_s", 0.0) or 0.0))
            latency += max(0.0, float(metadata.get("route_tx_time_s", 0.0) or 0.0))
            latency += max(0.0, float(metadata.get("cold_start_s", 0.0) or 0.0))
            cost += 0.5 * RESOURCE_LEVEL_VALUES[genome.compute_bins[node_idx]]
            cost += 0.3 * RESOURCE_LEVEL_VALUES[genome.bandwidth_bins[node_idx]]
            cost += 0.2 * max(0.0, float(metadata.get("wireless_hops", 0.0) or 0.0))
            reliability *= max(0.0, min(1.0, float(candidate.get("reliability_score", 0.99) or 0.99)))
            if float(metadata.get("route_available", 1.0) or 0.0) <= 0.0:
                violations += 100.0
            if float(metadata.get("resource_available_slots", 1.0) or 0.0) <= 0.0:
                violations += 100.0
            deadline_slack = float(metadata.get("deadline_slack_s", 0.0) or 0.0)
            if deadline_slack < 0.0:
                violations += 10.0 + abs(deadline_slack)
        return {
            "semantic": -semantic_quality,
            "latency": latency,
            "cost": cost,
            "reliability": -reliability,
            "violations": violations,
        }

    def _dominates(self, left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        if left["violations"] != right["violations"]:
            return left["violations"] < right["violations"]
        objectives = ("semantic", "latency", "cost", "reliability")
        return all(left[key] <= right[key] for key in objectives) and any(left[key] < right[key] for key in objectives)

    def _fast_non_dominated_sort(self, fitness: Sequence[Mapping[str, float]]) -> List[List[int]]:
        dominates = {index: [] for index in range(len(fitness))}
        dominated_count = {index: 0 for index in range(len(fitness))}
        fronts: List[List[int]] = [[]]
        for left in range(len(fitness)):
            for right in range(len(fitness)):
                if left == right:
                    continue
                if self._dominates(fitness[left], fitness[right]):
                    dominates[left].append(right)
                elif self._dominates(fitness[right], fitness[left]):
                    dominated_count[left] += 1
            if dominated_count[left] == 0:
                fronts[0].append(left)
        cursor = 0
        while cursor < len(fronts) and fronts[cursor]:
            next_front: List[int] = []
            for left in fronts[cursor]:
                for right in dominates[left]:
                    dominated_count[right] -= 1
                    if dominated_count[right] == 0:
                        next_front.append(right)
            if next_front:
                fronts.append(next_front)
            cursor += 1
        return fronts

    def _crowding_distance(self, front: Sequence[int], fitness: Sequence[Mapping[str, float]]) -> Dict[int, float]:
        distance = {index: 0.0 for index in front}
        if len(front) <= 2:
            return {index: float("inf") for index in front}
        for key in ("semantic", "latency", "cost", "reliability"):
            ordered = sorted(front, key=lambda index: fitness[index][key])
            distance[ordered[0]] = float("inf")
            distance[ordered[-1]] = float("inf")
            span = fitness[ordered[-1]][key] - fitness[ordered[0]][key]
            if abs(span) < 1e-12:
                continue
            for idx in range(1, len(ordered) - 1):
                distance[ordered[idx]] += (fitness[ordered[idx + 1]][key] - fitness[ordered[idx - 1]][key]) / span
        return distance

    def _tournament(self, fitness: Sequence[Mapping[str, float]], fronts: Sequence[Sequence[int]]) -> int:
        rank = {index: rank_idx for rank_idx, front in enumerate(fronts) for index in front}
        first, second = self.rng.randrange(len(fitness)), self.rng.randrange(len(fitness))
        if rank.get(first, 10**6) != rank.get(second, 10**6):
            return first if rank.get(first, 10**6) < rank.get(second, 10**6) else second
        return first if self._reference_score(fitness[first]) <= self._reference_score(fitness[second]) else second

    def _crossover(self, first: Genome, second: Genome, ordered_sets: Sequence[Mapping[str, Any]]) -> Tuple[Genome, Genome]:
        if self.rng.random() > self.crossover_p or len(first.candidate_indices) <= 1:
            return first, second
        point = self.rng.randrange(1, len(first.candidate_indices))
        return (
            Genome(first.candidate_indices[:point] + second.candidate_indices[point:], first.compute_bins[:point] + second.compute_bins[point:], first.bandwidth_bins[:point] + second.bandwidth_bins[point:]),
            Genome(second.candidate_indices[:point] + first.candidate_indices[point:], second.compute_bins[:point] + first.compute_bins[point:], second.bandwidth_bins[:point] + first.bandwidth_bins[point:]),
        )

    def _mutate(self, genome: Genome, ordered_sets: Sequence[Mapping[str, Any]]) -> Genome:
        candidates = list(genome.candidate_indices)
        compute = list(genome.compute_bins)
        bandwidth = list(genome.bandwidth_bins)
        for index, candidate_set in enumerate(ordered_sets):
            if self.rng.random() < self.mutation_p:
                candidates[index] = self.rng.randrange(max(1, len(candidate_set.get("raw_candidates", []) or [])))
            if self.rng.random() < self.mutation_p:
                compute[index] = self.rng.randrange(len(RESOURCE_LEVEL_VALUES))
            if self.rng.random() < self.mutation_p:
                bandwidth[index] = self.rng.randrange(len(RESOURCE_LEVEL_VALUES))
        return Genome(tuple(candidates), tuple(compute), tuple(bandwidth))

    @staticmethod
    def _reference_score(fitness: Mapping[str, float]) -> float:
        return (
            4.0 * float(fitness["violations"])
            + 2.5 * float(fitness["semantic"])
            + 0.8 * float(fitness["latency"])
            + 0.3 * float(fitness["cost"])
            + 0.5 * float(fitness["reliability"])
        )
