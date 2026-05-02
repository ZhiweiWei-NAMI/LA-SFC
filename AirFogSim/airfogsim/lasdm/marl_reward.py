from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class SFCRewardConfig:
    success: float = 1.0
    timeout: float = 0.0


class SFCReward:
    """Basic completion success-rate reward for SFC-oriented MARL."""

    def __init__(self, config: Optional[SFCRewardConfig] = None):
        self.config = config or SFCRewardConfig()

    def __call__(
        self,
        previous_summary: Mapping[str, Any],
        current_summary: Mapping[str, Any],
        aux: Optional[Mapping[str, Any]] = None,
    ) -> float:
        return compute_sfc_reward(previous_summary, current_summary, aux or {}, self.config)


def compute_sfc_reward(
    previous_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    aux: Mapping[str, Any],
    config: SFCRewardConfig,
) -> float:
    success_delta = _delta(previous_summary, current_summary, "succeeded")
    failed_delta = _delta(previous_summary, current_summary, "failed")
    timeout_delta = _delta(previous_summary, current_summary, "timed_out")
    completed_delta = success_delta + failed_delta + timeout_delta
    if completed_delta <= 0.0:
        return 0.0
    success_rate = success_delta / completed_delta
    failure_rate = (failed_delta + timeout_delta) / completed_delta
    return float(config.success * success_rate + config.timeout * failure_rate)


def reward_aux_from_observations(observations: Mapping[str, Mapping[str, Any]], overhead_bytes: float = 0.0) -> Dict[str, float]:
    top_scores = []
    stale_ratios = []
    churn_values = []
    for observation in observations.values():
        for candidate_set in observation.get("candidate_sets", []) or []:
            raw = candidate_set.get("raw_candidates", []) or []
            if raw:
                top_scores.append(float(raw[0].get("semantic_score", 0.0) or 0.0))
                stale_values = [float(item.get("staleness_s", 0.0) or 0.0) for item in raw if item.get("is_remote")]
                if stale_values:
                    stale_ratios.append(sum(1.0 for value in stale_values if value > 0.0) / len(stale_values))
        temporal = observation.get("temporal_features", {}) or {}
        if "edge_churn_rate" in temporal:
            churn_values.append(float(temporal["edge_churn_rate"]))
    return {
        "mean_semantic_top_score": _mean(top_scores),
        "stale_remote_ratio": _mean(stale_ratios),
        "message_overhead_bytes": float(overhead_bytes),
        "topology_churn": _mean(churn_values),
        "load_imbalance": 0.0,
    }


def _delta(previous: Mapping[str, Any], current: Mapping[str, Any], key: str) -> float:
    return float(current.get(key, 0.0) or 0.0) - float(previous.get(key, 0.0) or 0.0)



def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
