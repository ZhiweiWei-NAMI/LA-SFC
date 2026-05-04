from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class SFCRewardConfig:
    success: float = 10.0
    timeout: float = -5.0
    gs2l_stage_progress: float = 6.0
    gs2l_stage_success: float = 0.8
    gs2l_stage_failure_discount: float = 0.15
    gs2l_bandwidth_price: float = 0.5
    gs2l_resource_price: float = 1.0
    gs2l_quality_weight: float = 0.5
    gs2l_clip: float = 8.0
    route_unavailable: float = -2.0
    deadline_slack: float = 0.05
    semantic_score: float = 0.50
    semantic_cumulative: float = 0.50
    utility_prior: float = 1.0
    stale_remote: float = -0.75
    topology_risk: float = -0.50
    mobility_risk: float = -0.50
    cold_start: float = -0.05
    runtime_penalty: float = -0.10
    load_imbalance: float = -0.25
    route_hops: float = -0.10
    route_tx_time: float = -0.05
    rb_wait: float = -0.05
    wireless_pressure: float = -0.05
    resource_available: float = 0.25
    remaining_deadline: float = 0.25
    dense_clip: float = 3.0


class SFCReward:
    """SFC completion reward plus selected-candidate dense shaping."""

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
    reward = _terminal_reward(previous_summary, current_summary, config)
    reward += _gs2l_stage_reward(aux, config)
    dense = _selected_candidate_dense_reward(aux, config)
    if (
        float(aux.get("gs2l_stage_failure_delta", 0.0) or 0.0) > 0.0
        and float(aux.get("gs2l_stage_success_delta", 0.0) or 0.0) <= 0.0
        and float(aux.get("gs2l_chain_progress_delta", 0.0) or 0.0) <= 0.0
    ):
        dense = min(0.0, dense)
    reward += dense
    return float(reward)


def _terminal_reward(
    previous_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
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


def _gs2l_stage_reward(aux: Mapping[str, Any], config: SFCRewardConfig) -> float:
    progress_delta = float(aux.get("gs2l_chain_progress_delta", 0.0) or 0.0)
    stage_success_delta = float(aux.get("gs2l_stage_success_delta", 0.0) or 0.0)
    stage_failure_delta = float(aux.get("gs2l_stage_failure_delta", 0.0) or 0.0)
    if progress_delta <= 0.0 and stage_success_delta <= 0.0 and stage_failure_delta <= 0.0:
        return 0.0
    stage_value = float(config.gs2l_bandwidth_price) + float(config.gs2l_resource_price)
    stage_count = max(1.0, float(aux.get("gs2l_stage_count_mean", 1.0) or 1.0))
    quality = _mean(
        [
            float(aux.get("selected_route_available_mean", 0.0) or 0.0),
            float(aux.get("selected_resource_available_ratio_mean", 0.0) or 0.0),
            float(aux.get("selected_remaining_deadline_ratio_mean", 0.0) or 0.0),
            float(aux.get("selected_semantic_cumulative_quality_mean", 0.0) or 0.0),
        ]
    )
    quality_scale = 1.0 + max(0.0, float(config.gs2l_quality_weight)) * max(0.0, min(1.0, quality))
    reward = float(config.gs2l_stage_progress) * progress_delta * stage_value * quality_scale
    reward += float(config.gs2l_stage_success) * stage_success_delta * stage_value * max(0.0, min(1.0, quality))
    reward -= float(config.gs2l_stage_failure_discount) * stage_failure_delta * stage_value * stage_count
    clip = max(0.0, float(config.gs2l_clip))
    if clip > 0.0:
        reward = max(-clip, min(clip, reward))
    return float(reward)


def _selected_candidate_dense_reward(aux: Mapping[str, Any], config: SFCRewardConfig) -> float:
    if float(aux.get("selected_semantic_group_count", 0.0) or 0.0) <= 0.0:
        return 0.0
    reward = 0.0
    reward += float(config.route_unavailable) * float(aux.get("route_unavailable_ratio", 0.0) or 0.0)
    reward += float(config.deadline_slack) * _positive_clip(float(aux.get("selected_deadline_slack_mean", 0.0) or 0.0), 20.0)
    reward += float(config.semantic_score) * float(aux.get("mean_semantic_top_score", 0.0) or 0.0)
    reward += float(config.semantic_cumulative) * float(aux.get("selected_semantic_cumulative_quality_mean", 0.0) or 0.0)
    reward += float(config.utility_prior) * float(aux.get("utility_prior", 0.0) or 0.0)
    reward += float(config.stale_remote) * float(aux.get("selected_stale_remote_ratio", 0.0) or 0.0)
    reward += float(config.topology_risk) * float(aux.get("selected_topology_risk_mean", 0.0) or 0.0)
    reward += float(config.mobility_risk) * float(aux.get("selected_mobility_risk_mean", 0.0) or 0.0)
    reward += float(config.cold_start) * _positive_clip(float(aux.get("cold_start_s", 0.0) or 0.0), 20.0)
    reward += float(config.runtime_penalty) * _positive_clip(
        float(aux.get("selected_expected_runtime_penalty_mean", 0.0) or 0.0),
        20.0,
    )
    reward += float(config.load_imbalance) * float(aux.get("load_imbalance", 0.0) or 0.0)
    reward += float(config.route_hops) * _positive_clip(float(aux.get("route_hops", 0.0) or 0.0), 4.0)
    reward += float(config.route_tx_time) * _positive_clip(float(aux.get("route_tx_time_s", 0.0) or 0.0), 20.0)
    reward += float(config.rb_wait) * _positive_clip(float(aux.get("expected_rb_wait_s", 0.0) or 0.0), 20.0)
    reward += float(config.wireless_pressure) * _positive_clip(float(aux.get("wireless_pressure", 0.0) or 0.0), 8.0)
    reward += float(config.resource_available) * float(aux.get("selected_resource_available_ratio_mean", 0.0) or 0.0)
    reward += float(config.remaining_deadline) * float(aux.get("selected_remaining_deadline_ratio_mean", 0.0) or 0.0)
    clip = max(0.0, float(config.dense_clip))
    if clip > 0.0:
        reward = max(-clip, min(clip, reward))
    return float(reward)


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


def _positive_clip(value: float, scale: float) -> float:
    return max(0.0, min(1.0, float(value) / max(1e-9, float(scale))))



def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
