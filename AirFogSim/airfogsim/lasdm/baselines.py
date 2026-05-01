from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model import LASDMServiceChain


@dataclass(frozen=True)
class LASDMBaselinePolicy:
    name: str
    description: str
    score_weights: Dict[str, float] = field(default_factory=dict)
    allowed_node_types: Optional[Tuple[str, ...]] = None
    keep_preferred_region: bool = True
    fixed_order: bool = False
    completion_latency_factor: float = 1.0

    def apply_to_chain(self, chain: LASDMServiceChain) -> LASDMServiceChain:
        baseline_chain = copy.deepcopy(chain)
        context = dict(baseline_chain.context)
        context["baseline"] = self.name
        if self.allowed_node_types is not None:
            context["allowed_node_types"] = list(self.allowed_node_types)
        elif "allowed_node_types" in context:
            context.pop("allowed_node_types")
        if not self.keep_preferred_region:
            context.pop("preferred_region_id", None)
        if self.fixed_order:
            context["fixed_order"] = True
        baseline_chain.context = context
        return baseline_chain


BASELINE_REGISTRY: Dict[str, LASDMBaselinePolicy] = {
    "fixed_sfc": LASDMBaselinePolicy(
        name="fixed_sfc",
        description="Deterministic declarative SFC placement with region preference retained.",
        score_weights={
            "health": 1.0,
            "reliability": 1.0,
            "accuracy": 1.0,
            "trust": 1.0,
            "region": 2.0,
            "resource_headroom": 0.1,
            "load": 0.1,
            "cold_start": 0.0,
        },
        fixed_order=True,
        completion_latency_factor=1.15,
    ),
    "centralized_greedy": LASDMBaselinePolicy(
        name="centralized_greedy",
        description="Global greedy service instance selection.",
        keep_preferred_region=False,
        completion_latency_factor=1.0,
    ),
    "deployment_only": LASDMBaselinePolicy(
        name="deployment_only",
        description="Fixed SFC composition with deployment/cold-start cost retained.",
        score_weights={
            "health": 1.0,
            "reliability": 1.0,
            "accuracy": 1.0,
            "trust": 1.0,
            "region": 1.5,
            "resource_headroom": 0.25,
            "load": 0.25,
            "cold_start": 0.5,
        },
        fixed_order=True,
        completion_latency_factor=1.1,
    ),
    "nearest_edge": LASDMBaselinePolicy(
        name="nearest_edge",
        description="Edge-biased greedy placement that keeps the preferred region.",
        score_weights={"region": 3.0, "load": 0.25, "resource_headroom": 0.25, "cold_start": 0.0},
        keep_preferred_region=True,
        completion_latency_factor=1.05,
    ),
    "cats_style": LASDMBaselinePolicy(
        name="cats_style",
        description="CATS-style score baseline emphasizing confidence, availability, trust, and service quality.",
        score_weights={
            "health": 1.5,
            "reliability": 1.5,
            "accuracy": 1.5,
            "trust": 1.5,
            "region": 0.5,
            "resource_headroom": 1.0,
            "load": 1.0,
            "cold_start": 0.1,
        },
        keep_preferred_region=False,
        completion_latency_factor=0.98,
    ),
    "regional_distributed_greedy": LASDMBaselinePolicy(
        name="regional_distributed_greedy",
        description="Greedy selection with preferred-region bias.",
        score_weights={"region": 2.0, "load": 1.0, "resource_headroom": 0.5},
        keep_preferred_region=True,
        completion_latency_factor=0.95,
    ),
    "edge_only": LASDMBaselinePolicy(
        name="edge_only",
        description="Infrastructure-only placement; UAV-hosted services are excluded.",
        allowed_node_types=("rsu", "cloud_server"),
        keep_preferred_region=True,
        completion_latency_factor=1.05,
    ),
    "uav_only": LASDMBaselinePolicy(
        name="uav_only",
        description="UAV-hosted placement baseline; non-UAV service instances are excluded.",
        allowed_node_types=("uav",),
        keep_preferred_region=True,
        completion_latency_factor=1.2,
    ),
    "proposed": LASDMBaselinePolicy(
        name="proposed",
        description="LASDM QoS-aware greedy policy with region, trust, reliability, and load scoring.",
        score_weights={
            "health": 2.0,
            "reliability": 2.0,
            "accuracy": 1.0,
            "trust": 1.0,
            "region": 0.1,
            "resource_headroom": 3.0,
            "load": 3.0,
            "cold_start": 0.4,
        },
        keep_preferred_region=True,
        completion_latency_factor=0.85,
    ),
    "lasdm_static": LASDMBaselinePolicy(
        name="lasdm_static",
        description="LASDM fixed-order declarative SFC baseline; compatibility alias for fixed_sfc.",
        score_weights={
            "health": 1.0,
            "reliability": 1.0,
            "accuracy": 1.0,
            "trust": 1.0,
            "region": 2.0,
            "resource_headroom": 0.1,
            "load": 0.1,
            "cold_start": 0.0,
        },
        fixed_order=True,
        completion_latency_factor=1.15,
    ),
    "lasdm_greedy": LASDMBaselinePolicy(
        name="lasdm_greedy",
        description="LASDM QoS-aware greedy orchestrator; compatibility alias for the proposed MVP policy.",
        score_weights={
            "health": 2.0,
            "reliability": 2.0,
            "accuracy": 1.0,
            "trust": 1.0,
            "region": 0.1,
            "resource_headroom": 3.0,
            "load": 3.0,
            "cold_start": 0.4,
        },
        keep_preferred_region=True,
        completion_latency_factor=0.85,
    ),
}


def baseline_names() -> List[str]:
    return list(BASELINE_REGISTRY.keys())


def get_baseline_policy(name: str) -> LASDMBaselinePolicy:
    try:
        return BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown LASDM baseline: {name}") from exc


def merge_score_weights(base_weights: Dict[str, float], policy: LASDMBaselinePolicy) -> Dict[str, float]:
    merged = dict(base_weights)
    merged.update(policy.score_weights)
    return merged


def validate_baselines(names: Sequence[str]) -> List[str]:
    invalid = [name for name in names if name not in BASELINE_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown LASDM baselines: {', '.join(invalid)}")
    return list(names)
