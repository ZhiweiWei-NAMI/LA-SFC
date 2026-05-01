from __future__ import annotations

import random
from typing import List

import yaml

from airfogsim.service_orchestration.service_spec import (
    PayloadProfile,
    QoSProfile,
    ServiceCategory,
    ServiceIntent,
)


class StaticLowAltitudeIntentGenerator:
    def __init__(self, intents: List[ServiceIntent]):
        self.intents = list(intents)
        self._emitted = False

    @classmethod
    def from_yaml(cls, path: str) -> "StaticLowAltitudeIntentGenerator":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        intents = []
        for item in raw.get("intents", []):
            qos = item.get("qos", {})
            payload = item.get("payload", {})
            intents.append(
                ServiceIntent(
                    intent_id=item["intent_id"],
                    source_node_id=item["source_node_id"],
                    category=ServiceCategory(item["category"]),
                    required_capabilities=list(item.get("required_capabilities", [])),
                    payload=PayloadProfile(
                        semantic_type=payload.get("semantic_type", "any"),
                        input_mb=float(payload.get("input_mb", 0.0)),
                        output_ratio=float(payload.get("output_ratio", 1.0)),
                    ),
                    qos=QoSProfile(
                        deadline_s=float(qos.get("deadline_s", 1.0)),
                        priority=float(qos.get("priority", 1.0)),
                        reliability_min=float(qos.get("reliability_min", 0.0)),
                        accuracy_min=float(qos.get("accuracy_min", 0.0)),
                    ),
                    context=dict(item.get("context", {})),
                    sink_node_id=item.get("sink_node_id"),
                )
            )
        return cls(intents)

    def __call__(self, env) -> List[ServiceIntent]:
        if self._emitted:
            return []
        self._emitted = True
        return list(self.intents)


class PoissonLowAltitudeIntentGenerator:
    def __init__(self, arrival_prob: float = 0.2):
        self.arrival_prob = arrival_prob
        self.counter = 0

    def __call__(self, env) -> List[ServiceIntent]:
        if random.random() > self.arrival_prob:
            return []

        candidate_sources = list(getattr(env, "UAVs", {}).keys()) + list(getattr(env, "vehicles", {}).keys())
        if not candidate_sources:
            return []
        source = random.choice(candidate_sources)
        self.counter += 1

        scenario = random.choice([
            "urban_monitoring",
            "logistics_governance",
            "target_tracking_search",
            "c2_continuity_emergency",
        ])
        if scenario == "urban_monitoring":
            return [
                ServiceIntent(
                    intent_id=f"intent_{self.counter}",
                    source_node_id=source,
                    category=ServiceCategory.MISSION_APPLICATION,
                    required_capabilities=[
                        "video_preprocess",
                        "lightweight_inference",
                        "semantic_compression",
                        "situation_fusion",
                        "alert_publish",
                    ],
                    payload=PayloadProfile(semantic_type="video", input_mb=random.uniform(2.0, 8.0)),
                    qos=QoSProfile(deadline_s=random.uniform(1.0, 3.0), priority=0.8, accuracy_min=0.7),
                    context={"scenario": "urban_monitoring", "risk_level": random.uniform(0.1, 0.5)},
                    sink_node_id=source,
                )
            ]

        if scenario == "logistics_governance":
            return [
                ServiceIntent(
                    intent_id=f"intent_{self.counter}",
                    source_node_id=source,
                    category=ServiceCategory.AIRSPACE_GOVERNANCE,
                    required_capabilities=[
                        "remote_id",
                        "route_compliance",
                        "flight_authorization",
                        "conformance_monitoring",
                        "eta_update",
                    ],
                    payload=PayloadProfile(semantic_type="telemetry", input_mb=random.uniform(0.05, 0.2)),
                    qos=QoSProfile(deadline_s=random.uniform(0.2, 1.0), priority=1.0, reliability_min=0.99),
                    context={
                        "scenario": "logistics_governance",
                        "risk_level": random.uniform(0.3, 0.9),
                        "airspace_event": True,
                    },
                    sink_node_id=source,
                )
            ]

        if scenario == "target_tracking_search":
            return [
                ServiceIntent(
                    intent_id=f"intent_{self.counter}",
                    source_node_id=source,
                    category=ServiceCategory.SENSING_SITUATION,
                    required_capabilities=[
                        "video_preprocess",
                        "lightweight_inference",
                        "target_tracking",
                        "trajectory_replan",
                        "situation_fusion",
                        "alert_publish",
                    ],
                    payload=PayloadProfile(semantic_type="video", input_mb=random.uniform(4.0, 10.0)),
                    qos=QoSProfile(deadline_s=random.uniform(1.0, 2.5), priority=1.0, reliability_min=0.95),
                    context={
                        "scenario": "target_tracking_search",
                        "risk_level": random.uniform(0.7, 1.0),
                        "target_lost": True,
                    },
                    sink_node_id=source,
                )
            ]

        return [
            ServiceIntent(
                intent_id=f"intent_{self.counter}",
                source_node_id=source,
                category=ServiceCategory.CONTINUITY_RESILIENCE,
                required_capabilities=[
                    "video_preprocess",
                    "lightweight_inference",
                    "c2_monitoring",
                    "relay_select",
                    "object_verification",
                    "checkpoint",
                    "resilience",
                ],
                payload=PayloadProfile(semantic_type="video", input_mb=random.uniform(4.0, 12.0)),
                qos=QoSProfile(deadline_s=random.uniform(0.8, 2.0), priority=1.0, reliability_min=0.95),
                context={
                    "scenario": "c2_continuity_emergency",
                    "risk_level": random.uniform(0.7, 1.0),
                    "link_degraded": True,
                    "shared_prefix_capabilities": ["video_preprocess"],
                    "parallel_capability_groups": [
                        ["lightweight_inference", "object_verification"],
                        ["c2_monitoring", "relay_select"],
                    ],
                    "fusion_capabilities": ["checkpoint", "resilience"],
                },
                sink_node_id=source,
            )
        ]
