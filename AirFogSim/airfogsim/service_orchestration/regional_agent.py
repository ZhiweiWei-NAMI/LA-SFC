from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class CapabilityAdvertisement:
    region_id: str
    timestamp: float
    service_ids: Set[str]
    node_load: Dict[str, float]
    avg_latency_to_region: Dict[str, float] = field(default_factory=dict)


@dataclass
class ServiceProposal:
    src_region: str
    dst_region: str
    graph_id: str
    ms_id: str
    expected_payload_mb: float
    deadline_s: float
    utility: float = 0.0


class RegionalServiceAgent:
    """Regional heuristic agent for distributed service orchestration."""

    def __init__(self, region_id: str, managed_nodes: List[str]):
        self.region_id = region_id
        self.managed_nodes = set(managed_nodes)
        self.neighbor_ads: Dict[str, CapabilityAdvertisement] = {}

    def observe_local_state(self, env, deployment_registry, catalog=None) -> CapabilityAdvertisement:
        service_ids: Set[str] = set()
        node_load: Dict[str, float] = {}
        for node_id in self.managed_nodes:
            service_ids |= deployment_registry.deployed.get(node_id, set())
            if catalog is not None:
                for spec in catalog.all():
                    if deployment_registry.can_host(env, node_id, spec):
                        service_ids.add(spec.ms_id)
            try:
                node_load[node_id] = len(env.task_manager.getToComputeTasks(node_id))
            except Exception:
                node_load[node_id] = 0.0
        return CapabilityAdvertisement(
            region_id=self.region_id,
            timestamp=env.simulation_time,
            service_ids=service_ids,
            node_load=node_load,
        )

    def receive_advertisement(self, ad: CapabilityAdvertisement) -> None:
        self.neighbor_ads[ad.region_id] = ad

    def propose_remote_serving(
        self,
        graph_id: str,
        ms_id: str,
        expected_payload_mb: float,
        deadline_s: float,
    ) -> List[ServiceProposal]:
        proposals: List[ServiceProposal] = []
        for region_id, ad in self.neighbor_ads.items():
            if ms_id not in ad.service_ids:
                continue
            avg_load = sum(ad.node_load.values()) / max(1, len(ad.node_load))
            utility = -avg_load - 0.01 * expected_payload_mb + 1.0 / max(deadline_s, 1e-6)
            proposals.append(
                ServiceProposal(
                    src_region=self.region_id,
                    dst_region=region_id,
                    graph_id=graph_id,
                    ms_id=ms_id,
                    expected_payload_mb=expected_payload_mb,
                    deadline_s=deadline_s,
                    utility=utility,
                )
            )
        return sorted(proposals, key=lambda proposal: proposal.utility, reverse=True)
