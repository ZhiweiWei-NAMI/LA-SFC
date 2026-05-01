from __future__ import annotations

import uuid
from typing import Dict, List, Sequence, Tuple

from .service_catalog import ServiceCatalog
from .service_spec import MicroserviceSpec, ServiceGraphSpec, ServiceIntent


class RuleBasedServiceMatcher:
    """Rule-based task-intent to service-graph matcher."""

    def __init__(self, catalog: ServiceCatalog, policy: str = "adaptive_rule_based"):
        self.catalog = catalog
        self.policy = policy

    def _choose_candidate(
        self,
        capability: str,
        current_semantic: str,
        intent: ServiceIntent,
    ) -> MicroserviceSpec:
        candidates = self.catalog.find_by_capability(capability)
        if not candidates:
            raise ValueError(f"No microservice provides capability: {capability}")
        ranked = sorted(
            candidates,
            key=lambda ms: (
                ms.input_semantic not in [current_semantic, "any"],
                ms.input_semantic != intent.payload.semantic_type and ms.input_semantic != "any",
                ms.output_ratio,
                ms.cpu_per_mb,
            ),
        )
        return ranked[0]

    def _build_linear_chain(
        self,
        capabilities: Sequence[str],
        start_semantic: str,
        intent: ServiceIntent,
        nodes: Dict[str, MicroserviceSpec],
        edges: List[Tuple[str, str]],
    ) -> Tuple[List[str], str]:
        chain: List[str] = []
        current_semantic = start_semantic
        previous_node_id = None
        for capability in capabilities:
            spec = self._choose_candidate(capability, current_semantic, intent)
            node_id = spec.ms_id
            nodes[node_id] = spec
            if previous_node_id is not None:
                edges.append((previous_node_id, node_id))
            chain.append(node_id)
            previous_node_id = node_id
            current_semantic = spec.output_semantic
        return chain, current_semantic

    def match(self, intent: ServiceIntent) -> ServiceGraphSpec:
        nodes: Dict[str, MicroserviceSpec] = {}
        edges: List[Tuple[str, str]] = []

        if self.policy == "fixed_sfc":
            selected_chain, _ = self._build_linear_chain(
                intent.required_capabilities,
                intent.payload.semantic_type,
                intent,
                nodes,
                edges,
            )
        else:
            prefix_capabilities = list(intent.context.get("shared_prefix_capabilities", []))
            if not prefix_capabilities:
                prefix_capabilities = list(intent.required_capabilities)
            prefix_chain, prefix_semantic = self._build_linear_chain(
                prefix_capabilities,
                intent.payload.semantic_type,
                intent,
                nodes,
                edges,
            )
            selected_chain = list(prefix_chain)

            parallel_groups = list(intent.context.get("parallel_capability_groups", []))
            branch_ends: List[str] = []
            if parallel_groups:
                branch_root = prefix_chain[-1] if prefix_chain else None
                for group in parallel_groups:
                    branch_chain, _ = self._build_linear_chain(
                        list(group),
                        prefix_semantic,
                        intent,
                        nodes,
                        edges,
                    )
                    if branch_root is not None and branch_chain:
                        edges.append((branch_root, branch_chain[0]))
                    branch_ends.append(branch_chain[-1])
                    selected_chain.extend(branch_chain)

                fusion_capabilities = list(intent.context.get("fusion_capabilities", []))
                if fusion_capabilities:
                    fusion_chain, _ = self._build_linear_chain(
                        fusion_capabilities,
                        prefix_semantic,
                        intent,
                        nodes,
                        edges,
                    )
                    if fusion_chain:
                        for branch_end in branch_ends:
                            edges.append((branch_end, fusion_chain[0]))
                        selected_chain.extend(fusion_chain)

            if intent.context.get("risk_level", 0.0) >= 0.7:
                if "conflict_prediction" in self.catalog.microservices and "conflict_prediction" not in nodes:
                    conflict = self.catalog.get("conflict_prediction")
                    nodes[conflict.ms_id] = conflict
                    if selected_chain:
                        edges.append((selected_chain[-1], conflict.ms_id))
                    selected_chain.append(conflict.ms_id)

        deduped_edges: List[Tuple[str, str]] = []
        seen_edges = set()
        for edge in edges:
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            deduped_edges.append(edge)

        return ServiceGraphSpec(
            graph_id=f"sg_{intent.intent_id}_{uuid.uuid4().hex[:8]}",
            intent=intent,
            nodes=nodes,
            edges=deduped_edges,
        )
