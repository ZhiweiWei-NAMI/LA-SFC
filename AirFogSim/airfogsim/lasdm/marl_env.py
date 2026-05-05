from __future__ import annotations

import csv
import heapq
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .distributed_catalog import CatalogCandidate, DistributedServiceCatalog, split_directory_by_region
from .env_adapter import LASDMEnvAdapter
from .graph_observation import GraphObservationBuilder, GraphObservationConfig
from .instance_directory import ServiceInstanceDirectory
from .manager import LASDMManager
from .marl_reward import SFCReward, reward_aux_from_observations
from .model import GraphStatus, LASDMServiceChain, SFCFailureReason
from .orchestrator import LASDMDecision
from .runtime_bridge import LASDMRuntimeBridge
from .semantic_encoder import SemanticEncoder
from .semantic_exchange import SemanticCompressor, SemanticExchange, SemanticExchangeConfig
from .service_discovery_protocol import DistributedServiceDiscoveryProtocol
from .temporal_state_buffer import TemporalStateBuffer
from .topology_builder import DynamicTopology, TopologyBuilder
from .runtime_progress import RuntimeProgressConfig, advance_runtime, current_runtime_time


@dataclass(frozen=True)
class MARLEnvConfig:
    agent_type: str = "region"
    semantic_top_k: int = 8
    min_semantic_similarity: float = -1.0
    temporal_window: int = 4
    max_steps: int = 1000
    auto_exchange: bool = True
    include_remote_candidates: bool = True
    reset_underlying_env: bool = False
    include_topology_features: bool = True
    include_temporal_features: bool = True
    include_semantic_features: bool = True
    auto_plan_unassigned: bool = False
    semantic_exchange_ttl_s: float = 5.0
    semantic_exchange_radius_hops: int = 1
    semantic_exchange_fixed_delay_s: float = 0.10
    semantic_exchange_per_hop_delay_s: float = 0.02
    semantic_exchange_top_k_per_agent: int = 32
    semantic_exchange_compressed_dim: int = 64
    semantic_exchange_quantization_bits: int = 8
    semantic_encoder_backend: str = "hash"
    semantic_encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_encoder_hash_dim: int = 384
    semantic_encoder_batch_size: int = 64
    semantic_encoder_device: str = ""
    scenario_name: str = "default"
    service_role_sweep: str = "full_hybrid"
    runtime_ticks_per_marl_step: int = 1
    runtime_max_ticks_after_action: int = 25
    runtime_stop_when_progress: bool = True
    route_hop_floor_s: float = 1.0
    global_candidate_catalog: bool = False
    region_agents: Tuple[str, ...] = ()
    sequential_capacity_enabled: bool = True
    sequential_deadline_pruning_enabled: bool = True
    semantic_matrix: Optional[Any] = None
    enable_semantic_profiles: bool = True


@dataclass
class MARLStepResult:
    observations: Dict[str, Dict[str, Any]]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any]


_RESOURCE_LEVELS = tuple(round(0.1 * index, 1) for index in range(1, 11))


def _resource_level(value: Any, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    numeric = max(_RESOURCE_LEVELS[0], min(_RESOURCE_LEVELS[-1], numeric))
    return min(_RESOURCE_LEVELS, key=lambda level: abs(level - numeric))


def _normalize_resource_action(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "instance_id": str(value.get("instance_id", value.get("service_instance_id", "")) or ""),
            "compute_level": _resource_level(value.get("compute_level"), 1.0),
            "bandwidth_level": _resource_level(value.get("bandwidth_level"), 1.0),
        }
    return {"instance_id": str(value or ""), "compute_level": 1.0, "bandwidth_level": 1.0}


class SemanticTopologyMARLEnv:
    """Multi-agent environment binding semantic discovery, topology, and LASDM runtime.

    Agents are regional/RSU controllers by default. Their actions select service
    instance IDs for pending SFC function nodes. Accepted decisions are written
    into LASDMManager.decisions and then consumed by LASDMRuntimeBridge, so the
    action affects AirFogSim task creation/offloading when a live env is present.
    """

    def __init__(
        self,
        manager: LASDMManager,
        env: Optional[Any] = None,
        chains: Optional[Sequence[LASDMServiceChain]] = None,
        env_adapter: Optional[LASDMEnvAdapter] = None,
        runtime_bridge: Optional[LASDMRuntimeBridge] = None,
        topology_builder: Optional[TopologyBuilder] = None,
        observation_builder: Optional[GraphObservationBuilder] = None,
        discovery_protocol: Optional[DistributedServiceDiscoveryProtocol] = None,
        reward_fn: Optional[SFCReward] = None,
        config: Optional[MARLEnvConfig] = None,
    ):
        self.manager = manager
        self.env = env
        self.chains = list(chains or [])
        self.config = config or MARLEnvConfig()
        self.env_adapter = env_adapter or LASDMEnvAdapter(directory=manager.directory, manager=manager, chains=self.chains)
        self.runtime_bridge = runtime_bridge or LASDMRuntimeBridge(manager, env_adapter=self.env_adapter)
        self.topology_builder = topology_builder or TopologyBuilder(env_adapter=self.env_adapter, directory=manager.directory)
        self.observation_builder = observation_builder or GraphObservationBuilder(
            GraphObservationConfig(
                max_candidates=self.config.semantic_top_k,
                include_topology_features=self.config.include_topology_features,
                include_temporal_features=self.config.include_temporal_features,
                include_semantic_features=self.config.include_semantic_features,
            )
        )
        self.temporal_buffer = TemporalStateBuffer(maxlen=self.config.temporal_window)
        self.reward_fn = reward_fn or SFCReward()
        self.step_count = 0
        self.runtime_tick_count = 0
        self.last_topology: Optional[DynamicTopology] = None
        self.last_observations: Dict[str, Dict[str, Any]] = {}
        self.transition_trace: List[Dict[str, Any]] = []
        self.topology_trace: List[Dict[str, Any]] = []
        self.discovery_protocol = discovery_protocol or self._build_default_discovery_protocol()
        self._route_adjacency_topology_id: Optional[int] = None
        self._route_adjacency_cache: Dict[str, List[Tuple[str, float, float, bool, float]]] = {}
        self._route_reachability_cache: Dict[Tuple[str, str, float], Tuple[bool, int, float, float, float, int, float]] = {}
        self._route_tree_cache: Dict[Tuple[str, float], Dict[str, Tuple[bool, int, float, float, float, int, float]]] = {}
        self._wireless_capacity_cache_time: Optional[float] = None
        self._wireless_capacity_cache: Optional[Dict[str, float]] = None
        self._episode_start_time: float = 0.0
        self._pending_chain_ids: List[str] = []
        self._arrival_times: Dict[str, float] = {}
        self._chain_by_id: Dict[str, LASDMServiceChain] = {}

    def rebuild_config_dependent_components(self) -> None:
        """Recreate observation/discovery objects after MARLEnvConfig changes."""

        self.observation_builder = GraphObservationBuilder(
            GraphObservationConfig(
                max_candidates=self.config.semantic_top_k,
                include_topology_features=self.config.include_topology_features,
                include_temporal_features=self.config.include_temporal_features,
                include_semantic_features=self.config.include_semantic_features,
            )
        )
        self.discovery_protocol = self._build_default_discovery_protocol()
        if not self.config.include_semantic_features and isinstance(self.reward_fn, SFCReward):
            self.reward_fn = SFCReward(
                replace(
                    self.reward_fn.config,
                    semantic_score=0.0,
                    semantic_cumulative=0.0,
                    utility_prior=0.0,
                )
            )

    @property
    def agent_ids(self) -> List[str]:
        return sorted(self.discovery_protocol.catalogs)

    def reset(self, env: Optional[Any] = None, chains: Optional[Sequence[LASDMServiceChain]] = None) -> Dict[str, Dict[str, Any]]:
        if env is not None:
            self.env = env
        if chains is not None:
            self.chains = list(chains)
        if self.config.reset_underlying_env and self.env is not None and hasattr(self.env, "reset"):
            self.env.reset()
        self.step_count = 0
        self.runtime_tick_count = 0
        self.temporal_buffer.clear()
        self.transition_trace.clear()
        self.topology_trace.clear()
        self._clear_runtime_metric_caches()
        now = self._time()
        self._initialize_arrivals(now)
        self._submit_due_chains(now)
        topology = self._update_topology(now)
        if self.config.auto_exchange:
            self._semantic_exchange_tick(now, topology)
            warmup_time = now + max(
                0.0,
                float(self.config.semantic_exchange_fixed_delay_s)
                + float(self.config.semantic_exchange_per_hop_delay_s),
            )
            self._semantic_exchange_tick(warmup_time, topology)
        self.last_observations = self._build_observations(topology, now)
        return self.last_observations

    def observe(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.last_observations:
            topology = self._update_topology(self._time())
            self.last_observations = self._build_observations(topology, self._time())
        if agent_id is None:
            return self.last_observations
        return self.last_observations[str(agent_id)]

    def step(self, actions: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], bool, Dict[str, Any]]:
        now = self._time()
        previous_summary = dict(self.manager.summary())
        previous_gs2l_progress = _gs2l_chain_progress_snapshot(self.runtime_bridge, self.manager.chains)
        previous_done_tasks = len(getattr(self.runtime_bridge, "processed_done_tasks", set()) or set())
        previous_failed_tasks = len(getattr(self.runtime_bridge, "processed_failed_tasks", set()) or set())
        submitted_before_action = self._submit_due_chains(now)
        topology = self._update_topology(now)
        if self.config.auto_exchange:
            self._semantic_exchange_tick(now, topology)

        decoded_decisions = self._decode_actions(actions or {}, now)
        for decision in decoded_decisions:
            self.manager.decisions[decision.sfc_id] = decision
            if decision.rejected_reason is not None:
                self.manager.fail(decision.sfc_id, decision.rejected_reason, now)

        if self.config.auto_plan_unassigned:
            self.manager.step(current_time=now)
        else:
            decided = {decision.sfc_id for decision in decoded_decisions}
            for chain in list(self.manager.chains.values()):
                if chain.status == GraphStatus.RUNNING and chain.sfc_id not in decided and chain.sfc_id not in self.manager.decisions:
                    self.manager.fail(
                        chain.sfc_id,
                        SFCFailureReason.NO_CANDIDATE,
                        now,
                        details={"reason": "policy_returned_no_assignment"},
                    )
        runtime_report: Dict[str, Any] = {}
        if self.env is not None:
            bridge_prepare = self.runtime_bridge.prepare_airfogsim_step(
                self.env,
                chains=self._submitted_runtime_chains(),
                current_time=now,
            )
            progress = advance_runtime(
                self.env,
                RuntimeProgressConfig(
                    ticks_per_marl_step=int(self.config.runtime_ticks_per_marl_step),
                    max_ticks_after_action=int(self.config.runtime_max_ticks_after_action),
                    stop_when_report_has_progress=bool(self.config.runtime_stop_when_progress),
                ),
                bridge_sync=lambda: self.runtime_bridge.sync_from_airfogsim_tasks(
                    self.env,
                    current_time=self._time(),
                ),
                before_step=lambda: self._prepare_runtime_tick(),
            )
            self.runtime_tick_count += int(progress.get("ticks", 0) or 0)
            self.runtime_bridge.sync_from_airfogsim_tasks(self.env, current_time=self._time())
            runtime_report = {"bridge_prepare": bridge_prepare, "runtime_progress": progress}
            replanned = self._clear_partial_chain_decisions_for_replanning()
            if replanned:
                runtime_report["sequential_replanning"] = replanned
            self._expire_deadlines(self._time())
            submitted_after_runtime = self._submit_due_chains(self._time())
            submitted_during_step = submitted_before_action + submitted_after_runtime
            if submitted_during_step:
                runtime_report["arrival_submitted_sfc_ids"] = submitted_during_step
        else:
            raise RuntimeError("SemanticTopologyMARLEnv requires a live AirFogSim env")

        self.step_count += 1
        next_now = self._time()
        topology = self._update_topology(next_now)
        observations = self._build_observations(topology, next_now)
        overhead = self.discovery_protocol.exchange.overhead_summary().get("payload_bytes", 0.0)
        aux = reward_aux_from_observations(observations, overhead_bytes=overhead)
        aux.update(_reward_aux_from_decisions(decoded_decisions))
        aux.update(
            _gs2l_progress_aux(
                previous_gs2l_progress,
                _gs2l_chain_progress_snapshot(self.runtime_bridge, self.manager.chains),
                previous_done_tasks,
                previous_failed_tasks,
                self.runtime_bridge,
                self.manager.chains,
            )
        )
        reward_value = self.reward_fn(previous_summary, self.manager.summary(), aux)
        rewards = {agent_id: reward_value for agent_id in self.agent_ids}
        done = self.done()
        info = {
            "time_s": next_now,
            "step_count": self.step_count,
            "runtime_step_count": self.runtime_tick_count,
            "decisions": [decision.to_dict() for decision in decoded_decisions],
            "summary": self.manager.summary(),
            "runtime_report": runtime_report,
            "reward_aux": aux,
            "message_overhead": self.discovery_protocol.exchange.overhead_summary(),
            "arrival_queue": {
                "pending": len(self._pending_chain_ids),
                "submitted_this_step": submitted_during_step if self.env is not None else [],
            },
        }
        self.transition_trace.append(
            {
                "time_s": next_now,
                "step": self.step_count,
                "actions": _jsonable(actions or {}),
                "rewards": rewards,
                "done": done,
                "info": _transition_info_summary(info),
            }
        )
        self.last_observations = observations
        return observations, rewards, done, info

    def _clear_partial_chain_decisions_for_replanning(self) -> List[Dict[str, Any]]:
        """Release stale whole-chain decisions after a ready prefix finishes.

        Runtime execution is sequential: later SFC stages are not spawned until
        predecessors complete.  Keeping the original full-chain decision would
        force later stages to use route/source metadata from an older topology.
        """

        replanned: List[Dict[str, Any]] = []
        spawned = set(getattr(self.runtime_bridge, "sfc_node_to_task", {}) or {})
        completed_by_chain = getattr(self.runtime_bridge, "completed_nodes", {}) or {}
        for sfc_id, chain in list(self.manager.chains.items()):
            if chain.status != GraphStatus.RUNNING or sfc_id not in self.manager.decisions:
                continue
            completed = set(completed_by_chain.get(sfc_id, set()) or set())
            if not completed:
                continue
            order = list(chain.topological_order())
            unspawned = [node_id for node_id in order if (sfc_id, node_id) not in spawned]
            ready_unspawned = self._ready_unspawned_nodes(chain, spawned, completed)
            if not ready_unspawned:
                continue
            del self.manager.decisions[sfc_id]
            replanned.append(
                {
                    "sfc_id": sfc_id,
                    "completed_nodes": sorted(completed),
                    "unspawned_nodes": unspawned,
                    "ready_unspawned_nodes": ready_unspawned,
                }
            )
        return replanned

    def _ready_unspawned_nodes(
        self,
        chain: LASDMServiceChain,
        spawned: set[tuple[str, str]],
        completed: set[str],
    ) -> List[str]:
        ready: List[str] = []
        for node_id in chain.topological_order():
            if (chain.sfc_id, node_id) in spawned:
                continue
            if all(predecessor in completed for predecessor in chain.predecessors(node_id)):
                ready.append(node_id)
        return ready

    def _prepare_runtime_tick(self) -> None:
        if self.env is None:
            raise RuntimeError("runtime tick preparation requires a live AirFogSim env")
        self.env_adapter.schedule_returning(self.env)
        self.env_adapter.schedule_communication(self.env)
        self.env_adapter.schedule_computation(self.env)

    def done(self) -> bool:
        if self.step_count >= self.config.max_steps:
            return True
        if self._pending_chain_ids:
            return False
        summary = self.manager.summary()
        return bool(summary.get("submitted", 0)) and int(summary.get("active_graphs", 0) or 0) == 0

    def _initialize_arrivals(self, current_time: float) -> None:
        self._episode_start_time = float(current_time)
        self._chain_by_id = {}
        self._arrival_times = {}
        for chain in self.chains:
            chain.status = GraphStatus.PENDING
            chain.submit_time = None
            chain.finish_time = None
            chain.failure_reason = None
            self._chain_by_id[chain.sfc_id] = chain
            self._arrival_times[chain.sfc_id] = self._arrival_time_for_chain(chain)
        self._pending_chain_ids = sorted(
            self._chain_by_id,
            key=lambda sfc_id: (self._arrival_times.get(sfc_id, self._episode_start_time), sfc_id),
        )

    def _arrival_time_for_chain(self, chain: LASDMServiceChain) -> float:
        context = dict(getattr(chain, "context", {}) or {})
        raw = context.get("arrival_time_s", context.get("arrival_offset_s", 0.0))
        try:
            offset = max(0.0, float(raw))
        except (TypeError, ValueError):
            offset = 0.0
        return self._episode_start_time + offset

    def _max_active_sfcs(self) -> int:
        values: List[int] = []
        for chain in self.chains:
            context = dict(getattr(chain, "context", {}) or {})
            raw = context.get("max_concurrent_sfcs")
            if raw is None:
                continue
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        return min(values) if values else 0

    def _submit_due_chains(self, current_time: float) -> List[str]:
        submitted: List[str] = []
        max_active = self._max_active_sfcs()
        while self._pending_chain_ids:
            if max_active > 0 and self._active_chain_count() >= max_active:
                break
            sfc_id = self._pending_chain_ids[0]
            if self._arrival_times.get(sfc_id, self._episode_start_time) > float(current_time) + 1e-9:
                break
            self._pending_chain_ids.pop(0)
            if sfc_id in self.manager.chains:
                continue
            chain = self._chain_by_id[sfc_id]
            self.manager.submit(chain, current_time=current_time)
            submitted.append(sfc_id)
        return submitted

    def _active_chain_count(self) -> int:
        return sum(1 for chain in self.manager.chains.values() if chain.status == GraphStatus.RUNNING)

    def _submitted_runtime_chains(self) -> List[LASDMServiceChain]:
        return list(self.manager.chains.values())

    def write_traces(self, output_dir: str) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        _write_jsonl(target / "marl_transition_trace.jsonl", self.transition_trace)
        _write_jsonl(target / "topology_trace.jsonl", self.topology_trace)
        _write_csv(target / "semantic_candidate_trace.csv", self.discovery_protocol.semantic_candidate_trace_rows())
        _write_csv(target / "semantic_candidate_detail_trace.csv", self.discovery_protocol.semantic_candidate_detail_trace_rows())
        _write_csv(target / "distributed_exchange_trace.csv", self.discovery_protocol.message_overhead_rows())
        _write_csv(target / "message_overhead.csv", self.discovery_protocol.message_overhead_rows())
        lifecycle = getattr(self.runtime_bridge, "runtime_task_lifecycle_trace", None)
        if lifecycle is not None:
            _write_csv(target / "runtime_task_lifecycle_trace.csv", lifecycle)
        _write_json(target / "semantic_encoder_manifest.json", self._semantic_encoder_manifest())

    def _build_default_discovery_protocol(self) -> DistributedServiceDiscoveryProtocol:
        encoder = SemanticEncoder(
            backend=str(self.config.semantic_encoder_backend),
            model_name=str(self.config.semantic_encoder_model_name),
            hash_dim=int(self.config.semantic_encoder_hash_dim),
            batch_size=int(self.config.semantic_encoder_batch_size),
            device=str(self.config.semantic_encoder_device) or None,
        )
        _ = encoder.backend_active
        exchange_config = SemanticExchangeConfig(
            ttl_s=float(self.config.semantic_exchange_ttl_s),
            radius_hops=int(self.config.semantic_exchange_radius_hops),
            fixed_delay_s=float(self.config.semantic_exchange_fixed_delay_s),
            per_hop_delay_s=float(self.config.semantic_exchange_per_hop_delay_s),
            top_k_per_agent=int(self.config.semantic_exchange_top_k_per_agent),
            compressed_dim=int(self.config.semantic_exchange_compressed_dim),
            quantization_bits=int(self.config.semantic_exchange_quantization_bits),
        )
        compressor = SemanticCompressor(
            input_dim=encoder.embedding_dim,
            compressed_dim=exchange_config.compressed_dim,
            quantization_bits=exchange_config.quantization_bits,
        )
        exchange = SemanticExchange(exchange_config, compressor=compressor)
        per_region = split_directory_by_region(self.manager.directory)
        if not per_region:
            raise RuntimeError("semantic discovery requires at least one service region")
        if self.config.region_agents:
            configured = tuple(str(region) for region in self.config.region_agents if str(region))
            scoped = {region: per_region.get(region, ServiceInstanceDirectory()) for region in configured}
            shared_instances = [
                instance
                for region_id, directory in per_region.items()
                if region_id not in scoped
                for instance in directory.all()
            ]
            for directory in scoped.values():
                for instance in shared_instances:
                    directory.upsert(instance)
            per_region = scoped
        if self.config.global_candidate_catalog:
            per_region = {region_id: self.manager.directory for region_id in per_region}
        catalogs = {
            region_id: DistributedServiceCatalog(
                region_id,
                directory,
                encoder=encoder,
                compressor=compressor,
                respect_local_visibility=not self.config.global_candidate_catalog,
                semantic_matrix=self.config.semantic_matrix,
            )
            for region_id, directory in per_region.items()
        }
        regions = sorted(catalogs)
        neighbor_map = {region: [other for other in regions if other != region] for region in regions}
        return DistributedServiceDiscoveryProtocol(catalogs, exchange, neighbor_map=neighbor_map)

    def _semantic_encoder_manifest(self) -> Dict[str, Any]:
        encoder = getattr(next(iter(self.discovery_protocol.catalogs.values()), None), "encoder", None)
        if encoder is None:
            return {}
        return {
            "configured_backend": str(self.config.semantic_encoder_backend),
            "active_backend": str(getattr(encoder, "backend_active", "")),
            "model_name": str(getattr(encoder, "model_name", "")),
            "embedding_dim": int(getattr(encoder, "embedding_dim", 0) or 0),
            "device": str(getattr(encoder, "device", "") or ""),
        }

    def _semantic_exchange_tick(self, now: float, topology: DynamicTopology) -> None:
        neighbor_map = self.topology_builder.neighbor_map_by_region(topology)
        neighbor_map = _expand_neighbor_map_by_radius(neighbor_map, int(self.config.semantic_exchange_radius_hops))
        self.discovery_protocol.set_neighbors(neighbor_map)
        self.discovery_protocol.tick(now)

    def _build_observations(self, topology: DynamicTopology, now: float) -> Dict[str, Dict[str, Any]]:
        observations: Dict[str, Dict[str, Any]] = {}
        for agent_id in self.agent_ids:
            candidate_sets = self._candidate_sets_for_agent(agent_id, now)
            observations[agent_id] = self.observation_builder.build(
                agent_id=agent_id,
                topology=topology,
                candidate_sets=candidate_sets,
                temporal_buffer=self.temporal_buffer,
            )
        return observations

    def _candidate_sets_for_agent(self, agent_id: str, now: float) -> List[Dict[str, Any]]:
        sets: List[Dict[str, Any]] = []
        for chain in self.manager.chains.values():
            if chain.status != GraphStatus.RUNNING or chain.sfc_id in self.manager.decisions:
                continue
            controller_agent = self._controller_agent_for_chain(chain)
            if controller_agent and str(agent_id) != controller_agent:
                continue
            spawned = set(getattr(self.runtime_bridge, "sfc_node_to_task", {}) or {})
            completed = set(getattr(self.runtime_bridge, "completed_nodes", {}).get(chain.sfc_id, set()) or set())
            order = self._ready_unspawned_nodes(chain, spawned, completed)
            if not order:
                continue
            chain_order = list(chain.topological_order())
            discovered: Dict[str, List[CatalogCandidate]] = {}
            discovery_top_k = int(self.config.semantic_top_k)
            discovery_min_similarity = float(self.config.min_semantic_similarity)
            for sfc_node_id in order:
                link_input_semantic = self._semantic_input_for_sfc_node(chain, sfc_node_id)
                discovered[sfc_node_id] = self.discovery_protocol.discover_sfc_node(
                    agent_id,
                    chain,
                    sfc_node_id,
                    now_s=now,
                    top_k=discovery_top_k,
                    min_similarity=discovery_min_similarity,
                    include_remote=self.config.include_remote_candidates,
                    link_input_semantic=link_input_semantic,
                )
            enriched_by_node = self._enrich_candidate_sets_with_runtime_routes(chain, order, discovered)
            for index, sfc_node_id in enumerate(order):
                candidates = enriched_by_node.get(sfc_node_id, [])
                self.discovery_protocol.update_candidate_route_metadata(chain.sfc_id, sfc_node_id, candidates)
                sets.append(
                    {
                        "agent_id": agent_id,
                        "sfc_id": chain.sfc_id,
                        "sfc_node_id": sfc_node_id,
                        "sfc_node_index": chain_order.index(sfc_node_id) if sfc_node_id in chain_order else index,
                        "source_node_id": chain.source_node_id,
                        "predecessor_node_ids": chain.predecessors(sfc_node_id),
                        "candidates": candidates,
                    }
                )
        return sets

    def _controller_agent_for_chain(self, chain: LASDMServiceChain) -> str:
        agent_ids = self.agent_ids
        if not agent_ids:
            return ""
        context = dict(getattr(chain, "context", {}) or {})
        for key in ("preferred_region_id", "controller_agent_id", "source_region_id"):
            value = str(context.get(key, "") or "")
            if value and value in agent_ids:
                return value
        source = str(getattr(chain, "source_node_id", "") or "")
        if source in agent_ids:
            return source
        topology = self.last_topology
        if topology is not None and source in topology.nodes:
            source_region = str(topology.nodes[source].region_id or "")
            if source_region in agent_ids:
                return source_region
            ranked: List[Tuple[float, str]] = []
            payload_mb = max(0.0, float(getattr(chain, "payload_mb", 0.0) or 0.0))
            for candidate_agent in agent_ids:
                if candidate_agent not in topology.nodes:
                    continue
                available, hops, risk, tx_time, _wireless_tx, _wireless_hops, _bottleneck = self._candidate_route_reachability(
                    source,
                    candidate_agent,
                    payload_mb,
                )
                if available:
                    ranked.append((float(tx_time) + float(hops) * self._runtime_tick_s() + 4.0 * float(risk), candidate_agent))
            if ranked:
                ranked.sort(key=lambda item: (item[0], item[1]))
                return ranked[0][1]
        return sorted(agent_ids)[0]

    def _enrich_candidates_with_runtime_route(
        self,
        chain: LASDMServiceChain,
        candidates: Sequence[CatalogCandidate],
    ) -> List[CatalogCandidate]:
        source = str(chain.source_node_id)
        enriched: List[CatalogCandidate] = []
        for candidate in candidates:
            metadata = self._candidate_metadata_for_route_source(chain, candidate, source)
            enriched.append(replace(candidate, metadata=metadata))
        return enriched

    def _enrich_candidate_sets_with_runtime_routes(
        self,
        chain: LASDMServiceChain,
        order: Sequence[str],
        discovered: Mapping[str, Sequence[CatalogCandidate]],
    ) -> Dict[str, List[CatalogCandidate]]:
        enriched_by_node: Dict[str, List[CatalogCandidate]] = {}
        source_nodes_by_function: Dict[str, List[str]] = {}
        for sfc_node_id in order:
            predecessors = chain.predecessors(sfc_node_id)
            if predecessors:
                sources: List[str] = []
                for predecessor in predecessors:
                    runtime_source = self._completed_predecessor_output_node(chain.sfc_id, predecessor)
                    if runtime_source:
                        if runtime_source not in sources:
                            sources.append(runtime_source)
                        continue
                    for candidate in discovered.get(predecessor, []) or []:
                        node_id = str(candidate.node_id)
                        if node_id and node_id not in sources:
                            sources.append(node_id)
                if not sources:
                    sources = [str(chain.source_node_id)]
            else:
                sources = [str(chain.source_node_id)]
            source_nodes_by_function[sfc_node_id] = sources

            enriched: List[CatalogCandidate] = []
            for candidate in discovered.get(sfc_node_id, []) or []:
                metadata = dict(candidate.metadata or {})
                source_metrics = {
                    str(source): self._route_metric_fields(chain, candidate, str(source), sfc_node_id=sfc_node_id)
                    for source in source_nodes_by_function[sfc_node_id]
                }
                best_source, best_fields = min(
                    source_metrics.items(),
                    key=lambda item: (
                        float(item[1].get("expected_runtime_penalty_s", 0.0) or 0.0),
                        -float(item[1].get("semantic_score", 0.0) or 0.0),
                        item[0],
                    ),
                )
                metadata.update(best_fields)
                metadata["route_source_node_id"] = best_source
                metadata["candidate_route_sources"] = list(source_metrics.keys())
                metadata["route_available_by_source"] = {
                    source: float(fields.get("route_available", 0.0) or 0.0) for source, fields in source_metrics.items()
                }
                metadata["route_hops_by_source"] = {
                    source: int(float(fields.get("route_hops", 0.0) or 0.0)) for source, fields in source_metrics.items()
                }
                for field in (
                    "route_tx_time_s",
                    "route_topology_tx_time_s",
                    "route_wireless_tx_time_s",
                    "expected_rb_wait_s",
                    "expected_wireless_tx_time_s",
                    "wireless_pressure",
                    "wireless_hops",
                    "rb_slowdown",
                    "rb_slowdown_tx_extra_s",
                    "expected_runtime_penalty_s",
                    "deadline_slack_s",
                    "utility_prior",
                    "chain_deadline_s",
                    "chain_remaining_deadline_s",
                    "remaining_deadline_ratio",
                    "remaining_function_count",
                    "resource_capacity_total",
                    "resource_reserved_count",
                    "resource_available_slots",
                    "resource_available_ratio",
                    "capacity_exhausted",
                    "hops_from_prev_function",
                    "hops_from_prev_function_norm",
                    "sequential_deadline_feasible",
                    "link_similarity",
                    "node_template_similarity",
                    "semantic_quality_before",
                    "semantic_cumulative_quality_if_selected",
                    "semantic_link_truth_score",
                ):
                    metadata[f"{field}_by_source"] = {
                        source: float(fields.get(field, 0.0) or 0.0) for source, fields in source_metrics.items()
                    }
                enriched.append(replace(candidate, metadata=metadata))
            enriched_by_node[sfc_node_id] = enriched
        return enriched_by_node

    def _completed_predecessor_output_node(self, sfc_id: str, sfc_node_id: str) -> str:
        outputs = getattr(self.runtime_bridge, "node_outputs", {}) or {}
        output = outputs.get((str(sfc_id), str(sfc_node_id)))
        if not isinstance(output, Mapping):
            return ""
        return str(output.get("node_id", "") or "")

    def _candidate_metadata_for_route_source(
        self,
        chain: LASDMServiceChain,
        candidate: CatalogCandidate,
        source_node_id: str,
        sfc_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = dict(candidate.metadata or {})
        metadata.update(self._route_metric_fields(chain, candidate, source_node_id, sfc_node_id=sfc_node_id))
        metadata["route_source_node_id"] = str(source_node_id)
        return metadata

    def _route_metric_fields(
        self,
        chain: LASDMServiceChain,
        candidate: CatalogCandidate,
        source_node_id: str,
        sfc_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = dict(candidate.metadata or {})
        source = str(source_node_id)
        payload_mb = self._sfc_node_input_payload_mb(chain, sfc_node_id)
        (
            route_available,
            hop_count,
            route_risk,
            topology_tx_time_s,
            wireless_tx_time_s,
            wireless_hops,
            bottleneck_rate_mbps,
        ) = self._candidate_route_reachability(
            source,
            str(candidate.node_id),
            payload_mb,
        )
        route_hop_floor_s = max(1e-6, float(self.config.route_hop_floor_s))
        route_tx_time_s = max(topology_tx_time_s, float(hop_count) * route_hop_floor_s) if hop_count > 0 else 0.0
        wireless_capacity = self._wireless_capacity_metrics()
        rb_slowdown = float(wireless_capacity.get("rb_slowdown", 1.0) or 1.0)
        wireless_pressure = float(wireless_capacity.get("wireless_pressure", 0.0) or 0.0)
        expected_rb_wait_s = 0.0
        rb_slowdown_tx_extra_s = 0.0
        if wireless_hops > 0:
            rb_slowdown_tx_extra_s = max(0.0, wireless_tx_time_s * (rb_slowdown - 1.0))
            queue_rounds = max(0.0, math.ceil(wireless_pressure) - 1.0)
            expected_rb_wait_s = float(wireless_hops) * self._runtime_tick_s() * queue_rounds
            route_tx_time_s = max(topology_tx_time_s + rb_slowdown_tx_extra_s, float(hop_count) * route_hop_floor_s) + expected_rb_wait_s
        fields: Dict[str, Any] = {}
        fields["route_available"] = 1.0 if route_available else 0.0
        fields["route_hops"] = hop_count
        fields["route_hops_norm"] = min(1.0, float(hop_count) / 4.0)
        fields["route_tx_time_s"] = route_tx_time_s
        fields["route_topology_tx_time_s"] = topology_tx_time_s
        fields["route_wireless_tx_time_s"] = wireless_tx_time_s
        fields["wireless_hops"] = wireless_hops
        fields["wireless_pressure"] = wireless_pressure
        fields["wireless_queue_depth"] = float(wireless_capacity.get("wireless_queue_depth", 0.0) or 0.0)
        fields["effective_wireless_rb"] = float(wireless_capacity.get("effective_wireless_rb", 0.0) or 0.0)
        fields["full_wireless_rb"] = float(wireless_capacity.get("full_wireless_rb", 0.0) or 0.0)
        fields["rb_slowdown"] = rb_slowdown
        fields["rb_slowdown_tx_extra_s"] = rb_slowdown_tx_extra_s
        fields["expected_rb_wait_s"] = expected_rb_wait_s
        fields["expected_wireless_tx_time_s"] = route_tx_time_s
        fields["bottleneck_rate_mbps"] = bottleneck_rate_mbps
        topology_risk = float(metadata.get("topology_risk", 0.0) or 0.0)
        mobility_risk = float(metadata.get("mobility_risk", 0.0) or 0.0)
        if not route_available:
            topology_risk = max(topology_risk, 1.0)
            mobility_risk = max(mobility_risk, 0.75)
        elif hop_count > 1:
            topology_risk = min(1.0, topology_risk + 0.05 * float(hop_count - 1))
        topology_risk = min(1.0, topology_risk + route_risk)
        fields["topology_risk"] = topology_risk
        fields["mobility_risk"] = mobility_risk
        semantic_score = max(0.0, min(1.0, float(getattr(candidate, "semantic_score", 0.0) or 0.0)))
        semantic_quality_before = self._semantic_quality_before(chain, sfc_node_id)
        semantic_cumulative_quality = max(0.0, min(1.0, semantic_quality_before * semantic_score))
        chain_context = dict(getattr(chain, "context", {}) or {})
        semantic_min_score = max(0.0, float(chain_context.get("semantic_min_score", 0.0) or 0.0))
        semantic_shortfall = max(0.0, semantic_min_score - semantic_cumulative_quality)
        deadline_s = float(getattr(getattr(chain, "qos", None), "deadline_s", 0.0) or 0.0)
        elapsed_s = 0.0
        if getattr(chain, "submit_time", None) is not None:
            elapsed_s = max(0.0, self._time() - float(chain.submit_time or 0.0))
        remaining_deadline_s = max(0.0, deadline_s - elapsed_s) if deadline_s > 0.0 else 0.0
        remaining_function_count = self._remaining_function_count(chain, sfc_node_id)
        function_budget_s = (
            remaining_deadline_s / max(1, remaining_function_count)
            if remaining_deadline_s > 0.0
            else 0.0
        )
        mismatch_unit = float(chain_context.get("semantic_mismatch_penalty_s_per_unit", 0.0) or 0.0)
        mismatch_cost_s = mismatch_unit * max(0.0, 1.0 - semantic_score) ** 2
        cold_start_s = float(metadata.get("cold_start_s", 0.0) or 0.0)
        stale_cost_s = float(metadata.get("stale_latency_penalty_s", 0.0) or 0.0)
        task_cpu = self._sfc_node_task_cpu(chain, sfc_node_id)
        capacity_cpu = max(
            0.1,
            float(metadata.get("available_cpu", 0.0) or 0.0)
            if float(metadata.get("available_cpu", 0.0) or 0.0) > 0.0
            else float(metadata.get("capacity_cpu", 0.0) or 0.0),
        )
        max_concurrency = max(1, int(float(metadata.get("max_concurrency", 1) or 1)))
        current_load = max(0, int(float(metadata.get("current_load", 0) or 0)))
        load_ratio = max(0.0, min(1.0, float(metadata.get("load_ratio", 0.0) or 0.0)))
        reserved_count = self._node_capacity_reservations().get(str(candidate.node_id), 0)
        if not self.config.sequential_capacity_enabled:
            reserved_count = 0
        resource_available_slots = max(0, max_concurrency - int(current_load) - int(reserved_count))
        resource_available_ratio = max(0.0, min(1.0, float(resource_available_slots) / float(max_concurrency)))
        effective_cpu = max(0.1, (capacity_cpu / float(max_concurrency)) * max(0.1, 1.0 - 0.5 * load_ratio))
        estimated_compute_s = max(0.0, task_cpu / effective_cpu) if task_cpu > 0.0 else 0.0
        expected_penalty_s = (
            mismatch_cost_s
            + estimated_compute_s
            + cold_start_s
            + stale_cost_s
            + route_tx_time_s
            + 4.0 * topology_risk
            + 4.0 * mobility_risk
        )
        runtime_penalty_no_semantic_s = max(0.0, expected_penalty_s - mismatch_cost_s)
        fields["semantic_score"] = semantic_score
        fields["link_similarity"] = semantic_score
        fields["node_template_similarity"] = max(0.0, min(1.0, float(metadata.get("node_template_similarity", semantic_score) or 0.0)))
        fields["semantic_quality_before"] = semantic_quality_before
        fields["semantic_cumulative_quality_if_selected"] = semantic_cumulative_quality
        if "semantic_link_truth_score" in metadata:
            fields["semantic_link_truth_score"] = max(0.0, min(1.0, float(metadata.get("semantic_link_truth_score", 0.0) or 0.0)))
        fields["semantic_min_score"] = semantic_min_score
        fields["semantic_shortfall"] = semantic_shortfall
        fields["semantic_quality_violation"] = 0.0
        fields["task_cpu"] = task_cpu
        fields["candidate_capacity_cpu"] = capacity_cpu
        fields["effective_cpu"] = effective_cpu
        fields["estimated_compute_s"] = estimated_compute_s
        fields["expected_runtime_penalty_s"] = expected_penalty_s
        fields["expected_runtime_penalty_no_semantic_s"] = runtime_penalty_no_semantic_s
        fields["function_budget_s"] = function_budget_s
        fields["chain_deadline_s"] = deadline_s
        fields["chain_elapsed_s"] = elapsed_s
        fields["chain_remaining_deadline_s"] = remaining_deadline_s
        fields["scenario_request_count"] = max(1.0, float(chain_context.get("request_count", 1.0) or 1.0))
        fields["scenario_max_concurrent_sfcs"] = max(
            1.0,
            float(chain_context.get("max_concurrent_sfcs", 1.0) or 1.0),
        )
        fields["scenario_arrival_rate_sfc_per_s"] = max(
            1e-6,
            float(chain_context.get("arrival_rate_sfc_per_s", 1.0) or 1.0),
        )
        fields["remaining_function_count"] = remaining_function_count
        fields["remaining_deadline_ratio"] = (
            max(0.0, min(1.0, remaining_deadline_s / deadline_s)) if deadline_s > 0.0 else 1.0
        )
        fields["resource_capacity_total"] = max_concurrency
        fields["resource_reserved_count"] = reserved_count
        fields["resource_available_slots"] = resource_available_slots
        fields["resource_available_ratio"] = resource_available_ratio
        fields["capacity_exhausted"] = 1.0 if resource_available_slots <= 0 else 0.0
        fields["hops_from_prev_function"] = hop_count
        fields["hops_from_prev_function_norm"] = min(1.0, float(hop_count) / 4.0)
        fields["deadline_slack_s"] = function_budget_s - expected_penalty_s if function_budget_s > 0.0 else 0.0
        fields["sequential_deadline_feasible"] = (
            1.0
            if not self.config.sequential_deadline_pruning_enabled
            or deadline_s <= 0.0
            or expected_penalty_s <= remaining_deadline_s + 1e-9
            else 0.0
        )
        fields["utility_prior"] = semantic_cumulative_quality - (expected_penalty_s / max(1.0, function_budget_s or 1.0))
        fields["runtime_prior_no_semantic"] = -(runtime_penalty_no_semantic_s / max(1.0, function_budget_s or 1.0))
        return fields

    def _semantic_input_for_sfc_node(self, chain: LASDMServiceChain, sfc_node_id: str) -> str:
        predecessors = chain.predecessors(sfc_node_id)
        if not predecessors:
            return str(getattr(chain, "payload_semantic", "") or "any")
        for predecessor in predecessors:
            output = getattr(self.runtime_bridge, "node_outputs", {}).get((chain.sfc_id, predecessor))
            if isinstance(output, Mapping):
                semantic = str(output.get("semantic", "") or "")
                if semantic:
                    return semantic
        node = chain.nodes.get(sfc_node_id)
        return str(getattr(node, "input_semantic", "") or "any")

    def _semantic_quality_before(self, chain: LASDMServiceChain, sfc_node_id: Optional[str]) -> float:
        if not sfc_node_id:
            return 1.0
        predecessors = chain.predecessors(sfc_node_id)
        if not predecessors:
            return 1.0
        quality = 1.0
        for predecessor in predecessors:
            output = getattr(self.runtime_bridge, "node_outputs", {}).get((chain.sfc_id, predecessor))
            if isinstance(output, Mapping):
                quality *= max(0.0, min(1.0, float(output.get("semantic_cumulative_quality", 1.0) or 1.0)))
        return max(0.0, min(1.0, quality))

    def _remaining_function_count(self, chain: LASDMServiceChain, sfc_node_id: Optional[str]) -> int:
        order = list(chain.topological_order())
        if not order:
            return 1
        if sfc_node_id in order:
            return max(1, len(order) - order.index(str(sfc_node_id)))
        return max(1, len(order))

    def _node_capacity_reservations(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        outputs = getattr(self.runtime_bridge, "node_outputs", {}) or {}
        for chain in self.manager.chains.values():
            if chain.status != GraphStatus.RUNNING:
                continue
            for node_id in chain.topological_order():
                output = outputs.get((chain.sfc_id, node_id))
                if not isinstance(output, Mapping):
                    continue
                host = str(output.get("node_id", "") or "")
                if host:
                    counts[host] = counts.get(host, 0) + 1
        return counts

    def _sfc_node_task_cpu(self, chain: LASDMServiceChain, sfc_node_id: Optional[str]) -> float:
        if not sfc_node_id or sfc_node_id not in chain.nodes:
            return 0.0
        node = chain.nodes[sfc_node_id]
        metadata = dict(getattr(node, "metadata", {}) or {})
        for key in ("task_cpu", "cpu", "cpu_mb"):
            if key in metadata:
                return max(0.0, float(metadata[key]))
        return max(0.0, float(getattr(node, "cpu_mb", 0.0) or 0.0))

    def _sfc_node_input_payload_mb(self, chain: LASDMServiceChain, sfc_node_id: Optional[str]) -> float:
        payload = max(0.0, float(getattr(chain, "payload_mb", 0.0) or 0.0))
        if not sfc_node_id:
            return payload
        for node_id in chain.topological_order():
            if node_id == sfc_node_id:
                return payload
            node = chain.nodes.get(node_id)
            if node is None:
                continue
            metadata = dict(getattr(node, "metadata", {}) or {})
            if "output_payload_mb" in metadata:
                payload = max(0.0, float(metadata["output_payload_mb"]))
            else:
                payload *= max(0.0, float(metadata.get("output_ratio", 1.0) or 1.0))
        return payload

    def _runtime_tick_s(self) -> float:
        env = getattr(self, "env", None)
        for name in ("simulation_interval", "traffic_interval"):
            value = getattr(env, name, None)
            if value is not None:
                try:
                    return max(1e-6, float(value))
                except (TypeError, ValueError):
                    continue
        return 1.0

    def _wireless_capacity_metrics(self) -> Dict[str, float]:
        now = self._time() if self.env is not None else (self.last_topology.time_s if self.last_topology else 0.0)
        if self._wireless_capacity_cache_time == now and self._wireless_capacity_cache is not None:
            return dict(self._wireless_capacity_cache)
        scenario = dict(getattr(self.env_adapter, "scenario", {}) or {})
        effective_rb = 0.0
        full_rb = 0.0
        if self.env is not None:
            scheduler = getattr(self.env, "communication_scheduler", None) or getattr(self.env_adapter, "communication_scheduler", None)
            if scheduler is not None and hasattr(scheduler, "getNumberOfRB"):
                try:
                    full_rb = float(scheduler.getNumberOfRB(self.env))
                except Exception:
                    full_rb = 0.0
        budget = scenario.get("wireless_rb_budget")
        if budget is not None:
            try:
                effective_rb = max(1.0, min(full_rb or float(budget), float(budget)))
            except Exception:
                effective_rb = full_rb or 1.0
        else:
            effective_rb = max(1.0, full_rb or 1.0)
        contention = scenario.get("wireless_contention_factor")
        if contention is not None:
            try:
                effective_rb = max(1.0, effective_rb / max(1.0, float(contention)))
            except Exception:
                pass
        full_rb = max(effective_rb, full_rb or effective_rb)
        queue_depth = 0.0
        if self.env is not None:
            try:
                tasks = self.env_adapter.collect_task_snapshot(self.env)
                queue_depth = float(tasks.get("offloading", 0.0) or 0.0) + float(tasks.get("waiting_to_offload", 0.0) or 0.0)
            except Exception:
                queue_depth = 0.0
        pressure = queue_depth / max(1.0, effective_rb)
        result = {
            "effective_wireless_rb": float(effective_rb),
            "full_wireless_rb": float(full_rb),
            "rb_slowdown": max(1.0, full_rb / max(1.0, effective_rb)),
            "wireless_queue_depth": float(queue_depth),
            "wireless_pressure": float(pressure),
        }
        self._wireless_capacity_cache_time = now
        self._wireless_capacity_cache = dict(result)
        return result

    def _candidate_route_reachability(
        self,
        source_node_id: str,
        target_node_id: str,
        payload_mb: float = 0.0,
    ) -> Tuple[bool, int, float, float, float, int, float]:
        source = str(source_node_id)
        target = str(target_node_id)
        if source == target:
            return True, 0, 0.0, 0.0, 0.0, 0, 0.0
        topology = self.last_topology
        if topology is None or source not in topology.nodes or target not in topology.nodes:
            return False, 4, 1.0, 4.0, 4.0, 4, 0.1
        payload_key = round(max(0.0, float(payload_mb)), 6)
        cache_key = (source, target, payload_key)
        cached = self._route_reachability_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._route_reachability_tree(source, payload_key).get(target)
        if result is not None:
            self._route_reachability_cache[cache_key] = result
            return result
        result = (False, 4, 1.0, 4.0, 4.0, 4, 0.1)
        self._route_reachability_cache[cache_key] = result
        return result

    def _route_reachability_tree(
        self,
        source: str,
        payload_key: float,
    ) -> Dict[str, Tuple[bool, int, float, float, float, int, float]]:
        tree_key = (source, payload_key)
        cached = self._route_tree_cache.get(tree_key)
        if cached is not None:
            return cached
        adjacency = self._route_adjacency()
        payload_mbit = payload_key * 8.0
        frontier: List[Tuple[float, int, str, float, float, float, int, float]] = [(0.0, 0, source, 0.0, 0.0, 0.0, 0, float("inf"))]
        best: Dict[Tuple[str, int], float] = {(source, 0): 0.0}
        resolved: Dict[str, Tuple[bool, int, float, float, float, int, float]] = {}
        while frontier:
            cost, hops, node_id, risk, tx_time_s, wireless_tx_time_s, wireless_hops, bottleneck_rate = heapq.heappop(frontier)
            if node_id in resolved:
                continue
            bottleneck = 0.0 if math.isinf(bottleneck_rate) else float(bottleneck_rate)
            resolved[node_id] = (True, hops, min(1.0, risk), tx_time_s, wireless_tx_time_s, wireless_hops, bottleneck)
            if hops >= 4:
                continue
            for neighbor, edge_risk, edge_latency_s, is_wireless, rate_mbps in adjacency.get(node_id, []):
                if neighbor in resolved:
                    continue
                edge_tx_time_s = edge_latency_s + payload_mbit / rate_mbps
                next_hops = hops + 1
                next_risk = min(1.0, risk + edge_risk)
                next_tx_time_s = tx_time_s + edge_tx_time_s
                next_wireless_tx_time_s = wireless_tx_time_s + (edge_tx_time_s if is_wireless else 0.0)
                next_wireless_hops = wireless_hops + (1 if is_wireless else 0)
                next_bottleneck_rate = min(bottleneck_rate, rate_mbps) if is_wireless else bottleneck_rate
                next_cost = next_tx_time_s + 4.0 * next_risk
                key = (neighbor, next_hops)
                if next_cost >= best.get(key, float("inf")):
                    continue
                best[key] = next_cost
                heapq.heappush(
                    frontier,
                    (
                        next_cost,
                        next_hops,
                        neighbor,
                        next_risk,
                        next_tx_time_s,
                        next_wireless_tx_time_s,
                        next_wireless_hops,
                        next_bottleneck_rate,
                    ),
                )
        self._route_tree_cache[tree_key] = resolved
        return resolved

    def _route_adjacency(self) -> Dict[str, List[Tuple[str, float, float, bool, float]]]:
        topology = self.last_topology
        if topology is None:
            return {}
        topology_id = id(topology)
        if self._route_adjacency_topology_id == topology_id:
            return self._route_adjacency_cache
        adjacency: Dict[str, List[Tuple[str, float, float, bool, float]]] = {}
        for edge in topology.edges:
            edge_risk = max(0.0, min(1.0, 1.0 - float(edge.reliability)))
            rate_mbps = max(0.1, float(edge.rate_mbps or 0.0))
            edge_latency_s = max(0.0, float(edge.latency_s or 0.0))
            adjacency.setdefault(str(edge.src), []).append(
                (str(edge.dst), edge_risk, edge_latency_s, bool(edge.is_wireless), rate_mbps)
            )
        self._route_adjacency_topology_id = topology_id
        self._route_adjacency_cache = adjacency
        return adjacency

    def _clear_runtime_metric_caches(self) -> None:
        self._route_adjacency_topology_id = None
        self._route_adjacency_cache.clear()
        self._route_reachability_cache.clear()
        self._route_tree_cache.clear()
        self._wireless_capacity_cache_time = None
        self._wireless_capacity_cache = None

    def _decode_actions(self, actions: Mapping[str, Any], now: float) -> List[LASDMDecision]:
        if not actions:
            return []
        # Accepted formats:
        # {agent_id: {sfc_id: {sfc_node_id: {instance_id, compute_level, bandwidth_level}}}}
        # {sfc_id: {sfc_node_id: {instance_id, compute_level, bandwidth_level}}}
        nested_by_agent = any(key in self.agent_ids for key in actions.keys())
        chain_actions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if nested_by_agent:
            for _agent_id, payload in actions.items():
                if not isinstance(payload, Mapping):
                    continue
                for sfc_id, assignments in payload.items():
                    if isinstance(assignments, Mapping):
                        chain = self.manager.chains.get(str(sfc_id))
                        if chain is not None:
                            controller = self._controller_agent_for_chain(chain)
                            if controller and str(_agent_id) != controller:
                                continue
                        chain_actions.setdefault(str(sfc_id), {}).update(
                            {str(k): _normalize_resource_action(v) for k, v in assignments.items()}
                        )
        else:
            for sfc_id, assignments in actions.items():
                if isinstance(assignments, Mapping):
                    chain_actions.setdefault(str(sfc_id), {}).update(
                        {str(k): _normalize_resource_action(v) for k, v in assignments.items()}
                    )

        decisions: List[LASDMDecision] = []
        selected_lookup = self._selected_candidate_lookup()
        for sfc_id, assignments in chain_actions.items():
            chain = self.manager.chains.get(sfc_id)
            if chain is None:
                continue
            decision = LASDMDecision(sfc_id=sfc_id)
            selected_candidates: Dict[str, Any] = {}
            for sfc_node_id, action_value in assignments.items():
                instance_id = str(action_value.get("instance_id", "") or "")
                if sfc_node_id not in chain.nodes:
                    decision.rejected_reason = SFCFailureReason.INVALID_GRAPH
                    decision.diagnostics["invalid_sfc_node_id"] = sfc_node_id
                    break
                if not instance_id:
                    decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                    decision.diagnostics["missing_instance_id"] = instance_id
                    break
                try:
                    instance = self.manager.directory.get(instance_id)
                except KeyError:
                    decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                    decision.diagnostics["missing_instance_id"] = instance_id
                    break
                decision.assignments[sfc_node_id] = instance_id
                decision.node_mapping[sfc_node_id] = instance.node_id
                decision.resource_allocations[sfc_node_id] = {
                    "compute_level": _resource_level(action_value.get("compute_level"), 1.0),
                    "bandwidth_level": _resource_level(action_value.get("bandwidth_level"), 1.0),
                }
                source = chain.source_node_id
                predecessors = chain.predecessors(sfc_node_id)
                if predecessors:
                    source = (
                        self._completed_predecessor_output_node(sfc_id, predecessors[0])
                        or decision.node_mapping.get(predecessors[0], source)
                    )
                candidate_payload = selected_lookup.get((sfc_id, sfc_node_id, instance_id))
                if candidate_payload is not None:
                    candidate_payload = self._candidate_payload_for_route_source(candidate_payload, str(source))
                    selected_candidates[sfc_node_id] = candidate_payload
                    self.discovery_protocol.mark_selected_candidate(
                        sfc_id,
                        sfc_node_id,
                        instance_id,
                        selected_at_s=now,
                        selected_source_node_id=str(source),
                        selected_metadata=dict(candidate_payload.get("metadata", {}) or {}),
                    )
                decision.routes[sfc_node_id] = [source, instance.node_id] if source != instance.node_id else [instance.node_id]
            spawned = set(getattr(self.runtime_bridge, "sfc_node_to_task", {}) or {})
            completed = set(getattr(self.runtime_bridge, "completed_nodes", {}).get(sfc_id, set()) or set())
            ready_unspawned = self._ready_unspawned_nodes(chain, spawned, completed)
            required = {node_id for node_id in ready_unspawned if not chain.nodes[node_id].optional}
            missing = sorted(required - set(decision.assignments))
            if decision.rejected_reason is None and missing:
                decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                decision.diagnostics["missing_sfc_node_ids"] = missing
            if selected_candidates:
                decision.diagnostics["selected_candidates"] = selected_candidates
            if decision.resource_allocations:
                decision.diagnostics["selected_resource_allocations"] = {
                    key: dict(value) for key, value in decision.resource_allocations.items()
                }
            decisions.append(decision)
        return decisions

    def _selected_candidate_lookup(self) -> Dict[tuple[str, str, str], Dict[str, Any]]:
        lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for observation in (self.last_observations or {}).values():
            for candidate_set in observation.get("candidate_sets", []) or []:
                sfc_id = str(candidate_set.get("sfc_id", ""))
                sfc_node_id = str(candidate_set.get("sfc_node_id", ""))
                for candidate in candidate_set.get("raw_candidates", []) or []:
                    instance_id = str(candidate.get("instance_id", ""))
                    if instance_id:
                        lookup[(sfc_id, sfc_node_id, instance_id)] = dict(candidate)
        return lookup

    def _candidate_payload_for_route_source(self, candidate_payload: Mapping[str, Any], source_node_id: str) -> Dict[str, Any]:
        payload = dict(candidate_payload)
        metadata = dict(payload.get("metadata", {}) or {})
        source = str(source_node_id)
        source_fields = {
            "route_available": "route_available_by_source",
            "route_hops": "route_hops_by_source",
            "route_tx_time_s": "route_tx_time_s_by_source",
            "route_topology_tx_time_s": "route_topology_tx_time_s_by_source",
            "route_wireless_tx_time_s": "route_wireless_tx_time_s_by_source",
            "expected_rb_wait_s": "expected_rb_wait_s_by_source",
            "expected_wireless_tx_time_s": "expected_wireless_tx_time_s_by_source",
            "wireless_pressure": "wireless_pressure_by_source",
            "wireless_hops": "wireless_hops_by_source",
            "rb_slowdown": "rb_slowdown_by_source",
            "expected_runtime_penalty_s": "expected_runtime_penalty_s_by_source",
            "deadline_slack_s": "deadline_slack_s_by_source",
            "utility_prior": "utility_prior_by_source",
            "chain_deadline_s": "chain_deadline_s_by_source",
            "chain_remaining_deadline_s": "chain_remaining_deadline_s_by_source",
            "remaining_deadline_ratio": "remaining_deadline_ratio_by_source",
            "remaining_function_count": "remaining_function_count_by_source",
            "resource_capacity_total": "resource_capacity_total_by_source",
            "resource_reserved_count": "resource_reserved_count_by_source",
            "resource_available_slots": "resource_available_slots_by_source",
            "resource_available_ratio": "resource_available_ratio_by_source",
            "capacity_exhausted": "capacity_exhausted_by_source",
            "hops_from_prev_function": "hops_from_prev_function_by_source",
            "hops_from_prev_function_norm": "hops_from_prev_function_norm_by_source",
            "sequential_deadline_feasible": "sequential_deadline_feasible_by_source",
            "link_similarity": "link_similarity_by_source",
            "node_template_similarity": "node_template_similarity_by_source",
            "semantic_quality_before": "semantic_quality_before_by_source",
            "semantic_cumulative_quality_if_selected": "semantic_cumulative_quality_if_selected_by_source",
            "semantic_link_truth_score": "semantic_link_truth_score_by_source",
        }
        for field, mapping_name in source_fields.items():
            mapping = metadata.get(mapping_name)
            if isinstance(mapping, Mapping) and source in mapping:
                metadata[field] = mapping[source]
        metadata["actual_route_source_node_id"] = source
        payload["metadata"] = metadata
        payload["actual_route_source_node_id"] = source
        return payload

    def _expire_deadlines(self, current_time: float) -> None:
        for chain in list(self.manager.chains.values()):
            if chain.status != GraphStatus.RUNNING or chain.submit_time is None:
                continue
            if current_time - chain.submit_time > chain.qos.deadline_s:
                self.manager.fail(
                    chain.sfc_id,
                    SFCFailureReason.DEADLINE_MISSED,
                    current_time,
                    status=GraphStatus.TIMED_OUT,
                )

    def _update_topology(self, now: float) -> DynamicTopology:
        if self.env is None:
            raise RuntimeError("SemanticTopologyMARLEnv topology requires a live AirFogSim env")
        topology = self.topology_builder.build(self.env, current_time=now)
        self.last_topology = topology
        self._clear_runtime_metric_caches()
        self.temporal_buffer.append(topology)
        self.topology_trace.append(_topology_trace_summary(topology))
        return topology

    def _time(self) -> float:
        if self.env is None:
            raise RuntimeError("SemanticTopologyMARLEnv requires a live AirFogSim runtime clock")
        value = current_runtime_time(self.env)
        if value is None:
            raise RuntimeError("AirFogSim env does not expose a runtime clock")
        return float(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [_jsonable(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _transition_info_summary(info: Mapping[str, Any]) -> Dict[str, Any]:
    runtime_report = dict(info.get("runtime_report", {}) or {})
    runtime_progress = dict(runtime_report.get("runtime_progress", {}) or {})
    bridge_prepare = dict(runtime_report.get("bridge_prepare", {}) or {})
    reward_aux = {
        str(key): value
        for key, value in dict(info.get("reward_aux", {}) or {}).items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
    summary = dict(info.get("summary", {}) or {})
    summary.pop("decisions", None)
    payload = {
        "time_s": info.get("time_s"),
        "step_count": info.get("step_count"),
        "runtime_step_count": info.get("runtime_step_count"),
        "decision_count": len(info.get("decisions", []) or []),
        "summary": summary,
        "runtime_report": {
            "runtime_progress": {
                key: runtime_progress.get(key)
                for key in (
                    "ticks",
                    "progress",
                    "done_tasks",
                    "failed_tasks",
                    "active_tasks",
                    "created_tasks",
                    "elapsed_s",
                )
                if key in runtime_progress
            },
            "bridge_prepare": {
                key: bridge_prepare.get(key)
                for key in (
                    "submitted_tasks",
                    "pending_tasks",
                    "active_tasks",
                    "completed_tasks",
                    "failed_tasks",
                )
                if key in bridge_prepare
            },
            "sequential_replanning_count": len(runtime_report.get("sequential_replanning", []) or []),
        },
        "reward_aux": reward_aux,
        "message_overhead": info.get("message_overhead", {}),
    }
    return _jsonable(payload)


def _topology_trace_summary(topology: DynamicTopology) -> Dict[str, Any]:
    node_types: Dict[str, int] = {}
    regions: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}
    service_nodes = 0
    avg_load = 0.0
    for node in topology.nodes.values():
        node_types[str(node.node_type)] = node_types.get(str(node.node_type), 0) + 1
        regions[str(node.region_id)] = regions.get(str(node.region_id), 0) + 1
        service_nodes += 1 if int(node.service_count or 0) > 0 else 0
        avg_load += float(node.load_ratio or 0.0)
    for edge in topology.edges:
        edge_types[str(edge.link_type)] = edge_types.get(str(edge.link_type), 0) + 1
    node_count = len(topology.nodes)
    return {
        "time_s": float(topology.time_s),
        "node_count": node_count,
        "edge_count": len(topology.edges),
        "service_node_count": service_nodes,
        "avg_node_load_ratio": avg_load / max(1, node_count),
        "node_types": node_types,
        "regions": regions,
        "edge_types": edge_types,
        "metadata": dict(topology.metadata),
    }


def _reward_aux_from_decisions(decisions: Sequence[LASDMDecision]) -> Dict[str, float]:
    selected: List[Mapping[str, Any]] = []
    for decision in decisions:
        diagnostics = dict(getattr(decision, "diagnostics", {}) or {})
        for candidate in dict(diagnostics.get("selected_candidates", {}) or {}).values():
            if isinstance(candidate, Mapping):
                selected.append(candidate)
    if not selected:
        return {}
    semantic_scores = [_to_float(item.get("semantic_score"), 0.0) for item in selected]
    semantic_cumulative_qualities = []
    stale_values = []
    topology_risks = []
    mobility_risks = []
    load_ratios = []
    cold_starts = []
    route_hops = []
    expected_penalties = []
    deadline_slacks = []
    utility_priors = []
    route_tx_times = []
    rb_waits = []
    wireless_pressures = []
    wireless_hops = []
    route_available_values = []
    resource_available_ratios = []
    remaining_deadline_ratios = []
    function_budgets = []
    scenario_concurrency_values = []
    semantic_group_counts: Dict[str, int] = {}
    route_unavailable = 0
    for item in selected:
        metadata = dict(item.get("metadata", {}) or {})
        semantic_cumulative_qualities.append(
            _to_float(metadata.get("semantic_cumulative_quality_if_selected"), _to_float(item.get("semantic_score"), 1.0))
        )
        stale = (
            _to_float(item.get("staleness_s"), 0.0) > 0.0
            or bool(item.get("stale", False))
            or str(metadata.get("semantic_group", "")) == "stale_clone_exact"
        )
        stale_values.append(1.0 if stale else 0.0)
        topology_risks.append(_to_float(metadata.get("topology_risk"), _to_float(item.get("topology_risk"), 0.0)))
        mobility_risks.append(_to_float(metadata.get("mobility_risk"), _to_float(item.get("mobility_risk"), 0.0)))
        load_ratios.append(_to_float(metadata.get("load_ratio"), 0.0))
        cold_starts.append(_to_float(metadata.get("cold_start_s"), 0.0))
        route_hops.append(_to_float(metadata.get("route_hops"), _to_float(item.get("route_hops"), 0.0)))
        expected_penalties.append(_to_float(metadata.get("expected_runtime_penalty_s"), 0.0))
        deadline_slacks.append(_to_float(metadata.get("deadline_slack_s"), 0.0))
        utility_priors.append(_to_float(metadata.get("utility_prior"), 0.0))
        function_budgets.append(_to_float(metadata.get("function_budget_s"), 0.0))
        scenario_concurrency_values.append(_to_float(metadata.get("scenario_max_concurrent_sfcs"), 0.0))
        route_tx_times.append(_to_float(metadata.get("route_tx_time_s"), 0.0))
        rb_waits.append(_to_float(metadata.get("expected_rb_wait_s"), 0.0))
        wireless_pressures.append(_to_float(metadata.get("wireless_pressure"), 0.0))
        wireless_hops.append(_to_float(metadata.get("wireless_hops"), 0.0))
        route_available = _to_float(metadata.get("route_available"), _to_float(item.get("route_available"), 1.0))
        route_available_values.append(1.0 if route_available > 0.0 else 0.0)
        resource_available_ratios.append(_to_float(metadata.get("resource_available_ratio"), 1.0))
        remaining_deadline_ratios.append(_to_float(metadata.get("remaining_deadline_ratio"), 1.0))
        semantic_group = str(metadata.get("semantic_group", item.get("semantic_group", "unknown")) or "unknown")
        semantic_group_counts[semantic_group] = semantic_group_counts.get(semantic_group, 0) + 1
        truth_relation = str(metadata.get("semantic_link_truth_relation", "") or "unknown")
        semantic_group_counts[f"truth_relation_{truth_relation}"] = semantic_group_counts.get(f"truth_relation_{truth_relation}", 0) + 1
        if route_available <= 0.0:
            route_unavailable += 1
    aux = {
        "mean_semantic_top_score": _mean(semantic_scores),
        "stale_remote_ratio": _mean(stale_values),
        "load_imbalance": _mean(load_ratios),
        "topology_risk": _mean(topology_risks),
        "mobility_risk": _mean(mobility_risks),
        "cold_start_s": _mean(cold_starts),
        "route_hops": _mean(route_hops),
        "route_unavailable_ratio": route_unavailable / max(1, len(selected)),
        "expected_runtime_penalty_s": _mean(expected_penalties),
        "deadline_slack_s": _mean(deadline_slacks),
        "utility_prior": _mean(utility_priors),
        "route_tx_time_s": _mean(route_tx_times),
        "expected_rb_wait_s": _mean(rb_waits),
        "wireless_pressure": _mean(wireless_pressures),
        "wireless_hops": _mean(wireless_hops),
        "selected_route_available_mean": _mean(route_available_values),
        "selected_resource_available_ratio_mean": _mean(resource_available_ratios),
        "selected_remaining_deadline_ratio_mean": _mean(remaining_deadline_ratios),
        "selected_deadline_slack_mean": _mean(deadline_slacks),
        "selected_expected_runtime_penalty_mean": _mean(expected_penalties),
        "selected_topology_risk_mean": _mean(topology_risks),
        "selected_mobility_risk_mean": _mean(mobility_risks),
        "selected_stale_remote_ratio": _mean(stale_values),
        "selected_semantic_group_count": float(len(selected)),
        "selected_semantic_cumulative_quality_mean": _mean(semantic_cumulative_qualities),
        "reward_time_scale_s": max(1.0, _mean(function_budgets)),
        "reward_route_hop_scale": 4.0,
        "reward_wireless_pressure_scale": max(1.0, _mean(scenario_concurrency_values) / 8.0),
    }
    for group, count in semantic_group_counts.items():
        key = "".join(char if char.isalnum() else "_" for char in group.lower()).strip("_") or "unknown"
        aux[f"selected_semantic_group_{key}_count"] = float(count)
    return aux


def _gs2l_chain_progress_snapshot(
    bridge: LASDMRuntimeBridge,
    chains: Mapping[str, LASDMServiceChain],
) -> Dict[str, float]:
    completed_by_chain = getattr(bridge, "completed_nodes", {}) or {}
    progress: Dict[str, float] = {}
    for sfc_id, chain in chains.items():
        order = list(chain.topological_order())
        if not order:
            continue
        completed = set(completed_by_chain.get(sfc_id, set()) or set())
        progress[str(sfc_id)] = len(completed.intersection(order)) / max(1, len(order))
    return progress


def _gs2l_progress_aux(
    previous_progress: Mapping[str, float],
    current_progress: Mapping[str, float],
    previous_done_tasks: int,
    previous_failed_tasks: int,
    bridge: LASDMRuntimeBridge,
    chains: Mapping[str, LASDMServiceChain],
) -> Dict[str, float]:
    all_sfc_ids = set(previous_progress) | set(current_progress)
    progress_delta = 0.0
    for sfc_id in all_sfc_ids:
        progress_delta += max(0.0, float(current_progress.get(sfc_id, 0.0)) - float(previous_progress.get(sfc_id, 0.0)))
    current_done_tasks = len(getattr(bridge, "processed_done_tasks", set()) or set())
    current_failed_tasks = len(getattr(bridge, "processed_failed_tasks", set()) or set())
    stage_counts = [len(list(chain.topological_order())) for chain in chains.values() if len(list(chain.topological_order())) > 0]
    progress_values = [float(value) for value in current_progress.values()]
    return {
        "gs2l_chain_progress_delta": progress_delta,
        "gs2l_chain_progress_mean": _mean(progress_values),
        "gs2l_stage_success_delta": float(max(0, current_done_tasks - int(previous_done_tasks))),
        "gs2l_stage_failure_delta": float(max(0, current_failed_tasks - int(previous_failed_tasks))),
        "gs2l_stage_count_mean": _mean([float(value) for value in stage_counts]) if stage_counts else 1.0,
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    return sum(float(item) for item in values) / len(values) if values else 0.0


def _expand_neighbor_map_by_radius(neighbor_map: Mapping[str, Iterable[str]], radius_hops: int) -> Dict[str, Dict[str, int]]:
    radius = max(1, int(radius_hops or 1))
    adjacency = {str(region): {str(item) for item in values} for region, values in neighbor_map.items()}
    for region, values in list(adjacency.items()):
        for neighbor in values:
            adjacency.setdefault(neighbor, set()).add(region)
    expanded: Dict[str, Dict[str, int]] = {}
    for region in sorted(adjacency):
        distances: Dict[str, int] = {}
        frontier = {region}
        visited = {region}
        for hop in range(1, radius + 1):
            next_frontier: set[str] = set()
            for item in frontier:
                for neighbor in adjacency.get(item, set()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    distances[neighbor] = hop
            frontier = next_frontier
            if not frontier:
                break
        expanded[region] = dict(sorted(distances.items()))
    return expanded
