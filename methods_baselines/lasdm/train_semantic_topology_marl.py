from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

METHOD_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.graph_observation import flatten_observation
from airfogsim.lasdm.env_adapter import LASDMEnvAdapter
from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory
from airfogsim.lasdm.manager import LASDMManager
from airfogsim.lasdm.marl_env import MARLEnvConfig, SemanticTopologyMARLEnv
from airfogsim.lasdm.marl_policy import IPPOPolicy, policy_from_name
from airfogsim.lasdm.marl_reward import SFCReward, SFCRewardConfig
from airfogsim.lasdm.marl_trainer import HeuristicEvaluator, IPPOTrainer, write_reward_curve
from airfogsim.lasdm.model import LASDMServiceChain
from airfogsim.lasdm.orchestrator import LASDMOrchestrator
from airfogsim.lasdm.runtime_bridge import LASDMRuntimeBridge
from airfogsim.lasdm.topology_builder import TopologyBuilder

DEFAULT_CONFIG = os.path.join(METHOD_ROOT, "configs", "semantic_topology_marl.yaml")
PHYSICAL_VEHICLE_COUNT = 100
PHYSICAL_UAV_COUNT = 20
PHYSICAL_RSU_COUNT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate semantic-topology LASDM MARL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--policy", default="semantic_greedy", choices=["semantic_greedy", "topology_greedy", "random_valid", "ippo"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--service-role-sweep", default="full_hybrid")
    parser.add_argument("--output-dir", default="experiment_artifacts/raw_data/lasdm_semantic_topology_marl/train")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    episodes = int(args.episodes if args.episodes is not None else config.get("training", {}).get("episodes", 10))
    max_steps = int(args.max_steps if args.max_steps is not None else config.get("training", {}).get("max_steps", 100))
    env = build_offline_env(
        config,
        seed=args.seed,
        max_steps=max_steps,
        scenario=args.scenario,
        service_role_sweep=args.service_role_sweep,
        attach_runtime=True,
        baseline="proposed_semantic_topology_marl" if args.policy == "ippo" else args.policy,
    )
    try:
        observations = env.reset()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.policy == "ippo":
            obs_dim = max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
            marl_cfg = dict(config.get("marl", {}) or {})
            max_candidates = int(marl_cfg.get("max_candidates", 16))
            critic_agents = int(marl_cfg.get("ippo_critic_agent_count", len(observations) or 4) or 4)
            policy = IPPOPolicy(
                observation_dim=obs_dim,
                max_candidates=max_candidates,
                seed=args.seed,
                centralized_critic=bool(marl_cfg.get("ippo_centralized_critic", True)),
                critic_observation_dim=obs_dim * max(1, critic_agents),
                max_critic_agents=max(1, critic_agents),
                utility_prior_logit_weight=float(marl_cfg.get("ippo_utility_prior_logit_weight", 2.5) or 2.5),
                route_unavailable_penalty=float(
                    marl_cfg.get("ippo_route_unavailable_penalty", marl_cfg.get("ippo_expert_route_unavailable_penalty", 20.0))
                    or 20.0
                ),
                include_semantic_features=bool(marl_cfg.get("include_semantic_features", True)),
                include_topology_features=bool(marl_cfg.get("include_topology_features", True)),
                include_temporal_features=bool(marl_cfg.get("include_temporal_features", True)),
                use_region_encoder=bool(marl_cfg.get("ippo_use_region_encoder", True)),
                learnable_prior=bool(marl_cfg.get("ippo_learnable_prior", True)),
                prior_l2_coef=float(marl_cfg.get("ippo_prior_l2_coef", 1e-3) or 0.0),
                device=str(marl_cfg.get("ippo_device", "") or "") or None,
            )
            trainer = IPPOTrainer(env, policy)
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
        else:
            policy = policy_from_name(args.policy, seed=args.seed)
            evaluator = HeuristicEvaluator(env, policy)
            rows = evaluator.run(episodes=episodes, max_steps=max_steps)
            write_reward_curve(output_dir / "reward_curve.csv", rows)
            env.write_traces(str(output_dir))

        payload = {
            "completed": True,
            "policy": args.policy,
            "episodes": episodes,
            "max_steps": max_steps,
            "output_dir": str(output_dir),
            "last_metrics": rows[-1].to_dict() if rows else {},
        }
        (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        _close_runtime_env(getattr(env, "env", None))


def build_offline_env(
    config: Dict[str, Any],
    seed: int = 0,
    max_steps: int = 100,
    scenario: Optional[str | Mapping[str, Any]] = None,
    service_role_sweep: str = "full_hybrid",
    attach_runtime: bool = False,
    baseline: str = "semantic_topology_marl",
) -> SemanticTopologyMARLEnv:
    scenario_cfg = _resolve_scenario(config, scenario)
    run_config = _materialize_offline_config(config, scenario_cfg, service_role_sweep, seed)
    directory = ServiceInstanceDirectory()
    for item in run_config.get("service_instances", []):
        directory.register(ServiceInstance.from_dict(item))
    manager = LASDMManager(
        directory=directory,
        orchestrator=LASDMOrchestrator(directory, score_weights=run_config.get("orchestrator", {}).get("score_weights", {})),
    )
    chains = [LASDMServiceChain.from_dict(item) for item in run_config.get("service_chains", [])]
    marl_cfg = run_config.get("marl", {})
    exchange_cfg = run_config.get("semantic_exchange", {})
    topology_cfg = run_config.get("topology", {})
    env_cfg = MARLEnvConfig(
        semantic_top_k=int(marl_cfg.get("semantic_top_k", 8)),
        min_semantic_similarity=float(marl_cfg.get("min_semantic_similarity", -1.0)),
        temporal_window=int(marl_cfg.get("temporal_window", 4)),
        max_steps=max_steps,
        auto_exchange=bool(marl_cfg.get("auto_exchange", True)),
        include_remote_candidates=bool(marl_cfg.get("include_remote_candidates", True)),
        include_topology_features=bool(marl_cfg.get("include_topology_features", True)),
        include_temporal_features=bool(marl_cfg.get("include_temporal_features", True)),
        include_semantic_features=bool(marl_cfg.get("include_semantic_features", True)),
        auto_plan_unassigned=bool(marl_cfg.get("auto_plan_unassigned", False)),
        semantic_exchange_ttl_s=float(exchange_cfg.get("ttl_s", 5.0)),
        semantic_exchange_radius_hops=int(exchange_cfg.get("radius_hops", 1)),
        semantic_exchange_fixed_delay_s=float(exchange_cfg.get("fixed_delay_s", 0.10)),
        semantic_exchange_per_hop_delay_s=float(exchange_cfg.get("per_hop_delay_s", 0.02)),
        semantic_exchange_top_k_per_agent=int(exchange_cfg.get("top_k_per_agent", 32)),
        semantic_exchange_compressed_dim=int(exchange_cfg.get("compressed_dim", 64)),
        semantic_exchange_quantization_bits=int(exchange_cfg.get("quantization_bits", 8)),
        semantic_encoder_backend=str(exchange_cfg.get("encoder_backend", "hash")),
        semantic_encoder_model_name=str(exchange_cfg.get("sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_encoder_hash_dim=int(exchange_cfg.get("hash_dim", 384)),
        semantic_encoder_batch_size=int(exchange_cfg.get("encoder_batch_size", 64)),
        semantic_encoder_device=str(exchange_cfg.get("encoder_device", "")),
        scenario_name=str(scenario_cfg.get("name", "default")),
        service_role_sweep=str(service_role_sweep),
        runtime_ticks_per_marl_step=int(marl_cfg.get("runtime_ticks_per_marl_step", 1)),
        runtime_max_ticks_after_action=int(marl_cfg.get("runtime_max_ticks_after_action", 25)),
        runtime_stop_when_progress=bool(marl_cfg.get("runtime_stop_when_progress", True)),
        route_hop_floor_s=float(marl_cfg.get("route_hop_floor_s", 1.0) or 1.0),
        global_candidate_catalog=bool(marl_cfg.get("global_candidate_catalog", False)),
        region_agents=tuple(str(item) for item in topology_cfg.get("region_agents", []) or []),
    )
    semantic_env = SemanticTopologyMARLEnv(
        manager=manager,
        chains=chains,
        reward_fn=_build_reward_fn(marl_cfg),
        config=env_cfg,
    )
    if attach_runtime:
        _attach_runtime_env(semantic_env, config, scenario_cfg, seed, baseline)
    return semantic_env


def _attach_runtime_env(
    env: SemanticTopologyMARLEnv,
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    baseline: str,
) -> None:
    adapter = LASDMEnvAdapter(
        config=dict(config),
        runtime_config=dict(config.get("runtime", {}) or {}),
        baseline_name=str(baseline),
        seed=int(seed),
        scenario=dict(scenario),
        manager=env.manager,
        directory=env.manager.directory,
        chains=env.chains,
    )
    adapter._set_seed(seed)
    air_env = adapter._build_env(config.get("runtime", {}))
    adapter._warmup_env(air_env, int(dict(config.get("runtime", {}) or {}).get("warmup_steps", 0)))
    adapter.apply_node_aliases_to_directory(air_env)
    adapter.bind_service_instances_to_runtime_nodes(air_env, scenario)
    adapter.bind_chains_to_runtime_nodes(air_env, env.chains, scenario)
    env.env = air_env
    env.env_adapter = adapter
    env.runtime_bridge = LASDMRuntimeBridge(env.manager, env_adapter=adapter)
    env.topology_builder = TopologyBuilder(env_adapter=adapter, directory=env.manager.directory)


def _close_runtime_env(env: Any) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def _build_reward_fn(marl_cfg: Mapping[str, Any]) -> SFCReward:
    reward_cfg = dict(marl_cfg.get("reward", {}) or {})
    if not reward_cfg:
        return SFCReward()
    allowed = set(SFCRewardConfig.__dataclass_fields__)
    values = {}
    for key, value in reward_cfg.items():
        if key in allowed:
            values[key] = float(value)
    return SFCReward(SFCRewardConfig(**values))


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_scenario(config: Mapping[str, Any], scenario: Optional[str | Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(scenario, Mapping):
        return dict(scenario)
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [])
    if scenario is None:
        return dict(scenarios[0]) if scenarios else {"name": "default"}
    for item in scenarios:
        if str(item.get("name")) == str(scenario):
            return dict(item)
    raise ValueError(f"Unknown semantic-topology scenario: {scenario}")


def _materialize_offline_config(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    service_role_sweep: str,
    seed: int,
) -> Dict[str, Any]:
    run_config = copy.deepcopy(dict(config))
    role_name = str(service_role_sweep or "full_hybrid")
    allowed_node_types = _allowed_service_node_types(run_config, role_name)
    base_instances = [dict(item) for item in run_config.get("service_instances", [])]
    instances = [_scenario_instance(item, scenario) for item in base_instances if str(item.get("node_type")) in allowed_node_types]
    instances.extend(_materialized_service_instances(run_config, scenario, allowed_node_types, seed))
    instances = inject_semantic_topology_decoys(instances, scenario, seed, allowed_node_types)
    forced_service_node_id = scenario.get("forced_service_node_id") or scenario.get("force_service_node_id")
    if forced_service_node_id:
        forced_service_node_id = str(forced_service_node_id)
        instances = [item for item in instances if str(item.get("node_id")) == forced_service_node_id]
    instances = _apply_local_catalog_visibility(instances, scenario, seed)
    run_config["service_instances"] = instances

    exchange_cfg = dict(run_config.get("semantic_exchange", {}) or {})
    if scenario.get("exchange_ttl_s") is not None:
        exchange_cfg["ttl_s"] = float(scenario["exchange_ttl_s"])
    if scenario.get("exchange_radius_hops") is not None:
        exchange_cfg["radius_hops"] = int(scenario["exchange_radius_hops"])
    compressed_dim = scenario.get("compressed_dim", scenario.get("semantic_compressed_dim"))
    if compressed_dim is not None:
        exchange_cfg["compressed_dim"] = int(compressed_dim)
    if scenario.get("exchange_top_k") is not None:
        exchange_cfg["top_k_per_agent"] = int(scenario["exchange_top_k"])
    run_config["semantic_exchange"] = exchange_cfg
    run_config["service_chains"] = _scenario_chains(run_config, scenario, role_name, seed)
    return run_config


def _allowed_service_node_types(config: Mapping[str, Any], role_name: str) -> set[str]:
    sweeps = dict(config.get("topology", {}).get("service_role_sweeps", {}) or {})
    allowed = sweeps.get(role_name) or sweeps.get("full_hybrid") or ["rsu", "cloud_server", "vehicle", "uav"]
    return {str(item) for item in allowed}


def _scenario_instance(raw: Mapping[str, Any], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(dict(raw))
    load_multiplier = _load_multiplier(scenario)
    used = dict(item.get("used", {}) or {})
    capacity = dict(item.get("capacity", {}) or {})
    for key, value in list(used.items()):
        cap = float(capacity.get(key, value or 1.0) or 1.0)
        used[key] = min(0.98 * cap, float(value) * load_multiplier)
    item["used"] = used
    item["current_load"] = min(
        int(item.get("max_concurrency", 1) or 1),
        max(int(item.get("current_load", 0) or 0), int(math.floor(max(0.0, load_multiplier - 1.0)))),
    )
    item.setdefault("metadata", {})
    item["metadata"].update(
        {
            "scenario": scenario.get("name", "default"),
            "scenario_load_multiplier": load_multiplier,
        }
    )
    return item


def _materialized_service_instances(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    allowed_node_types: set[str],
    seed: int,
) -> List[Dict[str, Any]]:
    service_specs = _service_specs(config)
    service_nodes = dict(scenario.get("service_nodes", {}) or {})
    load_multiplier = _load_multiplier(scenario)
    supply_ratio = float(scenario.get("service_supply_ratio", 1.0) or 1.0)
    rng = random.Random(int(seed))
    generated: List[Dict[str, Any]] = []
    for node_type, max_nodes in (("vehicle", PHYSICAL_VEHICLE_COUNT), ("uav", PHYSICAL_UAV_COUNT), ("rsu", PHYSICAL_RSU_COUNT)):
        if node_type not in allowed_node_types:
            continue
        requested = int(service_nodes.get(node_type, 0) or 0)
        node_count = min(max_nodes, max(0, int(round(requested * supply_ratio))))
        for node_index in range(node_count):
            node_id = _materialized_node_id(node_type, node_index)
            region_id = f"RSU_{node_index % PHYSICAL_RSU_COUNT}" if node_type != "rsu" else node_id
            for spec_index, spec in enumerate(service_specs):
                capacity = _materialized_capacity(node_type)
                used_cpu = min(capacity["cpu"] * 0.9, (0.25 + 0.08 * ((node_index + spec_index) % 3)) * capacity["cpu"] * load_multiplier)
                generated.append(
                    {
                        "instance_id": f"{node_id.lower()}_{spec['service_id']}_{seed}_{spec_index}",
                        "service_id": spec["service_id"],
                        "node_id": node_id,
                        "node_type": node_type,
                        "region_id": region_id,
                        "capabilities": list(spec["required_capabilities"]),
                        "input_semantic": spec["input_semantic"],
                        "output_semantic": spec["output_semantic"],
                        "capacity": capacity,
                        "used": {
                            "cpu": used_cpu,
                            "memory": min(capacity["memory"] * 0.75, float(spec["memory_mb"]) * load_multiplier),
                            "storage": min(capacity["storage"] * 0.5, float(spec["storage_mb"]) * load_multiplier),
                        },
                        "max_concurrency": _materialized_concurrency(node_type),
                        "current_load": int(rng.random() < min(0.75, max(0.0, load_multiplier - 1.0) * 0.35)),
                        "reliability_score": _materialized_quality(node_type, 0.90, 0.985),
                        "accuracy_score": _materialized_quality(node_type, 0.88, 0.98),
                        "trust_score": _materialized_quality(node_type, 0.88, 0.99),
                        "cold_start_s": _materialized_cold_start(node_type, spec_index),
                        "metadata": {
                            "scenario_materialized": True,
                            "scenario": scenario.get("name", "default"),
                            "role_control": "service_supply",
                            "load_ratio_override": 0.20 if node_type in {"vehicle", "uav"} else 0.05,
                            "topology_risk": 0.30 if node_type in {"vehicle", "uav"} else 0.08,
                            "mobility_risk": 0.35 if node_type in {"vehicle", "uav"} else 0.05,
                        },
                    }
                )
    return generated


def _scenario_chains(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    service_role_sweep: str,
    seed: int,
) -> List[Dict[str, Any]]:
    base = [copy.deepcopy(dict(item)) for item in config.get("service_chains", [])]
    if not base:
        return []
    scenario_cfg = _scenario_with_default_task_classes(config, scenario)
    arrival_rate = float(scenario_cfg.get("arrival_rate_sfc_per_s", 0.4) or 0.4)
    if scenario_cfg.get("request_count") is not None:
        target_count = max(len(base), int(scenario_cfg.get("request_count") or len(base)))
    elif scenario_cfg.get("max_concurrent_sfcs") is not None:
        target_count = max(len(base), int(scenario_cfg.get("max_concurrent_sfcs") or len(base)))
    else:
        target_count = max(len(base), min(24, int(math.ceil(arrival_rate * 10.0))))
    task_nodes = dict(scenario_cfg.get("task_nodes", {}) or {})
    vehicle_tasks = max(0, int(task_nodes.get("vehicle", 0) or 0))
    uav_tasks = max(0, int(task_nodes.get("uav", 0) or 0))
    load_multiplier = _load_multiplier(scenario_cfg)
    chains: List[Dict[str, Any]] = []
    for index in range(target_count):
        raw = copy.deepcopy(base[index % len(base)])
        raw["sfc_id"] = f"{raw['sfc_id']}__{scenario_cfg.get('name', 'default')}__{service_role_sweep}__seed_{seed}__req_{index}"
        source = _task_source_node(index, vehicle_tasks, uav_tasks)
        raw["source_node_id"] = source
        raw["sink_node_id"] = source
        apply_semantic_scenario_overrides(raw, scenario_cfg, seed=seed, request_index=index)
        context = dict(raw.get("context", {}) or {})
        context.update(
            {
                "scenario": scenario_cfg.get("name", "default"),
                "service_role_sweep": service_role_sweep,
                "preferred_region_id": _scenario_preferred_region(scenario_cfg, index, source),
                "task_node_counts": task_nodes,
                "service_node_counts": dict(scenario_cfg.get("service_nodes", {}) or {}),
                "arrival_rate_sfc_per_s": arrival_rate,
                "load_multiplier": load_multiplier,
                "risk_level": float(scenario_cfg.get("risk_level", context.get("risk_level", 0.0)) or 0.0),
                "mobility_multiplier": _mobility_multiplier(scenario_cfg),
                "network_latency_multiplier": _network_latency_multiplier(scenario_cfg),
                "semantic_mismatch_penalty_s_per_unit": float(
                    scenario_cfg.get(
                        "semantic_mismatch_penalty_s_per_unit",
                        context.get("semantic_mismatch_penalty_s_per_unit", 0.0),
                    )
                    or 0.0
                ),
                "semantic_min_score": float(
                    context.get(
                        "semantic_min_score",
                        scenario_cfg.get("semantic_min_score", 0.0),
                    )
                    or 0.0
                ),
            }
        )
        raw["context"] = context
        chains.append(raw)
    return chains


def _scenario_with_default_task_classes(config: Mapping[str, Any], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_cfg = copy.deepcopy(dict(scenario))
    experiment_cfg = dict(config.get("semantic_topology_experiment", {}) or {})
    if "task_classes" not in scenario_cfg and "task_classes" in experiment_cfg:
        scenario_cfg["task_classes"] = copy.deepcopy(experiment_cfg["task_classes"])
    return scenario_cfg


def _select_task_class(
    scenario: Mapping[str, Any],
    seed: int,
    request_index: int,
) -> Optional[tuple[str, Dict[str, Any]]]:
    raw_classes = scenario.get("task_classes")
    if not isinstance(raw_classes, Mapping):
        return None
    classes: List[tuple[str, Dict[str, Any], float]] = []
    for name, raw in raw_classes.items():
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        ratio = max(0.0, float(item.get("ratio", 0.0) or 0.0))
        if ratio <= 0.0 or item.get("deadline_s") is None:
            continue
        classes.append((str(name), item, ratio))
    if not classes:
        return None
    total = sum(ratio for _name, _item, ratio in classes)
    threshold = random.Random(int(seed) * 104729 + int(request_index) * 9176 + 53).random() * total
    cumulative = 0.0
    for name, item, ratio in classes:
        cumulative += ratio
        if threshold <= cumulative:
            return name, item
    name, item, _ratio = classes[-1]
    return name, item


def apply_semantic_scenario_overrides(
    chain: Dict[str, Any],
    scenario: Mapping[str, Any],
    seed: int = 0,
    request_index: int = 0,
) -> Dict[str, Any]:
    """Apply semantic-runtime scenario overrides to a materialized SFC dict."""

    qos = dict(chain.get("qos", {}) or {})
    task_class = _select_task_class(scenario, seed, request_index)
    if task_class:
        class_name, class_cfg = task_class
        deadline = class_cfg.get("deadline_s")
        qos["deadline_s"] = float(deadline)
        context = dict(chain.get("context", {}) or {})
        context["task_class"] = class_name
        context["task_class_deadline_s"] = float(deadline)
        if class_cfg.get("semantic_min_score") is not None:
            context["semantic_min_score"] = float(class_cfg.get("semantic_min_score") or 0.0)
        allowed_types = class_cfg.get("allowed_node_types")
        if allowed_types:
            context["allowed_node_types"] = [str(item) for item in allowed_types]
        chain["context"] = context
    else:
        deadline = scenario.get("qos_deadline_s", scenario.get("deadline_s"))
        if deadline is not None:
            qos["deadline_s"] = float(deadline)
        elif "high_contention" in str(scenario.get("name", "")):
            qos["deadline_s"] = min(float(qos.get("deadline_s", 6.0)), 5.0)
    chain["qos"] = qos

    payload_range = scenario.get("payload_mb_range")
    if isinstance(payload_range, (list, tuple)) and len(payload_range) >= 2:
        low, high = float(payload_range[0]), float(payload_range[1])
        rng = random.Random(int(seed) * 1009 + int(request_index))
        chain["payload_mb"] = low if high <= low else rng.uniform(low, high)
    elif scenario.get("payload_mb") is not None:
        chain["payload_mb"] = float(scenario["payload_mb"])

    if scenario.get("preprocess_only"):
        chain["nodes"] = [dict(node) for node in chain.get("nodes", []) if str(node.get("node_id")) == "preprocess"]
        chain["edges"] = []

    output_ratio = scenario.get("output_ratio")
    if output_ratio is not None:
        ratio = max(0.0, float(output_ratio))
        for node in chain.get("nodes", []) or []:
            metadata = dict(node.get("metadata", {}) or {})
            if "output_ratio" in metadata or str(node.get("node_id")) != "verify":
                metadata["output_ratio"] = ratio
            node["metadata"] = metadata

    cpu_scale = scenario.get("service_cpu_scale")
    if cpu_scale is not None:
        scale = max(0.0, float(cpu_scale))
        for node in chain.get("nodes", []) or []:
            if "cpu_mb" in node:
                node["cpu_mb"] = float(node["cpu_mb"]) * scale
            metadata = dict(node.get("metadata", {}) or {})
            for key in ("task_cpu", "cpu", "cpu_mb", "cpu_per_mb"):
                if key in metadata:
                    metadata[key] = float(metadata[key]) * scale
            node["metadata"] = metadata

    forced_source = (
        scenario.get("forced_source_node_id")
        or scenario.get("runtime_source_node_id")
        or scenario.get("source_node_id")
    )
    if forced_source:
        chain["source_node_id"] = str(forced_source)
        chain["sink_node_id"] = str(scenario.get("forced_sink_node_id", scenario.get("sink_node_id", forced_source)))
    return chain


def _scenario_preferred_region(scenario: Mapping[str, Any], request_index: int, source_node_id: str) -> str:
    forced = scenario.get("preferred_region_id") or scenario.get("forced_preferred_region_id")
    if forced:
        return str(forced)
    skew = scenario.get("regional_skew_ratio", scenario.get("region_skew_ratio", scenario.get("skew_ratio")))
    if skew is not None:
        ratio = max(1.0, float(skew))
        hot_slots = max(1, int(round(ratio)))
        cycle = hot_slots + PHYSICAL_RSU_COUNT - 1
        if int(request_index) % cycle < hot_slots:
            return "RSU_0"
        return f"RSU_{1 + (int(request_index) % (PHYSICAL_RSU_COUNT - 1))}"
    source = str(source_node_id)
    digits = "".join(ch for ch in source if ch.isdigit())
    if digits:
        return f"RSU_{int(digits) % PHYSICAL_RSU_COUNT}"
    return f"RSU_{int(request_index) % PHYSICAL_RSU_COUNT}"


def inject_semantic_topology_decoys(
    instances: Sequence[Dict[str, Any]],
    scenario: Mapping[str, Any],
    seed: int,
    allowed_node_types: set[str],
) -> List[Dict[str, Any]]:
    """Inject controlled semantic/topology conflicts for stress calibration."""

    decoy_cfg = dict(scenario.get("candidate_decoys", {}) or {})
    if not decoy_cfg.get("enabled"):
        return [dict(item) for item in instances]
    scenario_name = str(scenario.get("name", "scenario"))
    rng = random.Random(int(seed) * 7919 + 17)
    out = [dict(item) for item in instances]
    service_templates: Dict[str, Dict[str, Any]] = {}
    for item in out:
        service_templates.setdefault(str(item.get("service_id")), item)

    decoy_specs = [
        ("semantic_high_topology_bad", int(decoy_cfg.get("semantic_high_topology_bad", 0) or 0)),
        ("semantic_medium_topology_good", int(decoy_cfg.get("semantic_medium_topology_good", 0) or 0)),
        ("semantic_low_topology_good", int(decoy_cfg.get("semantic_low_topology_good", 0) or 0)),
        ("stale_remote_candidates", int(decoy_cfg.get("stale_remote_candidates", 0) or 0)),
    ]
    for service_id, template in sorted(service_templates.items()):
        for decoy_kind, count in decoy_specs:
            for index in range(max(0, count)):
                node_type = _decoy_node_type(decoy_kind, allowed_node_types, index)
                node_id = _materialized_node_id(node_type, index + seed)
                region_id = _decoy_region_id(node_type, index, decoy_kind)
                profile = _decoy_profile(decoy_kind, node_type, rng)
                item = copy.deepcopy(template)
                item.update(
                    {
                        "instance_id": f"decoy_{scenario_name}_{decoy_kind}_{service_id}_{seed}_{index}",
                        "node_id": node_id,
                        "node_type": node_type,
                        "region_id": region_id,
                        "capacity": profile["capacity"],
                        "used": profile["used"],
                        "max_concurrency": profile["max_concurrency"],
                        "current_load": profile["current_load"],
                        "reliability_score": profile["reliability_score"],
                        "accuracy_score": profile["accuracy_score"],
                        "trust_score": profile["trust_score"],
                        "cold_start_s": profile["cold_start_s"],
                    }
                )
                metadata = dict(item.get("metadata", {}) or {})
                metadata.update(
                    {
                        "is_decoy": True,
                        "semantic_group": decoy_kind,
                        "semantic_score_bias": profile["semantic_score_bias"],
                        "load_ratio_override": profile["load_ratio_override"],
                        "topology_risk": profile["topology_risk"],
                        "mobility_risk": profile["mobility_risk"],
                        "stale_latency_penalty_s": float(
                            dict(scenario.get("stale_candidate_injection", {}) or {}).get("stale_latency_penalty_s", 0.0)
                            if decoy_kind == "stale_remote_candidates"
                            else 0.0
                        ),
                        "description": _decoy_description(service_id, decoy_kind),
                        "semantic_description": _decoy_description(service_id, decoy_kind),
                        "scenario": scenario_name,
                    }
                )
                item["metadata"] = metadata
                out.append(item)
    return out


def _apply_local_catalog_visibility(
    instances: Sequence[Dict[str, Any]],
    scenario: Mapping[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    visibility = scenario.get("local_catalog_visibility")
    if visibility is None:
        return [dict(item) for item in instances]
    ratio = max(0.0, min(1.0, float(visibility)))
    if ratio >= 0.999:
        out = [dict(item) for item in instances]
        for item in out:
            metadata = dict(item.get("metadata", {}) or {})
            metadata["local_catalog_visible"] = True
            item["metadata"] = metadata
        return out
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for raw in instances:
        item = copy.deepcopy(dict(raw))
        grouped.setdefault((str(item.get("region_id", "")), str(item.get("service_id", ""))), []).append(item)
    rng = random.Random(int(seed) * 104729 + 31)
    out: List[Dict[str, Any]] = []
    for _key, group in sorted(grouped.items()):
        group.sort(key=lambda item: str(item.get("instance_id", "")))
        keep_count = max(1, int(math.ceil(len(group) * ratio)))
        shuffled = list(group)
        rng.shuffle(shuffled)
        visible_ids = {str(item.get("instance_id", "")) for item in shuffled[:keep_count]}
        for item in group:
            metadata = dict(item.get("metadata", {}) or {})
            metadata["local_catalog_visible"] = str(item.get("instance_id", "")) in visible_ids
            metadata["local_catalog_visibility_ratio"] = ratio
            item["metadata"] = metadata
            out.append(item)
    return out


def _service_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for chain in config.get("service_chains", []) or []:
        for node in chain.get("nodes", []) or []:
            service_id = str(node.get("service_type", node.get("node_id", "")))
            specs.setdefault(
                service_id,
                {
                    "service_id": service_id,
                    "required_capabilities": list(node.get("required_capabilities", []) or []),
                    "input_semantic": str(node.get("input_semantic", "any")),
                    "output_semantic": str(node.get("output_semantic", "any")),
                    "memory_mb": float(node.get("memory_mb", 512.0) or 512.0),
                    "storage_mb": float(node.get("storage_mb", 256.0) or 256.0),
                },
            )
    return list(specs.values())


def _decoy_node_type(decoy_kind: str, allowed_node_types: set[str], index: int) -> str:
    mobile_order = ["uav", "vehicle"]
    if decoy_kind in {"semantic_high_topology_bad", "stale_remote_candidates"}:
        for node_type in mobile_order:
            if node_type in allowed_node_types:
                return node_type
    if decoy_kind == "semantic_medium_topology_good":
        if "rsu" in allowed_node_types:
            return "rsu"
        if "cloud_server" in allowed_node_types:
            return "cloud_server"
    if decoy_kind == "semantic_low_topology_good":
        if index % 2 == 0 and "vehicle" in allowed_node_types:
            return "vehicle"
        if "rsu" in allowed_node_types:
            return "rsu"
    for candidate_node_type in ("rsu", "cloud_server", "vehicle", "uav"):
        if candidate_node_type in allowed_node_types:
            return candidate_node_type
    raise ValueError("candidate_decoys require at least one allowed node type")


def _decoy_region_id(node_type: str, index: int, decoy_kind: str) -> str:
    if node_type == "cloud_server":
        return "cloud"
    if node_type == "rsu":
        return f"RSU_{index % PHYSICAL_RSU_COUNT}"
    if decoy_kind in {"semantic_high_topology_bad", "stale_remote_candidates"}:
        return f"RSU_{1 + (index % (PHYSICAL_RSU_COUNT - 1))}"
    return f"RSU_{index % PHYSICAL_RSU_COUNT}"


def _decoy_profile(decoy_kind: str, node_type: str, rng: random.Random) -> Dict[str, Any]:
    capacity = _materialized_capacity(node_type)
    if decoy_kind == "semantic_high_topology_bad":
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.35, "memory": capacity["memory"] * 0.20, "storage": capacity["storage"] * 0.10},
            "max_concurrency": 3,
            "current_load": 1,
            "reliability_score": 0.94,
            "accuracy_score": 0.97,
            "trust_score": 0.96,
            "cold_start_s": 0.55 + 0.10 * rng.random(),
            "semantic_score_bias": 0.90,
            "load_ratio_override": 0.92,
            "topology_risk": 0.88,
            "mobility_risk": 0.75 if node_type in {"uav", "vehicle"} else 0.25,
        }
    if decoy_kind == "semantic_medium_topology_good":
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.05, "memory": capacity["memory"] * 0.05, "storage": capacity["storage"] * 0.05},
            "max_concurrency": 8 if node_type == "rsu" else 4,
            "current_load": 0,
            "reliability_score": 0.97,
            "accuracy_score": 0.94,
            "trust_score": 0.98,
            "cold_start_s": 0.08 + 0.03 * rng.random(),
            "semantic_score_bias": 0.40,
            "load_ratio_override": 0.05,
            "topology_risk": 0.08,
            "mobility_risk": 0.05,
        }
    if decoy_kind == "semantic_low_topology_good":
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.04, "memory": capacity["memory"] * 0.05, "storage": capacity["storage"] * 0.05},
            "max_concurrency": 6 if node_type == "rsu" else 4,
            "current_load": 0,
            "reliability_score": 0.96,
            "accuracy_score": 0.90,
            "trust_score": 0.97,
            "cold_start_s": 0.05 + 0.02 * rng.random(),
            "semantic_score_bias": -0.20,
            "load_ratio_override": 0.02,
            "topology_risk": 0.05,
            "mobility_risk": 0.05 if node_type == "rsu" else 0.20,
        }
    return {
        "capacity": capacity,
        "used": {"cpu": capacity["cpu"] * 0.20, "memory": capacity["memory"] * 0.10, "storage": capacity["storage"] * 0.10},
        "max_concurrency": 4,
        "current_load": 1,
        "reliability_score": 0.92,
        "accuracy_score": 0.95,
            "trust_score": 0.94,
            "cold_start_s": 0.45 + 0.10 * rng.random(),
            "semantic_score_bias": 0.70,
            "load_ratio_override": 0.65,
            "topology_risk": 0.70,
        "mobility_risk": 0.70 if node_type in {"uav", "vehicle"} else 0.20,
    }


def _decoy_description(service_id: str, decoy_kind: str) -> str:
    service_text = str(service_id).replace("_", " ")
    if decoy_kind == "semantic_high_topology_bad":
        return f"high semantic match for {service_text} with specialized model but congested mobile topology"
    if decoy_kind == "semantic_medium_topology_good":
        return f"balanced {service_text} service with stable nearby topology and moderate semantic specificity"
    if decoy_kind == "semantic_low_topology_good":
        return f"generic nearby compute service for {service_text} with weak semantic specialization"
    return f"stale remote advertisement for {service_text} with formerly high semantic relevance"


def _load_multiplier(scenario: Mapping[str, Any]) -> float:
    if scenario.get("load_multiplier") is not None:
        return max(0.1, float(scenario["load_multiplier"]))
    arrival = float(scenario.get("arrival_rate_sfc_per_s", 0.6) or 0.6)
    return max(0.6, arrival / 0.6)


def _mobility_multiplier(scenario: Mapping[str, Any]) -> float:
    if scenario.get("uav_speed_scale") or scenario.get("vehicle_speed_scale") or scenario.get("speed_scale"):
        return 1.4
    name = str(scenario.get("name", ""))
    return 1.5 if "mobility" in name else 1.0


def _network_latency_multiplier(scenario: Mapping[str, Any]) -> float:
    if scenario.get("network_latency_multiplier") is not None:
        return max(1.0, float(scenario["network_latency_multiplier"]))
    name = str(scenario.get("name", ""))
    if "partial" in name or "contention" in name:
        return 1.25
    if "mobility" in name:
        return 1.2
    return 1.0


def _materialized_node_id(node_type: str, index: int) -> str:
    if node_type == "uav":
        return f"UAV_{index % PHYSICAL_UAV_COUNT}"
    if node_type == "rsu":
        return f"RSU_{index % PHYSICAL_RSU_COUNT}"
    if node_type == "cloud_server":
        return "cloudServer_0"
    return f"vehicle_{index % PHYSICAL_VEHICLE_COUNT}"


def _task_source_node(index: int, vehicle_tasks: int, uav_tasks: int) -> str:
    if uav_tasks and (index % 4 == 0 or not vehicle_tasks):
        return f"UAV_{index % min(PHYSICAL_UAV_COUNT, max(1, uav_tasks))}"
    if vehicle_tasks:
        return f"vehicle_{index % min(PHYSICAL_VEHICLE_COUNT, vehicle_tasks)}"
    return f"UAV_{index % min(PHYSICAL_UAV_COUNT, max(1, uav_tasks))}"


def _materialized_capacity(node_type: str) -> Dict[str, float]:
    if node_type == "vehicle":
        return {"cpu": 12.0, "memory": 4096.0, "storage": 4096.0}
    if node_type == "uav":
        return {"cpu": 6.0, "memory": 2048.0, "storage": 2048.0}
    return {"cpu": 60.0, "memory": 16384.0, "storage": 12000.0}


def _materialized_concurrency(node_type: str) -> int:
    return 3 if node_type == "vehicle" else 2 if node_type == "uav" else 5


def _materialized_quality(node_type: str, mobile_value: float, infra_value: float) -> float:
    return mobile_value if node_type in {"vehicle", "uav"} else infra_value


def _materialized_cold_start(node_type: str, service_index: int) -> float:
    base = 0.12 + 0.04 * service_index
    if node_type == "vehicle":
        return base + 0.18
    if node_type == "uav":
        return base + 0.10
    return base


if __name__ == "__main__":
    main()
