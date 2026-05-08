from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

LOGGER = logging.getLogger(__name__)
_LOGGED_V21_CHAIN_OVERWRITES: set[tuple[str, int, str, str]] = set()
_LOGGED_V21_MISSING_REQUEST_TYPES: set[str] = set()
_LOGGED_V21_CHAIN_EXPANSIONS: set[tuple[str, int, int]] = set()
_COMPRESSED_REQUEST_SERVICE_CHAINS: Dict[str, Dict[int, List[str]]] = {
    "forest_fire_monitoring": {
        2: ["aerial_video_preprocessing", "fire_candidate_detection"],
        3: ["aerial_video_preprocessing", "fire_candidate_detection", "fire_event_verification"],
        4: ["aerial_video_preprocessing", "fire_candidate_detection", "fire_event_verification", "emergency_alert_publish"],
        5: [
            "aerial_video_preprocessing",
            "fire_candidate_detection",
            "smoke_fire_fusion",
            "fire_event_verification",
            "emergency_alert_publish",
        ],
    },
    "traffic_surveillance": {
        2: ["aerial_video_preprocessing", "vehicle_detection_and_tracking"],
        3: ["aerial_video_preprocessing", "vehicle_detection_and_tracking", "trajectory_risk_assessment"],
        4: [
            "aerial_video_preprocessing",
            "vehicle_detection_and_tracking",
            "trajectory_risk_assessment",
            "emergency_alert_publish",
        ],
        5: [
            "aerial_video_preprocessing",
            "vehicle_detection_and_tracking",
            "trajectory_risk_assessment",
            "emergency_alert_publish",
            "surveillance_event_archive",
        ],
    },
    "urban_security": {
        2: ["aerial_video_preprocessing", "person_detection_and_reid"],
        3: ["aerial_video_preprocessing", "person_detection_and_reid", "identity_verification"],
        4: [
            "aerial_video_preprocessing",
            "person_detection_and_reid",
            "identity_verification",
            "trajectory_risk_assessment",
        ],
        5: [
            "aerial_video_preprocessing",
            "person_detection_and_reid",
            "identity_verification",
            "trajectory_risk_assessment",
            "emergency_alert_publish",
        ],
    },
    "industrial_inspection": {
        2: ["thermal_frame_preprocessing", "industrial_defect_detection"],
        3: ["thermal_frame_preprocessing", "industrial_defect_detection", "defect_classification"],
        4: [
            "thermal_frame_preprocessing",
            "industrial_defect_detection",
            "defect_classification",
            "inspection_report_generation",
        ],
        5: [
            "thermal_frame_preprocessing",
            "multi_spectral_fusion_preprocessing",
            "industrial_defect_detection",
            "defect_classification",
            "inspection_report_generation",
        ],
    },
}

METHOD_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.graph_observation import BASE_CANDIDATE_FEATURE_DIM, flatten_observation
from airfogsim.lasdm.env_adapter import LASDMEnvAdapter
from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory
from airfogsim.lasdm.manager import LASDMManager
from airfogsim.lasdm.marl_env import MARLEnvConfig, SemanticTopologyMARLEnv
from airfogsim.lasdm.marl_policy import MASACPolicy, policy_from_name
from airfogsim.lasdm.marl_reward import SFCReward, SFCRewardConfig
from airfogsim.lasdm.marl_trainer import HeuristicEvaluator, MASACTrainer, write_reward_curve
from airfogsim.lasdm.model import LASDMServiceChain
from airfogsim.lasdm.orchestrator import LASDMOrchestrator
from airfogsim.lasdm.runtime_bridge import LASDMRuntimeBridge
from airfogsim.lasdm.semantic_link_matrix import SERVICE_IO_DEFAULTS, DEFAULT_SEMANTIC_OUTPUT_ROOT, SemanticLinkMatrix
from airfogsim.lasdm.topology_builder import TopologyBuilder

DEFAULT_CONFIG = os.path.join(METHOD_ROOT, "configs", "semantic_topology_marl.yaml")
DEFAULT_SEMANTIC_OUTPUT_ROOT_ABS = os.path.join(WORKSPACE_ROOT, str(DEFAULT_SEMANTIC_OUTPUT_ROOT))
PHYSICAL_VEHICLE_COUNT = 100
PHYSICAL_UAV_COUNT = 20
PHYSICAL_RSU_COUNT = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate semantic-topology LASDM MARL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--policy",
        default="semantic_greedy",
        choices=[
            "semantic_greedy",
            "pure_semantic_greedy_no_exchange",
            "local_semantic_runtime_greedy",
            "topology_greedy",
            "utility_prior_with_exchange",
            "intra_region_only",
            "cross_region_auction",
            "centralized_planner",
            "nsga2_semantic_qos",
            "random_valid",
            "masac",
            "mappo",
            "iql",
        ],
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--service-role-sweep", default="full_hybrid")
    parser.add_argument("--output-dir", default=os.path.join(DEFAULT_SEMANTIC_OUTPUT_ROOT_ABS, "semantic_runtime_train", "default"))
    args = parser.parse_args()

    config = _load_yaml(args.config)
    episodes = int(args.episodes if args.episodes is not None else config.get("training", {}).get("episodes", 10))
    max_steps = int(args.max_steps if args.max_steps is not None else config.get("training", {}).get("max_steps", 100))
    baseline_name = {"masac": "proposed_semantic_topology_marl", "mappo": "mappo_ctde", "iql": "iql_offline"}.get(
        args.policy,
        args.policy,
    )
    env = build_offline_env(
        config,
        seed=args.seed,
        max_steps=max_steps,
        scenario=args.scenario,
        service_role_sweep=args.service_role_sweep,
        attach_runtime=True,
        baseline=baseline_name,
    )
    try:
        observations = env.reset()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        marl_cfg = dict(config.get("marl", {}) or {})
        policy_kwargs = _actor_policy_kwargs(config, observations, args.seed)
        def fresh_env(offset: int) -> SemanticTopologyMARLEnv:
            return build_offline_env(
                config,
                seed=args.seed * 100000 + int(offset),
                max_steps=max_steps,
                scenario=args.scenario,
                service_role_sweep=args.service_role_sweep,
                attach_runtime=True,
                baseline=baseline_name,
            )

        def close_semantic_env(item: SemanticTopologyMARLEnv) -> None:
            _close_runtime_env(getattr(item, "env", None))

        if args.policy == "masac":
            target_entropy_raw = marl_cfg.get("masac_target_entropy", None)
            policy = MASACPolicy(
                **policy_kwargs,
                q_lr=float(marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4)) or 3e-4),
                alpha=float(marl_cfg.get("masac_alpha", 0.20) or 0.20),
                auto_alpha=bool(marl_cfg.get("masac_auto_alpha", False)),
                alpha_lr=float(marl_cfg.get("masac_alpha_lr", 3e-4) or 3e-4),
                target_entropy=None if target_entropy_raw in (None, "") else float(target_entropy_raw),
                target_entropy_scale=float(marl_cfg.get("masac_target_entropy_scale", 0.90) or 0.90),
                alpha_min=float(marl_cfg.get("masac_alpha_min", 0.005) or 0.005),
                alpha_max=float(marl_cfg.get("masac_alpha_max", 0.25) or 0.25),
                tau=float(marl_cfg.get("masac_tau", 0.005) or 0.005),
                q_mlp_depth=int(marl_cfg.get("masac_q_mlp_depth", 3) or 3),
            )
            trainer = MASACTrainer(
                env,
                policy,
                env_factory=fresh_env,
                close_env=close_semantic_env,
                gamma=float(marl_cfg.get("masac_gamma", 0.99) or 0.99),
                tau=float(marl_cfg.get("masac_tau", 0.005) or 0.005),
                batch_size=int(marl_cfg.get("masac_batch_size", 128) or 128),
                replay_capacity=int(marl_cfg.get("masac_replay_capacity", 20000) or 20000),
                replay_warmup_steps=int(marl_cfg.get("masac_replay_warmup_steps", 128) or 128),
                update_interval=int(marl_cfg.get("masac_update_interval", 1) or 1),
                actor_update_interval=int(marl_cfg.get("masac_actor_update_interval", 2) or 2),
                updates_per_env_step=int(marl_cfg.get("masac_updates_per_env_step", 1) or 1),
                max_grad_norm=float(marl_cfg.get("masac_max_grad_norm", 10.0) or 10.0),
                reward_scale=float(marl_cfg.get("masac_reward_scale", 1.0) or 1.0),
                reward_normalization=bool(marl_cfg.get("masac_reward_normalization", True)),
                reward_clip=float(marl_cfg.get("masac_reward_clip", 5.0) or 5.0),
                replay_sample_strategy=str(marl_cfg.get("masac_replay_sample_strategy", "uniform") or "uniform"),
                seed=args.seed,
            )
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
        elif args.policy == "mappo":
            from airfogsim.lasdm.mappo_policy import MAPPOPolicy
            from airfogsim.lasdm.mappo_trainer import MAPPOTrainer

            policy = MAPPOPolicy(**policy_kwargs)
            trainer = MAPPOTrainer(
                env,
                policy,
                env_factory=fresh_env,
                close_env=close_semantic_env,
                gamma=float(marl_cfg.get("mappo_gamma", marl_cfg.get("masac_gamma", 0.99)) or 0.99),
                lam=float(marl_cfg.get("mappo_gae_lambda", 0.95) or 0.95),
                clip_eps=float(marl_cfg.get("mappo_clip_eps", 0.2) or 0.2),
                ppo_epochs=int(marl_cfg.get("mappo_ppo_epochs", 4) or 4),
                rollout_steps=int(marl_cfg.get("mappo_rollout_steps", 128) or 128),
                vf_coef=float(marl_cfg.get("mappo_vf_coef", 0.5) or 0.5),
                ent_coef=float(marl_cfg.get("mappo_ent_coef", 0.01) or 0.01),
                max_grad_norm=float(marl_cfg.get("mappo_max_grad_norm", 0.5) or 0.5),
            )
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
        elif args.policy == "iql":
            from airfogsim.lasdm.iql_policy import IQLPolicy
            from airfogsim.lasdm.iql_trainer import IQLTrainer

            policy = IQLPolicy(
                **policy_kwargs,
                q_lr=float(marl_cfg.get("iql_q_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
                expectile=float(marl_cfg.get("iql_expectile", 0.7) or 0.7),
                beta=float(marl_cfg.get("iql_beta", 3.0) or 3.0),
                v_lr=float(marl_cfg.get("iql_v_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
                tau=float(marl_cfg.get("masac_tau", 0.005) or 0.005),
                q_mlp_depth=int(marl_cfg.get("iql_q_mlp_depth", 3) or 3),
            )
            trainer = IQLTrainer(
                env,
                policy,
                env_factory=fresh_env,
                close_env=close_semantic_env,
                gamma=float(marl_cfg.get("iql_gamma", marl_cfg.get("masac_gamma", 0.99)) or 0.99),
                tau=float(marl_cfg.get("iql_tau", marl_cfg.get("masac_tau", 0.005)) or 0.005),
                batch_size=int(marl_cfg.get("iql_batch_size", marl_cfg.get("masac_batch_size", 256)) or 256),
                replay_capacity=int(marl_cfg.get("iql_replay_capacity", marl_cfg.get("masac_replay_capacity", 10000)) or 10000),
                replay_warmup_steps=int(marl_cfg.get("iql_replay_warmup_steps", marl_cfg.get("masac_replay_warmup_steps", 1024)) or 1024),
                update_interval=int(marl_cfg.get("iql_update_interval", marl_cfg.get("masac_update_interval", 50)) or 50),
                updates_per_env_step=int(marl_cfg.get("iql_updates_per_env_step", marl_cfg.get("masac_updates_per_env_step", 1)) or 1),
                max_grad_norm=float(marl_cfg.get("iql_max_grad_norm", marl_cfg.get("masac_max_grad_norm", 10.0)) or 10.0),
                reward_scale=float(marl_cfg.get("iql_reward_scale", marl_cfg.get("masac_reward_scale", 1.0)) or 1.0),
                reward_normalization=bool(marl_cfg.get("iql_reward_normalization", marl_cfg.get("masac_reward_normalization", True))),
                reward_clip=float(marl_cfg.get("iql_reward_clip", marl_cfg.get("masac_reward_clip", 5.0)) or 5.0),
                replay_sample_strategy=str(marl_cfg.get("iql_replay_sample_strategy", marl_cfg.get("masac_replay_sample_strategy", "uniform")) or "uniform"),
                seed=args.seed,
            )
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
        else:
            policy = policy_from_name(args.policy, seed=args.seed)
            evaluator = HeuristicEvaluator(env, policy, env_factory=fresh_env, close_env=close_semantic_env)
            rows = evaluator.run(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
            write_reward_curve(output_dir / "reward_curve.csv", rows)

        payload = {
            "completed": True,
            "policy": args.policy,
            "algorithm": _algorithm_name(args.policy),
            "seed": int(args.seed),
            "scenario": str(args.scenario or "default"),
            "service_role_sweep": str(args.service_role_sweep),
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
    semantic_matrix = run_config.get("_semantic_matrix")
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
        semantic_exchange_embedding_dim=int(exchange_cfg.get("embedding_dim", 384)),
        semantic_encoder_backend=str(exchange_cfg.get("encoder_backend", "sbert")),
        semantic_encoder_model_name=str(exchange_cfg.get("sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_encoder_hash_dim=int(exchange_cfg.get("hash_dim", 384)),
        semantic_encoder_batch_size=int(exchange_cfg.get("encoder_batch_size", 64)),
        semantic_encoder_device=str(exchange_cfg.get("encoder_device", "")),
        scenario_name=str(scenario_cfg.get("name", "default")),
        service_role_sweep=str(service_role_sweep),
        runtime_ticks_per_marl_step=int(marl_cfg.get("runtime_ticks_per_marl_step", 1)),
        runtime_max_ticks_after_action=int(marl_cfg.get("runtime_max_ticks_after_action", 25)),
        runtime_stop_when_progress=bool(marl_cfg.get("runtime_stop_when_progress", True)),
        route_hop_floor_s=float(marl_cfg.get("route_hop_floor_s", 0.10) or 0.10),
        global_candidate_catalog=bool(marl_cfg.get("global_candidate_catalog", False)),
        region_agents=tuple(str(item) for item in topology_cfg.get("region_agents", []) or []),
        sequential_capacity_enabled=bool(marl_cfg.get("sequential_capacity_enabled", True)),
        sequential_deadline_pruning_enabled=bool(marl_cfg.get("sequential_deadline_pruning_enabled", True)),
        semantic_matrix=semantic_matrix,
        enable_semantic_profiles=True,
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
    if not bool(marl_cfg.get("include_semantic_features", True)):
        reward_cfg["semantic_score"] = 0.0
        reward_cfg["semantic_cumulative"] = 0.0
        reward_cfg["utility_prior"] = 0.0
    allowed = set(SFCRewardConfig.__dataclass_fields__)
    values = {}
    for key, value in reward_cfg.items():
        if key not in allowed:
            raise ValueError(f"Unknown reward config key: {key}")
        values[key] = bool(value) if key == "scenario_normalized_dense" else float(value)
    return SFCReward(SFCRewardConfig(**values))


def _observation_candidate_feature_dim(observations: Mapping[str, Mapping[str, Any]]) -> Optional[int]:
    for observation in observations.values():
        for candidate_set in observation.get("candidate_sets", []) or []:
            width = _candidate_feature_width(candidate_set.get("candidate_features"))
            if width is not None:
                return width
    return None


def _configured_candidate_feature_dim(config: Mapping[str, Any]) -> int:
    semantic_cfg = dict(config.get("semantic_exchange", {}) or {})
    embedding_dim = int(semantic_cfg.get("embedding_dim", 0) or 0)
    if embedding_dim <= 0:
        raise ValueError("semantic_exchange.embedding_dim must be set to build learned policy candidate features.")
    return int(BASE_CANDIDATE_FEATURE_DIM + embedding_dim)


def _candidate_feature_width(features: Any) -> Optional[int]:
    shape = getattr(features, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"candidate_features must be 2-D, got shape {tuple(shape)}")
        width = int(shape[1])
        return width if width > 0 and int(shape[0]) > 0 else None
    if not features:
        return None
    first = features[0]
    if not hasattr(first, "__len__") or isinstance(first, (str, bytes)):
        raise ValueError("candidate_features must be a 2-D numeric sequence")
    width = len(first)
    if width <= 0:
        return None
    return int(width)


def _actor_policy_kwargs(
    config: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    marl_cfg = dict(config.get("marl", {}) or {})
    obs_dim = max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
    critic_agents = int(marl_cfg.get("ippo_critic_agent_count", len(observations) or 4) or 4)
    candidate_feature_dim = _observation_candidate_feature_dim(observations)
    if candidate_feature_dim is None:
        candidate_feature_dim = _configured_candidate_feature_dim(config)
    return {
        "observation_dim": obs_dim,
        "max_candidates": int(marl_cfg.get("max_candidates", 16) or 16),
        "candidate_feature_dim": candidate_feature_dim,
        "hidden_dim": int(marl_cfg.get("ippo_hidden_dim", 256) or 256),
        "mlp_depth": int(marl_cfg.get("ippo_mlp_depth", 3) or 3),
        "gnn_layers": int(marl_cfg.get("ippo_gnn_layers", 3) or 3),
        "lr": float(marl_cfg.get("ippo_lr", 3e-4) or 3e-4),
        "seed": int(seed),
        "centralized_critic": bool(marl_cfg.get("ippo_centralized_critic", True)),
        "critic_observation_dim": obs_dim * max(1, critic_agents),
        "max_critic_agents": max(1, critic_agents),
        "utility_prior_logit_weight": _config_float(marl_cfg, "ippo_utility_prior_logit_weight", 2.5),
        "route_unavailable_penalty": float(marl_cfg.get("ippo_route_unavailable_penalty", 20.0) or 20.0),
        "include_semantic_features": bool(marl_cfg.get("include_semantic_features", True)),
        "include_topology_features": bool(marl_cfg.get("include_topology_features", True)),
        "include_temporal_features": bool(marl_cfg.get("include_temporal_features", True)),
        "use_region_encoder": bool(marl_cfg.get("ippo_use_region_encoder", True)),
        "learnable_prior": bool(marl_cfg.get("ippo_learnable_prior", True)),
        "prior_l2_coef": float(marl_cfg.get("ippo_prior_l2_coef", 1e-3) or 0.0),
        "learned_logit_scale": float(
            marl_cfg.get("ippo_learned_logit_scale_init", marl_cfg.get("ippo_learned_logit_scale", 1.0)) or 1.0
        ),
        "prior_logit_scale": _config_float(
            marl_cfg,
            "ippo_prior_logit_scale_init",
            _config_float(marl_cfg, "ippo_prior_logit_scale", 1.0),
        ),
        "learnable_logit_blend": bool(marl_cfg.get("ippo_learnable_logit_blend", False)),
        "action_prior_enabled": bool(marl_cfg.get("ippo_action_prior_enabled", True)),
        "semantic_projection_dim": int(marl_cfg.get("semantic_projection_dim", 64) or 64),
        "cross_agent_attention_enabled": bool(marl_cfg.get("cross_agent_attention_enabled", True)),
        "cross_agent_attention_heads": int(marl_cfg.get("cross_agent_attention_heads", 4) or 4),
        "device": str(marl_cfg.get("ippo_device", "") or "") or None,
    }


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def _algorithm_name(policy_name: str) -> str:
    mapping = {
        "masac": "masac_discrete_ctde",
        "mappo": "mappo_ctde",
        "iql": "iql_offline",
        "nsga2_semantic_qos": "nsga2_semantic_qos",
    }
    return mapping.get(str(policy_name), str(policy_name))


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
        for item in scenarios:
            item_dict = dict(item)
            if bool(item_dict.get("include_in_default", item_dict.get("enabled", True))):
                return item_dict
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
    semantic_matrix = SemanticLinkMatrix.from_config(run_config, METHOD_ROOT, WORKSPACE_ROOT)
    instances = _materialized_service_instances(run_config, scenario, allowed_node_types, seed, semantic_matrix)
    instances = inject_semantic_topology_decoys(instances, scenario, seed, allowed_node_types, semantic_matrix)
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
    embedding_dim = scenario.get("embedding_dim", scenario.get("semantic_embedding_dim"))
    if embedding_dim is not None:
        exchange_cfg["embedding_dim"] = int(embedding_dim)
    if scenario.get("exchange_top_k") is not None:
        exchange_cfg["top_k_per_agent"] = int(scenario["exchange_top_k"])
    run_config["semantic_exchange"] = exchange_cfg
    run_config["service_chains"] = _scenario_chains(run_config, scenario, role_name, seed)
    _assign_v21_request_types(run_config["service_chains"], semantic_matrix)
    semantic_output_root = Path(
        str(dict(run_config.get("semantic_profiles", {}) or {}).get("output_root", DEFAULT_SEMANTIC_OUTPUT_ROOT_ABS))
    )
    _export_semantic_artifacts(semantic_matrix, semantic_output_root, instances, exchange_cfg, run_config)
    run_config["_semantic_matrix"] = semantic_matrix
    return run_config


def _export_semantic_artifacts(
    semantic_matrix: SemanticLinkMatrix,
    output_root: Path,
    instances: Sequence[Mapping[str, Any]],
    encoder_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    profile_cfg = dict(run_config.get("semantic_profiles", {}) or {})
    if not bool(profile_cfg.get("export_artifacts_once", True)):
        semantic_matrix.export_artifacts(output_root, materialized_instances=instances, encoder_config=encoder_config)
        return
    audit_path = output_root / "semantic_dataset_audit.json"
    if audit_path.exists():
        return
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".semantic_artifacts_once.lock"
    started = time.time()
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if audit_path.exists():
                return
            if time.time() - started > 300.0:
                raise TimeoutError(f"Timed out waiting for semantic artifact export lock: {lock_path}")
            time.sleep(0.5)
    try:
        if not audit_path.exists():
            semantic_matrix.export_artifacts(output_root, materialized_instances=instances, encoder_config=encoder_config)
    finally:
        try:
            lock_path.rmdir()
        except OSError:
            pass


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
    semantic_matrix: SemanticLinkMatrix,
) -> List[Dict[str, Any]]:
    service_nodes = dict(scenario.get("service_nodes", {}) or {})
    load_multiplier = _load_multiplier(scenario)
    supply_ratio = float(scenario.get("service_supply_ratio", 1.0) or 1.0)
    rng = random.Random(int(seed))
    generated: List[Dict[str, Any]] = []
    service_types = semantic_matrix.list_service_types()
    for node_type, max_nodes in (
        ("vehicle", PHYSICAL_VEHICLE_COUNT),
        ("uav", PHYSICAL_UAV_COUNT),
        ("rsu", PHYSICAL_RSU_COUNT),
        ("cloud_server", 1),
    ):
        if node_type not in allowed_node_types:
            continue
        requested = int(service_nodes.get(node_type, 1 if node_type == "cloud_server" else 0) or 0)
        if not bool(scenario.get("preprocess_only", False)) and node_type in {"rsu", "cloud_server"}:
            requested = max(1, requested)
        node_count = min(max_nodes, max(0, int(round(requested * supply_ratio))))
        for node_index in range(node_count):
            node_id = _materialized_node_id(node_type, node_index)
            region_id = f"RSU_{node_index % PHYSICAL_RSU_COUNT}" if node_type != "rsu" else node_id
            if node_type == "cloud_server":
                region_id = "cloud"
            for spec_index, service_type in enumerate(service_types):
                implementations = semantic_matrix.list_implementations_for(service_type, compatible_node_type=node_type)
                if not implementations:
                    continue
                instance_count = _instance_count_per_node(node_type, len(implementations))
                deployed = _select_deployed_implementations(
                    implementations,
                    scenario,
                    node_type,
                    service_type,
                    rng,
                    instance_count,
                )
                for replica_index, implementation in enumerate(deployed):
                    metadata = semantic_matrix.candidate_metadata_from_implementation(implementation)
                    profile = dict(implementation.raw)
                    resource_profile = dict(profile.get("resource_profile", {}) or {})
                    performance = dict(profile.get("performance_profile", {}) or {})
                    capacity = _materialized_capacity(node_type)
                    min_cpu = max(0.1, float(resource_profile.get("min_compute_cpu", 1.0) or 1.0))
                    min_memory = max(128.0, float(resource_profile.get("min_memory_mb", 512.0) or 512.0))
                    min_storage = max(128.0, float(resource_profile.get("min_storage_mb", 256.0) or 256.0))
                    used_cpu = min(capacity["cpu"] * 0.9, (0.10 + 0.04 * ((node_index + spec_index + replica_index) % 5)) * capacity["cpu"] * load_multiplier)
                    input_semantic = implementation.input_semantic_label
                    output_semantic = implementation.output_semantic_label
                    generated.append(
                        {
                            "instance_id": (
                                f"{node_id.lower()}_{implementation.implementation_id}_{seed}_{spec_index}_{replica_index}"
                            ),
                            "service_id": implementation.service_type,
                            "node_id": node_id,
                            "node_type": node_type,
                            "region_id": region_id,
                            "capabilities": list(dict.fromkeys([implementation.service_type, *profile.get("capabilities", [])])),
                            "input_semantic": input_semantic,
                            "output_semantic": output_semantic,
                            "capacity": capacity,
                            "used": {
                                "cpu": max(min_cpu * 0.20, used_cpu),
                                "memory": min(capacity["memory"] * 0.80, min_memory * load_multiplier),
                                "storage": min(capacity["storage"] * 0.60, min_storage * load_multiplier),
                            },
                            "max_concurrency": _materialized_concurrency(node_type),
                            "current_load": int(rng.random() < min(0.75, max(0.0, load_multiplier - 1.0) * 0.35)),
                            "reliability_score": _profile_quality(performance, "typical_reliability", node_type, 0.92, 0.99),
                            "accuracy_score": _profile_quality(performance, "typical_accuracy_f1", node_type, 0.86, 0.98),
                            "trust_score": _materialized_quality(node_type, 0.88, 0.99),
                            "cold_start_s": _materialized_cold_start(node_type, spec_index + replica_index),
                            "metadata": {
                                **metadata,
                                "scenario_materialized": True,
                                "scenario": scenario.get("name", "default"),
                                "role_control": "v21_semantic_profile",
                                "load_ratio_override": _scenario_node_type_value(
                                    scenario,
                                    node_type,
                                    "service_load_ratio",
                                    0.20 if node_type in {"vehicle", "uav"} else 0.05,
                                ),
                                "topology_risk": _scenario_node_type_value(
                                    scenario,
                                    node_type,
                                    "service_topology_risk",
                                    0.30 if node_type in {"vehicle", "uav"} else 0.08,
                                ),
                                "mobility_risk": _scenario_node_type_value(
                                    scenario,
                                    node_type,
                                    "service_mobility_risk",
                                    0.35 if node_type in {"vehicle", "uav"} else 0.05,
                                ),
                                "deployment_variant_weight": _deployment_variant_weight(
                                    scenario,
                                    node_type,
                                    implementation.variant_type,
                                    service_type,
                                ),
                            },
                        }
                    )
    return generated


def _scenario_node_type_value(
    scenario: Mapping[str, Any],
    node_type: str,
    suffix: str,
    default: float,
) -> float:
    direct_key = f"{node_type}_{suffix}"
    if direct_key in scenario:
        return float(scenario[direct_key])
    group = "mobile" if node_type in {"vehicle", "uav"} else "infrastructure"
    group_key = f"{group}_{suffix}"
    if group_key in scenario:
        return float(scenario[group_key])
    if group == "infrastructure":
        infra_key = f"infra_{suffix}"
        if infra_key in scenario:
            return float(scenario[infra_key])
    return float(default)


def _select_deployed_implementations(
    implementations: Sequence[Any],
    scenario: Mapping[str, Any],
    node_type: str,
    service_type: str,
    rng: random.Random,
    count: int,
) -> List[Any]:
    remaining = list(implementations)
    selected: List[Any] = []
    for _ in range(min(max(0, int(count)), len(remaining))):
        weights = [
            _deployment_variant_weight(scenario, node_type, item.variant_type, service_type)
            for item in remaining
        ]
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0.0:
            chosen_index = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            chosen_index = len(remaining) - 1
            for index, weight in enumerate(weights):
                cumulative += max(0.0, weight)
                if threshold <= cumulative:
                    chosen_index = index
                    break
        selected.append(remaining.pop(chosen_index))
    return selected


def _deployment_variant_weight(
    scenario: Mapping[str, Any],
    node_type: str,
    variant_type: str,
    service_type: str,
) -> float:
    cfg = dict(scenario.get("semantic_deployment_distribution", {}) or {})
    raw_weights = dict(cfg.get("variant_weights", {}) or {})
    weights = dict(raw_weights.get("default", {}) or {})
    weights.update(dict(raw_weights.get(str(node_type), {}) or {}))
    value = float(weights.get(str(variant_type), 1.0) or 0.0)
    service_exact_scale = dict(cfg.get("exact_service_scale", {}) or {})
    if str(variant_type) == "exact" and str(service_type) in service_exact_scale:
        value *= max(0.0, float(service_exact_scale[str(service_type)]))
    return max(0.0, value)


def _instance_count_per_node(node_type: str, available: int) -> int:
    target = 1
    if node_type == "rsu":
        target = 2
    elif node_type == "cloud_server":
        target = 3
    return max(1, min(int(available), target))


def _profile_quality(performance: Mapping[str, Any], key: str, node_type: str, low: float, high: float) -> float:
    value = performance.get(key)
    if value is not None:
        return max(0.0, min(1.0, float(value)))
    return _materialized_quality(node_type, low, high)


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
    task_nodes = dict(scenario_cfg.get("task_nodes", {}) or {})
    load_multiplier = _load_multiplier(scenario_cfg)
    arrival_plan = _per_task_node_poisson_arrival_plan(scenario_cfg, seed)
    target_count = len(arrival_plan)
    sampling_scenario = dict(scenario_cfg)
    sampling_scenario["generated_sfc_count"] = target_count
    horizon_s = _scenario_arrival_horizon_s(scenario_cfg)
    expected_arrival_rate = _scenario_expected_arrival_rate_sfc_per_s(scenario_cfg)
    chains: List[Dict[str, Any]] = []
    for index, arrival in enumerate(arrival_plan):
        raw = copy.deepcopy(base[index % len(base)])
        raw["sfc_id"] = f"{raw['sfc_id']}__{scenario_cfg.get('name', 'default')}__{service_role_sweep}__seed_{seed}__req_{index}"
        source = str(arrival["source_node_id"])
        raw["source_node_id"] = source
        raw["sink_node_id"] = source
        apply_semantic_scenario_overrides(raw, sampling_scenario, seed=seed, request_index=index)
        context = dict(raw.get("context", {}) or {})
        request_type = _select_request_type(sampling_scenario, seed, index)
        if request_type:
            context["request_type"] = request_type
        representative_length = _select_representative_service_chain_length(sampling_scenario, seed, index)
        if representative_length is not None:
            context["representative_service_chain_length"] = representative_length
        context.update(
            {
                "scenario": scenario_cfg.get("name", "default"),
                "preprocess_only": bool(scenario_cfg.get("preprocess_only", False)),
                "service_role_sweep": service_role_sweep,
                "preferred_region_id": _scenario_preferred_region(scenario_cfg, index, source),
                "task_node_counts": task_nodes,
                "service_node_counts": dict(scenario_cfg.get("service_nodes", {}) or {}),
                "generated_sfc_count": target_count,
                "arrival_rate_sfc_per_s": expected_arrival_rate,
                "arrival_process": "per_task_node_poisson",
                "arrival_horizon_s": horizon_s,
                "arrival_time_s": float(arrival.get("arrival_time_s", 0.0) or 0.0),
                "request_intensity_sfc_per_s": float(target_count) / max(1e-6, horizon_s),
                "logical_task_node_id": str(arrival["logical_task_node_id"]),
                "physical_task_node_id": source,
                "task_node_type": str(arrival["task_node_type"]),
                "per_task_node_arrival_rate_sfc_per_s": float(
                    arrival.get("per_task_node_arrival_rate_sfc_per_s", 0.0) or 0.0
                ),
                "per_task_node_arrival_interval_s": float(
                    arrival.get("per_task_node_arrival_interval_s", 0.0) or 0.0
                ),
                "theoretical_success_target": scenario_cfg.get("theoretical_success_target"),
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


def _scenario_arrival_horizon_s(scenario: Mapping[str, Any]) -> float:
    raw = scenario.get("arrival_horizon_s", scenario.get("episode_duration_s"))
    if raw is None:
        raise ValueError("per-task-node Poisson scenarios must set arrival_horizon_s")
    return max(0.1, float(raw))


def _per_task_node_poisson_arrival_plan(
    scenario: Mapping[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    task_sources = _scenario_task_sources(scenario)
    horizon_s = _scenario_arrival_horizon_s(scenario)
    per_node_rate = _per_task_node_arrival_rate_sfc_per_s(scenario)
    if per_node_rate <= 0.0:
        raise ValueError("per-task-node Poisson arrival rate must be positive")
    arrivals: List[Dict[str, Any]] = []
    for source_index, source in enumerate(task_sources):
        rng = random.Random(_stable_scenario_seed(scenario, seed, 41047) + source_index * 7919)
        current_time = 0.0
        while True:
            current_time += rng.expovariate(per_node_rate)
            if current_time > horizon_s + 1e-9:
                break
            item = dict(source)
            item["arrival_time_s"] = float(current_time)
            item["per_task_node_arrival_rate_sfc_per_s"] = float(per_node_rate)
            item["per_task_node_arrival_interval_s"] = float(1.0 / per_node_rate)
            arrivals.append(item)
    arrivals.sort(key=lambda item: (float(item["arrival_time_s"]), str(item["logical_task_node_id"])))
    return arrivals


def _scenario_task_sources(scenario: Mapping[str, Any]) -> List[Dict[str, Any]]:
    task_nodes = dict(scenario.get("task_nodes", {}) or {})
    sources: List[Dict[str, Any]] = []
    for node_type, physical_count, prefix in (
        ("vehicle", PHYSICAL_VEHICLE_COUNT, "vehicle"),
        ("uav", PHYSICAL_UAV_COUNT, "UAV"),
    ):
        logical_count = max(0, int(task_nodes.get(node_type, 0) or 0))
        for logical_index in range(logical_count):
            physical_index = logical_index % max(1, physical_count)
            physical_node_id = f"{prefix}_{physical_index}" if node_type == "uav" else f"vehicle_{physical_index}"
            sources.append(
                {
                    "logical_task_node_id": f"{node_type}_task_{logical_index}",
                    "source_node_id": physical_node_id,
                    "task_node_type": node_type,
                    "logical_task_node_index": logical_index,
                    "physical_task_node_index": physical_index,
                }
            )
    if not sources:
        raise ValueError("per-task-node Poisson scenarios must set positive task_nodes")
    return sources


def _per_task_node_arrival_rate_sfc_per_s(
    scenario: Mapping[str, Any],
) -> float:
    interval = scenario.get("per_task_node_arrival_interval_s")
    if interval is not None:
        return 1.0 / max(1e-6, float(interval))
    rate = scenario.get("per_task_node_arrival_rate_sfc_per_s")
    if rate is not None:
        return max(0.0, float(rate))
    raise ValueError("per-task-node Poisson scenarios must set per_task_node_arrival_interval_s or per_task_node_arrival_rate_sfc_per_s")


def _scenario_expected_arrival_rate_sfc_per_s(scenario: Mapping[str, Any]) -> float:
    sources = _scenario_task_sources(scenario)
    return max(1e-6, _per_task_node_arrival_rate_sfc_per_s(scenario) * len(sources))


def _select_representative_service_chain_length(
    scenario: Mapping[str, Any],
    seed: int,
    request_index: int,
) -> Optional[int]:
    distribution = scenario.get("sfc_length_distribution", scenario.get("chain_length_distribution"))
    if isinstance(distribution, Mapping):
        weighted: List[tuple[int, float]] = []
        for raw_length, raw_weight in distribution.items():
            weight = max(0.0, float(raw_weight or 0.0))
            if weight > 0.0:
                weighted.append((max(1, int(raw_length)), weight))
        if weighted:
            if bool(scenario.get("stratified_sfc_lengths", False)):
                return int(
                    _stratified_weighted_choice(
                        weighted,
                        scenario,
                        seed,
                        request_index,
                        int(scenario.get("generated_sfc_count", request_index + 1) or request_index + 1),
                        salt=52021,
                    )
                )
            total = sum(weight for _length, weight in weighted)
            rng = random.Random(_stable_scenario_seed(scenario, seed, 52021) + int(request_index) * 17)
            threshold = rng.random() * total
            cumulative = 0.0
            for length, weight in weighted:
                cumulative += weight
                if threshold <= cumulative:
                    return length
            return weighted[-1][0]
    raw_range = scenario.get("sfc_length_range", scenario.get("chain_length_range"))
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        low = max(1, int(raw_range[0]))
        high = max(low, int(raw_range[1]))
        rng = random.Random(_stable_scenario_seed(scenario, seed, 53047) + int(request_index) * 31)
        return rng.randint(low, high)
    raw_length = scenario.get("sfc_length", scenario.get("chain_length"))
    if raw_length is not None:
        return max(1, int(raw_length))
    return None


def _stable_scenario_seed(scenario: Mapping[str, Any], seed: int, salt: int) -> int:
    name = str(scenario.get("name", "scenario"))
    stable_name = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return int(seed) * 1000003 + stable_name + int(salt)


def _select_request_type(
    scenario: Mapping[str, Any],
    seed: int,
    request_index: int,
) -> Optional[str]:
    distribution = scenario.get("request_type_distribution")
    if not isinstance(distribution, Mapping):
        return None
    weighted = [
        (str(name), max(0.0, float(weight or 0.0)))
        for name, weight in distribution.items()
        if max(0.0, float(weight or 0.0)) > 0.0
    ]
    if not weighted:
        return None
    if bool(scenario.get("stratified_request_types", False)):
        return str(
            _stratified_weighted_choice(
                weighted,
                scenario,
                seed,
                request_index,
                int(scenario.get("generated_sfc_count", request_index + 1) or request_index + 1),
                salt=71,
            )
        )
    total = sum(weight for _name, weight in weighted)
    threshold = random.Random(int(seed) * 1000003 + int(request_index) * 9176 + 71).random() * total
    cumulative = 0.0
    for name, weight in weighted:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return weighted[-1][0]


def _stratified_weighted_choice(
    weighted: Sequence[tuple[Any, float]],
    scenario: Mapping[str, Any],
    seed: int,
    request_index: int,
    sample_count: int,
    salt: int,
) -> Any:
    total = sum(max(0.0, float(weight)) for _item, weight in weighted)
    if total <= 0.0:
        return weighted[-1][0]
    count = max(1, int(sample_count))
    phase = random.Random(_stable_scenario_seed(scenario, seed, salt)).random() / float(count)
    point = (((int(request_index) + 0.5) / float(count)) + phase) % 1.0
    threshold = point * total
    cumulative = 0.0
    for item, weight in weighted:
        cumulative += max(0.0, float(weight))
        if threshold <= cumulative:
            return item
    return weighted[-1][0]


def _scenario_with_default_task_classes(config: Mapping[str, Any], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_cfg = copy.deepcopy(dict(scenario))
    experiment_cfg = dict(config.get("semantic_topology_experiment", {}) or {})
    if "task_classes" not in scenario_cfg and "task_classes" in experiment_cfg:
        scenario_cfg["task_classes"] = copy.deepcopy(experiment_cfg["task_classes"])
    return scenario_cfg


def _assign_v21_request_types(chains: Sequence[Dict[str, Any]], semantic_matrix: SemanticLinkMatrix) -> None:
    request_cycle = [name for name in ("forest_fire_monitoring", "traffic_surveillance", "urban_security", "industrial_inspection") if name in semantic_matrix.request_types]
    if not request_cycle:
        request_cycle = sorted(semantic_matrix.request_types)
    if not request_cycle:
        raise ValueError("V21 semantic assignment requires at least one request_type in request_types.yaml")
    for index, chain in enumerate(chains):
        context = dict(chain.get("context", {}) or {})
        chain_id = str(chain.get("sfc_id", f"chain_{index}"))
        if not context.get("request_type"):
            context["request_type"] = request_cycle[index % len(request_cycle)]
            if str(context["request_type"]) not in _LOGGED_V21_MISSING_REQUEST_TYPES:
                _LOGGED_V21_MISSING_REQUEST_TYPES.add(str(context["request_type"]))
                LOGGER.warning(
                    "V21 chains missing request_type; assigning %s from deterministic request cycle.",
                    context["request_type"],
                )
        chain["context"] = context
        request_type = str(context["request_type"])
        sequence = _representative_service_sequence(request_type, semantic_matrix)
        if context.get("preprocess_only"):
            sequence = sequence[:1]
        target_length_raw = context.get("representative_service_chain_length")
        explicit_target_length = target_length_raw is not None
        if explicit_target_length:
            target_length = max(1, min(len(sequence), int(float(target_length_raw))))
            sequence = _compressed_representative_service_sequence(
                request_type,
                sequence,
                target_length,
                semantic_matrix,
            )
        nodes = list(chain.get("nodes", []) or [])
        if explicit_target_length and len(nodes) > len(sequence):
            nodes = [copy.deepcopy(node) for node in nodes[: len(sequence)]]
            chain["nodes"] = nodes
            node_ids = [str(node.get("node_id")) for node in nodes]
            chain["edges"] = [{"from": node_ids[i], "to": node_ids[i + 1]} for i in range(len(node_ids) - 1)]
        if len(nodes) < len(sequence):
            expansion_key = (request_type, len(nodes), len(sequence))
            if expansion_key not in _LOGGED_V21_CHAIN_EXPANSIONS:
                _LOGGED_V21_CHAIN_EXPANSIONS.add(expansion_key)
                LOGGER.warning(
                    "V21 request_type %s expands matching chains from %d structural nodes to %d representative services.",
                    request_type,
                    len(nodes),
                    len(sequence),
                )
            template = copy.deepcopy(nodes[-1]) if nodes else {}
            for extra_index in range(len(nodes), len(sequence)):
                service_type = sequence[extra_index]
                input_semantic, output_semantic = SERVICE_IO_DEFAULTS.get(service_type, ("any", "any"))
                node = copy.deepcopy(template)
                node["node_id"] = f"v21_{extra_index}_{service_type}"
                node["service_type"] = service_type
                node["required_capabilities"] = [service_type]
                node["input_semantic"] = input_semantic
                node["output_semantic"] = output_semantic
                node.setdefault("cpu_mb", float(template.get("cpu_mb", template.get("cpu", 4.0)) or 4.0))
                node.setdefault("memory_mb", float(template.get("memory_mb", 512.0) or 512.0))
                nodes.append(node)
            chain["nodes"] = nodes
            node_ids = [str(node.get("node_id")) for node in nodes]
            chain["edges"] = [{"from": node_ids[index], "to": node_ids[index + 1]} for index in range(len(node_ids) - 1)]
        if len(nodes) > len(sequence):
            LOGGER.warning(
                "V21 request_type %s has %d representative services but chain %s has %d nodes; "
                "terminal service %s is repeated for excess structural slots.",
                request_type,
                len(sequence),
                chain_id,
                len(nodes),
                sequence[-1],
            )
        for node_index, node in enumerate(nodes):
            service_type = sequence[node_index] if node_index < len(sequence) else sequence[-1]
            previous_service_type = str(node.get("service_type", "") or "")
            if previous_service_type and previous_service_type != service_type:
                warning_key = (request_type, node_index, previous_service_type, service_type)
                if warning_key not in _LOGGED_V21_CHAIN_OVERWRITES:
                    _LOGGED_V21_CHAIN_OVERWRITES.add(warning_key)
                    LOGGER.warning(
                        "V21 request_type %s overwrites structural chain node %d service_type %s -> %s. "
                        "This is intentional: request_types.yaml is the semantic source of truth.",
                        request_type,
                        node_index,
                        previous_service_type,
                        service_type,
                    )
            node["service_type"] = service_type
            node["required_capabilities"] = [service_type]
            input_semantic, output_semantic = SERVICE_IO_DEFAULTS.get(service_type, ("any", "any"))
            node["input_semantic"] = input_semantic
            node["output_semantic"] = output_semantic
            _apply_v21_runtime_cost_defaults(node, service_type, semantic_matrix, context)


def _representative_service_sequence(request_type: str, semantic_matrix: SemanticLinkMatrix) -> list[str]:
    request = dict(semantic_matrix.request_types.get(str(request_type), {}) or {})
    raw_sequence = request.get("representative_service_chain", request.get("service_sequence", [])) or []
    sequence = [str(item) for item in raw_sequence if str(item) in semantic_matrix.service_type_to_idx]
    if not sequence:
        raise ValueError(f"V21 request_type {request_type!r} has no valid representative_service_chain entries")
    return sequence


def _compressed_representative_service_sequence(
    request_type: str,
    full_sequence: Sequence[str],
    target_length: int,
    semantic_matrix: SemanticLinkMatrix,
) -> list[str]:
    target = max(1, int(target_length))
    full = [str(item) for item in full_sequence if str(item) in semantic_matrix.service_type_to_idx]
    if target >= len(full):
        return full
    templates = _COMPRESSED_REQUEST_SERVICE_CHAINS.get(str(request_type), {})
    exact = templates.get(target)
    if exact:
        sequence = [str(item) for item in exact if str(item) in semantic_matrix.service_type_to_idx]
        if len(sequence) == target:
            return sequence
    return _semantic_milestone_sequence(full, target, semantic_matrix)


def _semantic_milestone_sequence(
    full_sequence: Sequence[str],
    target_length: int,
    semantic_matrix: SemanticLinkMatrix,
) -> list[str]:
    full = [str(item) for item in full_sequence if str(item) in semantic_matrix.service_type_to_idx]
    if target_length >= len(full):
        return full
    by_domain: Dict[str, List[str]] = {}
    for service_type in full:
        raw = dict(semantic_matrix.service_types.get(str(service_type), {}) or {})
        domain = str(raw.get("functional_domain", "") or "")
        by_domain.setdefault(domain, []).append(str(service_type))
    milestones: List[str] = []
    for domain in ("preprocessing", "detection", "fusion", "verification", "alerting", "reporting", "archival"):
        for service_type in by_domain.get(domain, []):
            if service_type not in milestones:
                milestones.append(service_type)
                break
        if len(milestones) >= target_length:
            break
    if full[-1] not in milestones and len(milestones) >= 3:
        milestones[-1] = full[-1]
    for service_type in full:
        if len(milestones) >= target_length:
            break
        if service_type not in milestones:
            milestones.append(service_type)
    return milestones[:target_length]


def _apply_v21_runtime_cost_defaults(
    node: Dict[str, Any],
    service_type: str,
    semantic_matrix: SemanticLinkMatrix,
    context: Mapping[str, Any],
) -> None:
    metadata = dict(node.get("metadata", {}) or {})
    raw = dict(semantic_matrix.service_types.get(str(service_type), {}) or {})
    domain = str(raw.get("functional_domain", "") or "")
    base_cpu_by_domain = {
        "preprocessing": 16.0,
        "detection": 42.0,
        "fusion": 34.0,
        "verification": 38.0,
        "alerting": 14.0,
        "archival": 18.0,
        "reporting": 24.0,
    }
    cpu_scale = max(0.0, float(context.get("service_cpu_scale", 1.0) or 1.0))
    task_cpu = base_cpu_by_domain.get(domain, 24.0) * cpu_scale
    metadata["task_cpu"] = task_cpu
    metadata["cpu_mb"] = task_cpu
    node["cpu_mb"] = task_cpu
    node["metadata"] = metadata


def _service_type_for_request_position(request_type: str, position: int, semantic_matrix: SemanticLinkMatrix) -> str:
    sequence = _representative_service_sequence(request_type, semantic_matrix)
    return sequence[min(max(0, int(position)), len(sequence) - 1)]


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
        deadline = _sample_scenario_deadline_s(scenario, seed, request_index)
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

    target_sfc_length = scenario.get("sfc_length", scenario.get("chain_length"))
    if target_sfc_length is not None:
        _apply_sfc_length_override(chain, int(target_sfc_length))

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
        context = dict(chain.get("context", {}) or {})
        context["service_cpu_scale"] = scale
        chain["context"] = context
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


def _sample_scenario_deadline_s(
    scenario: Mapping[str, Any],
    seed: int,
    request_index: int,
) -> Optional[float]:
    raw_range = scenario.get("qos_deadline_range_s", scenario.get("deadline_range_s"))
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        low = float(raw_range[0])
        high = float(raw_range[1])
        if high <= low:
            return low
        rng = random.Random(_stable_scenario_seed(scenario, seed, 61001) + int(request_index) * 43)
        return rng.uniform(low, high)
    deadline = scenario.get("qos_deadline_s", scenario.get("deadline_s"))
    if deadline is None:
        return None
    return float(deadline)


def _apply_sfc_length_override(chain: Dict[str, Any], target_length: int) -> None:
    """Extend the inspection pipeline for chain-length sweeps."""

    nodes = [dict(node) for node in chain.get("nodes", []) or []]
    edges = [dict(edge) for edge in chain.get("edges", []) or []]
    if target_length <= 0 or len(nodes) >= target_length:
        chain["nodes"] = nodes
        chain["edges"] = edges
        return
    while len(nodes) < target_length:
        last = nodes[-1] if nodes else {}
        previous_id = str(last.get("node_id", "verify"))
        index = len(nodes) + 1
        node_id = "archive" if "archive" not in {str(node.get("node_id")) for node in nodes} else f"archive_{index}"
        input_semantic = str(last.get("output_semantic", "verified_event"))
        output_semantic = "archived_event" if node_id == "archive" else f"archived_event_{index}"
        nodes.append(
            {
                "node_id": node_id,
                "service_type": "surveillance_event_archive",
                "required_capabilities": ["surveillance_event_archive"],
                "input_semantic": input_semantic,
                "output_semantic": output_semantic,
                "cpu_mb": 4.0,
                "memory_mb": 1024.0,
                "metadata": {
                    "task_cpu": 24.0,
                    "output_ratio": 0.10,
                    "required_returned_size": 0.0,
                },
            }
        )
        edges.append({"from": previous_id, "to": node_id})
    chain["nodes"] = nodes
    chain["edges"] = edges


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
    semantic_matrix: SemanticLinkMatrix,
) -> List[Dict[str, Any]]:
    """Inject V21 truth-driven semantic/topology conflicts without score bias."""

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
        ("hard_negative_mismatch", "mismatch", int(decoy_cfg.get("hard_negative_mismatch", 0) or 0)),
        ("borderline_weak", "weak", int(decoy_cfg.get("borderline_weak", 0) or 0)),
        ("remote_exact", "exact", int(decoy_cfg.get("remote_exact", 0) or 0)),
        ("remote_compatible", "compatible", int(decoy_cfg.get("remote_compatible", 0) or 0)),
        ("stale_clone_exact", "exact", int(decoy_cfg.get("stale_clone_exact", 0) or 0)),
    ]
    for service_id, template in sorted(service_templates.items()):
        for decoy_kind, variant_type, count in decoy_specs:
            implementations = [
                item for item in semantic_matrix.list_implementations_for(service_id) if item.variant_type == variant_type
            ]
            if not implementations:
                continue
            for index in range(max(0, count)):
                implementation = implementations[index % len(implementations)]
                node_type = _decoy_node_type(decoy_kind, allowed_node_types, index)
                node_id = _materialized_node_id(node_type, index + seed)
                region_id = _decoy_region_id(node_type, index, decoy_kind)
                profile = _truth_driven_decoy_profile(decoy_kind, node_type, rng)
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
                        "input_semantic": implementation.input_semantic_label,
                        "output_semantic": implementation.output_semantic_label,
                    }
                )
                metadata = semantic_matrix.candidate_metadata_from_implementation(implementation)
                metadata.update(
                    {
                        "is_decoy": True,
                        "semantic_group": decoy_kind,
                        "load_ratio_override": profile["load_ratio_override"],
                        "topology_risk": profile["topology_risk"],
                        "mobility_risk": profile["mobility_risk"],
                        "stale_latency_penalty_s": float(
                            dict(scenario.get("stale_candidate_injection", {}) or {}).get("stale_latency_penalty_s", 0.0)
                            if decoy_kind == "stale_clone_exact"
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
    """Return only SFC service templates; V21 semantics come from YAML profiles."""

    specs: Dict[str, Dict[str, Any]] = {}
    for chain in config.get("service_chains", []) or []:
        for node in chain.get("nodes", []) or []:
            service_id = str(node.get("service_type", node.get("node_id", "")))
            specs.setdefault(
                service_id,
                {
                    "service_id": service_id,
                    "required_capabilities": list(node.get("required_capabilities", []) or []),
                    "memory_mb": float(node.get("memory_mb", 512.0) or 512.0),
                    "storage_mb": float(node.get("storage_mb", 256.0) or 256.0),
                },
            )
    return list(specs.values())


def _decoy_node_type(decoy_kind: str, allowed_node_types: set[str], index: int) -> str:
    mobile_order = ["uav", "vehicle"]
    if decoy_kind in {"remote_exact", "remote_compatible", "stale_clone_exact"}:
        for node_type in mobile_order:
            if node_type in allowed_node_types:
                return node_type
    if decoy_kind == "borderline_weak":
        if "rsu" in allowed_node_types:
            return "rsu"
        if "cloud_server" in allowed_node_types:
            return "cloud_server"
    if decoy_kind == "hard_negative_mismatch":
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
    if decoy_kind in {"remote_exact", "remote_compatible", "stale_clone_exact", "hard_negative_mismatch"}:
        return f"RSU_{1 + (index % (PHYSICAL_RSU_COUNT - 1))}"
    return f"RSU_{index % PHYSICAL_RSU_COUNT}"


def _truth_driven_decoy_profile(decoy_kind: str, node_type: str, rng: random.Random) -> Dict[str, Any]:
    capacity = _materialized_capacity(node_type)
    if decoy_kind in {"remote_exact", "remote_compatible"}:
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.35, "memory": capacity["memory"] * 0.20, "storage": capacity["storage"] * 0.10},
            "max_concurrency": 3,
            "current_load": 1,
            "reliability_score": 0.94,
            "accuracy_score": 0.97,
            "trust_score": 0.96,
            "cold_start_s": 0.55 + 0.10 * rng.random(),
            "load_ratio_override": 0.92,
            "topology_risk": 0.88,
            "mobility_risk": 0.75 if node_type in {"uav", "vehicle"} else 0.25,
        }
    if decoy_kind == "borderline_weak":
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.05, "memory": capacity["memory"] * 0.05, "storage": capacity["storage"] * 0.05},
            "max_concurrency": 8 if node_type == "rsu" else 4,
            "current_load": 0,
            "reliability_score": 0.97,
            "accuracy_score": 0.94,
            "trust_score": 0.98,
            "cold_start_s": 0.08 + 0.03 * rng.random(),
            "load_ratio_override": 0.05,
            "topology_risk": 0.08,
            "mobility_risk": 0.05,
        }
    if decoy_kind == "hard_negative_mismatch":
        return {
            "capacity": capacity,
            "used": {"cpu": capacity["cpu"] * 0.04, "memory": capacity["memory"] * 0.05, "storage": capacity["storage"] * 0.05},
            "max_concurrency": 6 if node_type == "rsu" else 4,
            "current_load": 0,
            "reliability_score": 0.96,
            "accuracy_score": 0.90,
            "trust_score": 0.97,
            "cold_start_s": 0.05 + 0.02 * rng.random(),
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
        "load_ratio_override": 0.65,
        "topology_risk": 0.70,
        "mobility_risk": 0.70 if node_type in {"uav", "vehicle"} else 0.20,
    }


def _decoy_description(service_id: str, decoy_kind: str) -> str:
    service_text = str(service_id).replace("_", " ")
    if decoy_kind == "remote_exact":
        return f"remote exact implementation for {service_text} with strong semantic fit but expensive route conditions"
    if decoy_kind == "remote_compatible":
        return f"remote compatible implementation for {service_text} that can complete the chain when local profiles are weak"
    if decoy_kind == "borderline_weak":
        return f"nearby weak implementation for {service_text} with stable topology and partial semantic coverage"
    if decoy_kind == "hard_negative_mismatch":
        return f"nearby mismatched implementation for {service_text} with attractive resources but wrong business semantics"
    return f"stale remote exact clone for {service_text} whose advertisement can survive long TTL settings"


def _load_multiplier(scenario: Mapping[str, Any]) -> float:
    if scenario.get("load_multiplier") is not None:
        return max(0.1, float(scenario["load_multiplier"]))
    arrival = _scenario_expected_arrival_rate_sfc_per_s(scenario)
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
