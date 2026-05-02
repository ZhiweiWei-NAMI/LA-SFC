from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METHOD_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT, WORKSPACE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.baselines import baseline_names
from airfogsim.lasdm.benchmark_adapter import run_lasdm_benchmark_suite
from airfogsim.lasdm.env_adapter import LASDMEnvAdapter
from airfogsim.lasdm.graph_observation import flatten_observation
from airfogsim.lasdm.marl_policy import IPPOPolicy, policy_from_name
from airfogsim.lasdm.marl_trainer import (
    PPORolloutStep,
    TrainingMetrics,
    ppo_update_policy,
    rollout_steps_from_policy_step,
    write_reward_curve,
)
from airfogsim.lasdm.runtime_bridge import LASDMRuntimeBridge
from airfogsim.lasdm.topology_builder import TopologyBuilder

from evaluate_semantic_topology_marl import (
    DEFAULT_BASELINES,
    canonical_ippo_baseline,
    checkpoint_subdir_for_baseline,
    expert_policy_for_ippo_baseline,
    is_ippo_checkpoint_baseline,
    _baseline_settings,
    _select_scenarios,
)
from semantic_runtime_guard import SemanticRuntimeGuardConfig, evaluate_guard, load_semantic_summary
from train_semantic_topology_marl import DEFAULT_CONFIG as DEFAULT_SEMANTIC_CONFIG
from train_semantic_topology_marl import build_offline_env, _load_yaml


DEFAULT_LASDM_CONFIG = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")
DEFAULT_OUTPUT_ROOT = os.path.join(WORKSPACE_ROOT, "experiment_artifacts", "raw_data", "complete_runtime_scheduler")

SUMMARY_FIELDS = [
    "family",
    "baseline",
    "scenario",
    "service_role_sweep",
    "seed",
    "status",
    "completed",
    "error",
    "submitted",
    "succeeded",
    "failed",
    "timed_out",
    "success_ratio",
    "qos_hit_ratio",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "task_done_num",
    "task_fail_num",
    "task_success_ratio",
    "runtime_step_count",
    "simulation_time_end",
    "avg_decision_time_ms",
    "p95_decision_time_ms",
    "selected_node_sequence",
    "selected_node_type_sequence",
    "remote_candidate_ratio",
    "stale_selected_ratio",
    "semantic_score_mean",
    "semantic_score_min",
    "topology_risk_mean",
    "queue_wait_time_mean",
    "wireless_tx_time_mean",
    "compute_time_mean",
    "return_time_mean",
    "policy_name",
    "policy_source",
    "semantic_encoder_backend",
    "semantic_encoder_model",
    "checkpoint_loaded",
    "checkpoint_path",
    "checkpoint_sha256",
    "run_dir",
]

FUNCTION_TRACE_FIELDS = [
    "family",
    "baseline",
    "seed",
    "scenario",
    "service_role_sweep",
    "sfc_id",
    "function_id",
    "selected_node",
    "route",
    "tx_delay",
    "compute_delay",
    "queue_delay",
    "candidate_runtime_cost_s",
    "semantic_mismatch_runtime_cost_s",
    "semantic_score",
    "semantic_min_score",
    "semantic_quality_violation",
    "task_id",
    "finish_time",
    "e2e_delay",
    "status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete LASDM/Semantic-Topology AirFogSim runtime scheduler experiments.")
    parser.add_argument("--lasdm-config", default=DEFAULT_LASDM_CONFIG)
    parser.add_argument("--semantic-config", default=DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--semantic-repair-config", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lasdm-baselines", nargs="+", default=sorted(baseline_names()))
    parser.add_argument("--semantic-baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--semantic-service-role-sweeps", nargs="+", default=["full_hybrid"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--semantic-scenarios", nargs="+", default=None)
    parser.add_argument("--semantic-exchange-ttl-sweep", nargs="+", type=float, default=None)
    parser.add_argument("--semantic-exchange-radius-sweep", nargs="+", type=int, default=None)
    parser.add_argument("--skip-lasdm-runtime", action="store_true")
    parser.add_argument("--skip-semantic-runtime", action="store_true")
    parser.add_argument("--skip-runtime-training", action="store_true")
    parser.add_argument("--semantic-guard", action="store_true")
    parser.add_argument("--disable-semantic-guard", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    semantic_config = _load_semantic_config(args.semantic_config, args.semantic_repair_config)
    repair_cfg = dict(semantic_config.get("runtime_repair", {}) or {})
    guard_raw = dict(
        repair_cfg.get("centralized_planner_success_guard")
        or repair_cfg.get("oracle_success_guard", {})
        or {}
    )
    guard_enabled = bool((args.semantic_guard or guard_raw.get("enabled", False)) and not args.disable_semantic_guard)
    train_seeds = args.train_seeds or [int(seed) for seed in semantic_config.get("training", {}).get("train_seeds", [0, 1, 2, 3, 4])]
    eval_seeds = args.eval_seeds or [int(seed) for seed in semantic_config.get("training", {}).get("eval_seeds", list(range(10)))]
    checkpoint_strategy = str(dict(semantic_config.get("marl", {}) or {}).get("ippo_eval_checkpoint_strategy", "exact_seed"))
    ippo_train_baselines = _requested_ippo_baselines(args.semantic_baselines)
    if (
        checkpoint_strategy == "exact_seed"
        and ippo_train_baselines
        and not args.skip_runtime_training
    ):
        train_seeds = sorted(set(int(seed) for seed in train_seeds) | set(int(seed) for seed in eval_seeds))
    train_episodes = int(args.train_episodes or semantic_config.get("training", {}).get("episodes", 20))
    max_steps = int(
        args.max_steps
        or semantic_config.get("runtime", {}).get("max_steps")
        or semantic_config.get("training", {}).get("max_steps", 100)
    )
    semantic_scenarios = _select_scenarios(semantic_config, args.semantic_scenarios)
    semantic_scenarios = _expand_semantic_exchange_sweep(
        semantic_scenarios,
        args.semantic_exchange_ttl_sweep,
        args.semantic_exchange_radius_sweep,
    )

    payload: Dict[str, Any] = {
        "started_at": _timestamp(),
        "output_root": str(output_root),
        "lasdm_config": args.lasdm_config,
        "semantic_config": args.semantic_config,
        "semantic_repair_config": args.semantic_repair_config or "",
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "train_episodes": train_episodes,
        "max_steps": max_steps,
        "semantic_exchange_ttl_sweep": args.semantic_exchange_ttl_sweep or [],
        "semantic_exchange_radius_sweep": args.semantic_exchange_radius_sweep or [],
    }

    checkpoint_root = output_root / "semantic_runtime_train"
    if guard_enabled and not args.skip_semantic_runtime:
        payload["semantic_runtime_guard"] = run_semantic_runtime_guard(
            semantic_config,
            output_root / "semantic_runtime_guard_calibration",
            args.semantic_service_role_sweeps[:1],
            checkpoint_root,
            max_steps,
        )
        _write_json(output_root / "semantic_runtime_guard.json", payload["semantic_runtime_guard"])
        if not payload["semantic_runtime_guard"].get("passed"):
            payload["finished_at"] = _timestamp()
            _write_json(output_root / "complete_runtime_manifest.json", payload)
            raise RuntimeError(
                "Semantic runtime guard failed before the full matrix: "
                f"{payload['semantic_runtime_guard'].get('reason')}"
            )

    if not args.skip_runtime_training:
        payload["semantic_runtime_train"] = train_semantic_ippo_variants_runtime(
            semantic_config,
            checkpoint_root,
            ippo_train_baselines,
            train_seeds,
            semantic_scenarios,
            args.semantic_service_role_sweeps[:1],
            train_episodes,
            max_steps,
        )

    if not args.skip_semantic_runtime:
        payload["semantic_runtime_eval"] = evaluate_semantic_runtime(
            semantic_config,
            output_root / "semantic_runtime_eval",
            args.semantic_baselines,
            semantic_scenarios,
            args.semantic_service_role_sweeps,
            eval_seeds,
            checkpoint_root,
            max_steps,
        )

    if not args.skip_lasdm_runtime:
        lasdm_output_root = output_root / "lasdm_runtime"
        exit_code, lasdm_payload = run_lasdm_benchmark_suite(
            config_path=args.lasdm_config,
            baseline_override=args.lasdm_baselines,
            seed_override=eval_seeds,
            scenario_override=None,
            output_root_override=str(lasdm_output_root),
            mode_override="airfogsim",
        )
        payload["lasdm_runtime"] = lasdm_payload
        payload["lasdm_runtime_exit_code"] = exit_code

    payload["finished_at"] = _timestamp()
    _write_json(output_root / "complete_runtime_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _requested_ippo_baselines(baselines: Sequence[str]) -> List[str]:
    requested = [canonical_ippo_baseline(str(item)) for item in baselines if is_ippo_checkpoint_baseline(str(item))]
    if not requested:
        requested = ["proposed_semantic_topology_marl"]
    ordered: List[str] = []
    for item in ["proposed_semantic_topology_marl", *requested]:
        if item in requested and item not in ordered:
            ordered.append(item)
    return ordered


def _checkpoint_root_for_baseline(checkpoint_root: Path, baseline: str) -> Path:
    subdir = checkpoint_subdir_for_baseline(baseline)
    return checkpoint_root if not subdir else checkpoint_root / subdir


def _config_with_baseline_updates(config: Mapping[str, Any], baseline: str) -> Dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    _policy_name, cfg_updates = _baseline_settings(baseline)
    if cfg_updates:
        updated.setdefault("marl", {}).update(cfg_updates)
    return updated


def train_semantic_ippo_variants_runtime(
    config: Dict[str, Any],
    output_root: Path,
    baselines: Sequence[str],
    seeds: Sequence[int],
    scenarios: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    variants = _requested_ippo_baselines(baselines)
    results: Dict[str, Any] = {}
    runs: List[Dict[str, Any]] = []
    for baseline in variants:
        variant_config = _config_with_baseline_updates(config, baseline)
        variant_root = _checkpoint_root_for_baseline(output_root, baseline)
        result = train_semantic_ippo_runtime(
            variant_config,
            variant_root,
            seeds,
            scenarios,
            roles,
            episodes,
            max_steps,
            baseline=baseline,
        )
        results[baseline] = result
        runs.extend(result.get("runs", []) or [])
    return {
        "completed": all(bool(item.get("completed", False)) for item in results.values()) if results else True,
        "variant_count": len(variants),
        "variants": variants,
        "output_dir": str(output_root),
        "runs": runs,
        "by_variant": results,
    }


def train_semantic_ippo_runtime(
    config: Dict[str, Any],
    output_root: Path,
    seeds: Sequence[int],
    scenarios: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    episodes: int,
    max_steps: int,
    baseline: str = "proposed_semantic_topology_marl",
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_root / f"ippo_seed_{seed}"
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        policy: Optional[IPPOPolicy] = None
        marl_cfg = dict(config.get("marl", {}) or {})
        policy_config = _config_with_baseline_updates(config, baseline)
        if baseline == "proposed_semantic_topology_marl":
            expert_policy_name = str(marl_cfg.get("ippo_expert_policy", "topology_greedy"))
        else:
            expert_policy_name = str(
                marl_cfg.get(f"{baseline}_ippo_expert_policy", expert_policy_for_ippo_baseline(baseline))
            )
        expert_policy = policy_from_name(
            expert_policy_name,
            seed=seed,
            branch_width=int(marl_cfg.get("ippo_expert_branch_width", 8) or 8),
            load_penalty=float(marl_cfg.get("ippo_expert_load_penalty", 0.5) or 0.0),
            topology_risk_penalty=float(marl_cfg.get("ippo_expert_topology_risk_penalty", 0.5) or 0.0),
            mobility_risk_penalty=float(marl_cfg.get("ippo_expert_mobility_risk_penalty", 0.2) or 0.0),
            route_hops_penalty=float(marl_cfg.get("ippo_expert_route_hops_penalty", 0.0) or 0.0),
            route_tx_penalty=float(marl_cfg.get("ippo_expert_route_tx_penalty", 0.0) or 0.0),
            route_unavailable_penalty=float(marl_cfg.get("ippo_expert_route_unavailable_penalty", 2.0) or 0.0),
            cold_start_penalty=float(marl_cfg.get("ippo_expert_cold_start_penalty", 0.0) or 0.0),
            deadline_violation_penalty=float(marl_cfg.get("ippo_expert_deadline_violation_penalty", 0.0) or 0.0),
            runtime_penalty=float(marl_cfg.get("ippo_expert_runtime_penalty", 0.0) or 0.0),
            semantic_mismatch_penalty=float(marl_cfg.get("ippo_expert_semantic_mismatch_penalty", 0.0) or 0.0),
            utility_prior_weight=float(marl_cfg.get("ippo_expert_utility_prior_weight", 2.0) or 0.0),
            remote_penalty=float(marl_cfg.get("ippo_expert_remote_penalty", 0.0) or 0.0),
            stale_penalty=float(marl_cfg.get("ippo_expert_stale_penalty", 0.05) or 0.0),
        )
        bc_pretrain_episodes = int(marl_cfg.get("ippo_behavior_clone_pretrain_episodes", 0) or 0)
        bc_pretrain_steps = int(marl_cfg.get("ippo_behavior_clone_steps_per_observation", 1) or 0)
        bc_finetune_steps = int(marl_cfg.get("ippo_behavior_clone_finetune_steps_per_observation", 0) or 0)
        reward_curve_policy_only = bool(marl_cfg.get("ippo_reward_curve_policy_only", False))
        selection_interval = int(marl_cfg.get("ippo_checkpoint_selection_interval", 0) or 0)
        early_selection_interval = int(marl_cfg.get("ippo_checkpoint_selection_early_interval", 0) or 0)
        early_selection_until = int(marl_cfg.get("ippo_checkpoint_selection_early_until_episode", 0) or 0)
        selection_max_steps = int(marl_cfg.get("ippo_checkpoint_selection_max_steps", max_steps) or max_steps)
        raw_seed_offsets = marl_cfg.get("ippo_checkpoint_selection_seed_offsets", [0, 1000])
        selection_seed_offsets = [int(item) for item in raw_seed_offsets] if isinstance(raw_seed_offsets, Sequence) and not isinstance(raw_seed_offsets, str) else [0, 1000]
        selection_mode = str(marl_cfg.get("ippo_checkpoint_selection_mode", "raw") or "raw")
        selection_metric = str(marl_cfg.get("ippo_checkpoint_selection_metric", "success_ratio") or "success_ratio")
        selection_window = max(1, int(marl_cfg.get("ippo_checkpoint_selection_robust_window", 1) or 1))
        selection_std_penalty = float(marl_cfg.get("ippo_checkpoint_selection_std_penalty", 0.0) or 0.0)
        best_state: Optional[Dict[str, Any]] = None
        best_score = -float("inf")
        best_metadata: Dict[str, Any] = {}
        bc_checkpoint_path = seed_dir / "ippo_bc_policy.pt"
        bc_checkpoint_saved = False
        bc_checkpoint_error = ""
        bc_metadata: Dict[str, Any] = {}
        bc_selection_rows: List[Dict[str, Any]] = []
        selection_rows: List[Dict[str, Any]] = []
        reward_rows: List[TrainingMetrics] = []
        progress_rows: List[Dict[str, Any]] = []
        ppo_diagnostic_rows: List[Dict[str, Any]] = []
        rollout_buffer: List[PPORolloutStep] = []
        rollout_episode_count = 0
        ppo_update_index = 0
        rollout_episodes_per_update = max(1, int(marl_cfg.get("ippo_rollout_episodes_per_update", 1) or 1))
        target_kl = float(marl_cfg.get("ippo_target_kl", 0.02) or 0.02)
        max_grad_norm = float(marl_cfg.get("ippo_max_grad_norm", 0.5) or 0.5)
        normalize_returns = bool(marl_cfg.get("ippo_return_norm_enabled", marl_cfg.get("ippo_normalize_returns", True)))
        return_norm_momentum = float(marl_cfg.get("ippo_return_norm_momentum", 0.95) or 0.95)
        return_norm_eps = float(marl_cfg.get("ippo_return_norm_eps", 1e-6) or 1e-6)
        bc_label_strategy = str(marl_cfg.get("ippo_bc_label_strategy", "expert") or "expert")
        bc_stale_relabel_margin = float(marl_cfg.get("ippo_bc_stale_relabel_margin", 0.05) or 0.0)
        bc_deadline_relabel_margin = float(marl_cfg.get("ippo_bc_deadline_relabel_margin", 0.05) or 0.0)
        per_function_rollout_samples = bool(marl_cfg.get("ippo_per_function_rollout_samples", False))
        share_reward_across_functions = bool(marl_cfg.get("ippo_share_reward_across_function_samples", True))
        per_function_reward_mode = str(marl_cfg.get("ippo_per_function_reward_mode", "shared") or "shared")
        for episode in range(int(episodes)):
            pretrain_episode = episode < bc_pretrain_episodes
            if pretrain_episode:
                scenario = _ippo_bc_scenario_for_episode(scenarios, marl_cfg, episode)
            else:
                scenario = _ippo_training_scenario_for_episode(
                    scenarios,
                    marl_cfg,
                    episode - bc_pretrain_episodes,
                    max(1, int(episodes) - bc_pretrain_episodes),
                )
            role = roles[episode % len(roles)]
            env, air_env = _build_semantic_runtime_env(config, scenario, role, seed, baseline, max_steps)
            episode_steps: List[PPORolloutStep] = []
            try:
                observations = env.reset()
                wrote_episode_traces = False
                episode_summary: Dict[str, Any] = {}
                episode_step_count = 0
                if policy is None:
                    obs_dim = max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
                    max_candidates = int(config.get("marl", {}).get("max_candidates", 16))
                    policy = IPPOPolicy(
                        **_ippo_policy_kwargs(policy_config, obs_dim, max_candidates, seed, observations=observations)
                    )
                total = 0.0
                bc_steps = bc_pretrain_steps if pretrain_episode else bc_finetune_steps
                bc_loss_weighted_sum = 0.0
                bc_sample_count = 0
                bc_relabel_count = 0
                bc_quality_rows: List[Dict[str, Any]] = []
                for step in range(int(max_steps)):
                    expert_actions = expert_policy.act(observations, deterministic=True)
                    for _ in range(max(0, bc_steps)):
                        policy.supervised_update(
                            observations,
                            expert_actions,
                            label_strategy=bc_label_strategy,
                            stale_relabel_margin=bc_stale_relabel_margin,
                            deadline_relabel_margin=bc_deadline_relabel_margin,
                        )
                        bc_loss_weighted_sum += float(getattr(policy, "last_supervised_loss", 0.0)) * float(
                            getattr(policy, "last_supervised_samples", 0) or 0
                        )
                        bc_sample_count += int(getattr(policy, "last_supervised_samples", 0) or 0)
                        bc_relabel_count += int(getattr(policy, "last_supervised_relabels", 0) or 0)
                    if bc_steps > 0:
                        bc_quality_rows.append(_ippo_bc_quality_metrics(policy, observations, expert_actions))
                    current_observations = observations
                    if pretrain_episode:
                        policy_step = None
                        actions = expert_actions
                    else:
                        policy_step = policy.act_with_logprobs(current_observations, deterministic=False, track_grad=False)
                        actions = policy_step.actions
                    observations, rewards, done, info = env.step(actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    if policy_step is not None:
                        episode_steps.extend(
                            rollout_steps_from_policy_step(
                                policy,
                                current_observations,
                                policy_step.actions,
                                mean_reward=float(mean_reward),
                                done=bool(done or step + 1 >= int(max_steps)),
                                episode=int(episode),
                                per_function_samples=per_function_rollout_samples,
                                share_reward_across_functions=share_reward_across_functions,
                                per_function_reward_mode=per_function_reward_mode,
                                reward_aux=info.get("reward_aux", {}),
                            )
                        )
                    summary_dict = info.get("summary", {})
                    episode_summary = dict(summary_dict)
                    episode_step_count = step + 1
                    if not (pretrain_episode and reward_curve_policy_only):
                        reward_rows.append(
                            TrainingMetrics(
                                episode=episode,
                                step=step,
                                mean_reward=mean_reward,
                                total_reward=total,
                                succeeded=int(summary_dict.get("succeeded", 0) or 0),
                                failed=int(summary_dict.get("failed", 0) or 0),
                                timed_out=int(summary_dict.get("timed_out", 0) or 0),
                                active_graphs=int(summary_dict.get("active_graphs", 0) or 0),
                            )
                        )
                    if done:
                        break
                if not pretrain_episode:
                    if episode_steps:
                        rollout_buffer.extend(episode_steps)
                        rollout_episode_count += 1
                    should_update = (
                        rollout_episode_count >= rollout_episodes_per_update
                        or episode == int(episodes) - 1
                    )
                    if should_update and rollout_buffer:
                        entropy_coef = _scheduled_ippo_entropy_coef(marl_cfg, episode)
                        metrics = _runtime_ippo_episode_update(
                            policy,
                            rollout_buffer,
                            gamma=float(marl_cfg.get("ippo_gamma", 0.99) or 0.99),
                            gae_lambda=float(marl_cfg.get("ippo_gae_lambda", 0.95) or 0.95),
                            clip_eps=float(marl_cfg.get("ippo_clip_eps", 0.2) or 0.2),
                            entropy_coef=entropy_coef,
                            actor_loss_coef=float(marl_cfg.get("ippo_actor_loss_coef", 1.0) or 1.0),
                            value_coef=float(marl_cfg.get("ippo_value_coef", 1.0) or 1.0),
                            update_epochs=int(marl_cfg.get("ippo_update_epochs", 4) or 4),
                            minibatch_size=int(marl_cfg.get("ippo_minibatch_size", 64) or 64),
                            prior_l2_coef=float(marl_cfg.get("ippo_prior_l2_coef", 1e-3) or 0.0),
                            target_kl=target_kl,
                            max_grad_norm=max_grad_norm,
                            normalize_returns=normalize_returns,
                            return_norm_momentum=return_norm_momentum,
                            return_norm_eps=return_norm_eps,
                        )
                        ppo_update_index += 1
                        ppo_diagnostic_rows.append(
                            {
                                "episode": int(episode),
                                "seed": int(seed),
                                "baseline": baseline,
                                "update_index": ppo_update_index,
                                "rollout_episodes": rollout_episode_count,
                                "rollout_steps": len(rollout_buffer),
                                "entropy_coef": entropy_coef,
                                **metrics,
                            }
                        )
                        _write_csv_dynamic(seed_dir / "ppo_diagnostics.csv", ppo_diagnostic_rows)
                        rollout_buffer = []
                        rollout_episode_count = 0
                if episode == episodes - 1:
                    env.write_traces(str(seed_dir))
                    _write_runtime_trace_files(seed_dir, "semantic_runtime_train", baseline, scenario, role, seed, env)
                    wrote_episode_traces = True
                selection_allowed = (episode + 1) >= bc_pretrain_episodes
                bc_boundary = bc_pretrain_episodes > 0 and (episode + 1) == bc_pretrain_episodes
                active_selection_interval = selection_interval
                if early_selection_interval > 0 and (episode + 1) <= max(0, early_selection_until):
                    active_selection_interval = early_selection_interval
                selection_due = (
                    active_selection_interval > 0
                    and ((episode + 1) % active_selection_interval == 0 or episode == episodes - 1 or bc_boundary)
                )
                if (
                    policy is not None
                    and selection_allowed
                    and selection_due
                ):
                    _close_env(air_env)
                    air_env = None
                    validation_seeds = [
                        int(seed) if offset == 0 else int(seed) + int(offset)
                        for offset in selection_seed_offsets
                    ]
                    validation_metrics = _evaluate_ippo_policy_selection_suite(
                        config,
                        scenarios,
                        roles,
                        validation_seeds,
                        policy,
                        selection_max_steps,
                        baseline,
                    )
                    raw_validation_score = _ippo_checkpoint_selection_score(validation_metrics, metric=selection_metric)
                    checkpoint_source = "bc" if bc_boundary else "ppo"
                    selection_row = {
                        "episode": episode,
                        "seed": seed,
                        "scenario": "validation_suite",
                        "role": ",".join(str(item) for item in roles),
                        "validation_seed": ",".join(str(item) for item in validation_seeds),
                        "checkpoint_source": checkpoint_source,
                        "selection_metric": selection_metric,
                        "raw_selection_score": raw_validation_score,
                        **validation_metrics,
                    }
                    selection_rows.append(selection_row)
                    robust_selection_score = _ippo_robust_checkpoint_selection_score(
                        selection_rows,
                        window=selection_window,
                        std_penalty=selection_std_penalty,
                    )
                    selection_score = (
                        robust_selection_score
                        if selection_mode in {"robust", "bc_or_ppo_robust", "robust_bc_or_ppo"}
                        else raw_validation_score
                    )
                    selection_row.update(
                        {
                            "selection_score": selection_score,
                            "robust_selection_score": robust_selection_score,
                            "selection_mode": selection_mode,
                            "robust_window": selection_window,
                            "robust_std_penalty": selection_std_penalty,
                            "bc_reference_score": bc_metadata.get("raw_selection_score", ""),
                            "bc_reference_success_ratio": bc_metadata.get("success_ratio", ""),
                            "bc_reference_min_success_ratio": bc_metadata.get("min_success_ratio", ""),
                        }
                    )
                    if bc_boundary:
                        bc_metadata = dict(selection_row)
                        selection_row.update(
                            {
                                "bc_reference_score": selection_row.get("raw_selection_score", ""),
                                "bc_reference_success_ratio": selection_row.get("success_ratio", ""),
                                "bc_reference_min_success_ratio": selection_row.get("min_success_ratio", ""),
                            }
                        )
                    seed_non_decrease_count = _ippo_validation_seed_non_decrease_count(
                        selection_row,
                        bc_metadata,
                        tolerance=float(
                            marl_cfg.get("ippo_checkpoint_selection_seed_non_decrease_tolerance", 0.0) or 0.0
                        ),
                    )
                    selection_row["validation_seed_success_non_decrease_count"] = seed_non_decrease_count
                    selection_row["validation_seed_success_non_decrease_required"] = int(
                        marl_cfg.get("ippo_checkpoint_selection_min_seed_non_decrease_count", 0) or 0
                    )
                    eligible_for_best = _ippo_checkpoint_eligible_for_best(selection_row, bc_metadata, marl_cfg)
                    selection_row["eligible_for_best_checkpoint"] = int(eligible_for_best)
                    if bc_boundary:
                        bc_metadata = dict(selection_row)
                    _write_csv_dynamic(seed_dir / "checkpoint_selection.csv", selection_rows)
                    if eligible_for_best and selection_score > best_score:
                        best_score = selection_score
                        best_state = copy.deepcopy(policy.model.state_dict())
                        best_metadata = dict(selection_row)
                    if bc_boundary:
                        try:
                            policy.torch.save(policy.model.state_dict(), bc_checkpoint_path)
                            bc_checkpoint_saved = bc_checkpoint_path.exists()
                            bc_checkpoint_error = ""
                        except Exception as exc:
                            bc_checkpoint_saved = False
                            bc_checkpoint_error = str(exc)
                        bc_selection_rows.append(
                            {
                                **selection_row,
                                "checkpoint_path": str(bc_checkpoint_path) if bc_checkpoint_saved else "",
                                "checkpoint_saved": bc_checkpoint_saved,
                                "checkpoint_sha256": _sha256_file(bc_checkpoint_path) if bc_checkpoint_saved else "",
                                "checkpoint_error": bc_checkpoint_error,
                            }
                        )
                        _write_csv_dynamic(seed_dir / "bc_checkpoint_selection.csv", bc_selection_rows)
                if episode == episodes - 1 and not wrote_episode_traces:
                    env.write_traces(str(seed_dir))
                    _write_runtime_trace_files(seed_dir, "semantic_runtime_train", baseline, scenario, role, seed, env)
                progress_row = _runtime_training_progress_row(
                    seed=seed,
                    episode=episode,
                    scenario=scenario,
                    role=role,
                    pretrain_episode=pretrain_episode,
                    bc_steps=bc_steps,
                    step_count=episode_step_count,
                    total_reward=total,
                    summary=episode_summary,
                )
                progress_row.update(
                    {
                        "bc_loss": bc_loss_weighted_sum / max(1, bc_sample_count),
                        "bc_samples": bc_sample_count,
                        "bc_relabels": bc_relabel_count,
                        **_mean_metric_rows(bc_quality_rows, prefix="bc_"),
                    }
                )
                progress_rows.append(progress_row)
                write_reward_curve(seed_dir / "reward_curve.csv", reward_rows)
                _write_csv_dynamic(seed_dir / "train_progress.csv", progress_rows)
                _write_csv_dynamic(seed_dir / "ppo_diagnostics.csv", ppo_diagnostic_rows)
                progress_guard = _evaluate_training_progress_guard(progress_rows, marl_cfg)
                if not progress_guard.get("passed", True):
                    _write_json(seed_dir / "training_progress_guard_failure.json", progress_guard)
                    raise RuntimeError(
                        "IPPO training progress guard failed: "
                        f"{progress_guard.get('reason')} | details={progress_guard.get('details')}"
                    )
            finally:
                _close_env(air_env)
        write_reward_curve(seed_dir / "reward_curve.csv", reward_rows)
        _write_csv_dynamic(seed_dir / "checkpoint_selection.csv", selection_rows)
        if bc_selection_rows:
            _write_csv_dynamic(seed_dir / "bc_checkpoint_selection.csv", bc_selection_rows)
        checkpoint_path = seed_dir / "ippo_policy.pt"
        checkpoint_saved = False
        checkpoint_error = ""
        if policy is not None:
            try:
                if best_state is not None:
                    policy.model.load_state_dict(best_state)
                policy.torch.save(policy.model.state_dict(), checkpoint_path)
                checkpoint_saved = checkpoint_path.exists()
            except Exception as exc:
                checkpoint_error = str(exc)
        train_summary = {
            "baseline": baseline,
            "seed": seed,
            "episodes": episodes,
            "max_steps": max_steps,
            "reward_rows": len(reward_rows),
            "output_dir": str(seed_dir),
            "checkpoint_saved": checkpoint_saved,
            "checkpoint_path": str(checkpoint_path) if checkpoint_saved else "",
            "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_saved else "",
            "checkpoint_error": checkpoint_error,
            "bc_checkpoint_saved": bc_checkpoint_saved,
            "bc_checkpoint_path": str(bc_checkpoint_path) if bc_checkpoint_saved else "",
            "bc_checkpoint_sha256": _sha256_file(bc_checkpoint_path) if bc_checkpoint_saved else "",
            "bc_checkpoint_error": bc_checkpoint_error,
            "bc_selection": bc_metadata,
            "behavior_clone_pretrain_episodes": bc_pretrain_episodes,
            "behavior_clone_pretrain_steps_per_observation": bc_pretrain_steps,
            "behavior_clone_finetune_steps_per_observation": bc_finetune_steps,
            "reward_curve_policy_only": reward_curve_policy_only,
            "expert_policy": expert_policy_name,
            "ippo_centralized_critic": bool(marl_cfg.get("ippo_centralized_critic", True)),
            "ippo_critic_agent_count": int(marl_cfg.get("ippo_critic_agent_count", 4) or 4),
            "checkpoint_selection_interval": selection_interval,
            "checkpoint_selection_early_interval": early_selection_interval,
            "checkpoint_selection_early_until_episode": early_selection_until,
            "checkpoint_selection_max_steps": selection_max_steps,
            "checkpoint_selection_seed_offsets": selection_seed_offsets,
            "best_selection_score": best_score if best_metadata else "",
            "best_selection": best_metadata,
            "prior_weights": policy.prior_weight_snapshot() if policy is not None and hasattr(policy, "prior_weight_snapshot") else {},
            "last_metrics": reward_rows[-1].to_dict() if reward_rows else {},
        }
        _write_json(seed_dir / "train_summary.json", train_summary)
        summary.append(train_summary)
    return {"completed": True, "seed_count": len(seeds), "output_dir": str(output_root), "runs": summary}


def evaluate_semantic_runtime(
    config: Dict[str, Any],
    output_root: Path,
    baselines: Sequence[str],
    scenarios: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    seeds: Sequence[int],
    checkpoint_root: Path,
    max_steps: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        for scenario in scenarios:
            for role in roles:
                for seed in seeds:
                    run_dir = output_root / f"{baseline}__{scenario['name']}__{role}__seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    record = _run_semantic_runtime_single(config, baseline, scenario, role, seed, checkpoint_root, max_steps, run_dir)
                    rows.append(record)
                    _write_json(run_dir / "run.json", record)
                    partial_guard = _evaluate_partial_result_guard(rows, config)
                    if not partial_guard.get("passed", True):
                        _write_json(output_root / "early_result_guard_failure.json", partial_guard)
                        raise RuntimeError(
                            "Early semantic runtime guard failed: "
                            f"{partial_guard.get('reason')} | details={partial_guard.get('details')}"
                        )
    _write_csv(output_root / "summary.csv", SUMMARY_FIELDS, rows)
    _write_aggregate_csv(output_root / "aggregate.csv", rows)
    _write_combined_runtime_outputs(output_root, rows)
    return {
        "completed": all(not row.get("error") for row in rows),
        "run_count": len(rows),
        "completed_run_count": sum(1 for row in rows if row.get("completed")),
        "failed_run_count": sum(1 for row in rows if row.get("error")),
        "output_dir": str(output_root),
    }


def _evaluate_partial_result_guard(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(dict(config.get("runtime_repair", {}) or {}).get("early_result_guard", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return {"passed": True, "reason": "early result guard disabled", "details": {}}

    min_completed = max(1, int(cfg.get("min_completed_per_group", 3) or 3))
    stage_count = max(1, int(cfg.get("stage_count", 3) or 3))
    chain_tolerance = float(cfg.get("chain_independence_tolerance", 0.08) or 0.08)
    min_chain_success = float(cfg.get("min_chain_success_ratio", 0.55) or 0.55)
    min_task_success = float(cfg.get("min_task_success_ratio", 0.55) or 0.55)
    zero_success_min_completed = max(1, int(cfg.get("zero_success_min_completed", min_completed) or min_completed))
    threshold_by_baseline = {
        str(key): float(value) for key, value in dict(cfg.get("min_success_by_baseline", {}) or {}).items()
    }
    threshold_by_group = {
        str(key): float(value) for key, value in dict(cfg.get("min_success_by_group", {}) or {}).items()
    }
    chain_guard_baselines = {
        str(name)
        for name in cfg.get("chain_model_guard_baselines", [])
    }

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("error"):
            return {"passed": False, "reason": "runtime row has error", "details": dict(row)}
        baseline = str(row.get("baseline", ""))
        scenario = str(row.get("scenario", ""))
        grouped.setdefault((baseline, scenario), []).append(row)

    checked: List[Dict[str, Any]] = []
    for (baseline, scenario), group_rows in grouped.items():
        completed = [row for row in group_rows if row.get("completed")]
        if len(completed) < min_completed:
            continue
        success = mean(float(row.get("success_ratio", 0.0) or 0.0) for row in completed)
        task_success = mean(float(row.get("task_success_ratio", 0.0) or 0.0) for row in completed)
        independent_chain_success = task_success**stage_count
        threshold = threshold_by_group.get(f"{baseline}::{scenario}", threshold_by_baseline.get(baseline))
        details = {
            "baseline": baseline,
            "scenario": scenario,
            "completed": len(completed),
            "success_ratio_mean": success,
            "task_success_ratio_mean": task_success,
            "task_success_power_stage_count": independent_chain_success,
            "stage_count": stage_count,
        }
        checked.append(details)
        if len(completed) >= zero_success_min_completed and success <= 0.0:
            return {"passed": False, "reason": "zero SFC success in partial results", "details": details, "checked": checked}
        if threshold is not None and success < threshold:
            details["threshold"] = threshold
            return {"passed": False, "reason": "partial SFC success below configured threshold", "details": details, "checked": checked}
        apply_chain_model_guard = baseline in chain_guard_baselines and (
            threshold is None or float(threshold) >= min_chain_success
        )
        chain_like_independent_failures = apply_chain_model_guard and abs(success - independent_chain_success) <= chain_tolerance
        if chain_like_independent_failures and success < min_chain_success and task_success >= min_task_success:
            details["chain_independence_tolerance"] = chain_tolerance
            details["min_chain_success_ratio"] = min_chain_success
            details["min_task_success_ratio"] = min_task_success
            return {"passed": False, "reason": "SFC success matches task_success^stage_count failure model", "details": details, "checked": checked}
    return {"passed": True, "reason": "partial result guard passed", "details": {"checked_groups": checked}}


def run_semantic_runtime_guard(
    config: Dict[str, Any],
    output_root: Path,
    roles: Sequence[str],
    checkpoint_root: Path,
    max_steps: int,
) -> Dict[str, Any]:
    repair_cfg = dict(config.get("runtime_repair", {}) or {})
    guard_raw = dict(
        repair_cfg.get("centralized_planner_success_guard")
        or repair_cfg.get("oracle_success_guard", {})
        or {}
    )
    baseline = str(guard_raw.get("baseline", "centralized_planner"))
    scenario_name = str(guard_raw.get("calibration_scenario", "semantic_runtime_calibration_easy"))
    seeds = [int(seed) for seed in guard_raw.get("seeds", [0, 1])]
    scenario = _scenario_by_name(config, scenario_name)
    result = evaluate_semantic_runtime(
        config,
        output_root,
        [baseline],
        [scenario],
        roles or ["full_hybrid"],
        seeds,
        checkpoint_root,
        max_steps,
    )
    guard_cfg = SemanticRuntimeGuardConfig(
        planner_baseline=baseline,
        calibration_scenario=scenario_name,
        min_planner_success_ratio=float(guard_raw.get("min_success_ratio", 0.50)),
        fail_on_all_zero_success=bool(repair_cfg.get("fail_fast_on_all_zero_success", True)),
    )
    guard = evaluate_guard(load_semantic_summary(output_root), guard_cfg)
    details = dict(guard.get("details", {}) or {})
    details.update(_semantic_runtime_trace_diagnostics(output_root))
    guard["details"] = details
    missing: List[str] = []
    if details.get("candidate_trace_rows", 0) <= 0:
        missing.append("candidate_trace_rows")
    if details.get("accepted_decisions", 0) <= 0:
        missing.append("accepted_decisions")
    if details.get("runtime_reports_with_tasks", 0) <= 0:
        missing.append("runtime_reports_with_tasks")
    if missing:
        guard["passed"] = False
        guard["reason"] = f"semantic runtime guard missing required trace evidence: {', '.join(missing)}"
    guard["calibration_run"] = result
    return guard


def _run_semantic_runtime_single(
    config: Dict[str, Any],
    baseline: str,
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    checkpoint_root: Path,
    max_steps: int,
    run_dir: Path,
) -> Dict[str, Any]:
    env = None
    air_env = None
    error = None
    reward_rows: List[TrainingMetrics] = []
    started = time.perf_counter()
    try:
        env, air_env = _build_semantic_runtime_env(config, scenario, role, seed, baseline, max_steps)
        observations = env.reset()
        policy = _semantic_policy_for_eval(config, baseline, seed, observations, checkpoint_root)
        policy_metadata = {
            "policy_name": type(policy).__name__,
            "policy_source": str(getattr(policy, "policy_source", "")),
            "checkpoint_loaded": bool(getattr(policy, "checkpoint_loaded", False)),
            "checkpoint_path": str(getattr(policy, "checkpoint_path", "")),
            "checkpoint_sha256": str(getattr(policy, "checkpoint_sha256", "")),
            "checkpoint_train_seed": str(getattr(policy, "checkpoint_train_seed", "")),
            "checkpoint_strategy": str(getattr(policy, "checkpoint_strategy", "")),
        }
        for step in range(int(max_steps)):
            actions = policy.act(observations, deterministic=True)
            observations, rewards, done, info = env.step(actions)
            mean_reward = sum(rewards.values()) / max(1, len(rewards))
            summary = info.get("summary", {})
            reward_rows.append(
                TrainingMetrics(
                    episode=0,
                    step=step,
                    mean_reward=mean_reward,
                    total_reward=sum(row.mean_reward for row in reward_rows) + mean_reward,
                    succeeded=int(summary.get("succeeded", 0) or 0),
                    failed=int(summary.get("failed", 0) or 0),
                    timed_out=int(summary.get("timed_out", 0) or 0),
                    active_graphs=int(summary.get("active_graphs", 0) or 0),
                )
            )
            if done:
                break
        metrics = dict(env.manager.summary())
        metrics["runtime_step_count"] = int(getattr(env, "runtime_tick_count", metrics.get("runtime_step_count", 0)) or 0)
        bridge = getattr(env, "runtime_bridge", None)
        if bridge is not None:
            task_done_num = len(getattr(bridge, "processed_done_tasks", []) or [])
            task_fail_num = len(getattr(bridge, "processed_failed_tasks", []) or [])
            metrics["task_done_num"] = task_done_num
            metrics["task_fail_num"] = task_fail_num
            metrics["task_success_ratio"] = task_done_num / max(1, task_done_num + task_fail_num)
        if hasattr(env, "_time"):
            metrics["current_time"] = float(env._time())
        metrics.update(policy_metadata)
        metrics.update(_semantic_runtime_explainability_metrics(env))
        env.write_traces(str(run_dir))
        _write_policy_metadata(run_dir, policy_metadata, env)
        write_reward_curve(run_dir / "reward_curve.csv", reward_rows)
        _write_runtime_trace_files(run_dir, "semantic_runtime_eval", baseline, scenario, role, seed, env)
    finally:
        _close_env(air_env)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _semantic_summary_row("semantic_runtime_eval", baseline, scenario, role, seed, metrics, run_dir, error, elapsed_ms)


def _build_semantic_runtime_env(
    config: Dict[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    baseline: str,
    max_steps: int,
) -> Tuple[Any, Any]:
    env = build_offline_env(config, seed=seed, max_steps=max_steps, scenario=scenario, service_role_sweep=role)
    policy_name, cfg_updates = _baseline_settings(baseline)
    env.config = replace(env.config, **cfg_updates)
    env.rebuild_config_dependent_components()
    adapter = LASDMEnvAdapter(
        config=config,
        runtime_config=config.get("runtime", {}),
        baseline_name=baseline,
        seed=seed,
        scenario=dict(scenario),
        manager=env.manager,
        directory=env.manager.directory,
        chains=env.chains,
    )
    adapter._set_seed(seed)
    air_env = adapter._build_env(config.get("runtime", {}))
    adapter._warmup_env(air_env, int(config.get("runtime", {}).get("warmup_steps", 0)))
    adapter.apply_node_aliases_to_directory(air_env)
    adapter.bind_service_instances_to_runtime_nodes(air_env, scenario)
    adapter.bind_chains_to_runtime_nodes(air_env, env.chains, scenario)
    env.env = air_env
    env.env_adapter = adapter
    env.runtime_bridge = LASDMRuntimeBridge(env.manager, env_adapter=adapter)
    env.topology_builder = TopologyBuilder(env_adapter=adapter, directory=env.manager.directory)
    return env, air_env


def _semantic_policy_for_eval(
    config: Mapping[str, Any],
    baseline: str,
    seed: int,
    observations: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path,
) -> Any:
    if is_ippo_checkpoint_baseline(baseline):
        policy_config = _config_with_baseline_updates(config, baseline)
        baseline_checkpoint_root = _checkpoint_root_for_baseline(checkpoint_root, baseline)
        strategy = str(dict(config.get("marl", {}) or {}).get("ippo_eval_checkpoint_strategy", "exact_seed"))
        marl_cfg = dict(config.get("marl", {}) or {})
        if strategy == "global_best_validation":
            if (
                baseline == "proposed_semantic_topology_marl"
                and bool(marl_cfg.get("ippo_eval_prefer_ppo_for_proposed", False))
            ):
                min_count = int(marl_cfg.get("ippo_eval_min_proposed_ppo_selected_count", 0) or 0)
                ppo_count = _selected_checkpoint_source_count(baseline_checkpoint_root, "ppo")
                if min_count > 0 and ppo_count < min_count:
                    raise RuntimeError(
                        "proposed_semantic_topology_marl failed PPO-uplift selection: "
                        f"selected PPO checkpoints={ppo_count}, required={min_count}"
                    )
                checkpoint, summary_path = _best_validation_checkpoint(
                    baseline_checkpoint_root,
                    checkpoint_source="ppo",
                )
            else:
                checkpoint, summary_path = _best_validation_checkpoint(baseline_checkpoint_root)
        elif strategy == "exact_seed":
            checkpoint = baseline_checkpoint_root / f"ippo_seed_{seed}" / "ippo_policy.pt"
            summary_path = baseline_checkpoint_root / f"ippo_seed_{seed}" / "train_summary.json"
        else:
            raise ValueError(f"Unknown ippo_eval_checkpoint_strategy: {strategy}")
        if not summary_path.exists():
            raise RuntimeError(
                f"{baseline} requires a same-run train_summary.json; "
                f"missing {summary_path}"
            )
        summary_seed = _read_checkpoint_summary_seed(summary_path)
        if strategy == "exact_seed" and summary_seed is not None and int(summary_seed) != int(seed):
            raise RuntimeError(
                f"{baseline} checkpoint seed mismatch: "
                f"expected {seed}, found {summary_seed} in {summary_path}"
            )
        if not checkpoint.exists():
            raise RuntimeError(
                f"{baseline} requires the exact seed IPPO checkpoint; "
                f"missing {checkpoint}"
            )
        import torch

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        obs_dim = _checkpoint_observation_dim(state) or (
            max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
        )
        action_dim = _checkpoint_action_dim(state) or int(policy_config.get("marl", {}).get("max_candidates", 16))
        policy = IPPOPolicy(
            **_ippo_policy_kwargs(policy_config, obs_dim, action_dim, seed, observations=observations, state=state)
        )
        strict = _checkpoint_has_critic_body(state)
        policy.model.load_state_dict(state, strict=strict)
        policy.checkpoint_loaded = True
        policy.checkpoint_path = str(checkpoint)
        policy.checkpoint_sha256 = _sha256_file(checkpoint)
        policy.policy_source = "ippo_checkpoint"
        policy.checkpoint_train_seed = summary_seed
        policy.checkpoint_strategy = strategy
        return policy
    policy_name, _ = _baseline_settings(baseline)
    policy = policy_from_name(policy_name, seed=seed, **_semantic_eval_policy_kwargs(config))
    policy.checkpoint_loaded = False
    policy.checkpoint_path = ""
    policy.checkpoint_sha256 = ""
    policy.policy_source = "heuristic_or_baseline"
    return policy


def _semantic_eval_policy_kwargs(config: Mapping[str, Any]) -> Dict[str, float]:
    marl_cfg = dict(config.get("marl", {}) or {})
    return {
        "load_penalty": float(marl_cfg.get("topology_greedy_load_penalty", marl_cfg.get("ippo_expert_load_penalty", 0.5)) or 0.0),
        "topology_risk_penalty": float(
            marl_cfg.get("topology_greedy_topology_risk_penalty", marl_cfg.get("ippo_expert_topology_risk_penalty", 0.5)) or 0.0
        ),
        "mobility_risk_penalty": float(
            marl_cfg.get("topology_greedy_mobility_risk_penalty", marl_cfg.get("ippo_expert_mobility_risk_penalty", 0.2)) or 0.0
        ),
        "route_hops_penalty": float(
            marl_cfg.get("topology_greedy_route_hops_penalty", marl_cfg.get("ippo_expert_route_hops_penalty", 0.0)) or 0.0
        ),
        "route_tx_penalty": float(
            marl_cfg.get("topology_greedy_route_tx_penalty", marl_cfg.get("ippo_expert_route_tx_penalty", 0.0)) or 0.0
        ),
        "route_unavailable_penalty": float(
            marl_cfg.get(
                "topology_greedy_route_unavailable_penalty",
                marl_cfg.get("ippo_expert_route_unavailable_penalty", 2.0),
            )
            or 0.0
        ),
        "cold_start_penalty": float(
            marl_cfg.get("topology_greedy_cold_start_penalty", marl_cfg.get("ippo_expert_cold_start_penalty", 0.0)) or 0.0
        ),
        "deadline_violation_penalty": float(
            marl_cfg.get(
                "topology_greedy_deadline_violation_penalty",
                marl_cfg.get("ippo_expert_deadline_violation_penalty", 0.0),
            )
            or 0.0
        ),
        "runtime_penalty": float(
            marl_cfg.get("topology_greedy_runtime_penalty", marl_cfg.get("ippo_expert_runtime_penalty", 0.0)) or 0.0
        ),
        "semantic_mismatch_penalty": float(
            marl_cfg.get(
                "topology_greedy_semantic_mismatch_penalty",
                marl_cfg.get("ippo_expert_semantic_mismatch_penalty", 0.0),
            )
            or 0.0
        ),
        "utility_prior_weight": float(
            marl_cfg.get("topology_greedy_utility_prior_weight", marl_cfg.get("ippo_expert_utility_prior_weight", 2.0)) or 0.0
        ),
        "remote_penalty": float(marl_cfg.get("topology_greedy_remote_penalty", marl_cfg.get("ippo_expert_remote_penalty", 0.0)) or 0.0),
        "stale_penalty": float(marl_cfg.get("topology_greedy_stale_penalty", marl_cfg.get("ippo_expert_stale_penalty", 0.05)) or 0.0),
    }


def _runtime_training_progress_row(
    seed: int,
    episode: int,
    scenario: Mapping[str, Any],
    role: str,
    pretrain_episode: bool,
    bc_steps: int,
    step_count: int,
    total_reward: float,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    submitted = int(summary.get("submitted", 0) or 0)
    succeeded = int(summary.get("succeeded", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    timed_out = int(summary.get("timed_out", 0) or 0)
    return {
        "seed": int(seed),
        "episode": int(episode),
        "scenario": str(scenario.get("name", "")),
        "service_role_sweep": str(role),
        "pretrain_episode": bool(pretrain_episode),
        "bc_steps_per_observation": int(bc_steps),
        "step_count": int(step_count),
        "total_reward": float(total_reward),
        "submitted": submitted,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "active_graphs": int(summary.get("active_graphs", 0) or 0),
        "success_ratio": succeeded / max(1, submitted),
    }


def _evaluate_training_progress_guard(
    progress_rows: Sequence[Mapping[str, Any]],
    marl_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    if not bool(marl_cfg.get("ippo_training_progress_guard_enabled", True)):
        return {"passed": True, "reason": "training progress guard disabled", "details": {}}
    min_episodes = max(1, int(marl_cfg.get("ippo_training_guard_min_episodes", 5) or 5))
    recent_window = max(1, int(marl_cfg.get("ippo_training_guard_recent_window", 3) or 3))
    min_recent_max_success = float(marl_cfg.get("ippo_training_guard_min_recent_max_success", 0.05) or 0.0)
    policy_rows = [row for row in progress_rows if not bool(row.get("pretrain_episode", False))]
    if len(policy_rows) < min_episodes:
        return {"passed": True, "reason": "insufficient policy episodes for training guard", "details": {"policy_episodes": len(policy_rows)}}
    recent = policy_rows[-recent_window:]
    max_success = max(float(row.get("success_ratio", 0.0) or 0.0) for row in recent)
    max_submitted = max(int(row.get("submitted", 0) or 0) for row in recent)
    details = {
        "policy_episodes": len(policy_rows),
        "recent_window": len(recent),
        "recent_max_success_ratio": max_success,
        "recent_max_submitted": max_submitted,
        "threshold": min_recent_max_success,
    }
    if max_submitted <= 0:
        return {"passed": False, "reason": "no SFCs submitted in recent training episodes", "details": details}
    if max_success < min_recent_max_success:
        return {"passed": False, "reason": "training success ratio stayed near zero", "details": details}
    return {"passed": True, "reason": "training progress guard passed", "details": details}


def _runtime_ippo_episode_update(
    policy: IPPOPolicy,
    episode_steps: Sequence[PPORolloutStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    actor_loss_coef: float = 1.0,
    value_coef: float = 1.0,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    prior_l2_coef: float = 0.0,
    target_kl: float = 0.02,
    max_grad_norm: float = 0.5,
    normalize_returns: bool = True,
    return_norm_momentum: float = 0.95,
    return_norm_eps: float = 1e-6,
) -> Dict[str, float]:
    return ppo_update_policy(
        policy,
        episode_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        actor_loss_coef=actor_loss_coef,
        value_coef=value_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        max_grad_norm=max_grad_norm,
        prior_l2_coef=prior_l2_coef,
        target_kl=target_kl,
        normalize_returns=normalize_returns,
        return_norm_momentum=return_norm_momentum,
        return_norm_eps=return_norm_eps,
    )


def _ippo_bc_quality_metrics(
    policy: IPPOPolicy,
    observations: Mapping[str, Mapping[str, Any]],
    expert_actions: Mapping[str, Any],
) -> Dict[str, Any]:
    torch = policy.torch
    was_training = bool(policy.model.training)
    policy.model.eval()
    top1 = 0
    top3 = 0
    count = 0
    utility_gaps: List[float] = []
    route_available: List[float] = []
    deadline_feasible: List[float] = []
    stale_remote: List[float] = []
    high_topology_bad: List[float] = []
    selected_ranks: List[float] = []
    with torch.no_grad():
        for agent_id, observation in observations.items():
            if policy.use_region_encoder:
                obs_tensor = policy.model.region_context_tensor(observation, policy.device)
                logits = None
            else:
                obs_tensor = torch.tensor(
                    policy._fit_dim(flatten_observation(observation)),
                    dtype=torch.float32,
                    device=policy.device,
                ).unsqueeze(0)
                logits = policy.model.actor_logits(obs_tensor)
            agent_payload = expert_actions.get(str(agent_id), {}) if isinstance(expert_actions, Mapping) else {}
            if not isinstance(agent_payload, Mapping):
                continue
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for candidate_set in observation.get("candidate_sets", []) or []:
                grouped.setdefault(str(candidate_set.get("sfc_id", "")), []).append(candidate_set)
            for sfc_id, candidate_sets in grouped.items():
                current_source = str(candidate_sets[0].get("source_node_id", "")) if candidate_sets else ""
                planned_node_load: Dict[str, int] = {}
                for candidate_set in sorted(candidate_sets, key=lambda item: int(item.get("sfc_node_index", 0) or 0)):
                    contextual = policy._contextual_candidate_set(candidate_set, current_source, planned_node_load)
                    ids = list(contextual.get("candidate_ids", []) or [])
                    mask_len = min(len(ids), policy.max_candidates)
                    if mask_len <= 0:
                        continue
                    sfc_node_id = str(contextual.get("sfc_node_id", ""))
                    target_id = None
                    if isinstance(agent_payload.get(sfc_id), Mapping):
                        target_id = agent_payload.get(sfc_id, {}).get(sfc_node_id)
                    if not target_id or str(target_id) not in ids[:mask_len]:
                        continue
                    base_logits = None if logits is None else logits[0, :mask_len]
                    scores = policy._candidate_scores(obs_tensor, base_logits, contextual, mask_len)
                    scores = policy._apply_route_penalty(scores, contextual, mask_len)
                    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
                    selected_idx = int(order[0])
                    target_idx = int(ids[:mask_len].index(str(target_id)))
                    selected_id = str(ids[selected_idx])
                    selected_candidate = policy._candidate_by_id(contextual, selected_id)
                    target_candidate = policy._candidate_by_id(contextual, str(target_id))
                    raw_candidates = list(contextual.get("raw_candidates", []) or [])[:mask_len]
                    if selected_candidate is None or target_candidate is None or not raw_candidates:
                        continue
                    utility_values = [_ippo_candidate_utility(item) for item in raw_candidates]
                    best_utility = max(utility_values) if utility_values else 0.0
                    selected_utility = _ippo_candidate_utility(selected_candidate)
                    top1 += int(selected_id == str(target_id))
                    top3 += int(target_idx in order[: min(3, len(order))])
                    count += 1
                    utility_gaps.append(best_utility - selected_utility)
                    route_available.append(_ippo_route_available(selected_candidate))
                    deadline_feasible.append(
                        1.0 if _ippo_metadata_float(selected_candidate, "deadline_slack_s", 0.0) >= 0.0 else 0.0
                    )
                    stale_remote.append(_ippo_candidate_stale(selected_candidate))
                    high_topology_bad.append(
                        1.0 if _ippo_candidate_group(selected_candidate) == "semantic_high_topology_bad" else 0.0
                    )
                    selected_ranks.append(float(_ippo_rank_desc(utility_values, selected_idx)))
                    node_id = str(target_candidate.get("node_id", ""))
                    if node_id:
                        current_source = node_id
                        planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
    if was_training:
        policy.model.train()
    return {
        "candidate_choice_samples": count,
        "top1_accuracy": top1 / max(1, count),
        "top3_accuracy": top3 / max(1, count),
        "policy_utility_gap_mean": mean(utility_gaps) if utility_gaps else 0.0,
        "selected_route_available_ratio": mean(route_available) if route_available else 0.0,
        "selected_deadline_feasible_ratio": mean(deadline_feasible) if deadline_feasible else 0.0,
        "selected_stale_remote_ratio": mean(stale_remote) if stale_remote else 0.0,
        "selected_high_topology_bad_ratio": mean(high_topology_bad) if high_topology_bad else 0.0,
        "selected_candidate_rank_mean": mean(selected_ranks) if selected_ranks else 0.0,
    }


def _mean_metric_rows(rows: Sequence[Mapping[str, Any]], prefix: str = "") -> Dict[str, Any]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                values.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {f"{prefix}{key}": mean(items) for key, items in values.items() if items}


def _ippo_candidate_utility(candidate: Mapping[str, Any]) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    return float(metadata.get("utility_prior", candidate.get("utility_prior", 0.0)) or 0.0)


def _ippo_metadata_float(candidate: Mapping[str, Any], key: str, default: float) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    return float(metadata.get(key, candidate.get(key, default)) or default)


def _ippo_candidate_group(candidate: Mapping[str, Any]) -> str:
    metadata = dict(candidate.get("metadata", {}) or {})
    return str(metadata.get("semantic_group", candidate.get("semantic_group", "")) or "")


def _ippo_candidate_stale(candidate: Mapping[str, Any]) -> float:
    return 1.0 if _ippo_candidate_group(candidate) == "stale_remote_candidates" else 0.0


def _ippo_route_available(candidate: Mapping[str, Any]) -> float:
    return 1.0 if _ippo_metadata_float(candidate, "route_available", 1.0) > 0.0 else 0.0


def _ippo_rank_desc(values: Sequence[float], index: int) -> int:
    if index < 0 or index >= len(values):
        return len(values)
    ranked = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    return ranked.index(index) + 1 if index in ranked else len(values)


def _ippo_training_scenario_for_episode(
    scenarios: Sequence[Mapping[str, Any]],
    marl_cfg: Mapping[str, Any],
    episode: int,
    total_episodes: int,
) -> Mapping[str, Any]:
    if not scenarios:
        return {"name": "default"}
    if not bool(marl_cfg.get("ippo_curriculum_enabled", False)):
        return scenarios[int(episode) % len(scenarios)]
    phases = list(marl_cfg.get("ippo_curriculum", []) or [])
    if not phases:
        return scenarios[int(episode) % len(scenarios)]
    scenario_by_name = {str(item.get("name", "")): item for item in scenarios}
    cursor = 0
    normalized: List[Tuple[int, List[Mapping[str, Any]]]] = []
    remaining_fraction = 1.0
    unspecified: List[Mapping[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        names = [str(item) for item in (phase.get("scenarios", []) or [])]
        phase_scenarios = [scenario_by_name[name] for name in names if name in scenario_by_name]
        if not phase_scenarios:
            continue
        if "episodes" in phase:
            phase_episodes = max(0, int(phase.get("episodes", 0) or 0))
            normalized.append((phase_episodes, phase_scenarios))
            continue
        if "fraction" in phase:
            fraction = max(0.0, min(1.0, float(phase.get("fraction", 0.0) or 0.0)))
            remaining_fraction = max(0.0, remaining_fraction - fraction)
            normalized.append((max(1, int(round(max(1, total_episodes) * fraction))), phase_scenarios))
            continue
        unspecified.append(phase)
    if unspecified:
        share = remaining_fraction / max(1, len(unspecified))
        for phase in unspecified:
            names = [str(item) for item in (phase.get("scenarios", []) or [])]
            phase_scenarios = [scenario_by_name[name] for name in names if name in scenario_by_name]
            if phase_scenarios:
                normalized.append((max(1, int(round(max(1, total_episodes) * share))), phase_scenarios))
    for phase_episodes, phase_scenarios in normalized:
        if phase_episodes <= 0:
            continue
        if int(episode) < cursor + phase_episodes:
            return phase_scenarios[(int(episode) - cursor) % len(phase_scenarios)]
        cursor += phase_episodes
    return scenarios[int(episode) % len(scenarios)]


def _ippo_bc_scenario_for_episode(
    scenarios: Sequence[Mapping[str, Any]],
    marl_cfg: Mapping[str, Any],
    episode: int,
) -> Mapping[str, Any]:
    if not scenarios:
        return {"name": "default"}
    names = [str(item) for item in (marl_cfg.get("ippo_bc_scenarios", []) or [])]
    if not names:
        return scenarios[int(episode) % len(scenarios)]
    scenario_by_name = {str(item.get("name", "")): item for item in scenarios}
    bc_scenarios = [scenario_by_name[name] for name in names if name in scenario_by_name]
    if not bc_scenarios:
        return scenarios[int(episode) % len(scenarios)]
    return bc_scenarios[int(episode) % len(bc_scenarios)]


def _scheduled_ippo_entropy_coef(marl_cfg: Mapping[str, Any], episode: int) -> float:
    fallback = float(marl_cfg.get("ippo_entropy_coef", 0.01) or 0.01)
    if "ippo_entropy_coef_start" not in marl_cfg or "ippo_entropy_coef_end" not in marl_cfg:
        return fallback
    decay_episodes = max(1, int(marl_cfg.get("ippo_entropy_decay_episodes", 1) or 1))
    start = float(marl_cfg.get("ippo_entropy_coef_start", fallback) or fallback)
    end = float(marl_cfg.get("ippo_entropy_coef_end", fallback) or fallback)
    progress = max(0.0, min(1.0, float(episode) / float(decay_episodes)))
    return start + (end - start) * progress


def _evaluate_ippo_policy_for_selection(
    config: Dict[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    policy: IPPOPolicy,
    max_steps: int,
    baseline: str = "proposed_semantic_topology_marl",
) -> Dict[str, Any]:
    """Evaluate the in-memory policy on a live runtime copy for checkpoint selection."""

    env = None
    air_env = None
    try:
        env, air_env = _build_semantic_runtime_env(config, scenario, role, seed, baseline, max_steps)
        observations = env.reset()
        for _step in range(int(max_steps)):
            actions = policy.act(observations, deterministic=True)
            observations, _rewards, done, _info = env.step(actions)
            if done:
                break
        metrics = dict(env.manager.summary())
        done_tasks = len(getattr(env.runtime_bridge, "processed_done_tasks", set()) or set())
        failed_tasks = len(getattr(env.runtime_bridge, "processed_failed_tasks", set()) or set())
        task_success_ratio = done_tasks / max(1, done_tasks + failed_tasks)
        return {
            "success_ratio": float(metrics.get("success_ratio", 0.0) or 0.0),
            "qos_hit_ratio": float(metrics.get("qos_hit_ratio", 0.0) or 0.0),
            "task_success_ratio": float(metrics.get("task_success_ratio", task_success_ratio) or task_success_ratio),
            "avg_graph_finish_time": float(metrics.get("avg_graph_finish_time", 0.0) or 0.0),
            "timed_out": int(metrics.get("timed_out", 0) or 0),
            "failed": int(metrics.get("failed", 0) or 0),
            "succeeded": int(metrics.get("succeeded", 0) or 0),
        }
    finally:
        _close_env(air_env)


def _evaluate_ippo_policy_selection_suite(
    config: Dict[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    seeds: Sequence[int],
    policy: IPPOPolicy,
    max_steps: int,
    baseline: str = "proposed_semantic_topology_marl",
) -> Dict[str, Any]:
    """Evaluate a checkpoint candidate across training stress scenarios and seeds."""

    rows: List[Dict[str, Any]] = []
    active_roles = list(roles or ["full_hybrid"])
    for validation_seed in seeds:
        for scenario in scenarios:
            for role in active_roles:
                metrics = _evaluate_ippo_policy_for_selection(
                    config,
                    scenario,
                    role,
                    int(validation_seed),
                    policy,
                    max_steps,
                    baseline,
                )
                metrics["scenario"] = str(scenario.get("name", "default"))
                metrics["role"] = role
                metrics["validation_seed"] = int(validation_seed)
                rows.append(metrics)
    if not rows:
        return {
            "success_ratio": 0.0,
            "qos_hit_ratio": 0.0,
            "task_success_ratio": 0.0,
            "avg_graph_finish_time": 0.0,
            "timed_out": 0,
            "failed": 0,
            "succeeded": 0,
            "validation_details": [],
        }
    success_values = [float(row.get("success_ratio", 0.0) or 0.0) for row in rows]
    finish_values = [float(row.get("avg_graph_finish_time", 0.0) or 0.0) for row in rows if float(row.get("avg_graph_finish_time", 0.0) or 0.0) > 0.0]
    return {
        "success_ratio": mean(success_values),
        "min_success_ratio": min(success_values),
        "qos_hit_ratio": mean(float(row.get("qos_hit_ratio", 0.0) or 0.0) for row in rows),
        "task_success_ratio": mean(float(row.get("task_success_ratio", 0.0) or 0.0) for row in rows),
        "avg_graph_finish_time": mean(finish_values) if finish_values else 0.0,
        "timed_out": sum(int(row.get("timed_out", 0) or 0) for row in rows),
        "failed": sum(int(row.get("failed", 0) or 0) for row in rows),
        "succeeded": sum(int(row.get("succeeded", 0) or 0) for row in rows),
        "validation_case_count": len(rows),
        "validation_details": rows,
    }


def _ippo_checkpoint_selection_score(metrics: Mapping[str, Any], metric: str = "success_ratio") -> float:
    """Checkpoint selector. Formal runs use SFC completion success ratio."""

    metric_name = str(metric or "success_ratio")
    success = float(metrics.get("success_ratio", 0.0) or 0.0)
    if metric_name in {"success_ratio", "completion_rate"}:
        return success
    min_success = float(metrics.get("min_success_ratio", success) or 0.0)
    if metric_name == "min_success_ratio":
        return min_success
    if metric_name != "composite":
        raise ValueError(f"Unknown IPPO checkpoint selection metric: {metric_name}")
    qos = float(metrics.get("qos_hit_ratio", 0.0) or 0.0)
    task_success = float(metrics.get("task_success_ratio", 0.0) or 0.0)
    finish = float(metrics.get("avg_graph_finish_time", 0.0) or 0.0)
    case_count = max(1.0, float(metrics.get("validation_case_count", 1.0) or 1.0))
    timed_out = float(metrics.get("timed_out", 0.0) or 0.0) / case_count
    failed = float(metrics.get("failed", 0.0) or 0.0) / case_count
    latency_penalty = 0.01 * finish if finish > 0.0 else 0.0
    return 100.0 * min_success + 50.0 * success + 10.0 * task_success + 5.0 * qos - latency_penalty - timed_out - failed


def _ippo_robust_checkpoint_selection_score(
    rows: Sequence[Mapping[str, Any]],
    window: int = 1,
    std_penalty: float = 0.0,
) -> float:
    """Trailing-window score used to avoid selecting one-off validation spikes."""

    tail = list(rows)[-max(1, int(window)) :]
    values = [float(row.get("raw_selection_score", row.get("selection_score", 0.0)) or 0.0) for row in tail]
    if not values:
        return -float("inf")
    return mean(values) - float(std_penalty) * (pstdev(values) if len(values) > 1 else 0.0)


def _ippo_checkpoint_eligible_for_best(
    row: Mapping[str, Any],
    bc_row: Mapping[str, Any],
    marl_cfg: Mapping[str, Any],
) -> bool:
    """Gate PPO checkpoints against the BC checkpoint unless explicitly disabled.

    Per-validation-seed non-decrease is logged as a robustness diagnostic, not
    as a hard gate. A policy can improve the robust aggregate by fixing the
    worst cases while slightly moving individual validation seeds.
    """

    source = str(row.get("checkpoint_source", "ppo") or "ppo")
    if source == "bc":
        return True
    require_bc_improvement = bool(marl_cfg.get("ippo_checkpoint_selection_require_ppo_improves_bc", False))
    if not require_bc_improvement or not bc_row:
        return True
    improvement_ratio = float(marl_cfg.get("ippo_checkpoint_selection_ppo_min_score_ratio", 1.0) or 1.0)
    current_score = float(row.get("selection_score", row.get("robust_selection_score", -float("inf"))) or -float("inf"))
    bc_score = float(bc_row.get("selection_score", bc_row.get("raw_selection_score", -float("inf"))) or -float("inf"))
    return current_score >= bc_score * improvement_ratio


def _ippo_validation_seed_non_decrease_count(
    row: Mapping[str, Any],
    bc_row: Mapping[str, Any],
    tolerance: float = 0.0,
) -> int:
    current = _ippo_validation_success_by_seed(row)
    baseline = _ippo_validation_success_by_seed(bc_row)
    if not current or not baseline:
        return 0
    count = 0
    for seed, success in current.items():
        if seed in baseline and success + float(tolerance) >= baseline[seed]:
            count += 1
    return count


def _ippo_validation_success_by_seed(row: Mapping[str, Any]) -> Dict[str, float]:
    details = row.get("validation_details", []) if isinstance(row, Mapping) else []
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return {}
    grouped: Dict[str, List[float]] = {}
    for item in details:
        if not isinstance(item, Mapping):
            continue
        seed = str(item.get("validation_seed", ""))
        if not seed:
            continue
        grouped.setdefault(seed, []).append(float(item.get("success_ratio", 0.0) or 0.0))
    return {seed: mean(values) for seed, values in grouped.items() if values}


def _ippo_policy_kwargs(
    config: Mapping[str, Any],
    obs_dim: int,
    max_candidates: int,
    seed: int,
    observations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    marl_cfg = dict(config.get("marl", {}) or {})
    checkpoint_critic_dim = _checkpoint_critic_observation_dim(state or {})
    centralized_critic = bool(marl_cfg.get("ippo_centralized_critic", True))
    if state is not None and not _checkpoint_has_critic_body(state):
        centralized_critic = False
    region_agents = list(dict(config.get("topology", {}) or {}).get("region_agents", []) or [])
    observed_agents = list((observations or {}).keys())
    critic_agents = int(
        marl_cfg.get(
            "ippo_critic_agent_count",
            len(region_agents) or len(observed_agents) or 4,
        )
        or 4
    )
    critic_dim = int(checkpoint_critic_dim or (int(obs_dim) * max(1, critic_agents)))
    device = str(marl_cfg.get("ippo_device", "") or "")
    candidate_feature_dim = _checkpoint_candidate_feature_dim(state or {}) or _observation_candidate_feature_dim(observations or {}) or 31
    return {
        "observation_dim": int(obs_dim),
        "max_candidates": int(max_candidates),
        "candidate_feature_dim": int(candidate_feature_dim),
        "seed": int(seed),
        "lr": float(marl_cfg.get("ippo_lr", 3e-4) or 3e-4),
        "utility_prior_logit_weight": float(marl_cfg.get("ippo_utility_prior_logit_weight", 2.5) or 2.5),
        "route_unavailable_penalty": float(
            marl_cfg.get("ippo_route_unavailable_penalty", marl_cfg.get("ippo_expert_route_unavailable_penalty", 20.0))
            or 20.0
        ),
        "include_semantic_features": bool(marl_cfg.get("include_semantic_features", True)),
        "include_topology_features": bool(marl_cfg.get("include_topology_features", True)),
        "include_temporal_features": bool(marl_cfg.get("include_temporal_features", True)),
        "centralized_critic": centralized_critic,
        "critic_observation_dim": critic_dim if centralized_critic else int(obs_dim),
        "max_critic_agents": max(1, critic_agents),
        "use_region_encoder": bool(marl_cfg.get("ippo_use_region_encoder", True)),
        "learnable_prior": bool(marl_cfg.get("ippo_learnable_prior", True)),
        "prior_l2_coef": float(marl_cfg.get("ippo_prior_l2_coef", 1e-3) or 0.0),
        "learned_logit_scale": float(
            marl_cfg.get("ippo_learned_logit_scale_init", marl_cfg.get("ippo_learned_logit_scale", 1.0)) or 1.0
        ),
        "prior_logit_scale": float(
            marl_cfg.get("ippo_prior_logit_scale_init", marl_cfg.get("ippo_prior_logit_scale", 1.0)) or 1.0
        ),
        "learnable_logit_blend": bool(marl_cfg.get("ippo_learnable_logit_blend", False)),
        "device": device or None,
    }


def _checkpoint_observation_dim(state: Mapping[str, Any]) -> Optional[int]:
    weight = state.get("body.0.weight") if isinstance(state, Mapping) else None
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return None


def _observation_candidate_feature_dim(observations: Mapping[str, Mapping[str, Any]]) -> Optional[int]:
    for observation in observations.values():
        for candidate_set in observation.get("candidate_sets", []) or []:
            features = candidate_set.get("candidate_features")
            shape = getattr(features, "shape", None)
            if shape is not None and len(shape) == 2 and int(shape[1]) > 0:
                return int(shape[1])
    return None


def _checkpoint_candidate_feature_dim(state: Mapping[str, Any]) -> Optional[int]:
    if not isinstance(state, Mapping):
        return None
    hidden_dim = None
    body_weight = state.get("body.0.weight")
    body_shape = getattr(body_weight, "shape", None)
    if body_shape is not None and len(body_shape) >= 1:
        hidden_dim = int(body_shape[0])
    for key in ("region_candidate_actor.0.weight", "candidate_actor.0.weight"):
        weight = state.get(key)
        shape = getattr(weight, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        if hidden_dim is not None and int(shape[1]) > hidden_dim:
            return int(shape[1]) - hidden_dim
    return None


def _checkpoint_action_dim(state: Mapping[str, Any]) -> Optional[int]:
    weight = state.get("actor.weight") if isinstance(state, Mapping) else None
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return None


def _checkpoint_critic_observation_dim(state: Mapping[str, Any]) -> Optional[int]:
    weight = state.get("critic_body.0.weight") if isinstance(state, Mapping) else None
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return None


def _checkpoint_has_critic_body(state: Mapping[str, Any]) -> bool:
    return isinstance(state, Mapping) and "critic_body.0.weight" in state


def _read_checkpoint_summary_seed(path: Path) -> Optional[int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "seed" in payload:
            return int(payload["seed"])
    except Exception:
        return None
    return None


def _best_validation_checkpoint(checkpoint_root: Path, checkpoint_source: Optional[str] = None) -> Tuple[Path, Path]:
    candidates: List[Tuple[float, Path, Path]] = []
    for summary_path in sorted(checkpoint_root.glob("ippo_seed_*/train_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = dict(payload.get("best_selection", {}) or {})
        if checkpoint_source and str(best.get("checkpoint_source", "") or "") != str(checkpoint_source):
            continue
        checkpoint_path = Path(str(payload.get("checkpoint_path") or summary_path.with_name("ippo_policy.pt")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            checkpoint_path = summary_path.with_name("ippo_policy.pt")
        if not checkpoint_path.exists():
            continue
        score = payload.get("best_selection_score", None)
        try:
            score_value = float(score)
        except Exception:
            score_value = _ippo_checkpoint_selection_score(best) if best else -float("inf")
        candidates.append((score_value, checkpoint_path, summary_path))
    if not candidates:
        suffix = f" with checkpoint_source={checkpoint_source}" if checkpoint_source else ""
        raise RuntimeError(f"No valid IPPO checkpoints found under {checkpoint_root}{suffix}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _selected_checkpoint_source_count(checkpoint_root: Path, checkpoint_source: str) -> int:
    count = 0
    for summary_path in sorted(checkpoint_root.glob("ippo_seed_*/train_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = dict(payload.get("best_selection", {}) or {})
        if str(best.get("checkpoint_source", "") or "") == str(checkpoint_source):
            count += 1
    return count


def _write_runtime_trace_files(
    run_dir: Path,
    family: str,
    baseline: str,
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    env: Any,
) -> None:
    function_rows = []
    for item in env.runtime_bridge.function_execution_trace:
        row = {field: item.get(field, "") for field in FUNCTION_TRACE_FIELDS}
        row.update(
            {
                "family": family,
                "baseline": baseline,
                "seed": seed,
                "scenario": scenario.get("name", ""),
                "service_role_sweep": role,
            }
        )
        function_rows.append(row)
    _write_csv(run_dir / "function_execution_trace.csv", FUNCTION_TRACE_FIELDS, function_rows)
    _write_json(run_dir / "failure_trace.json", env.runtime_bridge.failure_trace)
    _write_csv_dynamic(run_dir / "runtime_task_lifecycle_trace.csv", getattr(env.runtime_bridge, "runtime_task_lifecycle_trace", []))


def _write_policy_metadata(run_dir: Path, policy_metadata: Mapping[str, Any], env: Any) -> None:
    payload = dict(policy_metadata)
    payload["transition_trace_policy_metadata_injected"] = False
    transitions = getattr(env, "transition_trace", None)
    if isinstance(transitions, list) and transitions:
        info = transitions[0].setdefault("info", {})
        if isinstance(info, dict):
            info["policy_metadata"] = dict(policy_metadata)
            payload["transition_trace_policy_metadata_injected"] = True
            _write_jsonl(run_dir / "marl_transition_trace.jsonl", transitions)
    _write_json(run_dir / "policy_metadata.json", payload)


def _semantic_runtime_explainability_metrics(env: Any) -> Dict[str, Any]:
    function_rows = list(getattr(env.runtime_bridge, "function_execution_trace", []) or [])
    lifecycle_rows = list(getattr(env.runtime_bridge, "runtime_task_lifecycle_trace", []) or [])
    selected_nodes = [str(row.get("selected_node", "")) for row in function_rows if row.get("selected_node")]
    selected_node_types: List[str] = []
    selected_candidates: List[Mapping[str, Any]] = []
    for transition in getattr(env, "transition_trace", []) or []:
        info = dict(transition.get("info", {}) or {})
        for decision in info.get("decisions", []) or []:
            diagnostics = dict(decision.get("diagnostics", {}) or {})
            for candidate in dict(diagnostics.get("selected_candidates", {}) or {}).values():
                if isinstance(candidate, Mapping):
                    selected_candidates.append(candidate)
                    if candidate.get("node_type"):
                        selected_node_types.append(str(candidate.get("node_type")))
    if not selected_node_types and getattr(env, "env", None) is not None:
        for node_id in selected_nodes:
            try:
                selected_node_types.append(str(env.env_adapter.node_type(env.env, node_id)))
            except Exception:
                selected_node_types.append("")

    candidate_rows = list(getattr(env.discovery_protocol, "discovery_trace", []) or [])
    candidate_total = sum(int(getattr(row, "candidate_count", 0) or 0) for row in candidate_rows)
    remote_total = sum(int(getattr(row, "remote_candidate_count", 0) or 0) for row in candidate_rows)
    semantic_scores = [_float_metric(candidate.get("semantic_score")) for candidate in selected_candidates]
    semantic_scores = [value for value in semantic_scores if value is not None]
    topology_risks = [
        _float_metric(dict(candidate.get("metadata", {}) or {}).get("topology_risk"))
        for candidate in selected_candidates
    ]
    topology_risks = [value for value in topology_risks if value is not None]
    stale_selected = [
        1.0
        for candidate in selected_candidates
        if float(candidate.get("staleness_s", 0.0) or 0.0) > 0.0
        or str(dict(candidate.get("metadata", {}) or {}).get("semantic_group", "")) == "stale_remote_candidates"
    ]
    phase_durations = _task_phase_durations(lifecycle_rows)
    encoder_manifest = {}
    if hasattr(env, "_semantic_encoder_manifest"):
        encoder_manifest = env._semantic_encoder_manifest()
    return {
        "selected_node_sequence": "->".join(selected_nodes[:48]),
        "selected_node_type_sequence": "->".join(selected_node_types[:48]),
        "remote_candidate_ratio": remote_total / max(1, candidate_total),
        "stale_selected_ratio": len(stale_selected) / max(1, len(selected_candidates)),
        "semantic_score_mean": mean(semantic_scores) if semantic_scores else 0.0,
        "semantic_score_min": min(semantic_scores) if semantic_scores else 0.0,
        "topology_risk_mean": mean(topology_risks) if topology_risks else 0.0,
        "queue_wait_time_mean": mean(phase_durations["queue"]) if phase_durations["queue"] else 0.0,
        "wireless_tx_time_mean": mean(phase_durations["tx"]) if phase_durations["tx"] else 0.0,
        "compute_time_mean": mean(phase_durations["compute"]) if phase_durations["compute"] else 0.0,
        "return_time_mean": mean(phase_durations["return"]) if phase_durations["return"] else 0.0,
        "semantic_encoder_backend": str(encoder_manifest.get("active_backend", "")),
        "semantic_encoder_model": str(encoder_manifest.get("model_name", "")),
    }


def _task_phase_durations(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    by_task: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        state = str(row.get("queue_state", row.get("substate", "")))
        if not task_id or not state:
            continue
        time_s = _float_metric(row.get("time_s"))
        if time_s is None:
            continue
        by_task.setdefault(task_id, {}).setdefault(state, []).append(time_s)
    durations = {"queue": [], "tx": [], "compute": [], "return": []}
    for states in by_task.values():
        waiting = min(states.get("waiting_to_offload", []) or [None])
        offloading = min(states.get("offloading", []) or [None])
        computing = min(states.get("computing", []) or [None])
        returning = min(states.get("returning", []) or states.get("waiting_to_return", []) or [None])
        done = min(states.get("done", []) or [None])
        if waiting is not None and offloading is not None:
            durations["queue"].append(max(0.0, float(offloading) - float(waiting)))
        if offloading is not None and computing is not None:
            durations["tx"].append(max(0.0, float(computing) - float(offloading)))
        if computing is not None and (returning is not None or done is not None):
            end = returning if returning is not None else done
            durations["compute"].append(max(0.0, float(end) - float(computing)))
        if returning is not None and done is not None:
            durations["return"].append(max(0.0, float(done) - float(returning)))
    return durations


def _float_metric(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _semantic_summary_row(
    family: str,
    baseline: str,
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    metrics: Mapping[str, Any],
    run_dir: Path,
    error: Optional[str],
    elapsed_ms: float,
) -> Dict[str, Any]:
    overhead = dict(metrics.get("runtime_overhead", {}) or {})
    return {
        "family": family,
        "baseline": baseline,
        "scenario": scenario.get("name", ""),
        "service_role_sweep": role,
        "seed": seed,
        "status": "error" if error else _status_from_metrics(metrics),
        "completed": not error,
        "error": error or "",
        "submitted": int(metrics.get("submitted", 0) or 0),
        "succeeded": int(metrics.get("succeeded", 0) or 0),
        "failed": int(metrics.get("failed", 0) or 0),
        "timed_out": int(metrics.get("timed_out", 0) or 0),
        "success_ratio": float(metrics.get("success_ratio", 0.0) or 0.0),
        "qos_hit_ratio": float(metrics.get("qos_hit_ratio", 0.0) or 0.0),
        "avg_graph_finish_time": float(metrics.get("avg_graph_finish_time", metrics.get("avg_latency_s", 0.0)) or 0.0),
        "p95_graph_finish_time": float(metrics.get("p95_graph_finish_time", 0.0) or 0.0),
        "task_done_num": int(metrics.get("task_done_num", metrics.get("succeeded", 0)) or 0),
        "task_fail_num": int(metrics.get("task_fail_num", metrics.get("failed", 0) + metrics.get("timed_out", 0)) or 0),
        "task_success_ratio": float(metrics.get("task_success_ratio", metrics.get("success_ratio", 0.0)) or 0.0),
        "runtime_step_count": int(metrics.get("runtime_step_count", 0) or 0),
        "simulation_time_end": float(metrics.get("current_time", metrics.get("simulation_time_end", 0.0)) or 0.0),
        "avg_decision_time_ms": float(overhead.get("avg_decision_time_ms", elapsed_ms) or 0.0),
        "p95_decision_time_ms": float(overhead.get("p95_decision_time_ms", elapsed_ms) or 0.0),
        "selected_node_sequence": metrics.get("selected_node_sequence", ""),
        "selected_node_type_sequence": metrics.get("selected_node_type_sequence", ""),
        "remote_candidate_ratio": float(metrics.get("remote_candidate_ratio", 0.0) or 0.0),
        "stale_selected_ratio": float(metrics.get("stale_selected_ratio", 0.0) or 0.0),
        "semantic_score_mean": float(metrics.get("semantic_score_mean", 0.0) or 0.0),
        "semantic_score_min": float(metrics.get("semantic_score_min", 0.0) or 0.0),
        "topology_risk_mean": float(metrics.get("topology_risk_mean", 0.0) or 0.0),
        "queue_wait_time_mean": float(metrics.get("queue_wait_time_mean", 0.0) or 0.0),
        "wireless_tx_time_mean": float(metrics.get("wireless_tx_time_mean", 0.0) or 0.0),
        "compute_time_mean": float(metrics.get("compute_time_mean", 0.0) or 0.0),
        "return_time_mean": float(metrics.get("return_time_mean", 0.0) or 0.0),
        "policy_name": str(metrics.get("policy_name", "")),
        "policy_source": str(metrics.get("policy_source", "")),
        "semantic_encoder_backend": str(metrics.get("semantic_encoder_backend", "")),
        "semantic_encoder_model": str(metrics.get("semantic_encoder_model", "")),
        "checkpoint_loaded": bool(metrics.get("checkpoint_loaded", False)),
        "checkpoint_path": str(metrics.get("checkpoint_path", "")),
        "checkpoint_sha256": str(metrics.get("checkpoint_sha256", "")),
        "run_dir": str(run_dir),
    }


def _write_combined_runtime_outputs(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    function_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    semantic_detail_rows: List[Dict[str, Any]] = []
    exchange_rows: List[Dict[str, Any]] = []
    lifecycle_rows: List[Dict[str, Any]] = []
    overhead_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []
    topology_rows: List[str] = []
    transition_rows: List[str] = []

    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        metadata = {
            "family": row.get("family", ""),
            "baseline": row.get("baseline", ""),
            "scenario": row.get("scenario", ""),
            "service_role_sweep": row.get("service_role_sweep", ""),
            "seed": row.get("seed", ""),
        }
        function_rows.extend(_read_csv_with_metadata(run_dir / "function_execution_trace.csv", metadata))
        failure = _read_json(run_dir / "failure_trace.json", default=[])
        if isinstance(failure, list):
            for item in failure:
                if isinstance(item, dict):
                    failure_rows.append({**metadata, **item})
        semantic_rows.extend(_read_csv_with_metadata(run_dir / "semantic_candidate_trace.csv", metadata))
        semantic_detail_rows.extend(_read_csv_with_metadata(run_dir / "semantic_candidate_detail_trace.csv", metadata))
        exchange_rows.extend(_read_csv_with_metadata(run_dir / "distributed_exchange_trace.csv", metadata))
        lifecycle_rows.extend(_read_csv_with_metadata(run_dir / "runtime_task_lifecycle_trace.csv", metadata))
        overhead_rows.append(
            {
                **metadata,
                "avg_decision_time_ms": row.get("avg_decision_time_ms", ""),
                "p95_decision_time_ms": row.get("p95_decision_time_ms", ""),
                "runtime_step_count": row.get("runtime_step_count", ""),
            }
        )
        resource_rows.extend(_resource_rows_from_transition(run_dir / "marl_transition_trace.jsonl", metadata))
        topology_rows.extend(_read_jsonl_with_metadata(run_dir / "topology_trace.jsonl", metadata))
        transition_rows.extend(_read_jsonl_with_metadata(run_dir / "marl_transition_trace.jsonl", metadata))

    _write_csv(output_root / "function_execution_trace.csv", FUNCTION_TRACE_FIELDS, function_rows)
    _write_json(output_root / "failure_trace.json", failure_rows)
    _write_csv_dynamic(output_root / "semantic_candidate_trace.csv", semantic_rows)
    _write_csv_dynamic(output_root / "semantic_candidate_detail_trace.csv", semantic_detail_rows)
    _write_csv_dynamic(output_root / "distributed_exchange_trace.csv", exchange_rows)
    _write_csv_dynamic(output_root / "runtime_task_lifecycle_trace.csv", lifecycle_rows)
    _write_csv_dynamic(output_root / "message_overhead.csv", exchange_rows)
    _write_csv_dynamic(output_root / "runtime_overhead.csv", overhead_rows)
    _write_csv_dynamic(output_root / "resource_usage_timeseries.csv", resource_rows)
    (output_root / "topology_trace.jsonl").write_text("".join(topology_rows), encoding="utf-8")
    (output_root / "marl_transition_trace.jsonl").write_text("".join(transition_rows), encoding="utf-8")


def _resource_rows_from_transition(path: Path, metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in _iter_jsonl(path):
        info = dict(item.get("info", {}) or {})
        snapshot = dict((info.get("runtime_report", {}) or {}).get("env_snapshot", {}) or {})
        instances = dict(snapshot.get("instances", {}) or {})
        time_s = info.get("time_s", item.get("time_s", ""))
        for instance_id, instance in instances.items():
            rows.append(
                {
                    **metadata,
                    "time_s": time_s,
                    "node_id": instance.get("node_id", ""),
                    "instance_id": instance_id,
                    "cpu_utilization": _load_ratio(instance),
                    "rb_utilization": len((info.get("runtime_report", {}) or {}).get("wireless_rb", {}) or {}),
                    "backhaul_usage_mb": "",
                    "energy_consumption": "",
                    "computing_task_count": instance.get("current_load", ""),
                }
            )
    return rows


def _load_semantic_config(config_path: str, repair_config_path: Optional[str]) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    if repair_config_path:
        repair = _load_yaml(repair_config_path)
        config = _deep_merge(config, repair)
    return config


def _expand_semantic_exchange_sweep(
    scenarios: Sequence[Mapping[str, Any]],
    ttl_values: Optional[Sequence[float]],
    radius_values: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    if not ttl_values and not radius_values:
        return [dict(item) for item in scenarios]
    ttl_list = [float(item) for item in (ttl_values or [])] or sorted(
        {float(dict(item).get("exchange_ttl_s", 6.0) or 6.0) for item in scenarios}
    )
    radius_list = [int(item) for item in (radius_values or [])] or sorted(
        {int(dict(item).get("exchange_radius_hops", 1) or 1) for item in scenarios}
    )
    expanded: List[Dict[str, Any]] = []
    for scenario in scenarios:
        base = dict(scenario)
        base_name = str(base.get("name", "scenario"))
        for ttl in ttl_list:
            for radius in radius_list:
                item = copy.deepcopy(base)
                item["base_scenario"] = base_name
                item["name"] = f"{base_name}__ttl_{_token_float(ttl)}__radius_{radius}"
                item["exchange_ttl_s"] = float(ttl)
                item["exchange_radius_hops"] = int(radius)
                expanded.append(item)
    return expanded


def _token_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in dict(overlay).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _scenario_by_name(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [])
    for scenario in scenarios:
        if str(scenario.get("name")) == str(name):
            return dict(scenario)
    raise ValueError(f"Unknown semantic runtime guard scenario: {name}")


def _semantic_runtime_trace_diagnostics(root: Path) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "candidate_trace_rows": 0,
        "transition_rows": 0,
        "topology_rows": 0,
        "exchange_trace_rows": 0,
        "accepted_decisions": 0,
        "rejected_decisions": 0,
        "runtime_reports_with_tasks": 0,
    }
    top_scores: List[float] = []
    local_candidates = 0
    remote_candidates = 0
    payload_bytes: List[float] = []
    first_time: Optional[float] = None
    last_time: Optional[float] = None

    for candidate_path in root.rglob("semantic_candidate_trace.csv"):
        for row in _read_csv(candidate_path):
            diagnostics["candidate_trace_rows"] += 1
            score = _first_numeric(row, ("score", "similarity", "semantic_score", "top_score"))
            if score is not None:
                top_scores.append(score)
            source = str(row.get("source", row.get("candidate_source", ""))).lower()
            if source == "remote" or str(row.get("is_remote", "")).lower() in {"true", "1", "yes"}:
                remote_candidates += 1
            else:
                local_candidates += 1

    for exchange_path in root.rglob("distributed_exchange_trace.csv"):
        for row in _read_csv(exchange_path):
            diagnostics["exchange_trace_rows"] += 1
            size = _first_numeric(row, ("payload_bytes", "bytes", "message_bytes"))
            if size is not None:
                payload_bytes.append(size)

    for topology_path in root.rglob("topology_trace.jsonl"):
        for item in _iter_jsonl(topology_path):
            diagnostics["topology_rows"] += 1
            item_time = _first_numeric(item, ("time_s", "time", "current_time"))
            if item_time is not None:
                first_time = item_time if first_time is None else min(first_time, item_time)
                last_time = item_time if last_time is None else max(last_time, item_time)

    for transition_path in root.rglob("marl_transition_trace.jsonl"):
        for item in _iter_jsonl(transition_path):
            diagnostics["transition_rows"] += 1
            info = dict(item.get("info", {}) or {})
            decisions = info.get("decisions", []) or []
            if isinstance(decisions, list):
                for decision in decisions:
                    if not isinstance(decision, Mapping):
                        continue
                    if decision.get("rejected_reason"):
                        diagnostics["rejected_decisions"] += 1
                    else:
                        diagnostics["accepted_decisions"] += 1
            if "task" in json.dumps(info.get("runtime_report", {}), ensure_ascii=False).lower():
                diagnostics["runtime_reports_with_tasks"] += 1
            item_time = _first_numeric(info, ("time_s", "time", "current_time"))
            if item_time is not None:
                first_time = item_time if first_time is None else min(first_time, item_time)
                last_time = item_time if last_time is None else max(last_time, item_time)

    diagnostics["mean_local_candidates"] = float(local_candidates) / max(1, int(diagnostics["candidate_trace_rows"]))
    diagnostics["mean_remote_candidates"] = float(remote_candidates) / max(1, int(diagnostics["candidate_trace_rows"]))
    diagnostics["mean_top_score"] = mean(top_scores) if top_scores else 0.0
    diagnostics["payload_bytes"] = sum(payload_bytes)
    diagnostics["first_time_s"] = first_time if first_time is not None else 0.0
    diagnostics["last_time_s"] = last_time if last_time is not None else 0.0
    return diagnostics


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _first_numeric(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        try:
            value = row.get(key)
            if value in ("", None):
                continue
            return float(value)
        except Exception:
            continue
    return None


def _write_aggregate_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["baseline"]), str(row["scenario"])), []).append(row)
    out_rows: List[Dict[str, Any]] = []
    for (family, baseline, scenario), group in sorted(grouped.items()):
        record: Dict[str, Any] = {
            "family": family,
            "baseline": baseline,
            "scenario": scenario,
            "run_count": len(group),
            "completed_count": sum(1 for item in group if _truthy(item.get("completed"))),
            "error_count": sum(1 for item in group if item.get("error")),
        }
        for field in (
            "success_ratio",
            "qos_hit_ratio",
            "avg_graph_finish_time",
            "p95_graph_finish_time",
            "task_success_ratio",
            "avg_decision_time_ms",
            "remote_candidate_ratio",
            "stale_selected_ratio",
            "semantic_score_mean",
            "semantic_score_min",
            "topology_risk_mean",
            "queue_wait_time_mean",
            "wireless_tx_time_mean",
            "compute_time_mean",
            "return_time_mean",
        ):
            values = [float(item.get(field, 0.0) or 0.0) for item in group]
            record[f"{field}_mean"] = mean(values) if values else 0.0
            record[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out_rows.append(record)
    _write_csv_dynamic(path, out_rows)


def _status_from_metrics(metrics: Mapping[str, Any]) -> str:
    if int(metrics.get("timed_out", 0) or 0):
        return "timed_out"
    if int(metrics.get("failed", 0) or 0):
        return "failed"
    if int(metrics.get("submitted", 0) or 0) and int(metrics.get("succeeded", 0) or 0) >= int(metrics.get("submitted", 0) or 0):
        return "succeeded"
    return "completed"


def _close_env(env: Any) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def _load_ratio(instance: Mapping[str, Any]) -> float:
    return min(1.0, float(instance.get("current_load", 0) or 0) / max(1.0, float(instance.get("max_concurrency", 1) or 1)))


def _read_csv_with_metadata(path: Path, metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [{**metadata, **row} for row in csv.DictReader(file)]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_jsonl_with_metadata(path: Path, metadata: Mapping[str, Any]) -> List[str]:
    rows: List[str] = []
    for item in _iter_jsonl(path):
        rows.append(json.dumps({**metadata, **item}, ensure_ascii=False) + "\n")
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_csv_dynamic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_cell(value) for key, value in row.items()})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
