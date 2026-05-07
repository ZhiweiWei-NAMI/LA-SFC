from __future__ import annotations

import argparse
import copy
import csv
import faulthandler
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
from airfogsim.lasdm.graph_observation import BASE_CANDIDATE_FEATURE_DIM, flatten_observation
from airfogsim.lasdm.marl_policy import IPPOPolicy, MASACPolicy, policy_from_name
from airfogsim.lasdm.marl_trainer import (
    ReplayBuffer,
    RunningRewardNormalizer,
    TrainingMetrics,
    add_per_action_transitions,
    apply_terminal_credits,
    build_per_action_transitions,
    count_placement_actions,
    masac_update_policy,
    reward_config_from_env,
    write_reward_curve,
)
from airfogsim.lasdm.runtime_bridge import LASDMRuntimeBridge
from airfogsim.lasdm.topology_builder import TopologyBuilder

from evaluate_semantic_topology_marl import (
    DEFAULT_BASELINES,
    DEFAULT_EVAL_BASELINES,
    DEFAULT_TRAINED_BASELINES,
    canonical_ippo_baseline,
    checkpoint_subdir_for_baseline,
    is_ippo_checkpoint_baseline,
    _baseline_settings,
    _select_scenarios,
)
from semantic_runtime_guard import SemanticRuntimeGuardConfig, evaluate_guard, load_semantic_summary
from train_semantic_topology_marl import DEFAULT_CONFIG as DEFAULT_SEMANTIC_CONFIG
from train_semantic_topology_marl import build_offline_env, _load_yaml


DEFAULT_LASDM_CONFIG = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")
DEFAULT_OUTPUT_ROOT = os.path.join(WORKSPACE_ROOT, "experiment_artifacts", "raw_data", "complete_runtime_scheduler")
DEFAULT_RUNTIME_SEMANTIC_BASELINES = list(DEFAULT_EVAL_BASELINES)
TRAINED_CHECKPOINT_FILES = {
    "mappo_ctde": "mappo_policy.pt",
    "iql_offline": "iql_policy.pt",
}
_RUNTIME_STACK_WATCHDOG_FILES: Dict[str, Any] = {}


def _append_runtime_debug_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _arm_runtime_stack_watchdog(path)
    payload = {
        "time_s": time.time(),
        "pid": os.getpid(),
        "event": str(event),
        **fields,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
    heartbeat_path = path.with_name("runtime_heartbeat.json")
    tmp_path = heartbeat_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, heartbeat_path)


def _arm_runtime_stack_watchdog(debug_path: Path) -> None:
    timeout_s = int(os.environ.get("RUNTIME_DEBUG_STACK_TIMEOUT_S", "600") or "600")
    if timeout_s <= 0:
        return
    faulthandler.cancel_dump_traceback_later()
    stack_path = debug_path.with_name("runtime_stack_timeout.log")
    key = str(stack_path)
    handle = _RUNTIME_STACK_WATCHDOG_FILES.get(key)
    if handle is None or getattr(handle, "closed", False):
        handle = stack_path.open("a", encoding="utf-8")
        _RUNTIME_STACK_WATCHDOG_FILES[key] = handle
    faulthandler.dump_traceback_later(timeout_s, repeat=False, file=handle)

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
    "chain_progress_ratio_mean",
    "soft_completion_ratio",
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
    parser.add_argument("--semantic-baselines", nargs="+", default=DEFAULT_RUNTIME_SEMANTIC_BASELINES)
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
        debug_path = seed_dir / "runtime_debug.jsonl"
        _append_runtime_debug_event(
            debug_path,
            "seed_start",
            baseline=baseline,
            seed=int(seed),
            episodes=int(episodes),
            max_steps=int(max_steps),
        )
        policy: Optional[MASACPolicy] = None
        marl_cfg = dict(config.get("marl", {}) or {})
        policy_config = _config_with_baseline_updates(config, baseline)
        selection_interval = 0
        early_selection_interval = 0
        early_selection_until = 0
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
        selection_rows: List[Dict[str, Any]] = []
        reward_rows: List[TrainingMetrics] = []
        progress_rows: List[Dict[str, Any]] = []
        sac_diagnostic_rows: List[Dict[str, Any]] = []
        replay_buffer = ReplayBuffer(
            capacity=int(marl_cfg.get("masac_replay_capacity", 20000) or 20000),
            seed=int(seed),
        )
        reward_normalizer = (
            RunningRewardNormalizer(clip=float(marl_cfg.get("masac_reward_clip", 5.0) or 5.0))
            if bool(marl_cfg.get("masac_reward_normalization", True))
            else None
        )
        sac_update_index = 0
        sac_batch_size = int(marl_cfg.get("masac_batch_size", 128) or 128)
        sac_replay_warmup_steps = int(marl_cfg.get("masac_replay_warmup_steps", 128) or 128)
        sac_update_interval = max(1, int(marl_cfg.get("masac_update_interval", 1) or 1))
        sac_actor_update_interval = max(1, int(marl_cfg.get("masac_actor_update_interval", 2) or 2))
        sac_updates_per_env_step = int(marl_cfg.get("masac_updates_per_env_step", 1) or 1)
        sac_tau = float(marl_cfg.get("masac_tau", 0.005) or 0.005)
        sac_auto_alpha = bool(marl_cfg.get("masac_auto_alpha", False))
        sac_alpha_lr = float(marl_cfg.get("masac_alpha_lr", 3e-4) or 3e-4)
        sac_target_entropy = _float_metric(marl_cfg.get("masac_target_entropy", None))
        sac_target_entropy_scale = float(marl_cfg.get("masac_target_entropy_scale", 0.90) or 0.90)
        sac_alpha_min = float(marl_cfg.get("masac_alpha_min", 0.005) or 0.005)
        sac_alpha_max = float(marl_cfg.get("masac_alpha_max", 0.25) or 0.25)
        sac_reward_scale = float(marl_cfg.get("masac_reward_scale", 1.0) or 1.0)
        max_grad_norm = float(marl_cfg.get("masac_max_grad_norm", 10.0) or 10.0)
        sac_replay_sample_strategy = str(marl_cfg.get("masac_replay_sample_strategy", "uniform") or "uniform")
        for episode in range(int(episodes)):
            scenario = _ippo_training_scenario_for_episode(
                scenarios,
                marl_cfg,
                episode,
                max(1, int(episodes)),
            )
            role = roles[episode % len(roles)]
            _append_runtime_debug_event(
                debug_path,
                "episode_env_build_start",
                baseline=baseline,
                seed=int(seed),
                episode=int(episode),
                scenario=str(scenario.get("name", "default")),
                role=str(role),
            )
            env, air_env = _build_semantic_runtime_env(config, scenario, role, seed, baseline, max_steps)
            _append_runtime_debug_event(
                debug_path,
                "episode_env_build_end",
                baseline=baseline,
                seed=int(seed),
                episode=int(episode),
                scenario=str(scenario.get("name", "default")),
                role=str(role),
                service_chains=len(getattr(env, "chains", []) or []),
            )
            try:
                _append_runtime_debug_event(
                    debug_path,
                    "env_reset_start",
                    baseline=baseline,
                    seed=int(seed),
                    episode=int(episode),
                )
                observations = env.reset()
                _append_runtime_debug_event(
                    debug_path,
                    "env_reset_end",
                    baseline=baseline,
                    seed=int(seed),
                    episode=int(episode),
                    agent_count=len(observations),
                    candidate_set_count=sum(len(obs.get("candidate_sets", []) or []) for obs in observations.values()),
                )
                wrote_episode_traces = False
                episode_summary: Dict[str, Any] = {}
                episode_step_count = 0
                if policy is None:
                    _append_runtime_debug_event(
                        debug_path,
                        "policy_init_start",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                    )
                    obs_dim = max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
                    max_candidates = int(config.get("marl", {}).get("max_candidates", 16))
                    policy = MASACPolicy(
                        **_ippo_policy_kwargs(policy_config, obs_dim, max_candidates, seed, observations=observations),
                        q_lr=float(marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4)) or 3e-4),
                            alpha=float(marl_cfg.get("masac_alpha", 0.20) or 0.20),
                        auto_alpha=sac_auto_alpha,
                        alpha_lr=sac_alpha_lr,
                        target_entropy=sac_target_entropy,
                        target_entropy_scale=sac_target_entropy_scale,
                        alpha_min=sac_alpha_min,
                        alpha_max=sac_alpha_max,
                        tau=sac_tau,
                        q_mlp_depth=int(marl_cfg.get("masac_q_mlp_depth", 3) or 3),
                    )
                    _append_runtime_debug_event(
                        debug_path,
                        "policy_init_end",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                        obs_dim=int(obs_dim),
                        max_candidates=int(max_candidates),
                    )
                    total = 0.0
                    last_transition_by_chain: Dict[str, Any] = {}
                    step_diagnostic_totals = {
                        "env_action_count": 0.0,
                        "replay_transitions_added": 0.0,
                        "no_action_steps_skipped": 0.0,
                        "orphan_terminal_credit_count": 0.0,
                    }
                    for step in range(int(max_steps)):
                        current_observations = observations
                        _append_runtime_debug_event(
                            debug_path,
                            "step_policy_start",
                            baseline=baseline,
                            seed=int(seed),
                            episode=int(episode),
                            step=int(step),
                            replay_size=len(replay_buffer),
                        )
                        policy_step = policy.act_with_logprobs(current_observations, deterministic=False, track_grad=False)
                        actions = policy_step.actions
                        action_count = count_placement_actions(actions)
                        _append_runtime_debug_event(
                            debug_path,
                            "step_env_start",
                            baseline=baseline,
                            seed=int(seed),
                            episode=int(episode),
                            step=int(step),
                            action_count=int(action_count),
                        )
                        observations, rewards, done, info = env.step(actions)
                        mean_reward = sum(rewards.values()) / max(1, len(rewards))
                        total += mean_reward
                        transitions, replay_step_metrics = build_per_action_transitions(
                            current_observations,
                            actions,
                            observations,
                            bool(done or step + 1 >= int(max_steps)),
                            int(episode),
                            str(scenario.get("name", "default")),
                            reward_config_from_env(env),
                        )
                        add_per_action_transitions(
                            replay_buffer,
                            transitions,
                            last_transition_by_chain,
                            reward_normalizer=reward_normalizer,
                        )
                        terminal_metrics = apply_terminal_credits(
                            info.get("terminal_events", []) or [],
                            last_transition_by_chain,
                            replay_buffer,
                            reward_normalizer=reward_normalizer,
                        )
                        for metrics_source in (replay_step_metrics, terminal_metrics):
                            for key in step_diagnostic_totals:
                                step_diagnostic_totals[key] += float(metrics_source.get(key, 0.0) or 0.0)
                        replay_buffer.mark_env_step()
                        if replay_buffer.should_update(sac_replay_warmup_steps, sac_update_interval):
                            next_update_index = sac_update_index + 1
                            _append_runtime_debug_event(
                                debug_path,
                                "sac_update_start",
                                baseline=baseline,
                                seed=int(seed),
                                episode=int(episode),
                                step=int(step),
                                replay_size=len(replay_buffer),
                                replay_total_added=int(replay_buffer.total_added),
                                update_index=int(next_update_index),
                                update_actor=bool(next_update_index % sac_actor_update_interval == 0),
                            )
                            metrics = masac_update_policy(
                                policy,
                                replay_buffer,
                                batch_size=sac_batch_size,
                                updates=sac_updates_per_env_step,
                                gamma=float(marl_cfg.get("masac_gamma", 0.99) or 0.99),
                                tau=sac_tau,
                                max_grad_norm=max_grad_norm,
                                reward_scale=sac_reward_scale,
                                reward_transform=reward_normalizer.transform if reward_normalizer is not None else None,
                                sample_strategy=sac_replay_sample_strategy,
                                update_actor=bool(next_update_index % sac_actor_update_interval == 0),
                            )
                            if metrics and reward_normalizer is not None:
                                metrics = {**metrics, **reward_normalizer.snapshot()}
                            if metrics:
                                metrics = {**metrics, **step_diagnostic_totals}
                                step_diagnostic_totals = {
                                    "env_action_count": 0.0,
                                    "replay_transitions_added": 0.0,
                                    "no_action_steps_skipped": 0.0,
                                    "orphan_terminal_credit_count": 0.0,
                                }
                            _append_runtime_debug_event(
                                debug_path,
                                "sac_update_end",
                                baseline=baseline,
                                seed=int(seed),
                                episode=int(episode),
                                step=int(step),
                                replay_size=len(replay_buffer),
                                replay_total_added=int(replay_buffer.total_added),
                                update_index=int(next_update_index),
                                metric_count=len(metrics or {}),
                            )
                            if metrics:
                                sac_update_index += 1
                                sac_diagnostic_rows.append(
                                    {
                                        "episode": int(episode),
                                        "step": int(step),
                                        "seed": int(seed),
                                        "baseline": baseline,
                                        "update_index": sac_update_index,
                                        "replay_size": len(replay_buffer),
                                        "replay_total_added": int(replay_buffer.total_added),
                                        "batch_size": sac_batch_size,
                                        **metrics,
                                    }
                                )
                                _write_csv_dynamic(seed_dir / "sac_diagnostics.csv", sac_diagnostic_rows)
                    summary_dict = dict(info.get("summary", {}) or {})
                    summary_dict.update(_runtime_task_summary_from_env(env))
                    episode_summary = dict(summary_dict)
                    episode_step_count = step + 1
                    _append_runtime_debug_event(
                        debug_path,
                        "step_end",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                        step=int(step),
                        done=bool(done),
                        mean_reward=float(mean_reward),
                        total_reward=float(total),
                        succeeded=int(summary_dict.get("succeeded", 0) or 0),
                        failed=int(summary_dict.get("failed", 0) or 0),
                        timed_out=int(summary_dict.get("timed_out", 0) or 0),
                        active_graphs=int(summary_dict.get("active_graphs", 0) or 0),
                        task_done_num=int(summary_dict.get("task_done_num", 0) or 0),
                        task_fail_num=int(summary_dict.get("task_fail_num", 0) or 0),
                        task_success_ratio=float(summary_dict.get("task_success_ratio", 0.0) or 0.0),
                        next_candidate_set_count=sum(len(obs.get("candidate_sets", []) or []) for obs in observations.values()),
                    )
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
                            task_done_num=int(summary_dict.get("task_done_num", 0) or 0),
                            task_fail_num=int(summary_dict.get("task_fail_num", 0) or 0),
                            task_success_ratio=float(summary_dict.get("task_success_ratio", 0.0) or 0.0),
                            scenario=str(scenario.get("name", "default")),
                        )
                    )
                    if done:
                        break
                if episode == episodes - 1:
                    env.write_traces(str(seed_dir))
                    _write_runtime_trace_files(seed_dir, "semantic_runtime_train", baseline, scenario, role, seed, env)
                    wrote_episode_traces = True
                active_selection_interval = selection_interval
                if early_selection_interval > 0 and (episode + 1) <= max(0, early_selection_until):
                    active_selection_interval = early_selection_interval
                selection_due = (
                    active_selection_interval > 0
                    and (
                        (episode + 1) % active_selection_interval == 0
                        or (episode == episodes - 1 and int(episodes) >= active_selection_interval)
                    )
                )
                if policy is not None and selection_due:
                    _append_runtime_debug_event(
                        debug_path,
                        "checkpoint_selection_start",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                    )
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
                    _append_runtime_debug_event(
                        debug_path,
                        "checkpoint_selection_eval_end",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                        validation_success=float(validation_metrics.get("success_ratio_mean", 0.0) or 0.0),
                    )
                    raw_validation_score = _ippo_checkpoint_selection_score(validation_metrics, metric=selection_metric)
                    selection_row = {
                        "episode": episode,
                        "seed": seed,
                        "scenario": "validation_suite",
                        "role": ",".join(str(item) for item in roles),
                        "validation_seed": ",".join(str(item) for item in validation_seeds),
                        "checkpoint_source": "sac",
                        "selection_metric": selection_metric,
                        "raw_selection_score": raw_validation_score,
                        **validation_metrics,
                    }
                    selection_rows.append(selection_row)
                    if selection_mode not in {"raw", "robust"}:
                        raise ValueError(f"Unknown from-scratch MASAC checkpoint selection mode: {selection_mode}")
                    robust_selection_score = _ippo_robust_checkpoint_selection_score(
                        selection_rows,
                        window=selection_window,
                        std_penalty=selection_std_penalty,
                    )
                    selection_score = (
                        robust_selection_score
                        if selection_mode == "robust"
                        else raw_validation_score
                    )
                    selection_row.update(
                        {
                            "selection_score": selection_score,
                            "robust_selection_score": robust_selection_score,
                            "selection_mode": selection_mode,
                            "robust_window": selection_window,
                            "robust_std_penalty": selection_std_penalty,
                        }
                    )
                    selection_row["eligible_for_best_checkpoint"] = 1
                    if selection_score > best_score:
                        best_score = selection_score
                        best_state = copy.deepcopy(policy.sac_state_dict())
                        best_metadata = dict(selection_row)
                        best_candidate_path = seed_dir / f"masac_policy_best_ep{int(episode):04d}.pt"
                        try:
                            policy.torch.save(best_state, best_candidate_path)
                            selection_row["checkpoint_path"] = str(best_candidate_path)
                            selection_row["checkpoint_saved"] = 1
                            selection_row["checkpoint_sha256"] = _sha256_file(best_candidate_path)
                            selection_row["checkpoint_error"] = ""
                        except Exception as exc:
                            selection_row["checkpoint_path"] = str(best_candidate_path)
                            selection_row["checkpoint_saved"] = 0
                            selection_row["checkpoint_sha256"] = ""
                            selection_row["checkpoint_error"] = str(exc)
                    _write_csv_dynamic(seed_dir / "checkpoint_selection.csv", selection_rows)
                    _append_runtime_debug_event(
                        debug_path,
                        "checkpoint_selection_end",
                        baseline=baseline,
                        seed=int(seed),
                        episode=int(episode),
                        selection_score=float(selection_score),
                    )
                if episode == episodes - 1 and not wrote_episode_traces:
                    env.write_traces(str(seed_dir))
                    _write_runtime_trace_files(seed_dir, "semantic_runtime_train", baseline, scenario, role, seed, env)
                progress_row = _runtime_training_progress_row(
                    seed=seed,
                    episode=episode,
                    scenario=scenario,
                    role=role,
                    step_count=episode_step_count,
                    total_reward=total,
                    summary=episode_summary,
                )
                progress_rows.append(progress_row)
                write_reward_curve(seed_dir / "reward_curve.csv", reward_rows)
                _write_csv_dynamic(seed_dir / "train_progress.csv", progress_rows)
                _write_csv_dynamic(seed_dir / "sac_diagnostics.csv", sac_diagnostic_rows)
                progress_guard = _evaluate_training_progress_guard(progress_rows, marl_cfg)
                if not progress_guard.get("passed", True):
                    _write_json(seed_dir / "training_progress_guard_warning.json", progress_guard)
                _append_runtime_debug_event(
                    debug_path,
                    "episode_end",
                    baseline=baseline,
                    seed=int(seed),
                    episode=int(episode),
                    step_count=int(episode_step_count),
                    total_reward=float(total),
                    sac_update_index=int(sac_update_index),
                    succeeded=int(episode_summary.get("succeeded", 0) or 0),
                    failed=int(episode_summary.get("failed", 0) or 0),
                    timed_out=int(episode_summary.get("timed_out", 0) or 0),
                    task_done_num=int(episode_summary.get("task_done_num", 0) or 0),
                    task_fail_num=int(episode_summary.get("task_fail_num", 0) or 0),
                    task_success_ratio=float(episode_summary.get("task_success_ratio", 0.0) or 0.0),
                    active_graphs=int(episode_summary.get("active_graphs", 0) or 0),
                )
            finally:
                _append_runtime_debug_event(
                    debug_path,
                    "episode_close_env",
                    baseline=baseline,
                    seed=int(seed),
                    episode=int(episode),
                )
                _close_env(air_env)
        write_reward_curve(seed_dir / "reward_curve.csv", reward_rows)
        if selection_rows:
            _write_csv_dynamic(seed_dir / "checkpoint_selection.csv", selection_rows)
        checkpoint_path = seed_dir / "masac_policy.pt"
        checkpoint_saved = False
        checkpoint_error = ""
        if policy is not None:
            _append_runtime_debug_event(
                debug_path,
                "final_checkpoint_start",
                baseline=baseline,
                seed=int(seed),
            )
            try:
                if best_state is not None:
                    policy.load_sac_state_dict(best_state, strict=True)
                policy.torch.save(policy.sac_state_dict(), checkpoint_path)
                checkpoint_saved = checkpoint_path.exists()
            except Exception as exc:
                checkpoint_error = str(exc)
            _append_runtime_debug_event(
                debug_path,
                "final_checkpoint_end",
                baseline=baseline,
                seed=int(seed),
                checkpoint_saved=bool(checkpoint_saved),
                checkpoint_error=str(checkpoint_error),
            )
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
            "algorithm": "masac_discrete_ctde",
            "training_initialization": "from_scratch",
            "masac_centralized_critic": True,
            "masac_critic_agent_count": int(marl_cfg.get("ippo_critic_agent_count", 4) or 4),
            "masac_replay_size": len(replay_buffer),
            "masac_batch_size": sac_batch_size,
            "masac_update_interval": sac_update_interval,
            "masac_actor_update_interval": sac_actor_update_interval,
            "masac_alpha": float(marl_cfg.get("masac_alpha", 0.20) or 0.20),
            "masac_auto_alpha": sac_auto_alpha,
            "masac_alpha_lr": sac_alpha_lr,
            "masac_target_entropy": sac_target_entropy if sac_target_entropy is not None else "",
            "masac_target_entropy_scale": sac_target_entropy_scale,
            "masac_alpha_min": sac_alpha_min,
            "masac_alpha_max": sac_alpha_max,
            "masac_tau": sac_tau,
            "masac_replay_sample_strategy": sac_replay_sample_strategy,
            "masac_reward_normalization": bool(reward_normalizer is not None),
            "masac_reward_clip": float(marl_cfg.get("masac_reward_clip", 5.0) or 5.0),
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
        _append_runtime_debug_event(
            debug_path,
            "seed_end",
            baseline=baseline,
            seed=int(seed),
            checkpoint_saved=bool(checkpoint_saved),
            reward_rows=len(reward_rows),
        )
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
        policy = _semantic_policy_for_eval(
            config,
            baseline,
            seed,
            observations,
            checkpoint_root,
        )
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
            summary = dict(info.get("summary", {}) or {})
            summary.update(_runtime_task_summary_from_env(env))
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
                    task_done_num=int(summary.get("task_done_num", 0) or 0),
                    task_fail_num=int(summary.get("task_fail_num", 0) or 0),
                    task_success_ratio=float(summary.get("task_success_ratio", 0.0) or 0.0),
                    scenario=str(scenario.get("name", "default")),
                )
            )
            if done:
                break
        metrics = dict(env.runtime_bridge.collect_step_metrics(current_time=env._time()))
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
    canonical = canonical_ippo_baseline(baseline)
    if canonical in TRAINED_CHECKPOINT_FILES:
        return _trained_policy_for_eval(config, canonical, seed, observations, checkpoint_root)
    if is_ippo_checkpoint_baseline(baseline):
        policy_config = _config_with_baseline_updates(config, baseline)
        baseline_checkpoint_root = _checkpoint_root_for_baseline(checkpoint_root, baseline)
        strategy = str(dict(config.get("marl", {}) or {}).get("ippo_eval_checkpoint_strategy", "exact_seed"))
        marl_cfg = dict(config.get("marl", {}) or {})
        if strategy == "global_best_validation":
            if (
                baseline == "proposed_semantic_topology_marl"
                and bool(marl_cfg.get("masac_eval_prefer_sac_for_proposed", False))
            ):
                min_count = int(marl_cfg.get("masac_eval_min_proposed_sac_selected_count", 0) or 0)
                sac_count = _selected_checkpoint_source_count(baseline_checkpoint_root, "sac")
                if min_count > 0 and sac_count < min_count:
                    raise RuntimeError(
                        "proposed_semantic_topology_marl failed SAC-uplift selection: "
                        f"selected SAC checkpoints={sac_count}, required={min_count}"
                    )
                checkpoint, summary_path = _best_validation_checkpoint(
                    baseline_checkpoint_root,
                    checkpoint_source="sac",
                )
            else:
                checkpoint, summary_path = _best_validation_checkpoint(baseline_checkpoint_root)
        elif strategy == "exact_seed":
            checkpoint = baseline_checkpoint_root / f"ippo_seed_{seed}" / "masac_policy.pt"
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
                f"{baseline} requires the exact seed MASAC checkpoint; "
                f"missing {checkpoint}"
            )
        import torch

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        actor_state = state.get("actor", state) if isinstance(state, Mapping) else state
        if isinstance(actor_state, Mapping) and isinstance(state, Mapping) and "raw_candidate_feature_dim" in state:
            actor_state = dict(actor_state)
            actor_state["raw_candidate_feature_dim"] = state["raw_candidate_feature_dim"]
        obs_dim = _checkpoint_observation_dim(actor_state) or (
            max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
        )
        action_dim = _checkpoint_action_dim(actor_state) or int(policy_config.get("marl", {}).get("max_candidates", 16))
        policy = MASACPolicy(
            **_ippo_policy_kwargs(policy_config, obs_dim, action_dim, seed, observations=observations, state=actor_state),
            q_lr=float(marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4)) or 3e-4),
            alpha=float(marl_cfg.get("masac_alpha", 0.20) or 0.20),
            auto_alpha=bool(marl_cfg.get("masac_auto_alpha", False)),
            alpha_lr=float(marl_cfg.get("masac_alpha_lr", 3e-4) or 3e-4),
            target_entropy=_float_metric(marl_cfg.get("masac_target_entropy", None)),
            target_entropy_scale=float(marl_cfg.get("masac_target_entropy_scale", 0.90) or 0.90),
            alpha_min=float(marl_cfg.get("masac_alpha_min", 0.005) or 0.005),
            alpha_max=float(marl_cfg.get("masac_alpha_max", 0.25) or 0.25),
            tau=float(marl_cfg.get("masac_tau", 0.005) or 0.005),
            q_mlp_depth=int(marl_cfg.get("masac_q_mlp_depth", 3) or 3),
        )
        policy.load_sac_state_dict(state, strict=True)
        policy.checkpoint_loaded = True
        policy.checkpoint_path = str(checkpoint)
        policy.checkpoint_sha256 = _sha256_file(checkpoint)
        policy.policy_source = "masac_checkpoint"
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


def _trained_policy_for_eval(
    config: Mapping[str, Any],
    baseline: str,
    seed: int,
    observations: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path,
) -> Any:
    policy_config = _config_with_baseline_updates(config, baseline)
    marl_cfg = dict(config.get("marl", {}) or {})
    strategy = str(marl_cfg.get("ippo_eval_checkpoint_strategy", "exact_seed") or "exact_seed")
    checkpoint, summary_path = _trained_checkpoint_for_eval(checkpoint_root, baseline, seed, strategy)

    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if baseline == "mappo_ctde":
        from airfogsim.lasdm.mappo_policy import MAPPOPolicy

        actor_state = state.get("actor_critic", state) if isinstance(state, Mapping) else state
        obs_dim = _checkpoint_observation_dim(actor_state) or (
            max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
        )
        action_dim = _checkpoint_action_dim(actor_state) or int(policy_config.get("marl", {}).get("max_candidates", 16))
        policy = MAPPOPolicy(
            **_ippo_policy_kwargs(policy_config, obs_dim, action_dim, seed, observations=observations, state=actor_state),
        )
        policy.load_mappo_state_dict(state, strict=True)
        policy.policy_source = "mappo_checkpoint"
    elif baseline == "iql_offline":
        from airfogsim.lasdm.iql_policy import IQLPolicy

        actor_state = state.get("actor", state) if isinstance(state, Mapping) else state
        obs_dim = _checkpoint_observation_dim(actor_state) or (
            max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
        )
        action_dim = _checkpoint_action_dim(actor_state) or int(policy_config.get("marl", {}).get("max_candidates", 16))
        policy = IQLPolicy(
            **_ippo_policy_kwargs(policy_config, obs_dim, action_dim, seed, observations=observations, state=actor_state),
            q_lr=float(marl_cfg.get("iql_q_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
            alpha=float(marl_cfg.get("masac_alpha", 0.20) or 0.20),
            auto_alpha=bool(marl_cfg.get("masac_auto_alpha", False)),
            alpha_lr=float(marl_cfg.get("masac_alpha_lr", 3e-4) or 3e-4),
            target_entropy=_float_metric(marl_cfg.get("masac_target_entropy", None)),
            target_entropy_scale=float(marl_cfg.get("masac_target_entropy_scale", 0.90) or 0.90),
            alpha_min=float(marl_cfg.get("masac_alpha_min", 0.005) or 0.005),
            alpha_max=float(marl_cfg.get("masac_alpha_max", 0.25) or 0.25),
            tau=float(marl_cfg.get("iql_tau", marl_cfg.get("masac_tau", 0.005)) or 0.005),
            expectile=float(marl_cfg.get("iql_expectile", 0.7) or 0.7),
            beta=float(marl_cfg.get("iql_beta", 3.0) or 3.0),
            v_lr=float(marl_cfg.get("iql_v_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
        )
        policy.load_iql_state_dict(state, strict=True)
        policy.policy_source = "iql_checkpoint"
    else:
        raise ValueError(f"Unsupported trained checkpoint baseline: {baseline}")
    policy.checkpoint_loaded = True
    policy.checkpoint_path = str(checkpoint)
    policy.checkpoint_sha256 = _sha256_file(checkpoint)
    policy.checkpoint_train_seed = _read_checkpoint_summary_seed(summary_path)
    policy.checkpoint_strategy = strategy
    return policy


def _trained_checkpoint_for_eval(
    checkpoint_root: Path,
    baseline: str,
    seed: int,
    strategy: str,
) -> Tuple[Path, Path]:
    baseline_root = _checkpoint_root_for_baseline(checkpoint_root, baseline)
    filename = TRAINED_CHECKPOINT_FILES[baseline]
    if strategy == "exact_seed":
        candidates = [
            baseline_root / f"ippo_seed_{seed}" / filename,
            baseline_root / f"seed_{seed}" / filename,
            baseline_root / filename,
        ]
        for checkpoint in candidates:
            if checkpoint.exists():
                return checkpoint, checkpoint.with_name("train_summary.json")
        raise RuntimeError(f"{baseline} requires a trained checkpoint for seed {seed}; missing {candidates[0]}")
    if strategy == "global_best_validation":
        return _best_trained_checkpoint(baseline_root, filename, baseline)
    raise ValueError(f"Unknown trained checkpoint evaluation strategy: {strategy}")


def _best_trained_checkpoint(checkpoint_root: Path, filename: str, baseline: str) -> Tuple[Path, Path]:
    candidates: List[Tuple[float, Path, Path]] = []
    for summary_path in sorted(checkpoint_root.glob("**/train_summary.json")):
        checkpoint = summary_path.with_name(filename)
        if not checkpoint.exists():
            continue
        score = _trained_checkpoint_score(summary_path)
        candidates.append((score, checkpoint, summary_path))
    for checkpoint in sorted(checkpoint_root.glob(f"**/{filename}")):
        summary_path = checkpoint.with_name("train_summary.json")
        if any(existing == checkpoint for _score, existing, _summary in candidates):
            continue
        candidates.append((0.0, checkpoint, summary_path))
    if not candidates:
        raise RuntimeError(f"No valid {baseline} checkpoints found under {checkpoint_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _trained_checkpoint_score(summary_path: Path) -> float:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    for key in ("best_selection_score", "selection_score", "success_ratio"):
        try:
            return float(payload.get(key))
        except Exception:
            pass
    metrics = dict(payload.get("last_metrics", {}) or {})
    for key in ("success_ratio", "total_reward", "mean_reward"):
        try:
            return float(metrics.get(key))
        except Exception:
            pass
    return 0.0


def _semantic_eval_policy_kwargs(config: Mapping[str, Any]) -> Dict[str, float]:
    marl_cfg = dict(config.get("marl", {}) or {})
    return {
        "load_penalty": float(marl_cfg.get("topology_greedy_load_penalty", 0.5) or 0.0),
        "topology_risk_penalty": float(marl_cfg.get("topology_greedy_topology_risk_penalty", 0.5) or 0.0),
        "mobility_risk_penalty": float(marl_cfg.get("topology_greedy_mobility_risk_penalty", 0.2) or 0.0),
        "route_hops_penalty": float(marl_cfg.get("topology_greedy_route_hops_penalty", 0.0) or 0.0),
        "route_tx_penalty": float(marl_cfg.get("topology_greedy_route_tx_penalty", 0.0) or 0.0),
        "route_unavailable_penalty": float(marl_cfg.get("topology_greedy_route_unavailable_penalty", 2.0) or 0.0),
        "cold_start_penalty": float(marl_cfg.get("topology_greedy_cold_start_penalty", 0.0) or 0.0),
        "deadline_violation_penalty": float(marl_cfg.get("topology_greedy_deadline_violation_penalty", 0.0) or 0.0),
        "runtime_penalty": float(marl_cfg.get("topology_greedy_runtime_penalty", 0.0) or 0.0),
        "semantic_mismatch_penalty": float(marl_cfg.get("topology_greedy_semantic_mismatch_penalty", 0.0) or 0.0),
        "utility_prior_weight": float(marl_cfg.get("topology_greedy_utility_prior_weight", 0.0) or 0.0),
        "remote_penalty": float(marl_cfg.get("topology_greedy_remote_penalty", 0.0) or 0.0),
        "stale_penalty": float(marl_cfg.get("topology_greedy_stale_penalty", 0.05) or 0.0),
    }


def _runtime_training_progress_row(
    seed: int,
    episode: int,
    scenario: Mapping[str, Any],
    role: str,
    step_count: int,
    total_reward: float,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    submitted = int(summary.get("submitted", 0) or 0)
    succeeded = int(summary.get("succeeded", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    timed_out = int(summary.get("timed_out", 0) or 0)
    task_done_num = int(summary.get("task_done_num", 0) or 0)
    task_fail_num = int(summary.get("task_fail_num", 0) or 0)
    return {
        "seed": int(seed),
        "episode": int(episode),
        "scenario": str(scenario.get("name", "")),
        "service_role_sweep": str(role),
        "step_count": int(step_count),
        "total_reward": float(total_reward),
        "submitted": submitted,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "active_graphs": int(summary.get("active_graphs", 0) or 0),
        "success_ratio": succeeded / max(1, submitted),
        "task_done_num": task_done_num,
        "task_fail_num": task_fail_num,
        "task_success_ratio": task_done_num / max(1, task_done_num + task_fail_num),
    }


def _runtime_task_summary_from_env(env: Any) -> Dict[str, Any]:
    bridge = getattr(env, "runtime_bridge", None)
    if bridge is None:
        return {"task_done_num": 0, "task_fail_num": 0, "task_success_ratio": 0.0}
    done_tasks = len(getattr(bridge, "processed_done_tasks", set()) or set())
    failed_tasks = len(getattr(bridge, "processed_failed_tasks", set()) or set())
    return {
        "task_done_num": int(done_tasks),
        "task_fail_num": int(failed_tasks),
        "task_success_ratio": done_tasks / max(1, done_tasks + failed_tasks),
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
    policy_rows = list(progress_rows)
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


def _ippo_training_scenario_for_episode(
    scenarios: Sequence[Mapping[str, Any]],
    marl_cfg: Mapping[str, Any],
    episode: int,
    total_episodes: int,
) -> Mapping[str, Any]:
    if not scenarios:
        return {"name": "default"}
    sampling_mode = str(
        marl_cfg.get("masac_training_scenario_sampling", marl_cfg.get("ippo_training_scenario_sampling", "curriculum"))
        or "curriculum"
    )
    if sampling_mode in {"balanced", "round_robin", "cycle"}:
        return scenarios[int(episode) % len(scenarios)]
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
        metrics = dict(env.runtime_bridge.collect_step_metrics(current_time=env._time()))
        done_tasks = len(getattr(env.runtime_bridge, "processed_done_tasks", set()) or set())
        failed_tasks = len(getattr(env.runtime_bridge, "processed_failed_tasks", set()) or set())
        task_success_ratio = done_tasks / max(1, done_tasks + failed_tasks)
        chain_progress_ratio = _chain_progress_ratio_from_metrics(metrics)
        return {
            "success_ratio": float(metrics.get("success_ratio", 0.0) or 0.0),
            "qos_hit_ratio": float(metrics.get("qos_hit_ratio", 0.0) or 0.0),
            "task_success_ratio": float(metrics.get("task_success_ratio", task_success_ratio) or task_success_ratio),
            "chain_progress_ratio_mean": chain_progress_ratio,
            "soft_completion_ratio": max(float(metrics.get("success_ratio", 0.0) or 0.0), chain_progress_ratio),
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
        "chain_progress_ratio_mean": mean(float(row.get("chain_progress_ratio_mean", 0.0) or 0.0) for row in rows),
        "soft_completion_ratio": mean(float(row.get("soft_completion_ratio", 0.0) or 0.0) for row in rows),
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
    soft_completion = float(metrics.get("soft_completion_ratio", metrics.get("chain_progress_ratio_mean", 0.0)) or 0.0)
    if metric_name in {"soft_completion_ratio", "chain_progress_ratio"}:
        return soft_completion
    min_success = float(metrics.get("min_success_ratio", success) or 0.0)
    if metric_name == "min_success_ratio":
        return min_success
    if metric_name in {"success_min_blend", "success_min_robust"}:
        return 0.65 * success + 0.35 * min_success
    if metric_name != "composite":
        raise ValueError(f"Unknown MASAC checkpoint selection metric: {metric_name}")
    qos = float(metrics.get("qos_hit_ratio", 0.0) or 0.0)
    task_success = float(metrics.get("task_success_ratio", 0.0) or 0.0)
    finish = float(metrics.get("avg_graph_finish_time", 0.0) or 0.0)
    case_count = max(1.0, float(metrics.get("validation_case_count", 1.0) or 1.0))
    timed_out = float(metrics.get("timed_out", 0.0) or 0.0) / case_count
    failed = float(metrics.get("failed", 0.0) or 0.0) / case_count
    latency_penalty = 0.01 * finish if finish > 0.0 else 0.0
    return (
        100.0 * min_success
        + 50.0 * success
        + 25.0 * soft_completion
        + 10.0 * task_success
        + 5.0 * qos
        - latency_penalty
        - timed_out
        - failed
    )


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
    candidate_feature_dim = _checkpoint_candidate_feature_dim(state or {}) or _observation_candidate_feature_dim(observations or {})
    if candidate_feature_dim is None:
        candidate_feature_dim = _configured_candidate_feature_dim(config)
    return {
        "observation_dim": int(obs_dim),
        "max_candidates": int(max_candidates),
        "candidate_feature_dim": int(candidate_feature_dim),
        "seed": int(seed),
        "lr": float(marl_cfg.get("ippo_lr", 3e-4) or 3e-4),
        "utility_prior_logit_weight": _config_float(marl_cfg, "ippo_utility_prior_logit_weight", 2.5),
        "route_unavailable_penalty": float(marl_cfg.get("ippo_route_unavailable_penalty", 20.0) or 20.0),
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
        "prior_logit_scale": _config_float(
            marl_cfg,
            "ippo_prior_logit_scale_init",
            _config_float(marl_cfg, "ippo_prior_logit_scale", 1.0),
        ),
        "learnable_logit_blend": bool(marl_cfg.get("ippo_learnable_logit_blend", False)),
        "action_prior_enabled": bool(marl_cfg.get("ippo_action_prior_enabled", True)),
        "semantic_projection_dim": int(marl_cfg.get("semantic_projection_dim", 8) or 8),
        "cross_agent_attention_enabled": bool(marl_cfg.get("cross_agent_attention_enabled", True)),
        "cross_agent_attention_heads": int(marl_cfg.get("cross_agent_attention_heads", 4) or 4),
        "device": device or None,
    }


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def _checkpoint_observation_dim(state: Mapping[str, Any]) -> Optional[int]:
    weight = state.get("body.0.weight") if isinstance(state, Mapping) else None
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return None


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


def _checkpoint_candidate_feature_dim(state: Mapping[str, Any]) -> Optional[int]:
    if not isinstance(state, Mapping):
        return None
    value = state.get("raw_candidate_feature_dim")
    if value in (None, ""):
        return None
    return int(value)


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
        checkpoint_path = Path(str(payload.get("checkpoint_path") or summary_path.with_name("masac_policy.pt")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            checkpoint_path = summary_path.with_name("masac_policy.pt")
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
        raise RuntimeError(f"No valid MASAC checkpoints found under {checkpoint_root}{suffix}")
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
        or str(dict(candidate.get("metadata", {}) or {}).get("semantic_group", "")) == "stale_clone_exact"
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


def _chain_progress_ratio_from_metrics(metrics: Mapping[str, Any]) -> float:
    direct = metrics.get("chain_progress_ratio_mean")
    if direct not in (None, ""):
        return float(direct or 0.0)
    bridge = dict(metrics.get("runtime_bridge", {}) or {})
    chains = dict(bridge.get("chains", {}) or {})
    values = []
    for chain in chains.values():
        if isinstance(chain, Mapping):
            values.append(float(chain.get("chain_progress_ratio", 0.0) or 0.0))
    return mean(values) if values else 0.0


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
    success_ratio = float(metrics.get("success_ratio", 0.0) or 0.0)
    chain_progress_ratio = _chain_progress_ratio_from_metrics(metrics)
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
        "success_ratio": success_ratio,
        "qos_hit_ratio": float(metrics.get("qos_hit_ratio", 0.0) or 0.0),
        "avg_graph_finish_time": float(metrics.get("avg_graph_finish_time", metrics.get("avg_latency_s", 0.0)) or 0.0),
        "p95_graph_finish_time": float(metrics.get("p95_graph_finish_time", 0.0) or 0.0),
        "task_done_num": int(metrics.get("task_done_num", metrics.get("succeeded", 0)) or 0),
        "task_fail_num": int(metrics.get("task_fail_num", metrics.get("failed", 0) + metrics.get("timed_out", 0)) or 0),
        "task_success_ratio": float(metrics.get("task_success_ratio", metrics.get("success_ratio", 0.0)) or 0.0),
        "chain_progress_ratio_mean": chain_progress_ratio,
        "soft_completion_ratio": max(success_ratio, chain_progress_ratio),
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
            "chain_progress_ratio_mean",
            "soft_completion_ratio",
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
