#!/usr/bin/env python3
"""Run a small learnability audit for the IPPO candidate-choice actor.

The audit separates three questions that final runtime success cannot answer:

* Is the expert policy strong on the same runtime scenario?
* Can the IPPO actor behavior-clone the expert's per-function candidate choice?
* Does BC improve candidate quality metrics before PPO is attempted?

Outputs are diagnostic CSVs only. They are not paper figure data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORKSPACE = Path(__file__).resolve().parent.parent
for path in (WORKSPACE / "AirFogSim", WORKSPACE / "methods_baselines" / "lasdm", WORKSPACE):
    item = str(path)
    if item not in sys.path:
        sys.path.insert(0, item)


from airfogsim.lasdm.graph_observation import flatten_observation
from airfogsim.lasdm.marl_policy import IPPOPolicy, policy_from_name
from run_complete_runtime_experiment import (
    _build_semantic_runtime_env,
    _close_env,
    _ippo_checkpoint_selection_score,
    _ippo_policy_kwargs,
    _load_semantic_config,
    _semantic_eval_policy_kwargs,
)


DEFAULT_CONFIG = "methods_baselines/lasdm/configs/semantic_topology_marl.yaml"
DEFAULT_REPAIR_CONFIG = "methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml"
DEFAULT_OUTPUT = "experiment_artifacts/diagnostics/ippo_learnability_audit_20260501"
DEFAULT_SCENARIOS = [
    "semantic_runtime_calibration_easy",
    "semantic_runtime_contention_stress",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit IPPO BC learnability on real semantic-runtime env rollouts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repair-config", default=DEFAULT_REPAIR_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--role", default="full_hybrid")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--bc-episodes", type=int, default=6)
    parser.add_argument("--bc-steps-per-observation", type=int, default=2)
    parser.add_argument("--bc-label-strategy", default="expert")
    parser.add_argument("--bc-stale-relabel-margin", type=float, default=0.05)
    parser.add_argument("--bc-deadline-relabel-margin", type=float, default=0.05)
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.force:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_semantic_config(args.config, args.repair_config)
    scenarios = [_scenario_by_name(config, name) for name in args.scenarios]

    bc_rows: List[Dict[str, Any]] = []
    imitation_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for scenario in scenarios:
            scenario_name = str(scenario.get("name", "scenario"))
            print(f"[audit] seed={seed} scenario={scenario_name}", flush=True)
            policy = _new_ippo_policy(config, scenario, args.role, seed, args.max_steps)
            expert = _new_expert_policy(config, seed)

            runtime_rows.append(
                _run_runtime_episode(
                    config,
                    scenario,
                    args.role,
                    seed + args.eval_seed_offset,
                    expert,
                    args.max_steps,
                    method="expert_topology_greedy",
                    train_seed=seed,
                )
            )
            runtime_rows.append(
                _run_runtime_episode(
                    config,
                    scenario,
                    args.role,
                    seed + args.eval_seed_offset,
                    policy,
                    args.max_steps,
                    method="ippo_before_bc",
                    train_seed=seed,
                )
            )
            before_metrics, before_quality = _evaluate_imitation(
                config,
                scenario,
                args.role,
                seed + args.eval_seed_offset,
                policy,
                expert,
                args.max_steps,
                stage="before_bc",
                train_seed=seed,
            )
            imitation_rows.append(before_metrics)
            quality_rows.extend(before_quality)

            bc_rows.extend(
                _behavior_clone(
                    config,
                    scenario,
                    args.role,
                    seed,
                    policy,
                    expert,
                    bc_episodes=args.bc_episodes,
                    steps_per_observation=args.bc_steps_per_observation,
                    label_strategy=args.bc_label_strategy,
                    stale_relabel_margin=args.bc_stale_relabel_margin,
                    deadline_relabel_margin=args.bc_deadline_relabel_margin,
                    max_steps=args.max_steps,
                )
            )

            runtime_rows.append(
                _run_runtime_episode(
                    config,
                    scenario,
                    args.role,
                    seed + args.eval_seed_offset,
                    policy,
                    args.max_steps,
                    method="ippo_after_bc",
                    train_seed=seed,
                )
            )
            after_metrics, after_quality = _evaluate_imitation(
                config,
                scenario,
                args.role,
                seed + args.eval_seed_offset,
                policy,
                expert,
                args.max_steps,
                stage="after_bc",
                train_seed=seed,
            )
            imitation_rows.append(after_metrics)
            quality_rows.extend(after_quality)
            summary_rows.append(_summarize_pair(before_metrics, after_metrics))

    _write_csv(output_root / "bc_training.csv", bc_rows)
    _write_csv(output_root / "bc_accuracy.csv", imitation_rows)
    _write_csv(output_root / "candidate_quality.csv", quality_rows)
    _write_csv(output_root / "runtime_scores.csv", runtime_rows)
    _write_csv(output_root / "summary.csv", summary_rows)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "seeds": list(args.seeds),
                "scenarios": [str(item.get("name", "")) for item in scenarios],
                "bc_rows": len(bc_rows),
                "candidate_quality_rows": len(quality_rows),
                "mean_top1_before": _mean_stage(imitation_rows, "before_bc", "top1_accuracy"),
                "mean_top1_after": _mean_stage(imitation_rows, "after_bc", "top1_accuracy"),
                "mean_utility_gap_before": _mean_stage(imitation_rows, "before_bc", "policy_utility_gap_mean"),
                "mean_utility_gap_after": _mean_stage(imitation_rows, "after_bc", "policy_utility_gap_mean"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[audit] wrote {output_root}", flush=True)
    return 0


def _scenario_by_name(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [])
    for scenario in scenarios:
        item = dict(scenario)
        if str(item.get("name")) == str(name):
            return item
    raise ValueError(f"Unknown scenario: {name}")


def _new_expert_policy(config: Mapping[str, Any], seed: int) -> Any:
    marl_cfg = dict(config.get("marl", {}) or {})
    name = str(marl_cfg.get("ippo_expert_policy", "topology_greedy"))
    kwargs = dict(_semantic_eval_policy_kwargs(config))
    kwargs["seed"] = int(seed)
    kwargs["branch_width"] = int(marl_cfg.get("ippo_expert_branch_width", 8) or 8)
    return policy_from_name(name, **kwargs)


def _new_ippo_policy(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    max_steps: int,
) -> IPPOPolicy:
    env = None
    air_env = None
    try:
        env, air_env = _build_semantic_runtime_env(
            dict(config),
            scenario,
            role,
            int(seed),
            "proposed_semantic_topology_marl",
            max(1, int(max_steps)),
        )
        observations = env.reset()
        obs_dim = max((len(flatten_observation(obs)) for obs in observations.values()), default=1)
        max_candidates = int(dict(config.get("marl", {}) or {}).get("max_candidates", 16) or 16)
        return IPPOPolicy(**_ippo_policy_kwargs(config, obs_dim, max_candidates, int(seed), observations=observations))
    finally:
        _close_env(air_env)


def _behavior_clone(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    policy: IPPOPolicy,
    expert: Any,
    bc_episodes: int,
    steps_per_observation: int,
    label_strategy: str,
    stale_relabel_margin: float,
    deadline_relabel_margin: float,
    max_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenario_name = str(scenario.get("name", "scenario"))
    policy.model.train()
    for episode in range(max(0, int(bc_episodes))):
        train_seed = int(seed) + episode * 37
        env = None
        air_env = None
        try:
            env, air_env = _build_semantic_runtime_env(
                dict(config),
                scenario,
                role,
                train_seed,
                "proposed_semantic_topology_marl",
                max_steps,
            )
            observations = env.reset()
            for step in range(max_steps):
                expert_actions = expert.act(observations, deterministic=True)
                samples = 0
                last_loss = 0.0
                for _ in range(max(1, int(steps_per_observation))):
                    samples = policy.supervised_update(
                        observations,
                        expert_actions,
                        label_strategy=label_strategy,
                        stale_relabel_margin=stale_relabel_margin,
                        deadline_relabel_margin=deadline_relabel_margin,
                    )
                    last_loss = float(getattr(policy, "last_supervised_loss", 0.0) or 0.0)
                    relabels = int(getattr(policy, "last_supervised_relabels", 0) or 0)
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario_name,
                        "role": role,
                        "bc_episode": episode,
                        "step": step,
                        "samples": samples,
                        "loss": last_loss,
                        "relabels": relabels,
                        "label_strategy": label_strategy,
                    }
                )
                observations, _rewards, done, _info = env.step(expert_actions)
                if done:
                    break
        finally:
            _close_env(air_env)
    return rows


def _evaluate_imitation(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    policy: IPPOPolicy,
    expert: Any,
    max_steps: int,
    stage: str,
    train_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    top1 = 0
    top3 = 0
    count = 0
    utility_gaps: List[float] = []
    expert_gaps: List[float] = []
    route_available: List[float] = []
    deadline_feasible: List[float] = []
    high_topology_bad: List[float] = []
    stale_remote: List[float] = []
    selected_ranks: List[float] = []
    env = None
    air_env = None
    try:
        env, air_env = _build_semantic_runtime_env(
            dict(config),
            scenario,
            role,
            int(seed),
            "proposed_semantic_topology_marl",
            max_steps,
        )
        observations = env.reset()
        for step in range(max_steps):
            expert_actions = expert.act(observations, deterministic=True)
            step_rows = _compare_policy_to_expert(policy, observations, expert_actions)
            for row in step_rows:
                row.update(
                    {
                        "seed": train_seed,
                        "eval_seed": seed,
                        "scenario": str(scenario.get("name", "scenario")),
                        "role": role,
                        "stage": stage,
                        "step": step,
                    }
                )
                rows.append(row)
                count += 1
                top1 += int(row["top1_match"])
                top3 += int(row["top3_match"])
                utility_gaps.append(float(row["policy_utility_gap"]))
                expert_gaps.append(float(row["expert_utility_gap"]))
                route_available.append(float(row["selected_route_available"]))
                deadline_feasible.append(float(row["selected_deadline_feasible"]))
                high_topology_bad.append(float(row["selected_high_topology_bad"]))
                stale_remote.append(float(row["selected_stale_remote"]))
                selected_ranks.append(float(row["selected_candidate_rank"]))
            observations, _rewards, done, _info = env.step(expert_actions)
            if done:
                break
    finally:
        _close_env(air_env)
    metrics = {
        "seed": train_seed,
        "eval_seed": seed,
        "scenario": str(scenario.get("name", "scenario")),
        "role": role,
        "stage": stage,
        "samples": count,
        "top1_accuracy": top1 / max(1, count),
        "top3_accuracy": top3 / max(1, count),
        "policy_utility_gap_mean": mean(utility_gaps) if utility_gaps else 0.0,
        "expert_utility_gap_mean": mean(expert_gaps) if expert_gaps else 0.0,
        "selected_route_available_ratio": mean(route_available) if route_available else 0.0,
        "selected_deadline_feasible_ratio": mean(deadline_feasible) if deadline_feasible else 0.0,
        "selected_high_topology_bad_ratio": mean(high_topology_bad) if high_topology_bad else 0.0,
        "selected_stale_remote_ratio": mean(stale_remote) if stale_remote else 0.0,
        "selected_candidate_rank_mean": mean(selected_ranks) if selected_ranks else 0.0,
    }
    return metrics, rows


def _compare_policy_to_expert(
    policy: IPPOPolicy,
    observations: Mapping[str, Mapping[str, Any]],
    expert_actions: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    torch = policy.torch
    rows: List[Dict[str, Any]] = []
    policy.model.eval()
    with torch.no_grad():
        for agent_id, observation in observations.items():
            obs_tensor, logits = _policy_observation_tensors(policy, observation)
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
                    selected_id = str(ids[selected_idx])
                    target_idx = int(ids[:mask_len].index(str(target_id)))
                    target_candidate = policy._candidate_by_id(contextual, str(target_id))
                    selected_candidate = policy._candidate_by_id(contextual, selected_id)
                    if target_candidate is None or selected_candidate is None:
                        continue
                    raw_candidates = list(contextual.get("raw_candidates", []) or [])[:mask_len]
                    utility_values = [_candidate_utility(item) for item in raw_candidates]
                    best_idx = max(range(len(utility_values)), key=lambda idx: utility_values[idx]) if utility_values else selected_idx
                    best_candidate = raw_candidates[best_idx] if raw_candidates else selected_candidate
                    best_utility = utility_values[best_idx] if utility_values else 0.0
                    selected_utility = _candidate_utility(selected_candidate)
                    expert_utility = _candidate_utility(target_candidate)
                    utility_rank = _rank_desc(utility_values, selected_idx)
                    selected_group = _candidate_group(selected_candidate)
                    expert_group = _candidate_group(target_candidate)
                    best_group = _candidate_group(best_candidate)
                    rows.append(
                        {
                            "agent_id": str(agent_id),
                            "sfc_id": str(sfc_id),
                            "sfc_node_id": sfc_node_id,
                            "selected_instance_id": selected_id,
                            "expert_instance_id": str(target_id),
                            "top1_match": int(selected_id == str(target_id)),
                            "top3_match": int(target_idx in order[: min(3, len(order))]),
                            "num_candidates": mask_len,
                            "selected_slot_index": selected_idx,
                            "expert_slot_index": target_idx,
                            "selected_candidate_rank": utility_rank,
                            "selected_matches_best": int(selected_idx == best_idx),
                            "best_instance_id": str(best_candidate.get("instance_id", "")),
                            "best_utility": best_utility,
                            "selected_utility": selected_utility,
                            "expert_utility": expert_utility,
                            "selected_minus_expert_utility": selected_utility - expert_utility,
                            "policy_utility_gap": best_utility - selected_utility,
                            "expert_utility_gap": best_utility - expert_utility,
                            "selected_route_available": _route_available(selected_candidate),
                            "expert_route_available": _route_available(target_candidate),
                            "best_route_available": _route_available(best_candidate),
                            "selected_deadline_feasible": 1.0 if _metadata_float(selected_candidate, "deadline_slack_s", 0.0) >= 0.0 else 0.0,
                            "expert_deadline_feasible": 1.0 if _metadata_float(target_candidate, "deadline_slack_s", 0.0) >= 0.0 else 0.0,
                            "best_deadline_feasible": 1.0 if _metadata_float(best_candidate, "deadline_slack_s", 0.0) >= 0.0 else 0.0,
                            "selected_deadline_slack_s": _metadata_float(selected_candidate, "deadline_slack_s", 0.0),
                            "expert_deadline_slack_s": _metadata_float(target_candidate, "deadline_slack_s", 0.0),
                            "best_deadline_slack_s": _metadata_float(best_candidate, "deadline_slack_s", 0.0),
                            "selected_expected_runtime_penalty_s": _metadata_float(
                                selected_candidate,
                                "expected_runtime_penalty_s",
                                0.0,
                            ),
                            "expert_expected_runtime_penalty_s": _metadata_float(target_candidate, "expected_runtime_penalty_s", 0.0),
                            "best_expected_runtime_penalty_s": _metadata_float(best_candidate, "expected_runtime_penalty_s", 0.0),
                            "selected_topology_risk": _metadata_float(selected_candidate, "topology_risk", 0.0),
                            "expert_topology_risk": _metadata_float(target_candidate, "topology_risk", 0.0),
                            "best_topology_risk": _metadata_float(best_candidate, "topology_risk", 0.0),
                            "selected_mobility_risk": _metadata_float(selected_candidate, "mobility_risk", 0.0),
                            "expert_mobility_risk": _metadata_float(target_candidate, "mobility_risk", 0.0),
                            "best_mobility_risk": _metadata_float(best_candidate, "mobility_risk", 0.0),
                            "selected_trust_score": _metadata_float(selected_candidate, "trust_score", 1.0),
                            "expert_trust_score": _metadata_float(target_candidate, "trust_score", 1.0),
                            "best_trust_score": _metadata_float(best_candidate, "trust_score", 1.0),
                            "selected_high_topology_bad": 1.0 if selected_group == "semantic_high_topology_bad" else 0.0,
                            "expert_high_topology_bad": 1.0 if expert_group == "semantic_high_topology_bad" else 0.0,
                            "best_high_topology_bad": 1.0 if best_group == "semantic_high_topology_bad" else 0.0,
                            "selected_stale_remote": _candidate_stale(selected_candidate),
                            "expert_stale_remote": _candidate_stale(target_candidate),
                            "best_stale_remote": _candidate_stale(best_candidate),
                            "selected_semantic_group": selected_group,
                            "expert_semantic_group": expert_group,
                            "best_semantic_group": best_group,
                            "selected_semantic_score": float(selected_candidate.get("semantic_score", 0.0) or 0.0),
                            "expert_semantic_score": float(target_candidate.get("semantic_score", 0.0) or 0.0),
                            "best_semantic_score": float(best_candidate.get("semantic_score", 0.0) or 0.0),
                        }
                    )
                    chosen = target_candidate
                    node_id = str(chosen.get("node_id", ""))
                    if node_id:
                        current_source = node_id
                        planned_node_load[node_id] = planned_node_load.get(node_id, 0) + 1
    return rows


def _policy_observation_tensors(policy: IPPOPolicy, observation: Mapping[str, Any]) -> Tuple[Any, Optional[Any]]:
    torch = policy.torch
    if policy.use_region_encoder:
        return policy.model.region_context_tensor(observation, policy.device), None
    obs_tensor = torch.tensor(
        policy._fit_dim(flatten_observation(observation)),
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)
    return obs_tensor, policy.model.actor_logits(obs_tensor)


def _run_runtime_episode(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    policy: Any,
    max_steps: int,
    method: str,
    train_seed: int,
) -> Dict[str, Any]:
    env = None
    air_env = None
    try:
        env, air_env = _build_semantic_runtime_env(
            dict(config),
            scenario,
            role,
            int(seed),
            "proposed_semantic_topology_marl",
            max_steps,
        )
        observations = env.reset()
        for _step in range(max_steps):
            actions = policy.act(observations, deterministic=True)
            observations, _rewards, done, _info = env.step(actions)
            if done:
                break
        metrics = dict(env.manager.summary())
        row = {
            "method": method,
            "seed": train_seed,
            "eval_seed": seed,
            "scenario": str(scenario.get("name", "scenario")),
            "role": role,
            "success_ratio": float(metrics.get("success_ratio", 0.0) or 0.0),
            "qos_hit_ratio": float(metrics.get("qos_hit_ratio", 0.0) or 0.0),
            "avg_graph_finish_time": float(metrics.get("avg_graph_finish_time", 0.0) or 0.0),
            "timed_out": int(metrics.get("timed_out", 0) or 0),
            "failed": int(metrics.get("failed", 0) or 0),
            "succeeded": int(metrics.get("succeeded", 0) or 0),
        }
        row["selection_score"] = _ippo_checkpoint_selection_score(row)
        return row
    finally:
        _close_env(air_env)


def _candidate_utility(candidate: Mapping[str, Any]) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    return float(metadata.get("utility_prior", candidate.get("utility_prior", 0.0)) or 0.0)


def _metadata_float(candidate: Mapping[str, Any], key: str, default: float) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    return float(metadata.get(key, candidate.get(key, default)) or default)


def _candidate_group(candidate: Mapping[str, Any]) -> str:
    metadata = dict(candidate.get("metadata", {}) or {})
    group = str(metadata.get("semantic_group", candidate.get("semantic_group", "")) or "")
    return group or "real_non_decoy"


def _candidate_stale(candidate: Mapping[str, Any]) -> float:
    return 1.0 if _candidate_group(candidate) == "stale_remote_candidates" or float(candidate.get("staleness_s", 0.0) or 0.0) > 0.0 else 0.0


def _route_available(candidate: Mapping[str, Any]) -> float:
    return 1.0 if _metadata_float(candidate, "route_available", 1.0) > 0.0 else 0.0


def _rank_desc(values: Sequence[float], index: int) -> int:
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    return int(ordered.index(index) + 1) if index in ordered else 0


def _summarize_pair(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": before.get("seed"),
        "scenario": before.get("scenario"),
        "role": before.get("role"),
        "samples_before": before.get("samples", 0),
        "samples_after": after.get("samples", 0),
        "top1_before": before.get("top1_accuracy", 0.0),
        "top1_after": after.get("top1_accuracy", 0.0),
        "top1_delta": float(after.get("top1_accuracy", 0.0) or 0.0) - float(before.get("top1_accuracy", 0.0) or 0.0),
        "top3_before": before.get("top3_accuracy", 0.0),
        "top3_after": after.get("top3_accuracy", 0.0),
        "utility_gap_before": before.get("policy_utility_gap_mean", 0.0),
        "utility_gap_after": after.get("policy_utility_gap_mean", 0.0),
        "utility_gap_delta": float(after.get("policy_utility_gap_mean", 0.0) or 0.0)
        - float(before.get("policy_utility_gap_mean", 0.0) or 0.0),
    }


def _mean_stage(rows: Iterable[Mapping[str, Any]], stage: str, key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows if str(row.get("stage", "")) == stage]
    return mean(values) if values else 0.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


if __name__ == "__main__":
    raise SystemExit(main())
