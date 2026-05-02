#!/usr/bin/env python3
"""Decompose IPPO BC failures by environment, initialization, and BC data seed.

This is a diagnostic script. It runs real offline AirFogSim rollouts but does
not produce paper figure data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple


WORKSPACE = Path(__file__).resolve().parent.parent
for path in (WORKSPACE / "AirFogSim", WORKSPACE / "methods_baselines" / "lasdm", WORKSPACE, WORKSPACE / "scripts"):
    item = str(path)
    if item not in sys.path:
        sys.path.insert(0, item)


from airfogsim.lasdm.graph_observation import flatten_observation
from airfogsim.lasdm.marl_policy import IPPOPolicy
from audit_ippo_learnability import (
    _behavior_clone,
    _evaluate_imitation,
    _new_expert_policy,
    _run_runtime_episode,
)
from run_complete_runtime_experiment import (
    _build_semantic_runtime_env,
    _close_env,
    _config_with_baseline_updates,
    _ippo_policy_kwargs,
    _load_semantic_config,
)


DEFAULT_CONFIG = "methods_baselines/lasdm/configs/semantic_topology_marl.yaml"
DEFAULT_REPAIR_CONFIG = "methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml"
DEFAULT_OUTPUT = "experiment_artifacts/diagnostics/ippo_seed_decomposition_20260502"
DEFAULT_SCENARIOS = [
    "semantic_runtime_calibration_easy",
    "semantic_ambiguity_calibrated",
    "semantic_runtime_contention_stress",
    "semantic_runtime_mobility_staleness_stress",
    "distributed_service_discovery_calibrated",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether BC collapse is caused by env/init/BC-data seed.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repair-config", default=DEFAULT_REPAIR_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", choices=["seed4_minimal", "grid"], default="seed4_minimal")
    parser.add_argument("--env-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--bc-data-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fixed-env-seed", type=int, default=4)
    parser.add_argument("--fixed-init-seed", type=int, default=0)
    parser.add_argument("--fixed-bc-data-seed", type=int, default=0)
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--role", default="full_hybrid")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--bc-episodes", type=int, default=6)
    parser.add_argument("--bc-steps-per-observation", type=int, default=4)
    parser.add_argument("--baseline", default="proposed_semantic_topology_marl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.force:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_semantic_config(args.config, args.repair_config)
    scenarios = [_scenario_by_name(config, name) for name in args.scenarios]
    combos = _seed_combinations(args)
    marl_cfg = dict(config.get("marl", {}) or {})

    detail_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    imitation_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for group, env_seed, init_seed, bc_data_seed in combos:
        combo_rows: List[Dict[str, Any]] = []
        combo_runtime: List[Dict[str, Any]] = []
        print(
            f"[seed-audit] group={group} env_seed={env_seed} init_seed={init_seed} bc_data_seed={bc_data_seed}",
            flush=True,
        )
        for scenario in scenarios:
            policy = _new_decoupled_ippo_policy(config, scenario, args.role, env_seed, init_seed, args.max_steps, args.baseline)
            expert = _new_expert_policy(config, env_seed)
            _behavior_clone(
                config,
                scenario,
                args.role,
                bc_data_seed,
                policy,
                expert,
                bc_episodes=args.bc_episodes,
                steps_per_observation=args.bc_steps_per_observation,
                label_strategy=str(marl_cfg.get("ippo_bc_label_strategy", "expert") or "expert"),
                stale_relabel_margin=float(marl_cfg.get("ippo_bc_stale_relabel_margin", 0.05) or 0.0),
                deadline_relabel_margin=float(marl_cfg.get("ippo_bc_deadline_relabel_margin", 0.05) or 0.0),
                max_steps=args.max_steps,
            )
            eval_seed = int(env_seed) + int(args.eval_seed_offset)
            imitation, candidate_rows = _evaluate_imitation(
                config,
                scenario,
                args.role,
                eval_seed,
                policy,
                expert,
                args.max_steps,
                stage="after_bc",
                train_seed=env_seed,
            )
            runtime = _run_runtime_episode(
                config,
                scenario,
                args.role,
                eval_seed,
                policy,
                args.max_steps,
                method="ippo_after_bc",
                train_seed=env_seed,
            )
            expert_runtime = _run_runtime_episode(
                config,
                scenario,
                args.role,
                eval_seed,
                expert,
                args.max_steps,
                method="expert_reference",
                train_seed=env_seed,
            )
            common = {
                "audit_group": group,
                "env_seed": env_seed,
                "init_seed": init_seed,
                "bc_data_seed": bc_data_seed,
                "eval_seed": eval_seed,
                "scenario": str(scenario.get("name", "")),
                "role": args.role,
            }
            detail_rows.extend({**common, **row} for row in candidate_rows)
            imitation_rows.append({**common, **imitation})
            runtime_rows.append({**common, **runtime})
            runtime_rows.append({**common, **expert_runtime})
            combo_rows.append(imitation)
            combo_runtime.append(runtime)
        summary_rows.append(_summarize_combo(group, env_seed, init_seed, bc_data_seed, combo_rows, combo_runtime))
        _write_csv(output_root / "seed_decomposition_summary.csv", summary_rows)
        _write_csv(output_root / "seed_decomposition_runtime.csv", runtime_rows)
        _write_csv(output_root / "seed_decomposition_imitation.csv", imitation_rows)
        _write_csv(output_root / "seed_decomposition_candidates.csv", detail_rows)

    payload = {
        "output_root": str(output_root),
        "audit": args.audit,
        "scenarios": [str(item.get("name", "")) for item in scenarios],
        "combos": [
            {"audit_group": group, "env_seed": env_seed, "init_seed": init_seed, "bc_data_seed": bc_data_seed}
            for group, env_seed, init_seed, bc_data_seed in combos
        ],
        "summary_rows": len(summary_rows),
    }
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[seed-audit] wrote {output_root}", flush=True)
    return 0


def _seed_combinations(args: argparse.Namespace) -> List[Tuple[str, int, int, int]]:
    if args.audit == "grid":
        return [
            ("grid", int(env_seed), int(init_seed), int(bc_data_seed))
            for env_seed in args.env_seeds
            for init_seed in args.init_seeds
            for bc_data_seed in args.bc_data_seeds
        ]
    combos: List[Tuple[str, int, int, int]] = []
    combos.extend(
        ("fixed_env_vary_init", int(args.fixed_env_seed), int(init_seed), int(args.fixed_bc_data_seed))
        for init_seed in args.init_seeds
    )
    combos.extend(
        ("fixed_init_vary_env", int(env_seed), int(args.fixed_init_seed), int(args.fixed_bc_data_seed))
        for env_seed in args.env_seeds
    )
    combos.extend(
        ("fixed_env_init_vary_bc_data", int(args.fixed_env_seed), int(args.fixed_init_seed), int(bc_data_seed))
        for bc_data_seed in args.bc_data_seeds
    )
    seen = set()
    unique: List[Tuple[str, int, int, int]] = []
    for item in combos:
        key = item[1:]
        if (item[0], key) in seen:
            continue
        seen.add((item[0], key))
        unique.append(item)
    return unique


def _scenario_by_name(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [])
    for scenario in scenarios:
        item = dict(scenario)
        if str(item.get("name")) == str(name):
            return item
    raise ValueError(f"Unknown scenario: {name}")


def _new_decoupled_ippo_policy(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    env_seed: int,
    init_seed: int,
    max_steps: int,
    baseline: str,
) -> IPPOPolicy:
    env = None
    air_env = None
    try:
        env, air_env = _build_semantic_runtime_env(
            dict(config),
            scenario,
            role,
            int(env_seed),
            baseline,
            max(1, int(max_steps)),
        )
        observations = env.reset()
        obs_dim = max((len(flatten_observation(obs)) for obs in observations.values()), default=1)
        max_candidates = int(dict(config.get("marl", {}) or {}).get("max_candidates", 16) or 16)
        policy_config = _config_with_baseline_updates(config, baseline)
        return IPPOPolicy(
            **_ippo_policy_kwargs(policy_config, obs_dim, max_candidates, int(init_seed), observations=observations)
        )
    finally:
        _close_env(air_env)


def _summarize_combo(
    group: str,
    env_seed: int,
    init_seed: int,
    bc_data_seed: int,
    imitation_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "audit_group": group,
        "env_seed": env_seed,
        "init_seed": init_seed,
        "bc_data_seed": bc_data_seed,
        "scenario_count": len(imitation_rows),
        "top1_accuracy_mean": _mean(imitation_rows, "top1_accuracy"),
        "top3_accuracy_mean": _mean(imitation_rows, "top3_accuracy"),
        "policy_utility_gap_mean": _mean(imitation_rows, "policy_utility_gap_mean"),
        "selected_route_available_ratio": _mean(imitation_rows, "selected_route_available_ratio"),
        "selected_deadline_feasible_ratio": _mean(imitation_rows, "selected_deadline_feasible_ratio"),
        "selected_high_topology_bad_ratio": _mean(imitation_rows, "selected_high_topology_bad_ratio"),
        "selected_stale_remote_ratio": _mean(imitation_rows, "selected_stale_remote_ratio"),
        "runtime_selection_score_mean": _mean(runtime_rows, "selection_score"),
        "runtime_success_ratio_mean": _mean(runtime_rows, "success_ratio"),
        "runtime_min_success_ratio": min(
            (float(row.get("success_ratio", 0.0) or 0.0) for row in runtime_rows),
            default=0.0,
        ),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
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
