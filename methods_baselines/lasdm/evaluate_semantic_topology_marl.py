from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

METHOD_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_semantic_topology_marl import DEFAULT_CONFIG, build_offline_env, _close_runtime_env, _load_yaml
from airfogsim.lasdm.marl_policy import policy_from_name
from airfogsim.lasdm.marl_trainer import HeuristicEvaluator, write_reward_curve


DEFAULT_BASELINES = [
    "intra_region_only",
    "cross_region_auction",
    "pure_semantic_greedy_no_exchange",
    "local_semantic_runtime_greedy",
    "nsga2_semantic_qos",
    "mappo_ctde",
    "iql_offline",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_no_exchange",
    "marl_no_temporal",
    "marl_no_cross_region",
    "proposed_semantic_topology_marl",
]

IPPO_BASELINE_CONFIG_UPDATES: Dict[str, Dict[str, Any]] = {
    "proposed_semantic_topology_marl": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "marl_no_semantic": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": False,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "marl_topology_no_semantic": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": False,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "marl_semantic_no_topology": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": False,
        "include_temporal_features": True,
    },
    "marl_no_topology": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": False,
        "include_temporal_features": True,
    },
    "marl_no_exchange": {
        "auto_exchange": False,
        "include_remote_candidates": False,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "marl_no_temporal": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": False,
    },
    "marl_no_cross_region": {
        "auto_exchange": True,
        "include_remote_candidates": False,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "pure_semantic_greedy_no_exchange": {
        "auto_exchange": False,
        "include_remote_candidates": False,
        "include_semantic_features": True,
        "include_topology_features": False,
        "include_temporal_features": False,
    },
    "local_semantic_runtime_greedy": {
        "auto_exchange": False,
        "include_remote_candidates": False,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "nsga2_semantic_qos": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "mappo_ctde": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
    "iql_offline": {
        "auto_exchange": True,
        "include_remote_candidates": True,
        "include_semantic_features": True,
        "include_topology_features": True,
        "include_temporal_features": True,
    },
}

IPPO_BASELINE_ALIASES: Dict[str, str] = {
    "marl_topology_no_semantic": "marl_no_semantic",
    "marl_no_topology": "marl_semantic_no_topology",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic-topology LASDM baselines/ablations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--service-role-sweeps", nargs="+", default=["full_hybrid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="experiment_artifacts/raw_data/lasdm_semantic_topology_marl/eval")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    scenarios = _select_scenarios(config, args.scenarios)

    for baseline in args.baselines:
        for scenario in scenarios:
            for role in args.service_role_sweeps:
                for seed in args.seeds:
                    env = build_offline_env(
                        config,
                        seed=seed,
                        max_steps=args.max_steps,
                        scenario=scenario,
                        service_role_sweep=role,
                        attach_runtime=True,
                        baseline=baseline,
                    )
                    try:
                        policy_name, cfg_updates = _baseline_settings(baseline)
                        env.config = replace(env.config, **cfg_updates)
                        env.rebuild_config_dependent_components()
                        policy = policy_from_name(policy_name, seed=seed)
                        rows = HeuristicEvaluator(env, policy).run(episodes=1, max_steps=args.max_steps)
                        scenario_name = str(scenario.get("name", "default"))
                        run_dir = out / f"{baseline}__{scenario_name}__{role}__seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        write_reward_curve(run_dir / "reward_curve.csv", rows)
                        env.write_traces(str(run_dir))
                        last = rows[-1].to_dict() if rows else {}
                        metrics = dict(env.manager.summary())
                        summary_rows.append(
                            {
                                "baseline": baseline,
                                "scenario": scenario_name,
                                "service_role_sweep": role,
                                "seed": seed,
                                "task_node_counts": json.dumps(scenario.get("task_nodes", {}), sort_keys=True),
                                "service_node_counts": json.dumps(scenario.get("service_nodes", {}), sort_keys=True),
                                "arrival_rate_sfc_per_s": scenario.get("arrival_rate_sfc_per_s", ""),
                                "exchange_ttl_s": scenario.get("exchange_ttl_s", ""),
                                "compressed_dim": config.get("semantic_exchange", {}).get("compressed_dim", ""),
                                **last,
                                **_flat_metrics(metrics),
                            }
                        )
                    finally:
                        _close_runtime_env(getattr(env, "env", None))

    with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = sorted({key for row in summary_rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(json.dumps({"completed": True, "rows": len(summary_rows), "output_dir": str(out)}, indent=2))


def _baseline_settings(name: str) -> Tuple[str, Dict[str, Any]]:
    """Map experimental labels to executable policy/config ablations.

    The learned proposed method is invoked through train_semantic_topology_marl.py --policy masac.
    This evaluator gives deterministic smoke-test counterparts for every method label.
    """
    if name == "semantic_greedy_no_exchange":
        raise ValueError("semantic_greedy_no_exchange was removed in V21; use pure_semantic_greedy_no_exchange or local_semantic_runtime_greedy")
    if name == "pure_semantic_greedy_no_exchange":
        return "pure_semantic_greedy_no_exchange", dict(IPPO_BASELINE_CONFIG_UPDATES[name])
    if name == "local_semantic_runtime_greedy":
        return "local_semantic_runtime_greedy", dict(IPPO_BASELINE_CONFIG_UPDATES[name])
    if name == "nsga2_semantic_qos":
        return "nsga2_semantic_qos", dict(IPPO_BASELINE_CONFIG_UPDATES[name])
    if name in {"mappo_ctde", "iql_offline"}:
        raise ValueError(f"{name} requires a trained checkpoint; use train_semantic_topology_marl.py --policy {name.split('_')[0]}")
    if name == "intra_region_only":
        return "intra_region_only", {"auto_exchange": False, "include_remote_candidates": False}
    if name == "cross_region_auction":
        return "cross_region_auction", {"auto_exchange": True, "include_remote_candidates": True}
    if name == "semantic_greedy_with_exchange":
        raise ValueError("semantic_greedy_with_exchange was removed in V21; use utility_prior_with_exchange")
    if name == "utility_prior_with_exchange":
        return "utility_prior_with_exchange", {"auto_exchange": True, "include_remote_candidates": True}
    if name == "topology_greedy":
        return "topology_greedy", {"auto_exchange": True, "include_remote_candidates": True}
    canonical = canonical_ippo_baseline(name)
    if canonical in IPPO_BASELINE_CONFIG_UPDATES:
        return "proposed_semantic_topology_marl", dict(IPPO_BASELINE_CONFIG_UPDATES[canonical])
    if name in {"centralized_planner", "centralized_oracle"}:
        # centralized_oracle is a legacy alias. This is a full-information
        # heuristic planner, not a claim of mathematical optimality.
        # all candidates are visible, no exchange delay/staleness is charged.
        return "centralized_planner", {"auto_exchange": False, "include_remote_candidates": True, "global_candidate_catalog": True, "semantic_top_k": 64}
    return name, {}


def is_ippo_checkpoint_baseline(name: str) -> bool:
    canonical = canonical_ippo_baseline(name)
    if canonical in {"mappo_ctde", "iql_offline"}:
        return False
    return canonical in IPPO_BASELINE_CONFIG_UPDATES


def checkpoint_subdir_for_baseline(name: str) -> str:
    canonical = canonical_ippo_baseline(name)
    return "" if canonical == "proposed_semantic_topology_marl" else canonical


def canonical_ippo_baseline(name: str) -> str:
    return IPPO_BASELINE_ALIASES.get(str(name), str(name))


def _select_scenarios(config: Mapping[str, Any], names: List[str] | None) -> List[Dict[str, Any]]:
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [{"name": "default"}])
    normalized = [dict(item) if isinstance(item, Mapping) else {"name": str(item)} for item in scenarios]
    if not names:
        return normalized
    requested = {str(name) for name in names}
    selected = [item for item in normalized if str(item.get("name")) in requested]
    missing = sorted(requested - {str(item.get("name")) for item in selected})
    if missing:
        raise ValueError(f"Unknown semantic-topology scenario(s): {', '.join(missing)}")
    return selected


def _flat_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"decisions"}
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key in excluded:
            continue
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


if __name__ == "__main__":
    main()
