#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRFOGSIM_ROOT = REPO_ROOT / "AirFogSim"
TRAIN_ROOT = REPO_ROOT / "methods_baselines" / "lasdm"

for path in (str(AIRFOGSIM_ROOT), str(TRAIN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_semantic_topology_marl import DEFAULT_CONFIG, _close_runtime_env, _load_yaml, _resolve_scenario, build_offline_env

COMPATIBLE_SCORE_THRESHOLD = 0.58


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in dict(overlay).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config(config_path: str, repair_config_path: str | None) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    if repair_config_path and os.path.exists(repair_config_path):
        config = _deep_merge(config, _load_yaml(repair_config_path))
    return config


def _select_scenarios(config: Mapping[str, Any], names: Sequence[str] | None) -> List[Dict[str, Any]]:
    scenarios = list(config.get("semantic_topology_experiment", {}).get("scenarios", []) or [])
    if not scenarios:
        scenarios = list(config.get("experiment", {}).get("scenarios", []) or [{"name": "default"}])
    normalized = [dict(item) if isinstance(item, Mapping) else {"name": str(item)} for item in scenarios]
    if not names:
        return [item for item in normalized if bool(item.get("include_in_default", item.get("enabled", True)))]
    requested = {str(name) for name in names}
    selected = [item for item in normalized if str(item.get("name")) in requested]
    missing = sorted(requested - {str(item.get("name")) for item in selected})
    if missing:
        raise ValueError(f"Unknown semantic-topology scenario(s): {', '.join(missing)}")
    return selected


def _representative_service_sequence(request_type: str, semantic_matrix: Any) -> List[str]:
    request = dict(semantic_matrix.request_types.get(str(request_type), {}) or {})
    raw_sequence = request.get("representative_service_chain", request.get("service_sequence", [])) or []
    sequence = [str(item) for item in raw_sequence if str(item) in semantic_matrix.list_service_types()]
    if not sequence:
        raise ValueError(f"Request type {request_type!r} has no valid representative_service_chain entries")
    return sequence


def _compatible_candidate_count(
    source_semantic: str,
    target_instances: Sequence[Any],
    semantic_matrix: Any,
) -> int:
    return sum(1 for item in target_instances if semantic_matrix._io_score(source_semantic, str(item.input_semantic)) >= COMPATIBLE_SCORE_THRESHOLD)


def _check_chain_gates(
    env: Any,
    semantic_matrix: Any,
    scenario_name: str,
    service_role_sweep: str,
    seed: int,
    strict_io_compatibility: bool,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    io_warnings: List[Dict[str, Any]] = []
    instances_by_type = Counter()
    instance_index: Dict[str, List[Any]] = {}
    for instance in env.manager.directory.all():
        service_id = str(instance.service_id)
        instances_by_type[service_id] += 1
        instance_index.setdefault(service_id, []).append(instance)

    for chain in env.chains:
        chain_id = str(chain.sfc_id)
        context = dict(chain.context or {})
        request_type = str(context.get("request_type", "") or context.get("task_class", "") or "").strip()
        if request_type not in semantic_matrix.request_types:
            request_type = str(semantic_matrix.request_type_for_context(context, default=""))
        if request_type not in semantic_matrix.request_types:
            failures.append(
                {
                    "chain": chain_id,
                    "type": "missing_request_type",
                    "reason": f"Request type unavailable: {request_type or context}",
                }
            )
            continue

        try:
            representative_services = _representative_service_sequence(request_type, semantic_matrix)
        except ValueError as exc:
            failures.append({"chain": chain_id, "type": "invalid_representative_chain", "reason": str(exc)})
            continue

        if len(chain.nodes) < len(representative_services):
            failures.append(
                {
                    "chain": chain_id,
                    "type": "short_chain",
                    "reason": f"Chain has {len(chain.nodes)} nodes but request_type {request_type} expects {len(representative_services)}",
                }
            )

        for node_id in chain.nodes:
            service_type = str(chain.nodes[node_id].service_type)
            if instances_by_type[service_type] <= 0:
                failures.append(
                    {
                        "chain": chain_id,
                        "type": "missing_service_type",
                        "node_id": node_id,
                        "service_type": service_type,
                        "reason": "No materialized instances for service_type",
                    }
                )

        for source_id, target_id in chain.edges:
            source_node = chain.nodes[source_id]
            target_node = chain.nodes[target_id]
            source_type = str(source_node.service_type)
            target_type = str(target_node.service_type)
            source_count = instances_by_type.get(source_type, 0)
            target_count = instances_by_type.get(target_type, 0)
            target_instances = instance_index.get(target_type, [])
            compatible_count = _compatible_candidate_count(str(source_node.output_semantic or "any"), target_instances, semantic_matrix)
            print(
                f"  edge {chain_id}:{source_id}->{target_id}"
                f" source={source_type} target={target_type}"
                f" compatible_candidate_count={compatible_count}"
                f" source_instances={source_count} target_instances={target_count}"
            )
            if source_count <= 0 or target_count <= 0:
                failures.append(
                    {
                        "chain": chain_id,
                        "edge": f"{source_id}->{target_id}",
                        "source": source_type,
                        "target": target_type,
                        "compatible_candidate_count": compatible_count,
                        "source_instances": source_count,
                        "target_instances": target_count,
                        "reason": "No materialized source or target service instances",
                    }
                )
            elif compatible_count <= 0:
                item = {
                    "chain": chain_id,
                    "edge": f"{source_id}->{target_id}",
                    "source": source_type,
                    "target": target_type,
                    "compatible_candidate_count": compatible_count,
                    "source_instances": source_count,
                    "target_instances": target_count,
                    "reason": "Semantic I/O mismatch is soft in V21 and must affect delay/reward, not candidate feasibility.",
                }
                if strict_io_compatibility:
                    failures.append(item)
                else:
                    io_warnings.append(item)

    if failures:
        print(f"[{scenario_name}] seed={seed} service_role_sweep={service_role_sweep}: FAILED {len(failures)} checks")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print(f"[{scenario_name}] seed={seed} service_role_sweep={service_role_sweep}: PASS")
    if io_warnings:
        print(
            f"[{scenario_name}] seed={seed} service_role_sweep={service_role_sweep}: "
            f"{len(io_warnings)} soft semantic I/O warnings"
        )
        for item in io_warnings[:10]:
            print(f"  - SOFT_IO_WARNING {item}")
        if len(io_warnings) > 10:
            print(f"  - ... {len(io_warnings) - 10} more soft semantic I/O warnings")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate script for SFC chain I/O and materialization feasibility checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--repair-config",
        default=os.path.join(os.path.dirname(DEFAULT_CONFIG), "semantic_topology_runtime_repair.yaml"),
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--service-role-sweep", nargs="+", default=["full_hybrid"])
    parser.add_argument("--strict-io-compatibility", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config, args.repair_config)
    scenarios = _select_scenarios(config, args.scenarios)
    all_failures: List[Dict[str, Any]] = []

    for scenario in scenarios:
        scenario_name = str(scenario.get("name", "default"))
        scenario_cfg = _resolve_scenario(config, scenario)
        for service_role_sweep in args.service_role_sweep:
            for seed in args.seeds:
                env = build_offline_env(
                    config,
                    seed=seed,
                    scenario=scenario_cfg,
                    service_role_sweep=str(service_role_sweep),
                    attach_runtime=False,
                )
                try:
                    failures = _check_chain_gates(
                        env,
                        env.config.semantic_matrix,
                        scenario_name,
                        str(service_role_sweep),
                        seed,
                        strict_io_compatibility=bool(args.strict_io_compatibility),
                    )
                    for failure in failures:
                        failure.setdefault("scenario", scenario_name)
                        failure.setdefault("seed", seed)
                        failure.setdefault("service_role_sweep", str(service_role_sweep))
                        all_failures.append(failure)
                finally:
                    _close_runtime_env(getattr(env, "env", None))

    if all_failures:
        print("CHAIN_IO_GATE FAILED")
        for failure in all_failures:
            print(f"FAILED: {failure}")
        raise SystemExit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
