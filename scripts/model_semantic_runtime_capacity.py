from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = REPO_ROOT / "methods_baselines" / "lasdm"
AIRFOGSIM_ROOT = REPO_ROOT / "AirFogSim"
for path in (str(AIRFOGSIM_ROOT), str(METHOD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_complete_runtime_experiment import _load_semantic_config, _select_scenarios  # noqa: E402
from train_semantic_topology_marl import build_offline_env, _close_runtime_env  # noqa: E402


DEFAULT_CONFIG = METHOD_ROOT / "configs" / "semantic_topology_marl.yaml"
DEFAULT_REPAIR_CONFIG = METHOD_ROOT / "configs" / "semantic_topology_runtime_figures_aligned.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Model optimistic SFC capacity/deadline upper bounds from the real runtime config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--repair-config", default=str(DEFAULT_REPAIR_CONFIG))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--service-role-sweep", default="full_hybrid")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = _load_semantic_config(args.config, args.repair_config)
    scenarios = _select_scenarios(config, args.scenarios)
    rows = []
    for scenario in scenarios:
        for seed in args.seeds:
            rows.append(model_scenario(config, scenario, int(seed), args.service_role_sweep, int(args.max_steps)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "capacity_upper_bounds.csv", rows)
    summary = {
        "row_count": len(rows),
        "min_optimistic_success_upper_bound": min((row["optimistic_success_upper_bound"] for row in rows), default=0.0),
        "max_optimistic_success_upper_bound": max((row["optimistic_success_upper_bound"] for row in rows), default=0.0),
        "rows": rows,
    }
    (output_dir / "capacity_upper_bounds.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"completed": True, "rows": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


def model_scenario(
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    role: str,
    max_steps: int,
) -> Dict[str, Any]:
    env = build_offline_env(
        config,
        seed=seed,
        max_steps=max_steps,
        scenario=scenario,
        service_role_sweep=role,
        attach_runtime=True,
        baseline="centralized_planner",
    )
    try:
        chains = list(getattr(env, "chains", []) or [])
        instances = list(env.manager.directory.all())
        horizon_s = float(
            scenario.get(
                "arrival_horizon_s",
                dict(config.get("runtime", {}) or {}).get("airfogsim_config_overrides", {})
                .get("simulation", {})
                .get("max_simulation_time", 20.0),
            )
            or 20.0
        )
        demand_by_service = service_demand(chains)
        cpu_by_service = task_cpu_by_service(chains)
        throughput_by_service = optimistic_throughput(instances, demand_by_service, cpu_by_service)
        capacity_ratios = {
            service_id: throughput_by_service.get(service_id, 0.0) * horizon_s / max(1, demand)
            for service_id, demand in demand_by_service.items()
        }
        capacity_upper = min([1.0, *capacity_ratios.values()]) if capacity_ratios else 0.0
        deadline_feasible = deadline_feasible_ratio(chains, instances, cpu_by_service)
        chain_lengths = [len(getattr(chain, "nodes", {}) or {}) for chain in chains]
        request_count = len(chains)
        upper = min(1.0, float(capacity_upper), float(deadline_feasible))
        return {
            "scenario": str(scenario.get("name", "default")),
            "seed": int(seed),
            "request_count": int(request_count),
            "mean_sfc_length": mean(chain_lengths),
            "function_demand": int(sum(demand_by_service.values())),
            "service_type_count": int(len(demand_by_service)),
            "service_instance_count": int(len(instances)),
            "horizon_s": float(horizon_s),
            "min_service_capacity_ratio": float(capacity_upper),
            "deadline_feasible_ratio": float(deadline_feasible),
            "optimistic_success_upper_bound": float(upper),
            "bottleneck_service": bottleneck_service(capacity_ratios),
            "bottleneck_capacity_ratio": float(min(capacity_ratios.values()) if capacity_ratios else 0.0),
            "max_concurrent_sfcs": int(max_concurrent(chains)),
            "model": "min(capacity_throughput_upper, per_chain_deadline_lower_bound_feasible)",
        }
    finally:
        _close_runtime_env(getattr(env, "env", None))


def service_demand(chains: Sequence[Any]) -> Dict[str, int]:
    demand: Dict[str, int] = {}
    for chain in chains:
        for node in dict(getattr(chain, "nodes", {}) or {}).values():
            service_id = str(getattr(node, "service_type", "") or "")
            if service_id:
                demand[service_id] = demand.get(service_id, 0) + 1
    return demand


def task_cpu_by_service(chains: Sequence[Any]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for chain in chains:
        for node in dict(getattr(chain, "nodes", {}) or {}).values():
            service_id = str(getattr(node, "service_type", "") or "")
            metadata = dict(getattr(node, "metadata", {}) or {})
            cpu = float(metadata.get("task_cpu", getattr(node, "cpu_mb", 0.0)) or 0.0)
            if service_id and cpu > 0.0:
                sums[service_id] = sums.get(service_id, 0.0) + cpu
                counts[service_id] = counts.get(service_id, 0) + 1
    return {service_id: sums[service_id] / float(counts[service_id]) for service_id in sums}


def optimistic_throughput(
    instances: Sequence[Any],
    demand_by_service: Mapping[str, int],
    cpu_by_service: Mapping[str, float],
) -> Dict[str, float]:
    throughput: Dict[str, float] = {service_id: 0.0 for service_id in demand_by_service}
    for instance in instances:
        service_id = str(getattr(instance, "service_id", "") or "")
        if service_id not in throughput:
            continue
        task_cpu = float(cpu_by_service.get(service_id, representative_task_cpu(service_id, demand_by_service)) or 1.0)
        capacity = dict(getattr(instance, "capacity", {}) or {})
        used = dict(getattr(instance, "used", {}) or {})
        available_cpu = max(0.0, float(capacity.get("cpu", 0.0) or 0.0) - float(used.get("cpu", 0.0) or 0.0))
        throughput[service_id] += available_cpu / max(1e-9, task_cpu)
    return throughput


def representative_task_cpu(service_id: str, demand_by_service: Mapping[str, int]) -> float:
    cpu_by_domain = {
        "preprocess": 16.0,
        "detection": 42.0,
        "fusion": 34.0,
        "verification": 38.0,
        "alert": 14.0,
        "archive": 18.0,
        "report": 24.0,
    }
    key = str(service_id)
    for token, cpu in cpu_by_domain.items():
        if token in key:
            return cpu * 0.10
    return 24.0 * 0.10 if service_id in demand_by_service else 24.0


def deadline_feasible_ratio(chains: Sequence[Any], instances: Sequence[Any], cpu_by_service: Mapping[str, float]) -> float:
    if not chains:
        return 0.0
    best_latency_by_service = best_service_latency(instances, cpu_by_service)
    feasible = 0
    for chain in chains:
        lower_bound = 0.0
        for node in dict(getattr(chain, "nodes", {}) or {}).values():
            service_id = str(getattr(node, "service_type", "") or "")
            lower_bound += best_latency_by_service.get(service_id, math.inf)
        deadline = float(getattr(getattr(chain, "qos", None), "deadline_s", 0.0) or 0.0)
        if deadline > 0.0 and lower_bound <= deadline + 1e-9:
            feasible += 1
    return feasible / float(len(chains))


def best_service_latency(instances: Sequence[Any], cpu_by_service: Mapping[str, float]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for instance in instances:
        service_id = str(getattr(instance, "service_id", "") or "")
        if not service_id:
            continue
        capacity = dict(getattr(instance, "capacity", {}) or {})
        concurrency = max(1, int(getattr(instance, "max_concurrency", 1) or 1))
        cpu_per_slot = max(1e-9, float(capacity.get("cpu", 0.0) or 0.0) / float(concurrency))
        task_cpu = float(cpu_by_service.get(service_id, representative_task_cpu(service_id, {service_id: 1})) or 1.0)
        latency = task_cpu / cpu_per_slot + max(0.0, float(getattr(instance, "cold_start_s", 0.0) or 0.0))
        best[service_id] = min(best.get(service_id, math.inf), latency)
    return best


def max_concurrent(chains: Sequence[Any]) -> int:
    values = []
    for chain in chains:
        context = dict(getattr(chain, "context", {}) or {})
        if context.get("max_concurrent_sfcs") is not None:
            values.append(int(float(context.get("max_concurrent_sfcs") or 0)))
    return min([value for value in values if value > 0], default=0)


def bottleneck_service(capacity_ratios: Mapping[str, float]) -> str:
    if not capacity_ratios:
        return ""
    return min(capacity_ratios.items(), key=lambda item: item[1])[0]


def mean(values: Iterable[float]) -> float:
    items = [float(item) for item in values]
    return sum(items) / len(items) if items else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
