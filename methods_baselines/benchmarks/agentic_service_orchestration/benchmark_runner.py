from __future__ import annotations

import copy
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PROJECT_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airfogsim import AirFogSimEnv
from airfogsim.airfogsim_scheduler import AirFogSimScheduler
from airfogsim.service_orchestration.agentic_service_algorithm import (
    AgenticServiceAlgorithmModule,
)
from airfogsim.service_orchestration.service_catalog import ServiceCatalog
from airfogsim.service_orchestration.testing import (
    advance_algorithm_until_complete,
    build_smoke_env,
)
from examples.task_intents import (
    PoissonLowAltitudeIntentGenerator,
    StaticLowAltitudeIntentGenerator,
)


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "benchmark_config.yaml",
)

BASELINE_REGISTRY = {
    "fixed_sfc": {
        "matcher_policy": "fixed_sfc",
        "serving_policy": "centralized_greedy",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "deployment_only": {
        "matcher_policy": "fixed_sfc",
        "serving_policy": "nearest_edge",
        "regional": False,
        "predeploy": True,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "nearest_edge": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "nearest_edge",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "cats_style": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "cats_style_score",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "centralized_greedy": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "centralized_greedy",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "regional_distributed_greedy": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "centralized_greedy",
        "regional": True,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
    "edge_only": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "centralized_greedy",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": ["rsu", "cloud_server"],
        "strict_node_type_filter": True,
    },
    "uav_only": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "centralized_greedy",
        "regional": False,
        "predeploy": False,
        "allowed_node_types": ["uav"],
        "strict_node_type_filter": False,
    },
    "proposed": {
        "matcher_policy": "adaptive_rule_based",
        "serving_policy": "proposed",
        "regional": True,
        "predeploy": False,
        "allowed_node_types": None,
        "strict_node_type_filter": True,
    },
}

SUMMARY_FIELDS = [
    "baseline",
    "seed",
    "mode",
    "completed",
    "exit_code",
    "error",
    "matcher_policy",
    "serving_policy",
    "regional_agent_enabled",
    "predeploy",
    "allowed_node_types",
    "strict_node_type_filter",
    "cold_start_count",
    "deployment_cost",
    "payload_tx_mb",
    "cross_region_forward_count",
    "graph_submit_count",
    "graph_complete_count",
    "graph_completion_ratio",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "p99_graph_finish_time",
    "deadline_satisfaction_ratio",
    "wrong_service_composition_mean",
    "missing_service_capability_mean",
    "extra_service_capability_mean",
    "task_done_num",
    "task_fail_num",
    "task_success_ratio",
    "simulation_time_end",
]

NUMERIC_SUMMARY_FIELDS = [
    "cold_start_count",
    "deployment_cost",
    "payload_tx_mb",
    "cross_region_forward_count",
    "graph_submit_count",
    "graph_complete_count",
    "graph_completion_ratio",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "p99_graph_finish_time",
    "deadline_satisfaction_ratio",
    "wrong_service_composition_mean",
    "missing_service_capability_mean",
    "extra_service_capability_mean",
    "task_done_num",
    "task_fail_num",
    "task_success_ratio",
    "simulation_time_end",
]


def _ordered_unique(items: Sequence[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_repo_or_config_path(path: Optional[str], config_dir: str) -> Optional[str]:
    if path in (None, ""):
        return path
    if os.path.isabs(path):
        return os.path.abspath(path)
    repo_candidate = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    config_candidate = os.path.abspath(os.path.join(config_dir, path))
    if os.path.exists(repo_candidate):
        return repo_candidate
    if os.path.exists(config_candidate):
        return config_candidate
    return repo_candidate


def _resolve_output_root(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(WORKSPACE_ROOT, path))


def _is_placeholder_path(path: Optional[str]) -> bool:
    if not isinstance(path, str) or not path:
        return False
    normalized = path.replace("\\", "/")
    return "path/to/" in normalized or normalized.startswith("<")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _ensure_output_dirs(root_dir: str, experiment_name: str, overwrite: bool) -> Tuple[str, str]:
    suite_dir = os.path.join(root_dir, experiment_name, _timestamp())
    if os.path.exists(suite_dir) and not overwrite:
        raise FileExistsError(f"Output directory already exists: {suite_dir}")
    os.makedirs(os.path.join(suite_dir, "runs"), exist_ok=overwrite)
    return suite_dir, os.path.join(suite_dir, "runs")


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    raw = _load_yaml(config_path)

    experiment = raw.setdefault("experiment", {})
    intent_source = raw.setdefault("intent_source", {})
    output = raw.setdefault("output", {})

    experiment.setdefault("name", "aso_v1")
    experiment.setdefault("mode", "smoke")
    experiment.setdefault("baselines", list(BASELINE_REGISTRY.keys()))
    experiment.setdefault("seeds", [0])
    experiment.setdefault("catalog_path", "examples/service_catalog.yaml")
    experiment.setdefault("intents_path", "examples/task_intents.yaml")
    experiment.setdefault("airfogsim_config_path", "examples/config_agentic_service.yaml")
    experiment.setdefault("smoke_max_rounds", 12)
    experiment.setdefault("airfogsim_max_steps", 200)

    intent_source.setdefault("smoke", {"type": "static_yaml"})
    intent_source.setdefault(
        "airfogsim",
        {
            "type": "poisson_low_altitude",
            "arrival_prob": 0.3,
        },
    )

    output.setdefault("root_dir", "experiment_artifacts/raw_data/agentic_service_orchestration")
    output.setdefault("write_manifest", True)
    output.setdefault("write_per_run_json", True)
    output.setdefault("write_summary_csv", True)
    output.setdefault("write_aggregate_csv", True)
    output.setdefault("overwrite", False)

    baselines = _ordered_unique(experiment["baselines"])
    invalid = [baseline for baseline in baselines if baseline not in BASELINE_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown baselines in config: {', '.join(invalid)}")

    seeds = [int(seed) for seed in _ordered_unique(experiment["seeds"])]
    if not seeds:
        raise ValueError("At least one seed is required in experiment.seeds")

    mode = experiment["mode"]
    if mode not in {"smoke", "airfogsim"}:
        raise ValueError(f"Unsupported experiment.mode: {mode}")

    experiment["baselines"] = baselines
    experiment["seeds"] = seeds
    experiment["catalog_path"] = _resolve_repo_or_config_path(
        experiment["catalog_path"],
        config_dir,
    )
    experiment["intents_path"] = _resolve_repo_or_config_path(
        experiment["intents_path"],
        config_dir,
    )
    experiment["airfogsim_config_path"] = _resolve_repo_or_config_path(
        experiment["airfogsim_config_path"],
        config_dir,
    )
    output["root_dir"] = _resolve_output_root(output["root_dir"])
    raw["_meta"] = {
        "benchmark_config_path": config_path,
        "benchmark_config_dir": config_dir,
    }
    validate_catalog_and_intents(raw)
    return raw


def validate_catalog_and_intents(config: Dict[str, Any]) -> None:
    catalog_path = config["experiment"].get("catalog_path")
    intents_path = config["experiment"].get("intents_path")
    if not catalog_path or not os.path.exists(catalog_path):
        raise FileNotFoundError(f"catalog_path not found: {catalog_path}")
    if not intents_path or not os.path.exists(intents_path):
        raise FileNotFoundError(f"intents_path not found: {intents_path}")

    catalog = ServiceCatalog.from_yaml(catalog_path)
    raw_intents = _load_yaml(intents_path).get("intents", [])
    if not raw_intents:
        raise ValueError(f"intents_path has no intents: {intents_path}")
    for item in raw_intents:
        intent_id = item.get("intent_id", "<unknown>")
        required = list(item.get("required_capabilities", []))
        if not required:
            raise ValueError(f"Intent {intent_id} has no required_capabilities")
        try:
            catalog.validate_required_capabilities(required)
        except ValueError as exc:
            raise ValueError(f"Intent {intent_id}: {exc}") from exc


def select_runs(
    config: Dict[str, Any],
    baseline_override: Optional[str] = None,
    seed_override: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    baselines = (
        [baseline_override]
        if baseline_override is not None
        else list(config["experiment"]["baselines"])
    )
    if seed_override is not None:
        seeds = [int(seed_override)]
    elif baseline_override is not None:
        seeds = [int(config["experiment"]["seeds"][0])]
    else:
        seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    return baselines, seeds


def prime_deployments(algorithm, env) -> None:
    for ms in algorithm.catalog.all():
        for node_id in ["RSU_0", "cloudServer_0"]:
            if env._getNodeById(node_id) is not None:
                algorithm.deployment.deploy(node_id, ms.ms_id)


def _resolve_airfogsim_config_paths(config: Dict[str, Any], config_dir: str) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    path_keys = {
        "sumo": ["sumo_config", "sumo_osm", "sumo_net", "tripinfo_output"],
        "traffic": ["tripinfo", "uav_traffic_file"],
        "visualization": ["icon_path"],
    }
    for section, keys in path_keys.items():
        if section not in resolved:
            continue
        for key in keys:
            value = resolved[section].get(key)
            if not isinstance(value, str) or not value:
                continue
            if os.path.isabs(value):
                resolved[section][key] = value
            else:
                resolved[section][key] = os.path.abspath(os.path.join(config_dir, value))
    return resolved


def load_airfogsim_config(config_path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"airfogsim_config_path not found: {config_path}")
    config = _load_yaml(config_path)
    config = _resolve_airfogsim_config_paths(config, os.path.dirname(config_path))
    config.setdefault("task", {})
    config["task"]["task_generation_model"] = "None"
    return config


def validate_airfogsim_setup(config_path: str, config: Dict[str, Any]) -> None:
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"airfogsim_config_path not found: {config_path}")

    sumo_cfg = config.get("sumo", {})
    traffic_cfg = config.get("traffic", {})
    missing: List[str] = []

    def require_file(label: str, path: Optional[str]) -> None:
        if not path:
            missing.append(f"{label} is not set")
            return
        if _is_placeholder_path(path):
            missing.append(f"{label} is a placeholder path: {path}")
            return
        if not os.path.exists(path):
            missing.append(f"{label} not found: {path}")

    require_file("sumo.sumo_net", sumo_cfg.get("sumo_net"))
    require_file("sumo.sumo_config", sumo_cfg.get("sumo_config"))

    sumo_osm = sumo_cfg.get("sumo_osm")
    if isinstance(sumo_osm, str) and sumo_osm:
        require_file("sumo.sumo_osm", sumo_osm)

    traffic_mode = str(traffic_cfg.get("traffic_mode", "")).upper()
    if traffic_mode != "SUMO":
        require_file("traffic.tripinfo", traffic_cfg.get("tripinfo"))
    uav_traffic_file = traffic_cfg.get("uav_traffic_file")
    if isinstance(uav_traffic_file, str) and uav_traffic_file:
        require_file("traffic.uav_traffic_file", uav_traffic_file)

    if missing:
        raise FileNotFoundError("; ".join(missing))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        max(0, int(round((len(sorted_values) - 1) * percentile))),
    )
    return sorted_values[index]


def _build_algorithm(
    catalog_path: str,
    intent_generator,
    baseline_name: str,
) -> AgenticServiceAlgorithmModule:
    baseline_cfg = BASELINE_REGISTRY[baseline_name]
    return AgenticServiceAlgorithmModule(
        catalog_path=catalog_path,
        intent_generator=intent_generator,
        matcher_policy=baseline_cfg["matcher_policy"],
        serving_policy=baseline_cfg["serving_policy"],
        regional_agent_enabled=baseline_cfg["regional"],
        allowed_node_types=baseline_cfg["allowed_node_types"],
        strict_node_type_filter=baseline_cfg["strict_node_type_filter"],
    )


def _build_smoke_generator(config: Dict[str, Any]):
    smoke_cfg = config.get("intent_source", {}).get("smoke", {})
    source_type = smoke_cfg.get("type", "static_yaml")
    if source_type != "static_yaml":
        raise ValueError(f"Unsupported smoke intent_source type: {source_type}")
    return StaticLowAltitudeIntentGenerator.from_yaml(config["experiment"]["intents_path"])


def _build_airfogsim_generator(config: Dict[str, Any]):
    airfogsim_cfg = config.get("intent_source", {}).get("airfogsim", {})
    source_type = airfogsim_cfg.get("type", "poisson_low_altitude")
    if source_type != "poisson_low_altitude":
        raise ValueError(f"Unsupported airfogsim intent_source type: {source_type}")
    return PoissonLowAltitudeIntentGenerator(
        arrival_prob=float(airfogsim_cfg.get("arrival_prob", 0.3)),
    )


def _derive_summary_fields(
    raw_metrics: Optional[Dict[str, Any]],
    env,
) -> Dict[str, Optional[float]]:
    if raw_metrics is None:
        return {field: None for field in NUMERIC_SUMMARY_FIELDS}

    submit_times = dict(raw_metrics.get("graph_submit_time", {}))
    finish_times = dict(raw_metrics.get("service_graph_finish_time", {}))
    deadlines = dict(raw_metrics.get("service_graph_deadline_s", {}))
    composition = dict(raw_metrics.get("wrong_service_composition_ratio", {}))
    missing_composition = dict(raw_metrics.get("missing_service_capability_ratio", {}))
    extra_composition = dict(raw_metrics.get("extra_service_capability_ratio", {}))
    durations = []
    for graph_id, finish_time in finish_times.items():
        submit_time = submit_times.get(graph_id)
        durations.append(finish_time - submit_time if submit_time is not None else finish_time)
    deadline_hits = 0
    deadline_total = 0
    for graph_id, finish_time in finish_times.items():
        deadline = deadlines.get(graph_id)
        submit_time = submit_times.get(graph_id)
        if deadline is None or submit_time is None:
            continue
        deadline_total += 1
        if finish_time - submit_time <= deadline:
            deadline_hits += 1

    task_done_num = None
    task_fail_num = None
    simulation_time_end = None
    if env is not None:
        task_sched = AirFogSimScheduler.getTaskScheduler()
        task_done_num = task_sched.getDoneTaskNum(env)
        task_fail_num = task_sched.getOutOfDDLTasks(env)
        simulation_time_end = float(getattr(env, "simulation_time", 0.0))

    task_success_ratio = None
    if task_done_num is not None and task_fail_num is not None:
        task_success_ratio = task_done_num / max(1, task_done_num + task_fail_num)

    return {
        "cold_start_count": raw_metrics.get("cold_start_count"),
        "deployment_cost": raw_metrics.get("deployment_cost"),
        "payload_tx_mb": raw_metrics.get("payload_tx_mb"),
        "cross_region_forward_count": raw_metrics.get("cross_region_forward_count"),
        "graph_submit_count": len(submit_times),
        "graph_complete_count": len(finish_times),
        "graph_completion_ratio": len(finish_times) / max(1, len(submit_times)),
        "avg_graph_finish_time": mean(durations) if durations else 0.0,
        "p95_graph_finish_time": _percentile(durations, 0.95),
        "p99_graph_finish_time": _percentile(durations, 0.99),
        "deadline_satisfaction_ratio": deadline_hits / max(1, deadline_total),
        "wrong_service_composition_mean": mean(composition.values()) if composition else 0.0,
        "missing_service_capability_mean": mean(missing_composition.values()) if missing_composition else 0.0,
        "extra_service_capability_mean": mean(extra_composition.values()) if extra_composition else 0.0,
        "task_done_num": task_done_num,
        "task_fail_num": task_fail_num,
        "task_success_ratio": task_success_ratio,
        "simulation_time_end": simulation_time_end,
    }


def _build_run_record(
    baseline_name: str,
    seed: int,
    mode: str,
    raw_metrics: Optional[Dict[str, Any]],
    env,
    completed: bool,
    error: Optional[str],
) -> Dict[str, Any]:
    baseline_cfg = BASELINE_REGISTRY[baseline_name]
    record = {
        "baseline": baseline_name,
        "seed": seed,
        "mode": mode,
        "completed": completed,
        "exit_code": 0 if completed and error is None else 1,
        "error": error,
        "matcher_policy": baseline_cfg["matcher_policy"],
        "serving_policy": baseline_cfg["serving_policy"],
        "regional_agent_enabled": baseline_cfg["regional"],
        "predeploy": baseline_cfg["predeploy"],
        "allowed_node_types": ",".join(baseline_cfg["allowed_node_types"] or []),
        "strict_node_type_filter": baseline_cfg["strict_node_type_filter"],
        "raw_metrics": raw_metrics if raw_metrics is not None else {},
    }
    record.update(_derive_summary_fields(raw_metrics, env))
    return record


def run_single_baseline(
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
) -> Dict[str, Any]:
    mode = config["experiment"]["mode"]
    catalog_path = config["experiment"]["catalog_path"]
    raw_metrics: Optional[Dict[str, Any]] = None
    env = None
    algorithm = None
    completed = False
    error = None

    try:
        _set_seed(seed)
        if mode == "smoke":
            env = build_smoke_env(multi_region=True, include_cloud=True)
            algorithm = _build_algorithm(
                catalog_path=catalog_path,
                intent_generator=_build_smoke_generator(config),
                baseline_name=baseline_name,
            )
            algorithm.initialize(env)
            if BASELINE_REGISTRY[baseline_name]["predeploy"]:
                prime_deployments(algorithm, env)
            max_rounds = int(config["experiment"]["smoke_max_rounds"])
            completed = advance_algorithm_until_complete(
                algorithm,
                env,
                max_rounds=max_rounds,
            )
            if not completed:
                error = f"Service graph did not finish within {max_rounds} rounds."
        else:
            airfogsim_config_path = config["experiment"]["airfogsim_config_path"]
            airfogsim_config = load_airfogsim_config(airfogsim_config_path)
            validate_airfogsim_setup(airfogsim_config_path, airfogsim_config)
            env = AirFogSimEnv(airfogsim_config, interactive_mode=None)
            algorithm = _build_algorithm(
                catalog_path=catalog_path,
                intent_generator=_build_airfogsim_generator(config),
                baseline_name=baseline_name,
            )
            algorithm.initialize(env, airfogsim_config)
            if BASELINE_REGISTRY[baseline_name]["predeploy"]:
                prime_deployments(algorithm, env)
            max_steps = int(config["experiment"]["airfogsim_max_steps"])
            steps = 0
            while not env.isDone():
                if max_steps is not None and steps >= max_steps:
                    break
                algorithm.scheduleStep(env)
                env.step()
                steps += 1
            completed = True

        if algorithm is not None and env is not None:
            algorithm.runtime.on_airfogsim_step_finished(env)
        if algorithm is not None:
            raw_metrics = algorithm.metrics.summary()
    except Exception as exc:
        error = str(exc)
        if algorithm is not None:
            raw_metrics = algorithm.metrics.summary()
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    return _build_run_record(
        baseline_name=baseline_name,
        seed=seed,
        mode=mode,
        raw_metrics=raw_metrics,
        env=env,
        completed=completed,
        error=error,
    )


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _write_summary_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})


def _write_aggregate_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ["baseline", "run_count", "completed_count", "error_count"]
    for field in NUMERIC_SUMMARY_FIELDS:
        fieldnames.extend([f"{field}_mean", f"{field}_std"])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["baseline"], []).append(record)

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for baseline in _ordered_unique([record["baseline"] for record in records]):
            group = grouped[baseline]
            row: Dict[str, Any] = {
                "baseline": baseline,
                "run_count": len(group),
                "completed_count": sum(1 for record in group if record.get("completed")),
                "error_count": sum(1 for record in group if record.get("error")),
            }
            for field in NUMERIC_SUMMARY_FIELDS:
                values = [
                    float(record[field])
                    for record in group
                    if record.get(field) is not None
                ]
                row[f"{field}_mean"] = mean(values) if values else None
                row[f"{field}_std"] = pstdev(values) if values else None
            writer.writerow(row)


def _build_manifest(
    config: Dict[str, Any],
    output_dir: str,
    baselines: Sequence[str],
    seeds: Sequence[int],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    experiment = config["experiment"]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment["name"],
        "mode": experiment["mode"],
        "benchmark_config_path": config["_meta"]["benchmark_config_path"],
        "catalog_path": experiment["catalog_path"],
        "intents_path": experiment["intents_path"],
        "airfogsim_config_path": experiment["airfogsim_config_path"],
        "baselines": list(baselines),
        "seeds": list(seeds),
        "output_dir": output_dir,
        "git_commit": _get_git_commit(),
        "cli_overrides": cli_overrides,
        "parsed_config": config,
    }


def run_benchmark_suite(
    config_path: str = DEFAULT_CONFIG_PATH,
    baseline_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    output_root_override: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    config = load_benchmark_config(config_path)
    if baseline_override is not None and baseline_override not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {baseline_override}")

    if output_root_override is not None:
        config["output"]["root_dir"] = _resolve_output_root(output_root_override)

    baselines, seeds = select_runs(config, baseline_override, seed_override)
    output_dir, runs_dir = _ensure_output_dirs(
        root_dir=config["output"]["root_dir"],
        experiment_name=config["experiment"]["name"],
        overwrite=bool(config["output"].get("overwrite", False)),
    )

    records: List[Dict[str, Any]] = []
    for baseline_name in baselines:
        for seed in seeds:
            record = run_single_baseline(config, baseline_name, seed)
            record["output_dir"] = output_dir
            record["run_output_path"] = os.path.join(
                runs_dir,
                f"{baseline_name}__seed_{seed}.json",
            )
            records.append(record)
            if config["output"].get("write_per_run_json", True):
                _write_json(record["run_output_path"], record)

    cli_overrides = {
        "baseline": baseline_override,
        "seed": seed_override,
        "output_root": output_root_override,
    }

    if config["output"].get("write_manifest", True):
        _write_json(
            os.path.join(output_dir, "manifest.json"),
            _build_manifest(config, output_dir, baselines, seeds, cli_overrides),
        )
    if config["output"].get("write_summary_csv", True):
        _write_summary_csv(os.path.join(output_dir, "summary.csv"), records)
    if config["output"].get("write_aggregate_csv", True):
        _write_aggregate_csv(os.path.join(output_dir, "aggregate.csv"), records)

    exit_code = 0 if all(record["exit_code"] == 0 for record in records) else 1
    if len(records) == 1:
        payload = dict(records[0])
        payload["run_count"] = 1
    else:
        payload = {
            "experiment_name": config["experiment"]["name"],
            "mode": config["experiment"]["mode"],
            "run_count": len(records),
            "completed_run_count": sum(1 for record in records if record["completed"]),
            "failed_run_count": sum(1 for record in records if record["exit_code"] != 0),
            "baselines": baselines,
            "seeds": seeds,
        }
    payload["output_dir"] = output_dir
    payload["exit_code"] = exit_code
    return exit_code, payload
