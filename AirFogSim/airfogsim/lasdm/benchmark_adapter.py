from __future__ import annotations

import copy
import csv
import importlib
import inspect
import json
import os
import random
import subprocess
from dataclasses import asdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .baselines import (
    BASELINE_REGISTRY,
    get_baseline_policy,
    merge_score_weights,
    validate_baselines,
)
from .instance_directory import ServiceInstance, ServiceInstanceDirectory
from .manager import LASDMManager
from .model import LASDMServiceChain
from .orchestrator import LASDMOrchestrator


DEFAULT_OUTPUT_ROOT = "experiment_artifacts/raw_data/lasdm"
DEFAULT_EXPERIMENT_NAME = "lasdm_airfogsim"
DEFAULT_BENCHMARK_MODE = "offline"
DEFAULT_ENV_ADAPTER_IMPORT_PATH = "airfogsim.lasdm.env_adapter.LASDMEnvAdapter"
BENCHMARK_MODES = ("offline", "airfogsim")

DEFAULT_SENSITIVITY_DEADLINES = {
    "tight": {"deadline_s": 1.2},
    "medium": {"deadline_s": 3.0},
    "loose": {"deadline_s": 12.0},
}
DEFAULT_SENSITIVITY_NETWORKS = {
    "stable": {
        "network_fault": "none",
        "risk_level": 0.2,
        "network_latency_multiplier": 1.0,
    },
    "degraded": {
        "network_fault": "intermittent_v2u",
        "risk_level": 0.5,
        "network_latency_multiplier": 1.35,
    },
    "burst": {
        "network_fault": "burst",
        "risk_level": 0.75,
        "network_latency_multiplier": 1.8,
    },
}
DEFAULT_SENSITIVITY_LOADS = {
    "low": {"load_multiplier": 0.7},
    "medium": {"load_multiplier": 1.0},
    "high": {"load_multiplier": 2.0},
}

SUMMARY_FIELDS = [
    "baseline",
    "scenario",
    "seed",
    "status",
    "completed",
    "exit_code",
    "error",
    "submitted",
    "succeeded",
    "failed",
    "timed_out",
    "completed_terminal",
    "success_ratio",
    "terminal_ratio",
    "qos_hit_ratio",
    "avg_latency_s",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "p99_graph_finish_time",
    "graph_submit_count",
    "graph_submitted_count",
    "graph_complete_count",
    "graph_failed_count",
    "graph_timeout_count",
    "graph_completion_ratio",
    "deadline_satisfaction_ratio",
    "task_done_num",
    "task_fail_num",
    "task_success_ratio",
    "simulation_time_end",
    "accepted_decision_count",
    "rejected_decision_count",
    "avg_plan_score",
    "payload_tx_mb",
    "cross_region_forward_count",
    "total_candidate_count",
    "candidate_count_min",
    "candidate_count_max",
    "cold_start_count",
    "deployment_cost",
    "failure_reason",
    "failure_reason_count",
]

NUMERIC_SUMMARY_FIELDS = [
    "submitted",
    "succeeded",
    "failed",
    "timed_out",
    "completed_terminal",
    "success_ratio",
    "terminal_ratio",
    "qos_hit_ratio",
    "avg_latency_s",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "p99_graph_finish_time",
    "graph_submit_count",
    "graph_submitted_count",
    "graph_complete_count",
    "graph_failed_count",
    "graph_timeout_count",
    "graph_completion_ratio",
    "deadline_satisfaction_ratio",
    "task_done_num",
    "task_fail_num",
    "task_success_ratio",
    "simulation_time_end",
    "accepted_decision_count",
    "rejected_decision_count",
    "avg_plan_score",
    "payload_tx_mb",
    "cross_region_forward_count",
    "total_candidate_count",
    "candidate_count_min",
    "candidate_count_max",
    "cold_start_count",
    "deployment_cost",
]

SENSITIVITY_SUMMARY_FIELDS = [
    "baseline",
    "deadline",
    "network",
    "load",
    "scenario",
    "deadline_s",
    "network_fault",
    "load_multiplier",
    "run_count",
    "completed_count",
    "error_count",
]
for _field in NUMERIC_SUMMARY_FIELDS:
    SENSITIVITY_SUMMARY_FIELDS.extend([f"{_field}_mean", f"{_field}_std"])


def load_lasdm_benchmark_config(config_path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    experiment = config.setdefault("experiment", {})
    output = config.setdefault("output", {})
    runtime = config.setdefault("runtime", {})
    experiment.setdefault("name", DEFAULT_EXPERIMENT_NAME)
    experiment.setdefault("mode", DEFAULT_BENCHMARK_MODE)
    experiment.setdefault("seeds", [0])
    experiment.setdefault("baselines", list(BASELINE_REGISTRY.keys()))
    experiment.setdefault("scenarios", [{"name": "normal", "risk_level": 0.2, "network_fault": "none", "load_multiplier": 1.0}])
    runtime.setdefault("env_adapter_import_path", DEFAULT_ENV_ADAPTER_IMPORT_PATH)
    output.setdefault("root_dir", DEFAULT_OUTPUT_ROOT)
    output.setdefault("write_manifest", True)
    output.setdefault("write_per_run_json", True)
    output.setdefault("write_summary_csv", True)
    output.setdefault("write_aggregate_csv", True)
    output.setdefault("overwrite", False)

    experiment["baselines"] = validate_baselines(_ordered_unique(experiment["baselines"]))
    experiment["mode"] = _normalize_mode(experiment["mode"])
    experiment["seeds"] = [int(seed) for seed in _ordered_unique(experiment["seeds"])]
    if not experiment["seeds"]:
        raise ValueError("At least one LASDM seed is required in experiment.seeds")
    experiment["scenarios"] = _normalize_scenarios(experiment["scenarios"])
    if not config.get("service_instances"):
        raise ValueError("LASDM benchmark config must define service_instances")
    if not config.get("service_chains"):
        raise ValueError("LASDM benchmark config must define service_chains")

    output["root_dir"] = _resolve_output_root(output["root_dir"])
    config["_meta"] = {
        "benchmark_config_path": config_path,
        "benchmark_config_dir": os.path.dirname(config_path),
    }
    return config


def load_lasdm_sensitivity_config(config_path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    base_config_path = raw.get("base_config")
    if base_config_path:
        base_config_path = _resolve_relative_path(str(base_config_path), os.path.dirname(config_path))
        config = load_lasdm_benchmark_config(base_config_path)
        overlay = {key: value for key, value in raw.items() if key != "base_config"}
        _deep_update(config, overlay)
        config["_meta"]["base_benchmark_config_path"] = base_config_path
        config["_meta"]["sensitivity_config_path"] = config_path
    else:
        config = load_lasdm_benchmark_config(config_path)
        config["_meta"]["sensitivity_config_path"] = config_path

    experiment = config.setdefault("experiment", {})
    output = config.setdefault("output", {})
    experiment.setdefault("name", "lasdm_sensitivity")
    experiment["mode"] = _normalize_mode(experiment.get("mode", DEFAULT_BENCHMARK_MODE))
    experiment["baselines"] = validate_baselines(_ordered_unique(experiment.get("baselines", [])))
    experiment["seeds"] = [int(seed) for seed in _ordered_unique(experiment.get("seeds", []))]
    if not experiment["seeds"]:
        raise ValueError("At least one LASDM seed is required in experiment.seeds")
    output["root_dir"] = _resolve_output_root(output.get("root_dir", DEFAULT_OUTPUT_ROOT))
    output.setdefault("write_manifest", True)
    output.setdefault("write_per_run_json", True)
    output.setdefault("write_summary_csv", True)
    output.setdefault("write_aggregate_csv", True)
    output.setdefault("write_sensitivity_summary_csv", True)
    output.setdefault("overwrite", False)
    config["experiment"]["scenarios"] = build_lasdm_sensitivity_scenarios(config)
    return config


def build_lasdm_sensitivity_scenarios(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    sensitivity = config.get("sensitivity", {})
    deadlines = _sensitivity_levels(sensitivity.get("deadline"), DEFAULT_SENSITIVITY_DEADLINES)
    networks = _sensitivity_levels(sensitivity.get("network"), DEFAULT_SENSITIVITY_NETWORKS)
    loads = _sensitivity_levels(sensitivity.get("load"), DEFAULT_SENSITIVITY_LOADS)

    scenarios: List[Dict[str, Any]] = []
    for deadline_name, deadline_values in deadlines:
        for network_name, network_values in networks:
            for load_name, load_values in loads:
                scenario = {
                    "name": f"deadline_{deadline_name}__network_{network_name}__load_{load_name}",
                    "deadline_level": deadline_name,
                    "network_level": network_name,
                    "load_level": load_name,
                }
                scenario.update(deadline_values)
                scenario.update(network_values)
                scenario.update(load_values)
                scenario.setdefault("risk_level", 0.0)
                scenario.setdefault("network_fault", "none")
                scenario.setdefault("load_multiplier", 1.0)
                scenarios.append(scenario)
    return scenarios


def select_lasdm_runs(
    config: Dict[str, Any],
    baseline_override: Optional[Any] = None,
    seed_override: Optional[Any] = None,
    scenario_override: Optional[Any] = None,
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    if baseline_override is not None:
        baselines = [str(item) for item in _override_list(baseline_override)]
        validate_baselines(baselines)
    else:
        baselines = list(config["experiment"]["baselines"])

    if seed_override is not None:
        seeds = [int(seed) for seed in _override_list(seed_override)]
    elif baseline_override is not None or scenario_override is not None:
        seeds = [int(config["experiment"]["seeds"][0])]
    else:
        seeds = [int(seed) for seed in config["experiment"]["seeds"]]

    scenarios = list(config["experiment"]["scenarios"])
    if scenario_override is not None:
        scenario_names = {str(item) for item in _override_list(scenario_override)}
        scenarios = [scenario for scenario in scenarios if scenario["name"] in scenario_names]
        missing = sorted(scenario_names - {scenario["name"] for scenario in scenarios})
        if missing:
            raise ValueError(f"Unknown LASDM scenario: {', '.join(missing)}")
    return baselines, seeds, scenarios


def run_lasdm_single(
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    selected_mode = _normalize_mode(mode or config["experiment"].get("mode", DEFAULT_BENCHMARK_MODE))
    if selected_mode == "airfogsim":
        return _run_airfogsim_lasdm_single(config, baseline_name, seed, scenario, selected_mode)
    return _run_offline_lasdm_single(config, baseline_name, seed, scenario, selected_mode)


def _run_offline_lasdm_single(
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    completed = False
    error = None
    raw_metrics: Dict[str, Any] = {}
    decision_payloads: List[Dict[str, Any]] = []
    directory: Optional[ServiceInstanceDirectory] = None
    try:
        random.seed(seed)
        directory = _build_directory(config["service_instances"], scenario, seed)
        base_weights = dict(config.get("orchestrator", {}).get("score_weights", {}))
        policy = get_baseline_policy(baseline_name)
        weights = merge_score_weights(base_weights, policy)
        manager = LASDMManager(
            directory=directory,
            orchestrator=LASDMOrchestrator(directory, score_weights=weights),
        )

        chains = [_prepare_chain(item, baseline_name, scenario, seed) for item in config["service_chains"]]
        for chain in chains:
            manager.submit(policy.apply_to_chain(chain), current_time=0.0)
        decisions = manager.step(current_time=0.0)
        decision_payloads = [decision.to_dict() for decision in decisions]
        for chain in list(manager.chains.values()):
            decision = manager.decisions.get(chain.sfc_id)
            if decision is None or not decision.accepted:
                continue
            latency_s = _estimate_latency_s(chain, decision.to_dict(), directory, policy.completion_latency_factor, scenario)
            manager.complete(chain.sfc_id, current_time=latency_s)
        raw_metrics = manager.summary()
        completed = True
    except Exception as exc:
        error = str(exc)

    return _build_run_record(
        baseline_name=baseline_name,
        seed=seed,
        scenario=scenario,
        mode=mode,
        completed=completed,
        error=error,
        raw_metrics=raw_metrics,
        decisions=decision_payloads,
        directory=directory,
    )


def _run_airfogsim_lasdm_single(
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    completed = False
    error = None
    raw_metrics: Dict[str, Any] = {}
    decision_payloads: List[Dict[str, Any]] = []
    directory: Optional[ServiceInstanceDirectory] = None
    result: Any = None
    try:
        random.seed(seed)
        directory = _build_directory(config["service_instances"], scenario, seed)
        base_weights = dict(config.get("orchestrator", {}).get("score_weights", {}))
        policy = get_baseline_policy(baseline_name)
        weights = merge_score_weights(base_weights, policy)
        manager = LASDMManager(
            directory=directory,
            orchestrator=LASDMOrchestrator(directory, score_weights=weights),
        )
        chains = [_prepare_chain(item, baseline_name, scenario, seed) for item in config["service_chains"]]
        chains = [policy.apply_to_chain(chain) for chain in chains]
        adapter = _instantiate_env_adapter(config, baseline_name, seed, scenario, manager, directory, chains, policy)
        result = _run_env_adapter(adapter, config, baseline_name, seed, scenario, manager, directory, chains, policy)
        raw_metrics, decision_payloads, completed = _extract_runtime_result(result, manager)
        if isinstance(result, dict) and result.get("error"):
            error = str(result["error"])
    except Exception as exc:
        error = str(exc)

    record = _build_run_record(
        baseline_name=baseline_name,
        seed=seed,
        scenario=scenario,
        mode=mode,
        completed=completed,
        error=error,
        raw_metrics=raw_metrics,
        decisions=decision_payloads,
        directory=directory,
    )
    if isinstance(result, dict):
        for key in ("runtime_reports", "runtime_overhead", "resource_usage_timeseries"):
            if key in result:
                record[key] = result[key]
    return record


def run_lasdm_benchmark_suite(
    config_path: str,
    baseline_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    scenario_override: Optional[str] = None,
    output_root_override: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    config = load_lasdm_benchmark_config(config_path)
    if output_root_override is not None:
        config["output"]["root_dir"] = _resolve_output_root(output_root_override)
    if mode_override is not None:
        config["experiment"]["mode"] = _normalize_mode(mode_override)
    mode = _normalize_mode(config["experiment"].get("mode", DEFAULT_BENCHMARK_MODE))

    baselines, seeds, scenarios = select_lasdm_runs(config, baseline_override, seed_override, scenario_override)
    output_dir, runs_dir = _ensure_output_dirs(
        config["output"]["root_dir"],
        config["experiment"]["name"],
        bool(config["output"].get("overwrite", False)),
    )

    records: List[Dict[str, Any]] = []
    for baseline_name in baselines:
        for scenario in scenarios:
            for seed in seeds:
                record = run_lasdm_single(config, baseline_name, seed, scenario, mode=mode)
                record["output_dir"] = output_dir
                record["run_output_path"] = os.path.join(
                    runs_dir,
                    f"{baseline_name}__{scenario['name']}__seed_{seed}.json",
                )
                records.append(record)
                if config["output"].get("write_per_run_json", True):
                    _write_json(record["run_output_path"], record)

    cli_overrides = {
        "baseline": baseline_override,
        "seed": seed_override,
        "scenario": scenario_override,
        "output_root": output_root_override,
        "mode": mode_override,
    }
    if config["output"].get("write_manifest", True):
        _write_json(
            os.path.join(output_dir, "manifest.json"),
            _build_manifest(config, output_dir, baselines, seeds, scenarios, mode, cli_overrides),
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
            "mode": mode,
            "run_count": len(records),
            "completed_run_count": sum(1 for record in records if record["completed"]),
            "failed_run_count": sum(1 for record in records if record["exit_code"] != 0),
            "baselines": list(baselines),
            "seeds": list(seeds),
            "scenarios": [scenario["name"] for scenario in scenarios],
        }
    payload["output_dir"] = output_dir
    payload["mode"] = mode
    payload["exit_code"] = exit_code
    return exit_code, payload


def run_lasdm_sensitivity_suite(
    config_path: str,
    baseline_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    scenario_override: Optional[str] = None,
    output_root_override: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    config = load_lasdm_sensitivity_config(config_path)
    if output_root_override is not None:
        config["output"]["root_dir"] = _resolve_output_root(output_root_override)
    if mode_override is not None:
        config["experiment"]["mode"] = _normalize_mode(mode_override)
    mode = _normalize_mode(config["experiment"].get("mode", DEFAULT_BENCHMARK_MODE))

    baselines, seeds, scenarios = select_lasdm_runs(config, baseline_override, seed_override, scenario_override)
    output_dir, runs_dir = _ensure_output_dirs(
        config["output"]["root_dir"],
        config["experiment"]["name"],
        bool(config["output"].get("overwrite", False)),
    )

    records: List[Dict[str, Any]] = []
    for baseline_name in baselines:
        for scenario in scenarios:
            for seed in seeds:
                record = run_lasdm_single(config, baseline_name, seed, scenario, mode=mode)
                record["output_dir"] = output_dir
                record["run_output_path"] = os.path.join(
                    runs_dir,
                    f"{baseline_name}__{scenario['name']}__seed_{seed}.json",
                )
                records.append(record)
                if config["output"].get("write_per_run_json", True):
                    _write_json(record["run_output_path"], record)

    cli_overrides = {
        "baseline": baseline_override,
        "seed": seed_override,
        "scenario": scenario_override,
        "output_root": output_root_override,
        "mode": mode_override,
    }
    if config["output"].get("write_manifest", True):
        manifest = _build_manifest(config, output_dir, baselines, seeds, scenarios, mode, cli_overrides)
        manifest["sensitivity_dimensions"] = {
            "deadline": _scenario_levels(scenarios, "deadline_level"),
            "network": _scenario_levels(scenarios, "network_level"),
            "load": _scenario_levels(scenarios, "load_level"),
        }
        _write_json(os.path.join(output_dir, "manifest.json"), manifest)
    if config["output"].get("write_summary_csv", True):
        _write_summary_csv(os.path.join(output_dir, "summary.csv"), records)
    if config["output"].get("write_aggregate_csv", True):
        _write_aggregate_csv(os.path.join(output_dir, "aggregate.csv"), records)
    if config["output"].get("write_sensitivity_summary_csv", True):
        _write_sensitivity_summary_csv(os.path.join(output_dir, "sensitivity_summary.csv"), records)

    exit_code = 0 if all(record["exit_code"] == 0 for record in records) else 1
    payload = {
        "experiment_name": config["experiment"]["name"],
        "mode": mode,
        "run_count": len(records),
        "completed_run_count": sum(1 for record in records if record["completed"]),
        "failed_run_count": sum(1 for record in records if record["exit_code"] != 0),
        "baselines": list(baselines),
        "seeds": list(seeds),
        "scenarios": [scenario["name"] for scenario in scenarios],
        "sensitivity_summary_path": os.path.join(output_dir, "sensitivity_summary.csv"),
        "output_dir": output_dir,
        "exit_code": exit_code,
    }
    return exit_code, payload


def _normalize_mode(mode: Any) -> str:
    normalized = str(mode or DEFAULT_BENCHMARK_MODE).strip().lower()
    if normalized not in BENCHMARK_MODES:
        raise ValueError(f"Unknown LASDM benchmark mode: {mode}. Expected one of {', '.join(BENCHMARK_MODES)}")
    return normalized


def _resolve_relative_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _deep_update(target: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _sensitivity_levels(raw: Any, defaults: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    if raw is None:
        raw = defaults
    if isinstance(raw, dict):
        return [(str(name), dict(values or {})) for name, values in raw.items()]
    levels: List[Tuple[str, Dict[str, Any]]] = []
    for item in raw:
        if isinstance(item, str):
            if item not in defaults:
                raise ValueError(f"Unknown LASDM sensitivity level: {item}")
            levels.append((item, dict(defaults[item])))
        else:
            values = dict(item)
            name = str(values.pop("name"))
            levels.append((name, values))
    return levels


def _scenario_levels(scenarios: Sequence[Dict[str, Any]], key: str) -> List[str]:
    return _ordered_unique([str(scenario.get(key, "")) for scenario in scenarios if scenario.get(key) is not None])


def _normalize_scenarios(raw_scenarios: Iterable[Any]) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_scenarios):
        if isinstance(raw, str):
            scenario = {"name": raw}
        else:
            scenario = dict(raw)
        scenario.setdefault("name", f"scenario_{index}")
        scenario.setdefault("risk_level", 0.0)
        scenario.setdefault("network_fault", "none")
        scenario.setdefault("load_multiplier", 1.0)
        scenarios.append(scenario)
    return scenarios


def _instantiate_env_adapter(
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    manager: LASDMManager,
    directory: ServiceInstanceDirectory,
    chains: Sequence[LASDMServiceChain],
    policy: Any,
) -> Any:
    adapter_import_path = str(config.get("runtime", {}).get("env_adapter_import_path", DEFAULT_ENV_ADAPTER_IMPORT_PATH))
    module_name, _, class_name = adapter_import_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid LASDM env adapter import path: {adapter_import_path}")
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    kwargs = {
        "config": config,
        "runtime_config": config.get("runtime", {}),
        "baseline_name": baseline_name,
        "seed": seed,
        "scenario": scenario,
        "manager": manager,
        "directory": directory,
        "chains": list(chains),
        "policy": policy,
    }
    return _call_with_supported_kwargs(adapter_cls, kwargs)


def _run_env_adapter(
    adapter: Any,
    config: Dict[str, Any],
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    manager: LASDMManager,
    directory: ServiceInstanceDirectory,
    chains: Sequence[LASDMServiceChain],
    policy: Any,
) -> Any:
    kwargs = {
        "config": config,
        "runtime_config": config.get("runtime", {}),
        "baseline_name": baseline_name,
        "seed": seed,
        "scenario": scenario,
        "manager": manager,
        "directory": directory,
        "chains": list(chains),
        "policy": policy,
    }
    for method_name in ("run_benchmark", "run", "execute", "run_episode"):
        method = getattr(adapter, method_name, None)
        if method is not None:
            return _call_with_supported_kwargs(method, kwargs)
    raise AttributeError("LASDMEnvAdapter must expose run_benchmark(), run(), execute(), or run_episode()")


def _call_with_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return callable_obj(**supported)


def _extract_runtime_result(result: Any, manager: LASDMManager) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    if result is None:
        metrics = manager.summary()
        return metrics, _runtime_decisions_from_manager(manager), True
    if not isinstance(result, dict):
        if hasattr(result, "summary"):
            result = {"raw_metrics": result.summary()}
        elif hasattr(result, "collect_step_metrics"):
            result = {"raw_metrics": result.collect_step_metrics()}
        else:
            raise TypeError(f"Unsupported LASDMEnvAdapter result type: {type(result).__name__}")

    metrics = _first_dict(result, ("raw_metrics", "metrics", "summary")) or {}
    if not metrics and _looks_like_metrics(result):
        metrics = dict(result)
    decisions = result.get("decisions")
    if decisions is None:
        decisions = _runtime_decisions_from_manager(manager)
    completed = bool(result.get("completed", True))
    if result.get("error"):
        completed = False
    return dict(metrics), _normalize_decision_payloads(decisions), completed


def _first_dict(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _looks_like_metrics(payload: Dict[str, Any]) -> bool:
    metric_keys = {"submitted", "succeeded", "failed", "timed_out", "graph_submitted_count", "graph_complete_count"}
    return any(key in payload for key in metric_keys)


def _runtime_decisions_from_manager(manager: LASDMManager) -> List[Dict[str, Any]]:
    return [decision.to_dict() for decision in manager.decisions.values()]


def _normalize_decision_payloads(decisions: Any) -> List[Dict[str, Any]]:
    if decisions is None:
        return []
    if isinstance(decisions, dict):
        decisions = decisions.values()
    normalized: List[Dict[str, Any]] = []
    for decision in decisions:
        if hasattr(decision, "to_dict"):
            normalized.append(decision.to_dict())
        elif isinstance(decision, dict):
            normalized.append(dict(decision))
    return normalized


def _prepare_chain(raw_chain: Dict[str, Any], baseline_name: str, scenario: Dict[str, Any], seed: int) -> LASDMServiceChain:
    chain_raw = copy.deepcopy(raw_chain)
    chain_raw["sfc_id"] = f"{raw_chain['sfc_id']}__{baseline_name}__{scenario['name']}__seed_{seed}"
    qos = dict(chain_raw.get("qos", {}))
    if scenario.get("deadline_s") is not None:
        qos["deadline_s"] = float(scenario["deadline_s"])
    elif scenario.get("deadline_multiplier") is not None:
        qos["deadline_s"] = float(qos["deadline_s"]) * float(scenario["deadline_multiplier"])
    chain_raw["qos"] = qos
    context = dict(chain_raw.get("context", {}))
    context["risk_level"] = float(scenario.get("risk_level", context.get("risk_level", 0.0)))
    context["scenario"] = scenario["name"]
    context["network_fault"] = scenario.get("network_fault", "none")
    chain_raw["context"] = context
    return LASDMServiceChain.from_dict(chain_raw)


def _build_directory(raw_instances: Sequence[Dict[str, Any]], scenario: Dict[str, Any], seed: int) -> ServiceInstanceDirectory:
    directory = ServiceInstanceDirectory()
    load_multiplier = max(0.0, float(scenario.get("load_multiplier", 1.0)))
    network_fault = str(scenario.get("network_fault", "none"))
    for raw in raw_instances:
        item = copy.deepcopy(raw)
        used = {key: float(value) * load_multiplier for key, value in item.get("used", {}).items()}
        capacity = {key: float(value) for key, value in item.get("capacity", {}).items()}
        item["used"] = {
            key: min(capacity.get(key, value), value)
            for key, value in used.items()
        }
        if load_multiplier > 1.0:
            item["current_load"] = min(int(item.get("max_concurrency", 1)), int(load_multiplier) - 1 + seed % 2)
        if network_fault == "intermittent_v2u" and item.get("node_type") == "uav":
            item["health_score"] = min(float(item.get("health_score", 1.0)), 0.72)
            item["reliability_score"] = min(float(item.get("reliability_score", 1.0)), 0.91)
        elif network_fault == "rsu_congestion" and str(item.get("node_id", "")).startswith("RSU_"):
            item["health_score"] = min(float(item.get("health_score", 1.0)), 0.78)
            item["cold_start_s"] = float(item.get("cold_start_s", 0.0)) + 0.05
        elif network_fault == "burst" and item.get("node_type") in {"uav", "rsu"}:
            item["health_score"] = min(float(item.get("health_score", 1.0)), 0.68)
            item["reliability_score"] = min(float(item.get("reliability_score", 1.0)), 0.88)
            item["cold_start_s"] = float(item.get("cold_start_s", 0.0)) + 0.08
        directory.register(ServiceInstance.from_dict(item))
    return directory


def _estimate_latency_s(
    chain: LASDMServiceChain,
    decision: Dict[str, Any],
    directory: ServiceInstanceDirectory,
    baseline_factor: float,
    scenario: Dict[str, Any],
) -> float:
    processing = sum(max(0.05, node.cpu_mb * 0.12 + node.memory_mb * 0.0005) for node in chain.nodes.values())
    payload = chain.payload_mb * 0.02
    route_hops = sum(max(0, len(route) - 1) for route in decision.get("routes", {}).values())
    network_multiplier = max(0.0, float(scenario.get("network_latency_multiplier", 1.0)))
    network = route_hops * 0.08 * network_multiplier
    cold_start = 0.0
    for instance_id in decision.get("assignments", {}).values():
        try:
            cold_start += directory.get(instance_id).cold_start_s
        except KeyError:
            pass
    risk_penalty = float(scenario.get("risk_level", 0.0)) * 0.1
    load_penalty = max(0.0, float(scenario.get("load_multiplier", 1.0)) - 1.0) * 0.15
    latency_multiplier = max(0.0, float(scenario.get("latency_multiplier", 1.0)))
    return max(0.001, (processing + payload + network + cold_start + risk_penalty + load_penalty) * baseline_factor * latency_multiplier)


def _build_run_record(
    baseline_name: str,
    seed: int,
    scenario: Dict[str, Any],
    mode: str,
    completed: bool,
    error: Optional[str],
    raw_metrics: Dict[str, Any],
    decisions: Sequence[Dict[str, Any]],
    directory: Optional[ServiceInstanceDirectory] = None,
) -> Dict[str, Any]:
    metrics = dict(raw_metrics or {})
    failure_reasons = _failure_reason_count(metrics.get("failure_reason_count", {}))
    accepted = [decision for decision in decisions if decision.get("rejected_reason") is None]
    rejected = [decision for decision in decisions if decision.get("rejected_reason") is not None]
    candidate_counts = [
        count
        for decision in decisions
        for count in decision.get("diagnostics", {}).get("candidate_counts", {}).values()
    ]
    scores = [float(decision.get("score", 0.0)) for decision in accepted]
    submitted = _int_metric(metrics, "submitted", "graph_submitted_count", "graph_submit_count", "submit_count")
    succeeded = _int_metric(metrics, "succeeded", "graph_complete_count", "success_count")
    failed = _int_metric(metrics, "failed", "graph_failed_count", "failure_count")
    timed_out = _int_metric(metrics, "timed_out", "graph_timeout_count", "timeout_count")
    completed_terminal = _int_metric(metrics, "completed_terminal", default=succeeded + failed + timed_out)
    success_ratio = _float_metric(metrics, "success_ratio", default=succeeded / max(1, submitted))
    terminal_ratio = _float_metric(metrics, "terminal_ratio", default=completed_terminal / max(1, submitted))
    qos_hit_ratio = _float_metric(metrics, "qos_hit_ratio", "deadline_satisfaction_ratio")
    avg_latency_s = _float_metric(metrics, "avg_latency_s", "avg_graph_finish_time")
    avg_graph_finish_time = _float_metric(metrics, "avg_graph_finish_time", "avg_latency_s", default=avg_latency_s)
    p95_graph_finish_time = _float_metric(metrics, "p95_graph_finish_time", default=avg_graph_finish_time)
    p99_graph_finish_time = _float_metric(metrics, "p99_graph_finish_time", default=avg_graph_finish_time)
    accepted_count = _int_metric(metrics, "accepted_decision_count", default=len(accepted))
    rejected_count = _int_metric(metrics, "rejected_decision_count", default=len(rejected))
    task_done_num = _int_metric(metrics, "task_done_num", "done_task_num", default=succeeded)
    task_fail_num = _int_metric(metrics, "task_fail_num", "failed_task_num", default=failed + timed_out)
    task_success_ratio = _float_metric(
        metrics,
        "task_success_ratio",
        default=task_done_num / max(1, task_done_num + task_fail_num),
    )
    record: Dict[str, Any] = {
        "baseline": baseline_name,
        "scenario": scenario["name"],
        "seed": seed,
        "mode": mode,
        "status": _run_status(completed, error, submitted, succeeded, failed, timed_out),
        "completed": completed,
        "exit_code": 0 if completed and error is None else 1,
        "error": error,
        "submitted": submitted,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "completed_terminal": completed_terminal,
        "success_ratio": success_ratio,
        "terminal_ratio": terminal_ratio,
        "qos_hit_ratio": qos_hit_ratio,
        "avg_latency_s": avg_latency_s,
        "avg_graph_finish_time": avg_graph_finish_time,
        "p95_graph_finish_time": p95_graph_finish_time,
        "p99_graph_finish_time": p99_graph_finish_time,
        "graph_submit_count": submitted,
        "graph_submitted_count": submitted,
        "graph_complete_count": succeeded,
        "graph_failed_count": failed,
        "graph_timeout_count": timed_out,
        "graph_completion_ratio": succeeded / max(1, submitted),
        "deadline_satisfaction_ratio": qos_hit_ratio,
        "task_done_num": task_done_num,
        "task_fail_num": task_fail_num,
        "task_success_ratio": task_success_ratio,
        "simulation_time_end": _float_metric(metrics, "simulation_time_end", "current_time", default=0.0),
        "accepted_decision_count": accepted_count,
        "rejected_decision_count": rejected_count,
        "avg_plan_score": _float_metric(metrics, "avg_plan_score", default=mean(scores) if scores else 0.0),
        "payload_tx_mb": _float_metric(metrics, "payload_tx_mb", default=_payload_tx_mb(decisions)),
        "cross_region_forward_count": _int_metric(metrics, "cross_region_forward_count", default=_cross_region_forward_count(decisions)),
        "total_candidate_count": _int_metric(metrics, "total_candidate_count", default=sum(int(count) for count in candidate_counts)),
        "candidate_count_min": _int_metric(metrics, "candidate_count_min", default=min(candidate_counts) if candidate_counts else 0),
        "candidate_count_max": _int_metric(metrics, "candidate_count_max", default=max(candidate_counts) if candidate_counts else 0),
        "cold_start_count": _int_metric(metrics, "cold_start_count", default=_cold_start_count(decisions, directory)),
        "deployment_cost": _float_metric(metrics, "deployment_cost", default=_deployment_cost(decisions, directory)),
        "failure_reason": _dominant_failure_reason(failure_reasons),
        "failure_reason_count": json.dumps(failure_reasons, sort_keys=True),
        "raw_metrics": metrics,
        "decisions": list(decisions),
        "scenario_config": dict(scenario),
        "baseline_config": asdict(get_baseline_policy(baseline_name)),
    }
    return record


def _payload_tx_mb(decisions: Sequence[Dict[str, Any]]) -> float:
    # The adapter does not run the network simulator; this is a deterministic route load proxy.
    return float(sum(max(0, len(route) - 1) for decision in decisions for route in decision.get("routes", {}).values()))


def _cross_region_forward_count(decisions: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for decision in decisions:
        for route in decision.get("routes", {}).values():
            if len(set(route)) > 1:
                count += 1
    return count


def _run_status(
    completed: bool,
    error: Optional[str],
    submitted: int,
    succeeded: int,
    failed: int,
    timed_out: int,
) -> str:
    if error is not None:
        return "error"
    if timed_out:
        return "timed_out"
    if failed:
        return "failed"
    if submitted and succeeded >= submitted:
        return "succeeded"
    if completed:
        return "completed"
    return "unknown"


def _dominant_failure_reason(failure_reasons: Dict[str, int]) -> str:
    if not failure_reasons:
        return "none"
    return max(sorted(failure_reasons), key=lambda key: int(failure_reasons.get(key, 0)))


def _assigned_instances(
    decisions: Sequence[Dict[str, Any]],
    directory: Optional[ServiceInstanceDirectory],
) -> List[ServiceInstance]:
    if directory is None:
        return []
    instances: List[ServiceInstance] = []
    for decision in decisions:
        for instance_id in decision.get("assignments", {}).values():
            try:
                instances.append(directory.get(str(instance_id)))
            except KeyError:
                continue
    return instances


def _cold_start_count(
    decisions: Sequence[Dict[str, Any]],
    directory: Optional[ServiceInstanceDirectory],
) -> int:
    return sum(1 for instance in _assigned_instances(decisions, directory) if instance.cold_start_s > 0.0)


def _deployment_cost(
    decisions: Sequence[Dict[str, Any]],
    directory: Optional[ServiceInstanceDirectory],
) -> float:
    return float(sum(instance.cold_start_s for instance in _assigned_instances(decisions, directory)))


def _int_metric(metrics: Dict[str, Any], *keys: str, default: int = 0) -> int:
    value = _metric_value(metrics, keys, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_metric(metrics: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value = _metric_value(metrics, keys, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metric_value(metrics: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return default


def _failure_reason_count(value: Any) -> Dict[str, int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {value: 1} if value and value != "none" else {}
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, count in value.items():
        try:
            normalized[str(key)] = int(count)
        except (TypeError, ValueError):
            normalized[str(key)] = 1
    return normalized


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
    fieldnames = ["baseline", "scenario", "run_count", "completed_count", "error_count"]
    for field in NUMERIC_SUMMARY_FIELDS:
        fieldnames.extend([f"{field}_mean", f"{field}_std"])
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["baseline"], record["scenario"]), []).append(record)

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in _ordered_unique([(record["baseline"], record["scenario"]) for record in records]):
            group = grouped[key]
            row: Dict[str, Any] = {
                "baseline": key[0],
                "scenario": key[1],
                "run_count": len(group),
                "completed_count": sum(1 for record in group if record.get("completed")),
                "error_count": sum(1 for record in group if record.get("error")),
            }
            for field in NUMERIC_SUMMARY_FIELDS:
                values = [float(record[field]) for record in group if record.get(field) is not None]
                row[f"{field}_mean"] = mean(values) if values else None
                row[f"{field}_std"] = pstdev(values) if values else None
            writer.writerow(row)


def _write_sensitivity_summary_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for record in records:
        scenario_config = record.get("scenario_config", {})
        key = (
            str(record["baseline"]),
            str(scenario_config.get("deadline_level", "")),
            str(scenario_config.get("network_level", "")),
            str(scenario_config.get("load_level", "")),
        )
        grouped.setdefault(key, []).append(record)

    ordered_keys = _ordered_unique(
        [
            (
                str(record["baseline"]),
                str(record.get("scenario_config", {}).get("deadline_level", "")),
                str(record.get("scenario_config", {}).get("network_level", "")),
                str(record.get("scenario_config", {}).get("load_level", "")),
            )
            for record in records
        ]
    )
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SENSITIVITY_SUMMARY_FIELDS)
        writer.writeheader()
        for key in ordered_keys:
            group = grouped[key]
            scenario_config = group[0].get("scenario_config", {})
            row: Dict[str, Any] = {
                "baseline": key[0],
                "deadline": key[1],
                "network": key[2],
                "load": key[3],
                "scenario": group[0].get("scenario"),
                "deadline_s": scenario_config.get("deadline_s"),
                "network_fault": scenario_config.get("network_fault"),
                "load_multiplier": scenario_config.get("load_multiplier"),
                "run_count": len(group),
                "completed_count": sum(1 for record in group if record.get("completed")),
                "error_count": sum(1 for record in group if record.get("error")),
            }
            for field in NUMERIC_SUMMARY_FIELDS:
                values = [float(record[field]) for record in group if record.get(field) is not None]
                row[f"{field}_mean"] = mean(values) if values else None
                row[f"{field}_std"] = pstdev(values) if values else None
            writer.writerow(row)


def _build_manifest(
    config: Dict[str, Any],
    output_dir: str,
    baselines: Sequence[str],
    seeds: Sequence[int],
    scenarios: Sequence[Dict[str, Any]],
    mode: str,
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": config["experiment"]["name"],
        "mode": mode,
        "benchmark_config_path": config["_meta"]["benchmark_config_path"],
        "baselines": list(baselines),
        "seeds": list(seeds),
        "scenarios": [scenario["name"] for scenario in scenarios],
        "output_dir": output_dir,
        "git_commit": _get_git_commit(),
        "cli_overrides": cli_overrides,
        "parsed_config": config,
        "summary_fields": list(SUMMARY_FIELDS),
    }


def _ensure_output_dirs(root_dir: str, experiment_name: str, overwrite: bool) -> Tuple[str, str]:
    output_dir = os.path.join(root_dir, experiment_name, _timestamp())
    if os.path.exists(output_dir) and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=overwrite)
    return output_dir, runs_dir


def _resolve_output_root(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    return os.path.abspath(os.path.join(workspace_root, path))


def _override_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _get_git_commit() -> Optional[str]:
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    repo_root = os.path.join(workspace_root, "AirFogSim")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _ordered_unique(items: Sequence[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered
