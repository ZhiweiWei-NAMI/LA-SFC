from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


METHOD_ROOT = Path("methods_baselines/lasdm").resolve()
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from run_complete_runtime_experiment import evaluate_semantic_runtime, _load_semantic_config, _scenario_by_name


TRACE_FILES = [
    "summary.csv",
    "aggregate.csv",
    "function_execution_trace.csv",
    "runtime_task_lifecycle_trace.csv",
    "message_overhead.csv",
    "distributed_exchange_trace.csv",
    "semantic_candidate_trace.csv",
    "semantic_candidate_detail_trace.csv",
    "resource_usage_timeseries.csv",
    "runtime_overhead.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real evaluation sweeps required by plans/figures_req.md.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--semantic-config", default="methods_baselines/lasdm/configs/semantic_topology_marl.yaml")
    parser.add_argument("--semantic-repair-config", default="methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["service_nodes", "task_nodes", "skew", "ttl", "ablation", "uav", "sfc_length", "speed", "centralized"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.force:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    suite_root = output_root / "suites"
    combined_root = output_root / "semantic_runtime_eval"
    suite_root.mkdir(parents=True, exist_ok=True)
    combined_root.mkdir(parents=True, exist_ok=True)

    cfg = _load_semantic_config(args.semantic_config, args.semantic_repair_config)
    cfg.setdefault("marl", {})["ippo_eval_checkpoint_strategy"] = "global_best_validation"
    cfg.setdefault("runtime_repair", {}).setdefault("early_result_guard", {})["enabled"] = False
    cfg.setdefault("runtime_repair", {})["fail_fast_on_all_zero_success"] = False

    manifest: list[dict[str, Any]] = []
    for suite_name in args.suites:
        scenarios, baselines = build_suite(cfg, suite_name)
        if not scenarios:
            manifest.append({"suite": suite_name, "completed": False, "reason": "empty suite"})
            continue
        out = suite_root / suite_name / "semantic_runtime_eval"
        result = evaluate_semantic_runtime(
            cfg,
            out,
            baselines,
            scenarios,
            ["full_hybrid"],
            [int(seed) for seed in args.seeds],
            Path(args.checkpoint_root),
            int(args.max_steps),
        )
        manifest.append(
            {
                "suite": suite_name,
                "completed": bool(result.get("completed")),
                "baselines": baselines,
                "scenario_count": len(scenarios),
                "seed_count": len(args.seeds),
                "output_dir": str(out),
                "result": result,
            }
        )
    combine_suite_outputs(suite_root, combined_root)
    (output_root / "figures_req_eval_manifest.json").write_text(
        json.dumps({"suites": manifest, "combined_eval_dir": str(combined_root)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "combined_eval_dir": str(combined_root), "suites": len(manifest)}, indent=2))
    return 0


def build_suite(cfg: Mapping[str, Any], name: str) -> tuple[list[dict[str, Any]], list[str]]:
    suite = str(name)
    if suite == "service_nodes":
        methods = ["proposed_semantic_topology_marl", "topology_greedy", "utility_prior_with_exchange", "cross_region_auction", "intra_region_only"]
        scenarios = [
            scenario_variant(
                cfg,
                "semantic_runtime_contention_stress",
                f"fig1_service_nodes_{count}",
                service_nodes=service_node_counts(count),
                task_nodes={"vehicle": 100, "uav": 0},
                request_count=100,
                max_concurrent_sfcs=100,
            )
            for count in [20, 40, 60, 80]
        ]
        return scenarios, methods
    if suite == "task_nodes":
        methods = ["proposed_semantic_topology_marl", "topology_greedy", "utility_prior_with_exchange", "cross_region_auction", "intra_region_only"]
        scenarios = [
            scenario_variant(
                cfg,
                "semantic_runtime_contention_stress",
                f"fig2_task_nodes_{count}",
                service_nodes=service_node_counts(60),
                task_nodes={"vehicle": count, "uav": 0},
                request_count=count,
                max_concurrent_sfcs=count,
            )
            for count in [20, 50, 100, 150, 200]
        ]
        return scenarios, methods
    if suite == "skew":
        methods = ["proposed_semantic_topology_marl", "topology_greedy", "cross_region_auction", "intra_region_only"]
        scenarios = [
            scenario_variant(
                cfg,
                "semantic_runtime_contention_stress",
                f"fig3_skew_{str(ratio).replace('.', 'p')}",
                service_nodes=service_node_counts(60),
                task_nodes={"vehicle": 100, "uav": 0},
                request_count=100,
                max_concurrent_sfcs=100,
                regional_skew_ratio=float(ratio),
            )
            for ratio in [1, 2, 4, 6, 8, 10]
        ]
        return scenarios, methods
    if suite == "ttl":
        methods = ["proposed_semantic_topology_marl"]
        scenarios = [
            scenario_variant(
                cfg,
                "distributed_service_discovery_calibrated",
                f"fig6_ttl_{int(ttl)}",
                exchange_ttl_s=float(ttl),
                exchange_radius_hops=3,
            )
            for ttl in [2, 4, 6, 8, 10, 12]
        ]
        return scenarios, methods
    if suite == "ablation":
        methods = [
            "proposed_semantic_topology_marl",
            "marl_no_semantic",
            "marl_semantic_no_topology",
            "marl_no_exchange",
            "marl_no_temporal",
            "marl_no_cross_region",
        ]
        scenarios = [scenario_variant(cfg, "semantic_runtime_contention_stress", "fig9_ablation_contention")]
        return scenarios, methods
    if suite == "uav":
        methods = ["proposed_semantic_topology_marl", "utility_prior_with_exchange", "cross_region_auction"]
        scenarios = [
            scenario_variant(
                cfg,
                "semantic_runtime_mobility_staleness_stress",
                f"fig11_uav_{count}",
                service_nodes={"rsu": 4, "vehicle": 10, "uav": count, "cloud_server": 1},
                active_uav_count=count,
                uav_count=count,
            )
            for count in [2, 4, 6, 8, 10, 12]
        ]
        return scenarios, methods
    if suite == "sfc_length":
        methods = ["proposed_semantic_topology_marl", "topology_greedy"]
        scenarios = []
        for length in [3, 4]:
            for count in [20, 50, 100, 150]:
                scenarios.append(
                    scenario_variant(
                        cfg,
                        "semantic_runtime_contention_stress",
                        f"fig12_sfc_length_{length}_task_nodes_{count}",
                        service_nodes=service_node_counts(60),
                        task_nodes={"vehicle": count, "uav": 0},
                        request_count=count,
                        max_concurrent_sfcs=count,
                        sfc_length=length,
                    )
                )
        return scenarios, methods
    if suite == "speed":
        methods = ["proposed_semantic_topology_marl", "marl_no_temporal", "utility_prior_with_exchange"]
        scenarios = [
            scenario_variant(
                cfg,
                "semantic_runtime_mobility_staleness_stress",
                f"fig13_speed_{speed}",
                speed_scale=float(speed) / 50.0,
                vehicle_speed_scale=float(speed) / 50.0,
                uav_speed_scale=max(1.0, float(speed) / 50.0),
            )
            for speed in [30, 50, 70, 90]
        ]
        return scenarios, methods
    if suite == "centralized":
        methods = ["centralized_planner", "proposed_semantic_topology_marl", "topology_greedy"]
        scenarios = [
            scenario_variant(cfg, "semantic_runtime_contention_stress", "fig14_contention"),
            scenario_variant(cfg, "semantic_runtime_mobility_staleness_stress", "fig14_mobility"),
        ]
        return scenarios, methods
    raise ValueError(f"Unknown suite: {name}")


def scenario_variant(cfg: Mapping[str, Any], base_name: str, name: str, **updates: Any) -> dict[str, Any]:
    scenario = copy.deepcopy(_scenario_by_name(cfg, base_name))
    scenario["name"] = name
    scenario.update({key: value for key, value in updates.items() if value is not None})
    return scenario


def service_node_counts(total: int) -> dict[str, int]:
    total = max(5, int(total))
    mobile = max(0, total - 5)
    uav = min(20, max(2, int(round(mobile * 0.25))))
    vehicle = max(0, min(100, mobile - uav))
    return {"rsu": 4, "vehicle": vehicle, "uav": uav, "cloud_server": 1}


def combine_suite_outputs(suite_root: Path, combined_root: Path) -> None:
    combined_root.mkdir(parents=True, exist_ok=True)
    for file_name in TRACE_FILES:
        frames = []
        for path in suite_root.glob(f"*/semantic_runtime_eval/{file_name}"):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["figure_suite"] = path.parent.parent.name
            frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(combined_root / file_name, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
