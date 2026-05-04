from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BASELINES = [
    "intra_region_only",
    "cross_region_auction",
    "pure_semantic_greedy_no_exchange",
    "local_semantic_runtime_greedy",
    "nsga2_semantic_qos",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_no_exchange",
    "marl_no_temporal",
    "marl_no_cross_region",
    "proposed_semantic_topology_marl",
    "centralized_planner",
]

DEFAULT_TRAINED_BASELINES = [
    "mappo_ctde",
    "iql_offline",
]

TRAINED_CHECKPOINT_FILES = {
    "mappo_ctde": "mappo_policy.pt",
    "iql_offline": "iql_policy.pt",
}

COMBINED_FILES = [
    "function_execution_trace.csv",
    "runtime_task_lifecycle_trace.csv",
    "message_overhead.csv",
    "distributed_exchange_trace.csv",
    "semantic_candidate_trace.csv",
    "semantic_candidate_detail_trace.csv",
    "resource_usage_timeseries.csv",
    "runtime_overhead.csv",
    "marl_transition_trace.jsonl",
    "topology_trace.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Semantic-Topology stage2 evaluation by eval-seed shards.")
    parser.add_argument("--root", required=True, help="Stage1 root containing semantic_runtime_train.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--baselines", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument(
        "--seed-concurrency",
        type=int,
        default=2,
        help="Number of eval seed shards to run at once.",
    )
    parser.add_argument("--combine-only", action="store_true", help="Reuse existing shard outputs and only write final summaries.")
    parser.add_argument("--copy-run-dirs", action="store_true", help="Copy per-run raw directories into the final output.")
    parser.add_argument("--include-traces", action="store_true", help="Combine large trace files into the final output.")
    args = parser.parse_args()

    root = Path(args.root)
    shards = root / "stage2_eval_shards"
    final = root / "semantic_runtime_eval"
    log_dir = root / "logs" / "stage2_eval_shards"
    checkpoint_root = root / "semantic_runtime_train"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Missing checkpoint root: {checkpoint_root}")
    baselines = list(args.baselines) if args.baselines else default_stage2_baselines(checkpoint_root)
    if args.force and not args.combine_only:
        shutil.rmtree(shards, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
    shards.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.combine_only:
        processes = [
            {"seed": seed, "out": shards / f"seed_{seed}" / "semantic_runtime_eval"}
            for seed in args.seeds
        ]
        failed = validate_existing_shards(processes)
    else:
        processes = []
        seed_concurrency = max(1, int(args.seed_concurrency or 1))
        failed = []
        for seed_start in range(0, len(args.seeds), seed_concurrency):
            batch = list(args.seeds[seed_start : seed_start + seed_concurrency])
            batch_processes = [
                start_shard(seed, shards, checkpoint_root, baselines, log_dir)
                for seed in batch
            ]
            processes.extend(batch_processes)
            (root / "stage2_eval_pids.json").write_text(
                json.dumps(
                    [{"seed": item["seed"], "pid": item["proc"].pid} for item in processes],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                "started stage2 eval shards: "
                + ", ".join(f"seed {item['seed']} pid {item['proc'].pid}" for item in batch_processes),
                flush=True,
            )

            batch_failed = monitor_shards(batch_processes, args.poll_s)
            failed.extend(batch_failed)
            for item in batch_processes:
                code = item["proc"].wait()
                item["log"].close()
                if code != 0 and item["seed"] not in {entry["seed"] for entry in failed}:
                    failed.append(
                        {
                            "seed": item["seed"],
                            "returncode": code,
                            "log": str(log_dir / f"seed_{item['seed']}.log"),
                        }
                    )
            if failed:
                break

    manifest: dict[str, Any] = {
        "completed": False,
        "failed": failed,
        "seeds": list(args.seeds),
        "baselines": baselines,
        "checkpoint_root": str(checkpoint_root),
        "shard_root": str(shards),
        "output_dir": str(final),
    }
    if not failed:
        if args.combine_only or args.force:
            shutil.rmtree(final, ignore_errors=True)
        combine_shards(processes, final, copy_run_dirs=args.copy_run_dirs, include_traces=args.include_traces)
        summary = pd.read_csv(final / "summary.csv")
        manifest.update(
            {
                "completed": True,
                "run_count": int(len(summary)),
                "summary_csv": str(final / "summary.csv"),
                "aggregate_csv": str(final / "aggregate.csv"),
                "copy_run_dirs": bool(args.copy_run_dirs),
                "include_traces": bool(args.include_traces),
            }
        )
    (root / "stage2_eval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if manifest["completed"] else 1


def default_stage2_baselines(checkpoint_root: Path) -> list[str]:
    baselines = list(DEFAULT_BASELINES)
    baselines.extend(
        baseline
        for baseline in DEFAULT_TRAINED_BASELINES
        if trained_checkpoint_available(checkpoint_root, baseline)
    )
    return baselines


def trained_checkpoint_available(checkpoint_root: Path, baseline: str) -> bool:
    filename = TRAINED_CHECKPOINT_FILES.get(str(baseline))
    if not filename:
        return False
    return any((checkpoint_root / str(baseline)).glob(f"**/{filename}"))


def start_shard(seed: int, shards: Path, checkpoint_root: Path, baselines: list[str], log_dir: Path) -> dict[str, Any]:
    shard_out = shards / f"seed_{seed}" / "semantic_runtime_eval"
    shard_out.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"seed_{seed}.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [
        sys.executable,
        "-u",
        "-c",
        SHARD_CODE,
        str(seed),
        str(shard_out),
        str(checkpoint_root),
        json.dumps(baselines),
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"seed": seed, "proc": proc, "log": log, "out": shard_out}


def validate_existing_shards(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = []
    for item in processes:
        summary = item["out"] / "summary.csv"
        if not summary.exists():
            failed.append({"seed": item["seed"], "returncode": None, "error": f"missing {summary}"})
            continue
        try:
            pd.read_csv(summary)
        except Exception as exc:
            failed.append({"seed": item["seed"], "returncode": None, "error": str(exc)})
    return failed


def monitor_shards(processes: list[dict[str, Any]], poll_s: float) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    while True:
        alive = []
        for item in processes:
            code = item["proc"].poll()
            if code is None:
                alive.append(item)
            elif code != 0 and item["seed"] not in {entry["seed"] for entry in failed}:
                failed.append({"seed": item["seed"], "returncode": code})
        print_progress(processes, alive)
        if failed:
            for item in alive:
                item["proc"].terminate()
            return failed
        if not alive:
            return failed
        time.sleep(max(1.0, float(poll_s)))


def print_progress(processes: list[dict[str, Any]], alive: list[dict[str, Any]]) -> None:
    summaries = []
    for item in processes:
        summary = item["out"] / "summary.csv"
        if not summary.exists():
            summaries.append(f"seed {item['seed']}: no-summary")
            continue
        try:
            df = pd.read_csv(summary)
            summaries.append(f"seed {item['seed']}: rows={len(df)}")
        except Exception as exc:
            summaries.append(f"seed {item['seed']}: read-error={exc}")
    print(time.strftime("%F %T"), "alive=", [item["seed"] for item in alive], "; ".join(summaries), flush=True)


def combine_shards(
    processes: list[dict[str, Any]],
    final: Path,
    *,
    copy_run_dirs: bool = False,
    include_traces: bool = False,
) -> None:
    final.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    aggregate_frames = []
    csv_rows: dict[str, list[pd.DataFrame]] = {name: [] for name in COMBINED_FILES if name.endswith(".csv")}
    jsonl_names = [name for name in COMBINED_FILES if name.endswith(".jsonl")]
    failure_rows: list[Any] = []
    for item in processes:
        shard_eval = item["out"]
        summary_frames.append(pd.read_csv(shard_eval / "summary.csv"))
        aggregate_path = shard_eval / "aggregate.csv"
        if aggregate_path.exists():
            aggregate_frames.append(pd.read_csv(aggregate_path))
        if copy_run_dirs:
            for child in shard_eval.iterdir():
                if child.is_dir() and (child / "run.json").exists():
                    target = final / child.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(child, target)
        if include_traces:
            for name in csv_rows:
                path = shard_eval / name
                if not path.exists():
                    continue
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                if not frame.empty:
                    csv_rows[name].append(frame)
        failure_path = shard_eval / "failure_trace.json"
        if failure_path.exists():
            try:
                failure_rows.extend(json.loads(failure_path.read_text(encoding="utf-8")) or [])
            except Exception:
                pass
    pd.concat(summary_frames, ignore_index=True).to_csv(final / "summary.csv", index=False)
    if aggregate_frames:
        pd.concat(aggregate_frames, ignore_index=True).to_csv(final / "aggregate.csv", index=False)
    for name, frames in csv_rows.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(final / name, index=False)
    if include_traces:
        for name in jsonl_names:
            target = final / name
            with target.open("w", encoding="utf-8") as out_file:
                wrote = False
                for item in processes:
                    path = item["out"] / name
                    if not path.exists():
                        continue
                    with path.open("r", encoding="utf-8") as in_file:
                        shutil.copyfileobj(in_file, out_file)
                    wrote = True
                if not wrote:
                    target.unlink(missing_ok=True)
    (final / "failure_trace.json").write_text(json.dumps(failure_rows, indent=2, ensure_ascii=False), encoding="utf-8")


SHARD_CODE = r"""
import json
import sys
from pathlib import Path

method_root = Path("methods_baselines/lasdm").resolve()
if str(method_root) not in sys.path:
    sys.path.insert(0, str(method_root))

from run_complete_runtime_experiment import evaluate_semantic_runtime, _load_semantic_config, _select_scenarios

seed = int(sys.argv[1])
out = Path(sys.argv[2])
checkpoint_root = Path(sys.argv[3])
baselines = json.loads(sys.argv[4])
cfg = _load_semantic_config(
    "methods_baselines/lasdm/configs/semantic_topology_marl.yaml",
    "methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml",
)
cfg.setdefault("marl", {})["ippo_eval_checkpoint_strategy"] = "exact_seed"
cfg.setdefault("runtime_repair", {}).setdefault("early_result_guard", {})["enabled"] = False
scenarios = _select_scenarios(cfg, None)
result = evaluate_semantic_runtime(
    cfg,
    out,
    baselines,
    scenarios,
    ["full_hybrid"],
    [seed],
    checkpoint_root,
    int(cfg.get("training", {}).get("max_steps", 100)),
)
(out / "shard_result.json").write_text(
    json.dumps({"seed": seed, "result": result}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps({"seed": seed, "result": result}, indent=2, ensure_ascii=False))
"""


if __name__ == "__main__":
    raise SystemExit(main())
