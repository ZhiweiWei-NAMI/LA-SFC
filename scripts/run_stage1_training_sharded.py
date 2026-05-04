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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Semantic-Topology stage1 MASAC training by seed shards.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["proposed_semantic_topology_marl"],
        help="MASAC variants to train; ablations write under semantic_runtime_train/<variant>.",
    )
    parser.add_argument(
        "--variant-concurrency",
        type=int,
        default=1,
        help="Number of variants to train at once. Seeds inside each variant still run in parallel.",
    )
    parser.add_argument(
        "--seed-concurrency",
        type=int,
        default=1,
        help="Number of seed shards to train at once inside each variant batch.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--episodes", type=int, default=None, help="Override configured training episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override configured max steps per episode.")
    parser.add_argument(
        "--cuda-devices",
        default="",
        help=(
            "Comma-separated physical CUDA device ids for shard workers, or 'auto' to use idle GPUs. "
            "When set, each shard gets one CUDA_VISIBLE_DEVICES value in round-robin order."
        ),
    )
    parser.add_argument("--idle-gpu-max-used-mib", type=int, default=2048)
    parser.add_argument("--idle-gpu-max-util", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    train_root = root / "semantic_runtime_train"
    log_dir = root / "logs"
    if args.force:
        shutil.rmtree(train_root, ignore_errors=True)
        for name in ("stage1_training_manifest.json", "stage1_training_pids.json"):
            (root / name).unlink(missing_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_processes: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    cuda_devices = select_cuda_devices(
        str(args.cuda_devices or ""),
        max_used_mib=int(args.idle_gpu_max_used_mib),
        max_util=int(args.idle_gpu_max_util),
    )
    if cuda_devices:
        print("using CUDA devices for stage1 shards: " + ", ".join(cuda_devices), flush=True)
    variant_concurrency = max(1, int(args.variant_concurrency or 1))
    for batch_start in range(0, len(args.variants), variant_concurrency):
        variant_batch = list(args.variants[batch_start : batch_start + variant_concurrency])
        pending = []
        skipped = []
        for variant in variant_batch:
            for seed in args.seeds:
                if args.skip_completed and shard_completed(seed, variant, root):
                    skipped.append((variant, seed))
                else:
                    pending.append((variant, seed))
        if skipped:
            print(
                "skipping completed stage1 shards: "
                + ", ".join(f"{variant} seed {seed}" for variant, seed in skipped),
                flush=True,
            )
        if not pending:
            continue
        seed_concurrency = max(1, int(args.seed_concurrency or 1))
        for pending_start in range(0, len(pending), seed_concurrency):
            pending_chunk = pending[pending_start : pending_start + seed_concurrency]
            processes = [
                start_shard(
                    seed,
                    variant,
                    root,
                    log_dir,
                    args.episodes,
                    args.max_steps,
                    cuda_devices[index % len(cuda_devices)] if cuda_devices else None,
                )
                for index, (variant, seed) in enumerate(pending_chunk)
            ]
            all_processes.extend(processes)
            (root / "stage1_training_pids.json").write_text(
                json.dumps(
                    [
                        {
                            "seed": item["seed"],
                            "variant": item["variant"],
                            "pid": item["proc"].pid,
                            "cuda_device": item.get("cuda_device", ""),
                        }
                        for item in all_processes
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                "started stage1 training shards: "
                + ", ".join(f"{item['variant']} seed {item['seed']} pid {item['proc'].pid}" for item in processes),
                flush=True,
            )
            batch_failed = monitor_shards(processes, args.poll_s)
            failed.extend(batch_failed)
            for item in processes:
                code = item["proc"].wait()
                item["log"].close()
                failed_keys = {(entry.get("variant"), entry.get("seed")) for entry in failed}
                if code != 0 and (item["variant"], item["seed"]) not in failed_keys:
                    failed.append(
                        {
                            "seed": item["seed"],
                            "variant": item["variant"],
                            "returncode": code,
                            "log": str(log_dir / f"stage1_{item['variant']}_seed_{item['seed']}.log"),
                        }
                    )
            if failed:
                break
        if failed:
            break

    runs = []
    for variant in args.variants:
        for seed in args.seeds:
            result_path = log_dir / f"stage1_{variant}_seed_{seed}_result.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            runs.extend(payload.get("runs", []) or [])
    unique_runs = {}
    for run in runs:
        key = (run.get("baseline"), run.get("seed"), run.get("checkpoint_path") or run.get("output_dir"))
        unique_runs[key] = run
    runs = list(unique_runs.values())

    manifest: dict[str, Any] = {
        "completed": not failed and len(runs) == len(args.seeds) * len(args.variants),
        "failed": failed,
        "seed_count": len(runs),
        "requested_seeds": list(args.seeds),
        "variants": list(args.variants),
        "output_dir": str(train_root),
        "runs": runs,
    }
    (root / "stage1_training_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if manifest["completed"] else 1


def start_shard(
    seed: int,
    variant: str,
    root: Path,
    log_dir: Path,
    episodes: int | None,
    max_steps: int | None,
    cuda_device: str | None = None,
) -> dict[str, Any]:
    log = (log_dir / f"stage1_{variant}_seed_{seed}.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cuda_device:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    cmd = [
        sys.executable,
        "-u",
        "-c",
        SHARD_CODE,
        str(seed),
        str(root),
        str(variant),
        "" if episodes is None else str(episodes),
        "" if max_steps is None else str(max_steps),
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    variant_root = root / "semantic_runtime_train" if variant == "proposed_semantic_topology_marl" else root / "semantic_runtime_train" / variant
    return {
        "seed": seed,
        "variant": variant,
        "proc": proc,
        "log": log,
        "out": variant_root / f"ippo_seed_{seed}",
        "cuda_device": str(cuda_device or ""),
    }


def select_cuda_devices(spec: str, *, max_used_mib: int, max_util: int) -> list[str]:
    value = str(spec or "").strip()
    if not value:
        return []
    if value.lower() != "auto":
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    devices: list[str] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = parts[0]
            used_mib = int(float(parts[1]))
            util = int(float(parts[2]))
        except ValueError:
            continue
        if used_mib <= max(0, int(max_used_mib)) and util <= max(0, int(max_util)):
            devices.append(index)
    return devices


def shard_completed(seed: int, variant: str, root: Path) -> bool:
    variant_root = root / "semantic_runtime_train" if variant == "proposed_semantic_topology_marl" else root / "semantic_runtime_train" / variant
    checkpoint = variant_root / f"ippo_seed_{seed}" / "masac_policy.pt"
    result = root / "logs" / f"stage1_{variant}_seed_{seed}_result.json"
    return checkpoint.exists() and result.exists()


def monitor_shards(processes: list[dict[str, Any]], poll_s: float) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    while True:
        alive = []
        for item in processes:
            code = item["proc"].poll()
            if code is None:
                alive.append(item)
            elif code != 0 and (item["variant"], item["seed"]) not in {
                (entry.get("variant"), entry.get("seed")) for entry in failed
            }:
                failed.append({"seed": item["seed"], "variant": item["variant"], "returncode": code})
        print_progress(processes, alive)
        if failed:
            for item in alive:
                item["proc"].terminate()
            return failed
        if not alive:
            return failed
        time.sleep(max(1.0, float(poll_s)))


def print_progress(processes: list[dict[str, Any]], alive: list[dict[str, Any]]) -> None:
    parts = []
    for item in processes:
        progress = item["out"] / "train_progress.csv"
        if not progress.exists():
            parts.append(f"{item['variant']} seed {item['seed']}: no-progress")
            continue
        try:
            df = pd.read_csv(progress)
            episode = int(df["episode"].max()) if not df.empty and "episode" in df else -1
            recent = df.tail(5)
            success = float(pd.to_numeric(recent.get("success_ratio", pd.Series(dtype=float)), errors="coerce").mean()) if not recent.empty else 0.0
            parts.append(f"{item['variant']} seed {item['seed']}: rows={len(df)} episode={episode} recent_success={success:.3f}")
        except Exception as exc:
            parts.append(f"{item['variant']} seed {item['seed']}: read-error={exc}")
    print(time.strftime("%F %T"), "alive=", [(item["variant"], item["seed"]) for item in alive], "; ".join(parts), flush=True)


SHARD_CODE = r"""
import json
import sys
from pathlib import Path

method_root = Path("methods_baselines/lasdm").resolve()
if str(method_root) not in sys.path:
    sys.path.insert(0, str(method_root))

from run_complete_runtime_experiment import train_semantic_ippo_variants_runtime, _load_semantic_config, _select_scenarios

seed = int(sys.argv[1])
root = Path(sys.argv[2])
variant = str(sys.argv[3])
episodes_override = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
max_steps_override = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
cfg = _load_semantic_config(
    "methods_baselines/lasdm/configs/semantic_topology_marl.yaml",
    "methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml",
)
scenarios = _select_scenarios(cfg, None)
result = train_semantic_ippo_variants_runtime(
    cfg,
    root / "semantic_runtime_train",
    [variant],
    [seed],
    scenarios,
    ["full_hybrid"],
    int(episodes_override or cfg.get("training", {}).get("episodes", 100)),
    int(max_steps_override or cfg.get("training", {}).get("max_steps", 100)),
)
(root / "logs" / f"stage1_{variant}_seed_{seed}_result.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))
"""


if __name__ == "__main__":
    raise SystemExit(main())
