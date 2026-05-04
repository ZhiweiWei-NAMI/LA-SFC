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


TRAINED_CHECKPOINT_FILES = {
    "mappo_ctde": "mappo_policy.pt",
    "iql_offline": "iql_policy.pt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Semantic-Topology stage1 learned-policy training by seed shards.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["proposed_semantic_topology_marl"],
        help="Learned variants to train; non-proposed variants write under semantic_runtime_train/<variant>.",
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
        "--proposed-seeds",
        nargs="+",
        type=int,
        default=None,
        help="When set, train proposed_semantic_topology_marl on these seeds and all other variants on --other-seeds.",
    )
    parser.add_argument(
        "--other-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seed list for non-proposed variants when --proposed-seeds is set.",
    )
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
    parser.add_argument("--gpu-mib-per-process", type=int, default=4096)
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
    requested_jobs = build_job_plan(args)
    seed_concurrency = max(1, int(args.seed_concurrency or 1))
    cuda_devices = select_cuda_devices(
        str(args.cuda_devices or ""),
        max_used_mib=int(args.idle_gpu_max_used_mib),
        max_util=int(args.idle_gpu_max_util),
        mib_per_process=int(args.gpu_mib_per_process),
        required_slots=min(seed_concurrency, len(requested_jobs)),
    )
    if cuda_devices:
        print("using CUDA devices for stage1 shards: " + ", ".join(cuda_devices), flush=True)
    if args.skip_completed:
        skipped = [(variant, seed) for variant, seed in requested_jobs if shard_completed(seed, variant, root)]
        requested_jobs = [(variant, seed) for variant, seed in requested_jobs if not shard_completed(seed, variant, root)]
        if skipped:
            print(
                "skipping completed stage1 shards: "
                + ", ".join(f"{variant} seed {seed}" for variant, seed in skipped),
                flush=True,
            )
    for pending_start in range(0, len(requested_jobs), seed_concurrency):
        pending_chunk = requested_jobs[pending_start : pending_start + seed_concurrency]
        if pending_chunk:
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

    runs = []
    for variant, seed in build_job_plan(args):
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
        "completed": not failed and len(runs) == len(build_job_plan(args)),
        "failed": failed,
        "seed_count": len(runs),
        "requested_seeds": list(args.seeds),
        "requested_jobs": [{"variant": variant, "seed": seed} for variant, seed in build_job_plan(args)],
        "variants": list(args.variants),
        "output_dir": str(train_root),
        "runs": runs,
    }
    (root / "stage1_training_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if manifest["completed"] else 1


def build_job_plan(args: argparse.Namespace) -> list[tuple[str, int]]:
    variants = [str(item) for item in args.variants]
    if args.proposed_seeds is None:
        return [(variant, int(seed)) for seed in args.seeds for variant in variants]
    proposed = [int(seed) for seed in args.proposed_seeds]
    other = [int(seed) for seed in (args.other_seeds if args.other_seeds is not None else args.seeds)]
    jobs: list[tuple[str, int]] = []
    for variant in variants:
        seeds = proposed if variant == "proposed_semantic_topology_marl" else other
        for seed in seeds:
            jobs.append((variant, seed))
    return jobs


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


def select_cuda_devices(
    spec: str,
    *,
    max_used_mib: int,
    max_util: int,
    mib_per_process: int,
    required_slots: int,
) -> list[str]:
    value = str(spec or "").strip()
    if not value:
        return []
    if value.lower() != "auto":
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    candidates: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = parts[0]
            total_mib = int(float(parts[1]))
            used_mib = int(float(parts[2]))
            util = int(float(parts[3]))
        except ValueError:
            continue
        if used_mib <= max(0, int(max_used_mib)) and util <= max(0, int(max_util)):
            free_mib = max(0, total_mib - used_mib)
            slots = max(0, free_mib // max(1, int(mib_per_process)))
            if slots > 0:
                candidates.append((index, slots))
    if not candidates:
        return []
    needed = max(1, int(required_slots or 1))
    selected: list[tuple[str, int]] = []
    total_slots = 0
    for item in sorted(candidates, key=lambda pair: pair[1], reverse=True):
        selected.append(item)
        total_slots += item[1]
        if total_slots >= needed:
            break
    counts = {index: 0 for index, _slots in selected}
    devices: list[str] = []
    for _ in range(min(needed, total_slots)):
        available = [(index, slots) for index, slots in selected if counts[index] < slots]
        if not available:
            break
        index, _slots = min(available, key=lambda pair: (counts[pair[0]], -pair[1], pair[0]))
        counts[index] += 1
        devices.append(index)
    return devices


def shard_completed(seed: int, variant: str, root: Path) -> bool:
    variant_root = root / "semantic_runtime_train" if variant == "proposed_semantic_topology_marl" else root / "semantic_runtime_train" / variant
    checkpoint = variant_root / f"ippo_seed_{seed}" / checkpoint_name_for_variant(variant)
    result = root / "logs" / f"stage1_{variant}_seed_{seed}_result.json"
    return checkpoint.exists() and result.exists()


def checkpoint_name_for_variant(variant: str) -> str:
    return TRAINED_CHECKPOINT_FILES.get(str(variant), "masac_policy.pt")


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
        if not alive:
            return failed
        time.sleep(max(1.0, float(poll_s)))


def print_progress(processes: list[dict[str, Any]], alive: list[dict[str, Any]]) -> None:
    parts = []
    for item in processes:
        progress = next(
            (
                path
                for path in (
                    item["out"] / "train_progress.csv",
                    item["out"] / "reward_curve.csv",
                    item["out"] / "iql_eval_reward_curve.csv",
                )
                if path.exists()
            ),
            None,
        )
        if progress is None:
            parts.append(f"{item['variant']} seed {item['seed']}: no-progress")
            continue
        try:
            df = pd.read_csv(progress)
            episode = int(df["episode"].max()) if not df.empty and "episode" in df else -1
            recent = df.tail(10)
            success_column = "success_ratio" if "success_ratio" in recent else "succeeded"
            success = (
                float(pd.to_numeric(recent.get(success_column, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean())
                if not recent.empty
                else 0.0
            )
            parts.append(f"{item['variant']} seed {item['seed']}: rows={len(df)} episode={episode} avg10_success={success:.3f}")
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
from train_semantic_topology_marl import build_offline_env, _actor_policy_kwargs, _close_runtime_env

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
episodes = int(episodes_override if episodes_override is not None else cfg.get("training", {}).get("episodes", 100))
max_steps = int(max_steps_override if max_steps_override is not None else cfg.get("training", {}).get("max_steps", 100))

if variant in {"mappo_ctde", "iql_offline"}:
    scenario = dict(scenarios[0]) if scenarios else {"name": "default"}
    output_dir = root / "semantic_runtime_train" / variant / f"ippo_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_offline_env(
        cfg,
        seed=seed,
        max_steps=max_steps,
        scenario=scenario,
        service_role_sweep="full_hybrid",
        attach_runtime=True,
        baseline=variant,
    )
    try:
        observations = env.reset()
        marl_cfg = dict(cfg.get("marl", {}) or {})
        policy_kwargs = _actor_policy_kwargs(cfg, observations, env.config.semantic_scorer, seed)
        def fresh_env(offset):
            return build_offline_env(
                cfg,
                seed=seed * 100000 + int(offset),
                max_steps=max_steps,
                scenario=scenario,
                service_role_sweep="full_hybrid",
                attach_runtime=True,
                baseline=variant,
            )

        def close_semantic_env(item):
            _close_runtime_env(getattr(item, "env", None))

        if variant == "mappo_ctde":
            from airfogsim.lasdm.mappo_policy import MAPPOPolicy
            from airfogsim.lasdm.mappo_trainer import MAPPOTrainer

            policy = MAPPOPolicy(**policy_kwargs)
            trainer = MAPPOTrainer(
                env,
                policy,
                env_factory=fresh_env,
                close_env=close_semantic_env,
                gamma=float(marl_cfg.get("mappo_gamma", marl_cfg.get("masac_gamma", 0.99)) or 0.99),
                lam=float(marl_cfg.get("mappo_gae_lambda", 0.95) or 0.95),
                clip_eps=float(marl_cfg.get("mappo_clip_eps", 0.2) or 0.2),
                ppo_epochs=int(marl_cfg.get("mappo_ppo_epochs", 4) or 4),
                rollout_steps=int(marl_cfg.get("mappo_rollout_steps", 128) or 128),
                vf_coef=float(marl_cfg.get("mappo_vf_coef", 0.5) or 0.5),
                ent_coef=float(marl_cfg.get("mappo_ent_coef", 0.01) or 0.01),
                max_grad_norm=float(marl_cfg.get("mappo_max_grad_norm", 0.5) or 0.5),
            )
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
            checkpoint = output_dir / "mappo_policy.pt"
        else:
            from airfogsim.lasdm.iql_policy import IQLPolicy
            from airfogsim.lasdm.iql_trainer import IQLTrainer
            from airfogsim.lasdm.marl_policy import policy_from_name

            policy = IQLPolicy(
                **policy_kwargs,
                q_lr=float(marl_cfg.get("iql_q_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
                expectile=float(marl_cfg.get("iql_expectile", 0.7) or 0.7),
                beta=float(marl_cfg.get("iql_beta", 3.0) or 3.0),
                v_lr=float(marl_cfg.get("iql_v_lr", marl_cfg.get("masac_q_lr", marl_cfg.get("ippo_lr", 3e-4))) or 3e-4),
                tau=float(marl_cfg.get("iql_tau", marl_cfg.get("masac_tau", 0.005)) or 0.005),
            )
            behavior_policy = policy_from_name(str(marl_cfg.get("iql_behavior_policy", "utility_prior_with_exchange")), seed=seed)
            trainer = IQLTrainer(
                env,
                policy,
                behavior_policy=behavior_policy,
                env_factory=fresh_env,
                eval_env_factory=lambda: fresh_env(900000),
                close_env=close_semantic_env,
                gamma=float(marl_cfg.get("iql_gamma", marl_cfg.get("masac_gamma", 0.99)) or 0.99),
                tau=float(marl_cfg.get("iql_tau", marl_cfg.get("masac_tau", 0.005)) or 0.005),
                batch_size=int(marl_cfg.get("iql_batch_size", marl_cfg.get("masac_batch_size", 128)) or 128),
                replay_capacity=int(marl_cfg.get("iql_replay_capacity", marl_cfg.get("masac_replay_capacity", 20000)) or 20000),
                offline_updates=int(marl_cfg.get("iql_offline_updates", 0) or 0),
                updates_per_transition=float(marl_cfg.get("iql_updates_per_transition", 1.0) or 1.0),
                max_grad_norm=float(marl_cfg.get("iql_max_grad_norm", marl_cfg.get("masac_max_grad_norm", 1.0)) or 1.0),
                reward_scale=float(marl_cfg.get("iql_reward_scale", marl_cfg.get("masac_reward_scale", 1.0)) or 1.0),
                seed=seed,
            )
            rows = trainer.train(episodes=episodes, max_steps=max_steps, output_dir=str(output_dir))
            checkpoint = output_dir / "iql_policy.pt"
    finally:
        _close_runtime_env(getattr(env, "env", None))
    if not checkpoint.exists():
        raise RuntimeError(f"{variant} did not write checkpoint: {checkpoint}")
    run = {
        "completed": True,
        "baseline": variant,
        "algorithm": variant,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "scenario": str(scenario.get("name", "default")),
        "service_role_sweep": "full_hybrid",
        "checkpoint_path": str(checkpoint),
        "output_dir": str(output_dir),
        "last_metrics": rows[-1].to_dict() if rows else {},
    }
    (output_dir / "train_summary.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {"completed": True, "runs": [run], "output_dir": str(root / "semantic_runtime_train")}
else:
    result = train_semantic_ippo_variants_runtime(
        cfg,
        root / "semantic_runtime_train",
        [variant],
        [seed],
        scenarios,
        ["full_hybrid"],
        episodes,
        max_steps,
    )
(root / "logs" / f"stage1_{variant}_seed_{seed}_result.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))
"""


if __name__ == "__main__":
    raise SystemExit(main())
