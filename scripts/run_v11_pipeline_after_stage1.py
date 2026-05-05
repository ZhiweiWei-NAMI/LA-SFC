from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for Stage1, then run Stage2 evaluation and figures_req plotting.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--poll-s", type=float, default=600.0)
    parser.add_argument("--stage2-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--figure-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--semantic-repair-config",
        default="methods_baselines/lasdm/configs/semantic_topology_runtime_figures_aligned.yaml",
    )
    args = parser.parse_args()

    root = Path(args.root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_manifest = root / "v11_pipeline_manifest.json"
    state: dict[str, Any] = {
        "root": str(root),
        "started_at": timestamp(),
        "steps": [],
        "completed": False,
    }
    write_json(pipeline_manifest, state)
    if not wait_for_stage1(root, float(args.poll_s), state, pipeline_manifest):
        return 1

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env.setdefault(key, "1")

    commands = [
        (
            "stage2_eval",
            [
                sys.executable,
                "-u",
                "scripts/run_stage2_eval_sharded.py",
                "--root",
                str(root),
                "--seeds",
                *[str(seed) for seed in args.stage2_seeds],
                "--seed-concurrency",
                "2",
                "--force",
                "--include-traces",
                "--semantic-repair-config",
                str(args.semantic_repair_config),
                "--poll-s",
                "300",
            ],
        ),
        (
            "figures_req_eval",
            [
                sys.executable,
                "-u",
                "scripts/run_figures_req_eval.py",
                "--output-root",
                str(root / "figures_req_eval"),
                "--checkpoint-root",
                str(root / "semantic_runtime_train"),
                "--seeds",
                *[str(seed) for seed in args.figure_seeds],
                "--max-steps",
                str(int(args.max_steps)),
                "--semantic-repair-config",
                str(args.semantic_repair_config),
                "--force",
            ],
        ),
        (
            "figures_req_plot",
            [
                sys.executable,
                "-u",
                "scripts/plot_figures_req.py",
                "--root",
                str(root),
                "--eval-dir",
                str(root / "figures_req_eval" / "semantic_runtime_eval"),
                "--train-dir",
                str(root / "semantic_runtime_train"),
                "--output-dir",
                str(root / "figures_req"),
                "--format",
                "both",
                "--config",
                str(args.semantic_repair_config),
            ],
        ),
    ]
    for name, command in commands:
        if run_step(name, command, env, log_dir, state, pipeline_manifest) != 0:
            state["completed"] = False
            state["failed_at"] = timestamp()
            write_json(pipeline_manifest, state)
            return 1
    state["completed"] = True
    state["finished_at"] = timestamp()
    write_json(pipeline_manifest, state)
    return 0


def wait_for_stage1(root: Path, poll_s: float, state: dict[str, Any], manifest_path: Path) -> bool:
    stage1_manifest = root / "stage1_training_manifest.json"
    stage1_pid_file = root / "stage1_nohup.pid"
    while True:
        if stage1_manifest.exists():
            try:
                payload = json.loads(stage1_manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                state["stage1_manifest_error"] = str(exc)
            else:
                if payload.get("completed"):
                    state["stage1_completed_at"] = timestamp()
                    write_json(manifest_path, state)
                    return True
                if payload.get("failed"):
                    state["stage1_failed"] = payload.get("failed")
                    state["completed"] = False
                    write_json(manifest_path, state)
                    return False
        if stage1_pid_file.exists():
            pid_text = stage1_pid_file.read_text(encoding="utf-8").strip()
            if pid_text and not process_alive(pid_text):
                state["stage1_process_missing"] = pid_text
                state["completed"] = False
                write_json(manifest_path, state)
                return False
        state["last_stage1_poll_at"] = timestamp()
        write_json(manifest_path, state)
        time.sleep(max(30.0, poll_s))


def run_step(
    name: str,
    command: Sequence[str],
    env: dict[str, str],
    log_dir: Path,
    state: dict[str, Any],
    manifest_path: Path,
) -> int:
    log_path = log_dir / f"v11_pipeline_{name}.log"
    step = {"name": name, "command": list(command), "log": str(log_path), "started_at": timestamp()}
    state["steps"].append(step)
    write_json(manifest_path, state)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(list(command), stdout=log_file, stderr=subprocess.STDOUT, env=env)
        step["pid"] = proc.pid
        write_json(manifest_path, state)
        code = proc.wait()
    step["returncode"] = code
    step["finished_at"] = timestamp()
    write_json(manifest_path, state)
    return int(code)


def process_alive(pid_text: str) -> bool:
    try:
        os.kill(int(pid_text), 0)
        return True
    except Exception:
        return False


def timestamp() -> str:
    return time.strftime("%F %T")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
