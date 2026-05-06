from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RAW_ROOT = Path("experiment_artifacts/raw_data")
DEFAULT_CONFIG = Path("methods_baselines/lasdm/configs/semantic_topology_marl.yaml")
DEFAULT_REPAIR_CONFIG = Path("methods_baselines/lasdm/configs/semantic_topology_runtime_figures_aligned.yaml")
DEFAULT_EPISODES = 200
DEFAULT_SCENARIOS = (
    "semantic_runtime_probe_preprocess_only",
    "semantic_runtime_calibration_easy",
    "semantic_runtime_contention_stress",
    "distributed_service_discovery_calibrated",
    "semantic_ambiguity_calibrated",
)
DEFAULT_LEARNING_VARIANTS = (
    "proposed_semantic_topology_marl",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_no_exchange",
    "marl_no_temporal",
    "marl_no_cross_region",
)
OPTIONAL_MARL_COMPARISON_VARIANTS = ("mappo_ctde", "iql_offline")
DEFAULT_NONLEARNING_BASELINES = (
    "intra_region_only",
    "cross_region_auction",
    "pure_semantic_greedy_no_exchange",
    "local_semantic_runtime_greedy",
    "nsga2_semantic_qos",
    "topology_greedy",
    "centralized_planner",
)
DEFAULT_LEARNING_EVAL_BASELINES = (*DEFAULT_LEARNING_VARIANTS, *OPTIONAL_MARL_COMPARISON_VARIANTS)
CHECKPOINT_NAMES = ("masac_policy.pt", "mappo_policy.pt", "iql_policy.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Stage1 semantic-topology training and runtime evaluation status.")
    parser.add_argument("--root", type=Path, default=None, help="Stage1 root. Defaults to the newest experiment_artifacts/raw_data/stage1_v* directory.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Expected training episodes per learning variant.")
    parser.add_argument("--eval-seed-count", type=int, default=1, help="Expected eval seed count for each baseline/scenario.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Semantic topology config used to infer scenario names.")
    parser.add_argument("--repair-config", type=Path, default=DEFAULT_REPAIR_CONFIG, help="Optional runtime repair config merged over --config.")
    parser.add_argument("--watch-s", type=float, default=0.0, help="Refresh every N seconds. Omit or set 0 for one-shot output.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between watch refreshes.")
    parser.add_argument("--include-eval", action="store_true", help="Also print runtime evaluation tables.")
    parser.add_argument("--include-artifacts", action="store_true", help="Also print semantic artifact checks.")
    args = parser.parse_args()

    root = args.root or newest_stage1_root(DEFAULT_RAW_ROOT)
    if root is None:
        print("No stage1_v* root found under experiment_artifacts/raw_data", file=sys.stderr)
        return 2
    root = root.resolve()
    config = load_merged_config(args.config, args.repair_config)
    scenarios = scenario_names(config) or list(DEFAULT_SCENARIOS)

    while True:
        if args.watch_s > 0 and not args.no_clear:
            print("\033[2J\033[H", end="")
        print(
            render_status(
                root,
                expected_episodes=max(1, args.episodes),
                eval_seed_count=max(1, args.eval_seed_count),
                scenarios=scenarios,
                include_eval=bool(args.include_eval),
                include_artifacts=bool(args.include_artifacts),
            )
        )
        if args.watch_s <= 0:
            return 0
        time.sleep(max(1.0, float(args.watch_s)))


def newest_stage1_root(raw_root: Path) -> Path | None:
    candidates = [path for path in raw_root.glob("stage1_v*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def render_status(
    root: Path,
    expected_episodes: int,
    eval_seed_count: int,
    scenarios: Sequence[str],
    include_eval: bool = False,
    include_artifacts: bool = False,
) -> str:
    now = time.time()
    process_rows = ps_rows()
    root_tokens = {str(root)}
    try:
        root_tokens.add(str(root.relative_to(Path.cwd())))
    except ValueError:
        pass
    self_pid = str(os.getpid())
    active_lines = [
        row
        for row in process_rows
        if row.get("pid") != self_pid
        and "monitor_stage1_status.py" not in row.get("args", "")
        and any(token and token in row.get("args", "") for token in root_tokens)
    ]
    lines: list[str] = []
    lines.append(f"Root: {root}")
    lines.append(f"Generated: {datetime.fromtimestamp(now).strftime('%F %T')}")
    lines.append(f"Training completion threshold: first {expected_episodes} episodes")
    lines.append(f"Active related processes: {len(active_lines)}")
    for row in active_lines[:8]:
        lines.append(
            f"  PID {row['pid']:>7} {row['stat']:<4} elapsed={row['etime']:<9} cpu={row['pcpu']:>5} "
            f"cmd={shorten(row['args'], 120)}"
        )
    if len(active_lines) > 8:
        lines.append(f"  ... {len(active_lines) - 8} more")
    lines.append("")
    lines.extend(render_training(root, expected_episodes, process_rows, now))
    if include_eval:
        lines.append("")
        lines.extend(render_eval_suite(
            root / "semantic_runtime_eval",
            "Runtime Eval",
            [*DEFAULT_NONLEARNING_BASELINES, *DEFAULT_LEARNING_EVAL_BASELINES],
            expected_runs=(len(DEFAULT_NONLEARNING_BASELINES) + len(DEFAULT_LEARNING_EVAL_BASELINES)) * len(scenarios) * eval_seed_count,
            process_rows=process_rows,
            now=now,
        ))
        if (root / "nonlearning_runtime_eval").exists() or (root / "learning_runtime_eval").exists():
            lines.append("")
            lines.extend(render_eval_suite(
                root / "nonlearning_runtime_eval" / "semantic_runtime_eval",
                "Nonlearning Eval",
                DEFAULT_NONLEARNING_BASELINES,
                expected_runs=len(DEFAULT_NONLEARNING_BASELINES) * len(scenarios) * eval_seed_count,
                process_rows=process_rows,
                now=now,
            ))
            lines.append("")
            lines.extend(render_eval_suite(
                root / "learning_runtime_eval" / "semantic_runtime_eval",
                "Learning Eval",
                DEFAULT_LEARNING_EVAL_BASELINES,
                expected_runs=len(DEFAULT_LEARNING_EVAL_BASELINES) * len(scenarios) * eval_seed_count,
                process_rows=process_rows,
                now=now,
            ))
    if include_artifacts:
        lines.append("")
        lines.extend(render_artifacts(root))
    return "\n".join(lines)


def render_training(root: Path, expected_episodes: int, process_rows: Sequence[Mapping[str, str]], now: float) -> list[str]:
    title = "Training Progress"
    rows = []
    pid_entries = read_json(root / "stage1_training_pids.json")
    pid_by_variant: dict[tuple[str, str], int] = {}
    if isinstance(pid_entries, list):
        for item in pid_entries:
            if isinstance(item, Mapping):
                variant = str(item.get("variant", ""))
                seed = str(item.get("seed", ""))
                pid = safe_int(item.get("pid"))
                if variant and seed and pid is not None:
                    pid_by_variant[(variant, seed)] = pid

    for variant, seed in training_targets(root):
        seed_dir = training_seed_dir(root, variant, seed)
        progress_path = first_existing(seed_dir / "train_progress.csv", seed_dir / "reward_curve.csv")
        selection_path = seed_dir / "checkpoint_selection.csv"
        summary_path = seed_dir / "train_summary.json"
        rows.append(training_row(
            variant,
            seed,
            progress_path,
            selection_path,
            summary_path,
            expected_episodes,
            pid_by_variant.get((variant, seed)),
            process_rows,
            now,
        ))

    headers = [
        "variant",
        "state",
        "progress",
        "last_update",
        "ep",
        "last_sfc",
        "last_task",
        "last_reward",
        "best",
        "w10(sfc/task/r)",
        "avg(sfc/task/r)",
    ]
    return [title, table(headers, rows)]


def training_seed_dir(root: Path, variant: str, seed: str) -> Path:
    train_root = root / "semantic_runtime_train"
    if variant == "proposed_semantic_topology_marl":
        return train_root / f"ippo_seed_{seed}"
    return train_root / variant / f"ippo_seed_{seed}"


def training_row(
    variant: str,
    seed: str,
    progress_path: Path,
    selection_path: Path,
    summary_path: Path,
    expected_episodes: int,
    pid: int | None,
    process_rows: Sequence[Mapping[str, str]],
    now: float,
) -> list[str]:
    rows = read_csv_rows(progress_path)
    summary = read_json(summary_path)
    filtered = filter_expected_rows(rows, expected_episodes)
    latest = filtered[-1] if filtered else (rows[-1] if rows else {})
    latest_ep = safe_int(latest.get("episode")) if latest else None
    progress_num = 0
    if filtered:
        progress_num = min(expected_episodes, max(0, (latest_ep or 0) + 1))
    elif rows:
        progress_num = min(expected_episodes, len(rows))
    progress = f"{progress_num}/{expected_episodes} ({progress_num / expected_episodes:.0%})"
    threshold_met = progress_num >= expected_episodes
    state = "PENDING"
    if pid is not None and pid_alive(pid, process_rows):
        state = f"READY:{pid}" if threshold_met else f"RUN:{pid}"
    elif threshold_met:
        state = "DONE"
    elif checkpoint_exists(progress_path.parent) or isinstance(summary, Mapping):
        state = "CHECKPOINT_PARTIAL"
    elif rows:
        state = "PARTIAL"
    activity_paths = [progress_path, selection_path, summary_path]
    last_activity = newest_mtime(activity_paths)
    last_update = age(last_activity, now) if last_activity else "-"
    best = best_selection(selection_path, summary, progress_path.parent)
    return [
        variant,
        state,
        progress,
        last_update,
        str(latest_ep if latest_ep is not None else "-"),
        fmt_float(row_success(latest)),
        fmt_float(row_task_success(latest)),
        fmt_float(row_reward(latest)),
        best,
        metric_triplet(filtered or rows, tail_count=10),
        metric_triplet(filtered or rows, tail_count=None),
    ]


def filter_expected_rows(rows: Sequence[Mapping[str, str]], expected_episodes: int) -> list[Mapping[str, str]]:
    filtered = []
    for row in rows:
        episode = safe_int(row.get("episode"))
        if episode is None or episode < expected_episodes:
            filtered.append(row)
    return filtered


def terminal_episode_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    latest_by_episode: dict[int, Mapping[str, str]] = {}
    passthrough: list[Mapping[str, str]] = []
    for row in rows:
        episode = safe_int(row.get("episode"))
        if episode is None:
            passthrough.append(row)
            continue
        latest_by_episode[int(episode)] = row
    if latest_by_episode:
        return [latest_by_episode[key] for key in sorted(latest_by_episode)]
    return passthrough


def best_selection(path: Path, summary: Any, seed_dir: Path) -> str:
    rows = read_csv_rows(path)
    if not rows:
        checkpoint = first_existing(*(seed_dir / name for name in CHECKPOINT_NAMES))
        if checkpoint.exists():
            algorithm = str(summary.get("algorithm", checkpoint.stem) if isinstance(summary, Mapping) else checkpoint.stem)
            return f"{algorithm}:{checkpoint.name}"
        return "-"
    def key(row: Mapping[str, str]) -> float:
        value = safe_float(row.get("selection_score"))
        return value if value is not None else -float("inf")
    best = max(rows, key=key)
    source = str(best.get("checkpoint_source", "") or "?")
    episode = best.get("episode", "?")
    score = fmt_float(best.get("selection_score"))
    min_success = fmt_float(best.get("min_success_ratio"))
    return f"{source}@{episode}:{score}/{min_success}"


def metric_triplet(rows: Sequence[Mapping[str, str]], tail_count: int | None) -> str:
    if not rows:
        return "-"
    episode_rows = terminal_episode_rows(rows)
    tail = list(episode_rows[-tail_count:]) if tail_count is not None else list(episode_rows)
    success_values = [value for row in tail if (value := row_success(row)) is not None]
    task_values = [value for row in tail if (value := row_task_success(row)) is not None]
    reward_values = [value for row in tail if (value := row_reward(row)) is not None]
    if not success_values and not task_values and not reward_values:
        return "-"
    return f"{fmt_mean(success_values)}/{fmt_mean(task_values)}/{fmt_mean(reward_values)}"


def render_eval_suite(
    suite_root: Path,
    title: str,
    expected_baselines: Sequence[str],
    expected_runs: int,
    process_rows: Sequence[Mapping[str, str]],
    now: float,
) -> list[str]:
    run_paths = eval_run_paths(suite_root)
    runs = []
    for path in run_paths:
        item = read_json(path)
        if isinstance(item, Mapping):
            payload = dict(item)
            payload["_run_dir"] = str(path.parent)
            runs.append(payload)
    if not runs:
        runs = read_csv_rows(suite_root / "summary.csv")
    status = "PENDING"
    suite_args = str(suite_root.parent)
    if any(suite_args in row.get("args", "") for row in process_rows):
        status = "RUN"
    elif (suite_root / "summary.csv").exists() or len(runs) >= expected_runs:
        status = "DONE"
    elif runs:
        status = "PARTIAL"
    complete = f"{len(runs)}/{expected_runs} ({(len(runs) / max(1, expected_runs)):.0%})"
    last = newest_mtime([suite_root / "summary.csv", suite_root.parent / "complete_runtime_manifest.json", *run_paths])
    lines = [
        f"{title}: state={status} completion={complete} last_update={age(last, now) if last else '-'}",
    ]
    rows = eval_rows_by_baseline(runs, suite_root, expected_baselines)
    headers = ["baseline", "runs", "succ_avg", "min_succ", "task_avg", "soft_avg", "reward_avg", "timeouts"]
    lines.append(table(headers, rows))
    return lines


def eval_run_paths(suite_root: Path) -> list[Path]:
    paths = sorted(suite_root.glob("*/run.json"))
    if paths:
        return paths
    shard_root = suite_root.parent / "stage2_eval_shards"
    return sorted(shard_root.glob("seed_*/semantic_runtime_eval/*/run.json"))


def eval_rows_by_baseline(
    runs: Sequence[Mapping[str, Any]],
    suite_root: Path,
    expected_baselines: Sequence[str],
) -> list[list[str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("baseline", "unknown"))].append(run)
    rows = []
    baselines = list(dict.fromkeys([*expected_baselines, *sorted(grouped)]))
    for baseline in baselines:
        items = grouped.get(baseline, [])
        rewards = []
        for item in items:
            run_dir_text = str(item.get("_run_dir", "") or "")
            if run_dir_text:
                run_dir = Path(run_dir_text)
            else:
                scenario = str(item.get("scenario", ""))
                role = str(item.get("service_role_sweep", "full_hybrid") or "full_hybrid")
                seed = str(item.get("seed", "0"))
                run_dir = suite_root / f"{baseline}__{scenario}__{role}__seed_{seed}"
            reward = last_reward(run_dir / "reward_curve.csv")
            if reward is not None:
                rewards.append(reward)
        successes = [value for item in items if (value := safe_float(item.get("success_ratio"))) is not None]
        tasks = [value for item in items if (value := safe_float(item.get("task_success_ratio"))) is not None]
        soft = [
            value
            for item in items
            if (
                value := safe_float(
                    item.get("soft_completion_ratio", item.get("chain_progress_ratio_mean"))
                )
            )
            is not None
        ]
        timeouts = [value for item in items if (value := safe_float(item.get("timed_out"))) is not None]
        rows.append([
            baseline,
            str(len(items)),
            fmt_mean(successes),
            fmt_float(min(successes) if successes else None),
            fmt_mean(tasks),
            fmt_mean(soft),
            fmt_mean(rewards),
            fmt_mean(timeouts),
        ])
    return rows


def last_reward(path: Path) -> float | None:
    rows = read_csv_rows(path)
    if not rows:
        return None
    return row_reward(rows[-1])


def training_targets(root: Path) -> list[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()

    manifest = read_json(root / "stage1_training_manifest.json")
    if isinstance(manifest, Mapping):
        variants = [str(item) for item in manifest.get("variants", []) or []]
        seeds = [str(item) for item in manifest.get("requested_seeds", []) or []]
        for variant in variants:
            for seed in seeds:
                targets.add((variant, seed))
        for run in manifest.get("runs", []) or []:
            if isinstance(run, Mapping):
                variant = str(run.get("baseline", run.get("variant", "")) or "")
                seed = str(run.get("seed", "") or "")
                if variant and seed:
                    targets.add((variant, seed))

    complete_manifest = read_json(root / "complete_runtime_manifest.json")
    if isinstance(complete_manifest, Mapping):
        train_payload = complete_manifest.get("semantic_runtime_train", {})
        if isinstance(train_payload, Mapping):
            variants = [str(item) for item in train_payload.get("variants", []) or []]
            seeds = [str(item) for item in complete_manifest.get("train_seeds", []) or []]
            for variant in variants:
                for seed in seeds:
                    targets.add((variant, seed))
            for run in train_payload.get("runs", []) or []:
                if isinstance(run, Mapping):
                    variant = str(run.get("baseline", run.get("variant", "")) or "")
                    seed = str(run.get("seed", "") or "")
                    if variant and seed:
                        targets.add((variant, seed))

    pids = read_json(root / "stage1_training_pids.json")
    if isinstance(pids, list):
        for item in pids:
            if isinstance(item, Mapping):
                variant = str(item.get("variant", "") or "")
                seed = str(item.get("seed", "") or "")
                if variant and seed:
                    targets.add((variant, seed))

    train_root = root / "semantic_runtime_train"
    for seed_dir in sorted(train_root.glob("ippo_seed_*")):
        seed = seed_dir.name.removeprefix("ippo_seed_")
        targets.add(("proposed_semantic_topology_marl", seed))
    variant_dirs = sorted(path for path in train_root.iterdir() if path.is_dir()) if train_root.exists() else []
    for variant_dir in variant_dirs:
        if variant_dir.name.startswith("ippo_seed_"):
            continue
        for seed_dir in sorted(variant_dir.glob("ippo_seed_*")):
            seed = seed_dir.name.removeprefix("ippo_seed_")
            targets.add((variant_dir.name, seed))

    if not targets:
        for variant in DEFAULT_LEARNING_VARIANTS:
            targets.add((variant, "0"))

    order = {name: index for index, name in enumerate([*DEFAULT_LEARNING_VARIANTS, *OPTIONAL_MARL_COMPARISON_VARIANTS])}
    return sorted(targets, key=lambda item: (order.get(item[0], 999), safe_int(item[1]) if safe_int(item[1]) is not None else 999, item[0], item[1]))


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else Path("")


def checkpoint_exists(seed_dir: Path) -> bool:
    return any((seed_dir / name).exists() for name in CHECKPOINT_NAMES)


def row_success(row: Mapping[str, Any]) -> float | None:
    value = safe_float(row.get("success_ratio"))
    if value is not None:
        return value
    succeeded = safe_float(row.get("succeeded"))
    submitted = safe_float(row.get("submitted"))
    failed = safe_float(row.get("failed"))
    timed_out = safe_float(row.get("timed_out"))
    active = safe_float(row.get("active_graphs"))
    if succeeded is None:
        return None
    total = float(submitted) if submitted is not None else float(succeeded) + float(failed or 0.0) + float(timed_out or 0.0) + float(active or 0.0)
    return float(succeeded) / total if total > 0.0 else None


def row_task_success(row: Mapping[str, Any]) -> float | None:
    value = safe_float(row.get("task_success_ratio"))
    if value is not None:
        return value
    done = safe_float(row.get("task_done_num"))
    failed = safe_float(row.get("task_fail_num"))
    if done is None:
        return None
    total = float(done) + float(failed or 0.0)
    return float(done) / total if total > 0.0 else None


def row_reward(row: Mapping[str, Any]) -> float | None:
    value = safe_float(row.get("total_reward"))
    if value is not None:
        return value
    return safe_float(row.get("mean_reward"))


def render_artifacts(root: Path) -> list[str]:
    required = [
        "semantic_link_truth.csv",
        "io_compatibility_matrix.csv",
        "semantic_profile_pool.jsonl",
        "semantic_dataset_audit.json",
        "semantic_embedding_cache.pt",
        "semantic_embedding_report.json",
    ]
    rows = []
    audit = read_json(root / "semantic_dataset_audit.json")
    gate_summary = "-"
    if isinstance(audit, Mapping):
        gates = dict(audit.get("quality_gates", {}) or {})
        passed = sum(1 for value in gates.values() if bool(value))
        gate_summary = f"{passed}/{len(gates)}"
    for name in required:
        path = root / name
        rows.append([name, "yes" if path.exists() else "no", str(path.stat().st_size) if path.exists() else "-"])
    return ["V21 Semantic Artifacts", f"audit_quality_gates={gate_summary}", table(["artifact", "exists", "bytes"], rows)]


def load_merged_config(config_path: Path, repair_config_path: Path | None) -> dict[str, Any]:
    config = read_yaml(config_path)
    if repair_config_path is not None and repair_config_path.exists():
        config = deep_merge(config, read_yaml(repair_config_path))
    return config


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return dict(value) if isinstance(value, Mapping) else {}
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return {}


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in dict(overlay).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def scenario_names(config: Mapping[str, Any]) -> list[str]:
    scenarios = config.get("semantic_topology_experiment", {}).get("scenarios", []) if isinstance(config.get("semantic_topology_experiment"), Mapping) else []
    if not scenarios and isinstance(config.get("experiment"), Mapping):
        scenarios = config.get("experiment", {}).get("scenarios", [])
    names = []
    for scenario in scenarios or []:
        if isinstance(scenario, Mapping) and scenario.get("name"):
            if not bool(scenario.get("include_in_default", scenario.get("enabled", True))):
                continue
            names.append(str(scenario["name"]))
    return names


def ps_rows() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,pcpu,pmem,args"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        rows.append({
            "pid": parts[0],
            "ppid": parts[1],
            "stat": parts[2],
            "etime": parts[3],
            "pcpu": parts[4],
            "pmem": parts[5],
            "args": parts[6],
        })
    return rows


def pid_alive(pid: int, process_rows: Sequence[Mapping[str, str]]) -> bool:
    target = str(pid)
    return any(row.get("pid") == target for row in process_rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def newest_mtime(paths: Iterable[Path]) -> float | None:
    values = []
    for path in paths:
        try:
            if path.exists():
                values.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(values) if values else None


def path_age(path: Path, now: float) -> str:
    try:
        return age(path.stat().st_mtime, now) if path.exists() else "-"
    except OSError:
        return "-"


def age(timestamp: float, now: float) -> str:
    seconds = max(0, int(now - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m ago"


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fmt_float(value: Any) -> str:
    number = safe_float(value)
    return "-" if number is None else f"{number:.3f}"


def fmt_mean(values: Sequence[float]) -> str:
    return "-" if not values else f"{mean(values):.3f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    normalized = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], min(len(cell), 42))
    sep = "  "
    lines = [sep.join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append(sep.join("-" * width for width in widths))
    for row in normalized:
        lines.append(sep.join(shorten(cell, widths[index]).ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
