from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize runtime calibration traces by run directory.")
    parser.add_argument("--root", required=True, help="Evaluation output root containing per-run directories.")
    parser.add_argument("--out", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [summarize_run(path) for path in sorted(run_dirs(root))]
    frame = pd.DataFrame(rows)
    if frame.empty:
        print(f"No runtime run directories found under {root}")
        return 1
    columns = [
        "baseline",
        "scenario",
        "seed",
        "service_nodes",
        "request_count",
        "success_ratio",
        "succeeded",
        "failed",
        "timed_out",
        "active",
        "arrival_count",
        "arrival_span_s",
        "arrival_gap_mean_s",
        "arrival_gap_cv",
        "completion_span_s",
        "completion_gap_cv",
        "finish_rate_sfc_per_s",
        "candidate_count_mean",
        "selected_count",
        "selected_compute_mean_s",
        "selected_route_mean_s",
        "selected_semantic_mismatch_mean_s",
        "selected_runtime_mean_s",
        "selected_budget_mean_s",
        "selected_slack_mean_s",
        "selected_rsu_ratio",
        "selected_mobile_ratio",
        "top_failure_reason",
        "last_failed_tasks",
        "last_done_tasks",
        "last_active_tasks",
    ]
    frame = frame[[column for column in columns if column in frame.columns]]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
    print(frame.to_string(index=False))
    return 0


def run_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "reward_curve.csv").exists()
    ]


def summarize_run(path: Path) -> dict[str, Any]:
    baseline, scenario, seed = parse_run_name(path.name)
    row: dict[str, Any] = {
        "run_dir": str(path),
        "baseline": baseline,
        "scenario": scenario,
        "seed": seed,
        "service_nodes": service_node_count_from_scenario(scenario),
        "request_count": request_count_from_scenario(scenario),
    }
    reward = read_csv(path / "reward_curve.csv")
    if not reward.empty:
        last = reward.iloc[-1]
        succeeded = to_float(last.get("succeeded"))
        failed = to_float(last.get("failed"))
        timed_out = to_float(last.get("timed_out"))
        active = to_float(last.get("active_graphs"))
        denom = max(1.0, succeeded + failed + timed_out + active)
        row.update(
            {
                "succeeded": succeeded,
                "failed": failed,
                "timed_out": timed_out,
                "active": active,
                "success_ratio": succeeded / denom,
            }
        )
        row.update(completion_stats(reward))
    candidates = read_csv(path / "semantic_candidate_trace.csv")
    if not candidates.empty:
        row.update(arrival_stats(candidates))
        row["candidate_count_mean"] = float(candidates["candidate_count"].mean()) if "candidate_count" in candidates else np.nan
    detail = read_csv(path / "semantic_candidate_detail_trace.csv")
    if not detail.empty:
        row.update(selected_candidate_stats(detail))
    lifecycle_path = path / "runtime_task_lifecycle_trace.csv"
    if lifecycle_path.exists() and lifecycle_path.stat().st_size > 0:
        row.update(lifecycle_stats(lifecycle_path))
    return row


def parse_run_name(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    if len(parts) >= 4:
        baseline = parts[0]
        scenario = parts[1]
        seed = parts[-1].replace("seed_", "")
        return baseline, scenario, seed
    return name, "", ""


def service_node_count_from_scenario(name: str) -> float:
    token = "service_nodes_"
    if token not in name:
        return np.nan
    suffix = name.split(token, 1)[1].split("_", 1)[0]
    return float(suffix) if suffix.isdigit() else np.nan


def request_count_from_scenario(name: str) -> float:
    token = "request_count_"
    if token not in name:
        return np.nan
    suffix = name.split(token, 1)[1].split("_", 1)[0]
    return float(suffix) if suffix.isdigit() else np.nan


def arrival_stats(candidates: pd.DataFrame) -> dict[str, float]:
    if "sfc_id" not in candidates or "time_s" not in candidates:
        return {}
    arrivals = candidates.groupby("sfc_id")["time_s"].min().sort_values().to_numpy(dtype=float)
    if arrivals.size == 0:
        return {"arrival_count": 0}
    gaps = np.diff(arrivals)
    return {
        "arrival_count": int(arrivals.size),
        "arrival_span_s": float(arrivals.max() - arrivals.min()) if arrivals.size else 0.0,
        "arrival_gap_mean_s": float(gaps.mean()) if gaps.size else np.nan,
        "arrival_gap_cv": float(gaps.std() / gaps.mean()) if gaps.size and gaps.mean() > 0 else np.nan,
    }


def completion_stats(reward: pd.DataFrame) -> dict[str, float]:
    if "step" not in reward:
        return {}
    frame = reward.copy()
    frame["time_s"] = pd.to_numeric(frame["step"], errors="coerce").fillna(0.0) * 0.1
    deltas = []
    for column in ("succeeded", "failed", "timed_out"):
        if column in frame:
            delta = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).diff().fillna(frame[column]).clip(lower=0.0)
            events = frame.loc[delta > 0, "time_s"].repeat(delta[delta > 0].astype(int))
            deltas.extend(float(item) for item in events)
    if not deltas:
        return {"completion_span_s": np.nan, "completion_gap_cv": np.nan, "finish_rate_sfc_per_s": 0.0}
    times = np.array(sorted(deltas), dtype=float)
    gaps = np.diff(times)
    span = max(1e-9, float(times.max() - times.min()))
    return {
        "completion_span_s": span,
        "completion_gap_cv": float(gaps.std() / gaps.mean()) if gaps.size and gaps.mean() > 0 else np.nan,
        "finish_rate_sfc_per_s": float(times.size / span),
    }


def selected_candidate_stats(detail: pd.DataFrame) -> dict[str, float]:
    if "selected" not in detail:
        return {}
    selected = detail[detail["selected"].astype(str).str.lower().isin({"true", "1"})].copy()
    if selected.empty:
        return {"selected_count": 0}
    semantic_score = pd.to_numeric(selected.get("semantic_score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    mismatch = 0.60 * (1.0 - semantic_score) ** 2
    node_type = selected.get("node_type", pd.Series(dtype=str)).astype(str)
    return {
        "selected_count": int(len(selected)),
        "selected_compute_mean_s": mean_column(selected, "estimated_compute_s"),
        "selected_route_mean_s": mean_column(selected, "route_tx_time_s"),
        "selected_semantic_mismatch_mean_s": float(mismatch.mean()),
        "selected_runtime_mean_s": mean_column(selected, "expected_runtime_penalty_s"),
        "selected_budget_mean_s": mean_column(selected, "function_budget_s"),
        "selected_slack_mean_s": mean_column(selected, "deadline_slack_s"),
        "selected_rsu_ratio": float((node_type == "rsu").mean()) if len(node_type) else np.nan,
        "selected_mobile_ratio": float(node_type.isin(["vehicle", "uav"]).mean()) if len(node_type) else np.nan,
    }


def lifecycle_stats(path: Path) -> dict[str, Any]:
    phase_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    latest_by_task: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(path, chunksize=500000, low_memory=False):
        if "phase" in chunk:
            for key, value in chunk["phase"].value_counts(dropna=False).items():
                phase_counts[str(key)] = phase_counts.get(str(key), 0) + int(value)
        if "failure_reason" in chunk:
            for key, value in chunk["failure_reason"].dropna().astype(str).value_counts().items():
                if key and key.lower() != "nan":
                    failure_counts[key] = failure_counts.get(key, 0) + int(value)
        if {"task_id", "time_s"}.issubset(chunk.columns):
            idx = chunk.groupby("task_id")["time_s"].idxmax()
            for record in chunk.loc[idx].to_dict("records"):
                latest_by_task[str(record.get("task_id"))] = record
    latest = pd.DataFrame(latest_by_task.values()) if latest_by_task else pd.DataFrame()
    top_failure = max(failure_counts, key=failure_counts.get) if failure_counts else ""
    if latest.empty or "substate" not in latest:
        return {"top_failure_reason": top_failure}
    substate = latest["substate"].astype(str)
    return {
        "top_failure_reason": top_failure,
        "last_failed_tasks": int((substate == "failed").sum()),
        "last_done_tasks": int((substate == "done").sum()),
        "last_active_tasks": int((~substate.isin(["failed", "done"])).sum()),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def mean_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def to_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
