from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class SemanticRuntimeGuardConfig:
    planner_baseline: str = "centralized_planner"
    calibration_scenario: str = "semantic_runtime_calibration_easy"
    min_planner_success_ratio: float = 0.50
    require_time_progress: bool = True
    require_runtime_steps: bool = True
    fail_on_all_zero_success: bool = True
    oracle_baseline: Optional[str] = None
    min_oracle_success_ratio: Optional[float] = None


def load_semantic_summary(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    candidates = [
        root / "ablation_summary.csv",
        root / "summary.csv",
        root / "semantic_runtime_summary.csv",
        root / "analysis" / "semantic_topology" / "ablation_summary.csv",
        root / "analysis" / "semantic_topology" / "summary.csv",
        root / "raw" / "semantic_runtime_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            for col in ["success_ratio", "qos_hit_ratio", "runtime_step_count", "simulation_time_end", "submitted", "succeeded", "timed_out"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    return pd.DataFrame()


def evaluate_guard(summary: pd.DataFrame, cfg: SemanticRuntimeGuardConfig = SemanticRuntimeGuardConfig()) -> Dict[str, Any]:
    if summary.empty:
        return {"passed": False, "reason": "semantic summary is empty", "details": {}}
    details: Dict[str, Any] = {"rows": int(len(summary))}
    success = pd.to_numeric(summary.get("success_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0)
    details["success_ratio_max"] = float(success.max()) if len(success) else 0.0
    details["success_ratio_mean"] = float(success.mean()) if len(success) else 0.0
    if cfg.fail_on_all_zero_success and details["success_ratio_max"] <= 0:
        return {"passed": False, "reason": "all semantic success ratios are zero", "details": details}

    calibration = summary.copy()
    planner_baseline = str(cfg.oracle_baseline or cfg.planner_baseline)
    min_success_ratio = (
        float(cfg.min_oracle_success_ratio)
        if cfg.min_oracle_success_ratio is not None
        else float(cfg.min_planner_success_ratio)
    )
    if "baseline" in calibration.columns:
        calibration = calibration[calibration["baseline"].astype(str).eq(planner_baseline)]
    if "scenario" in calibration.columns and cfg.calibration_scenario in set(calibration["scenario"].astype(str)):
        calibration = calibration[calibration["scenario"].astype(str).eq(cfg.calibration_scenario)]
    if calibration.empty:
        return {"passed": False, "reason": "centralized planner calibration rows are missing", "details": details}

    planner_success = pd.to_numeric(calibration.get("success_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0)
    details["centralized_planner_success_ratio_mean"] = float(planner_success.mean()) if len(planner_success) else 0.0
    if details["centralized_planner_success_ratio_mean"] < min_success_ratio:
        return {"passed": False, "reason": "centralized planner success guard failed", "details": details}

    if cfg.require_time_progress and "simulation_time_end" in calibration.columns:
        sim_time = pd.to_numeric(calibration["simulation_time_end"], errors="coerce").fillna(0)
        details["simulation_time_end_max"] = float(sim_time.max())
        if details["simulation_time_end_max"] <= 0:
            return {"passed": False, "reason": "simulation time did not advance", "details": details}

    if cfg.require_runtime_steps and "runtime_step_count" in calibration.columns:
        steps = pd.to_numeric(calibration["runtime_step_count"], errors="coerce").fillna(0)
        details["runtime_step_count_max"] = float(steps.max())
        if details["runtime_step_count_max"] <= 0:
            return {"passed": False, "reason": "runtime step counter did not advance", "details": details}

    return {"passed": True, "reason": "semantic runtime guard passed", "details": details}


def assert_guard(summary_path: str | Path, cfg: SemanticRuntimeGuardConfig = SemanticRuntimeGuardConfig()) -> Dict[str, Any]:
    result = evaluate_guard(load_semantic_summary(summary_path), cfg)
    if not result.get("passed"):
        raise RuntimeError(f"Semantic runtime guard failed: {result['reason']} | details={result.get('details', {})}")
    return result
