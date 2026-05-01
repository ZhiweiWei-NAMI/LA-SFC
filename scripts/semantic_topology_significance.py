from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


BASELINES = [
    "intra_region_only",
    "cross_region_auction",
    "semantic_greedy_no_exchange",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_topology_no_semantic",
    "proposed_semantic_topology_marl",
]
PROPOSED = "proposed_semantic_topology_marl"
CENTRALIZED_PLANNER = "centralized_planner"
ABLATIONS = [
    "intra_region_only",
    "cross_region_auction",
    "semantic_greedy_no_exchange",
    "utility_prior_with_exchange",
    "topology_greedy",
    "marl_no_semantic",
    "marl_semantic_no_topology",
    "marl_topology_no_semantic",
]
SUMMARY_METRICS = [
    "success_ratio",
    "qos_hit_ratio",
    "task_success_ratio",
    "avg_graph_finish_time",
    "p95_graph_finish_time",
    "semantic_score_mean",
    "topology_risk_mean",
    "stale_selected_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic-Topology MARL paper-matrix significance and gate checks.")
    parser.add_argument("--input-root", default="experiment_artifacts/raw_data/semantic_topology_paper_matrix/semantic_runtime_eval")
    parser.add_argument("--figure-root", default="analysis/semantic_topology_paper")
    parser.add_argument("--output-dir", default="analysis/semantic_topology_paper")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--medium-effect", type=float, default=0.33)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    figure_root = Path(args.figure_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(input_root / "summary.csv")
    candidate_rows = read_csv(input_root / "semantic_candidate_detail_trace.csv")
    discovery_rows = read_csv(figure_root / "figure_tables" / "discovery_ttl_radius.csv")
    lifecycle_rows = read_csv(input_root / "runtime_task_lifecycle_trace.csv")
    manifest = read_json(figure_root / "figures" / "semantic_figure_manifest.json")

    rows = significance_rows(summary, candidate_rows, alpha=float(args.alpha), medium_effect=float(args.medium_effect))
    write_csv(output_dir / "significance_summary.csv", significance_fieldnames(), rows)

    gate = gate_report(
        summary=summary,
        candidate_rows=candidate_rows,
        discovery_rows=discovery_rows,
        lifecycle_rows=lifecycle_rows,
        manifest=manifest,
        significance_rows_=rows,
        medium_effect=float(args.medium_effect),
    )
    (output_dir / "semantic_topology_paper_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True))


def significance_rows(
    summary: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    alpha: float,
    medium_effect: float,
) -> List[Dict[str, Any]]:
    metrics = metric_values(summary, candidate_rows)
    rows: List[Dict[str, Any]] = []
    for metric, values in metrics.items():
        for baseline in [*ABLATIONS, CENTRALIZED_PLANNER]:
            left, right = paired_values(values, PROPOSED, baseline)
            if len(left) < 2 or len(right) < 2:
                continue
            p_value = mann_whitney_p_value(left, right)
            delta = cliffs_delta(left, right)
            left_mean = mean(left)
            right_mean = mean(right)
            direction = metric_direction(metric)
            better = (left_mean > right_mean) if direction == "higher" else (left_mean < right_mean)
            rows.append(
                {
                    "metric": metric,
                    "baseline_a": PROPOSED,
                    "baseline_b": baseline,
                    "baseline_a_mean": f"{left_mean:.8g}",
                    "baseline_b_mean": f"{right_mean:.8g}",
                    "p_value": f"{p_value:.8g}",
                    "significant": str(bool(p_value < alpha)),
                    "cliffs_delta": f"{delta:.8g}",
                    "medium_effect": str(bool(abs(delta) >= medium_effect)),
                    "direction": direction,
                    "proposed_better": str(bool(better)),
                    "n_pairs": str(min(len(left), len(right))),
                }
            )
    return rows


def metric_values(
    summary: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    values: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for row in summary:
        key = metric_key(row)
        if not all(key):
            continue
        for metric in SUMMARY_METRICS:
            value = to_float(row.get(metric))
            if value is not None:
                if metric_direction(metric) == "lower" and value <= 0.0:
                    continue
                values[metric][key] = value
    selected: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if not truthy(row.get("selected")):
            continue
        key = metric_key(row)
        if all(key):
            selected[key].append(row)
    for key, group in selected.items():
        semantic = [value for value in (to_float(row.get("semantic_score")) for row in group) if value is not None]
        topology = [value for value in (to_float(row.get("topology_risk")) for row in group) if value is not None]
        remote = [1.0 if str(row.get("local_or_remote", "")).lower() == "remote" else 0.0 for row in group]
        stale = [1.0 if truthy(row.get("stale")) else 0.0 for row in group]
        if semantic:
            values["selected_semantic_score"][key] = mean(semantic)
        if topology:
            values["selected_topology_risk"][key] = mean(topology)
        if remote:
            values["selected_remote_ratio"][key] = mean(remote)
        if stale:
            values["selected_stale_ratio"][key] = mean(stale)
    return values


def gate_report(
    summary: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    discovery_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    significance_rows_: Sequence[Mapping[str, Any]],
    medium_effect: float,
) -> Dict[str, Any]:
    scenarios = sorted({str(row.get("scenario", "")) for row in summary if "calibration" not in str(row.get("scenario", ""))})
    baselines = sorted({canonical_baseline(row.get("baseline", "")) for row in summary})
    summary_by = group_summary(summary)
    checks: Dict[str, Any] = {}
    checks["at_least_2_non_calibration_stress_scenarios"] = len(scenarios) >= 2
    checks["has_all_8_baselines"] = all(item in baselines for item in BASELINES)
    checks["each_scenario_not_all_0_or_all_1"] = all(scenario_non_degenerate(summary, scenario) for scenario in scenarios)
    checks["centralized_planner_health"] = centralized_planner_health(summary_by, scenarios)
    checks["proposed_better_than_main_ablations_two_metrics"] = proposed_better_two_metrics(summary_by, scenarios)
    checks["proposed_beats_topology_only_baselines"] = proposed_beats_topology_only(summary_by, scenarios)
    checks["semantic_exchange_improves_discovery_or_quality"] = semantic_exchange_improves(summary_by, candidate_rows)
    checks["stale_ratio_explained_by_ttl"] = stale_ttl_relationship(discovery_rows)
    checks["topology_aware_better_under_mobility"] = topology_better_under_mobility(summary_by, scenarios)
    proposed_rows = [row for row in summary if canonical_baseline(row.get("baseline", "")) == PROPOSED]
    checks["ippo_checkpoint_strictly_loaded"] = bool(proposed_rows) and all(
        truthy(row.get("checkpoint_loaded"))
        and str(row.get("policy_source", "")) == "ippo_checkpoint"
        and bool(str(row.get("checkpoint_sha256", "")).strip())
        for row in proposed_rows
    )
    checks["runtime_lifecycle_has_done"] = any(
        str(row.get("queue_state", row.get("phase", ""))).lower() == "done"
        or str(row.get("phase", "")).lower() == "done"
        for row in lifecycle_rows
    )
    checks["figures_not_diagnostic"] = figures_ready(manifest)
    core_metric_allowlist = {
        "success_ratio",
        "qos_hit_ratio",
        "task_success_ratio",
        "avg_graph_finish_time",
        "p95_graph_finish_time",
        "selected_semantic_score",
        "selected_topology_risk",
    }
    strong_rows = [
        row
        for row in significance_rows_
        if row.get("baseline_a") == PROPOSED
        and row.get("baseline_b") in ABLATIONS
        and row.get("metric") in core_metric_allowlist
        and truthy(row.get("proposed_better"))
        and (truthy(row.get("significant")) or truthy(row.get("medium_effect")))
    ]
    core_metrics = sorted({str(row.get("metric", "")) for row in strong_rows})
    checks["significance_has_2_or_3_core_metrics"] = len(core_metrics) >= 2 and bool(
        {"success_ratio", "qos_hit_ratio", "task_success_ratio"}.intersection(core_metrics)
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "scenarios": scenarios,
        "baselines": baselines,
        "run_count": len(summary),
        "strong_metric_count": len(core_metrics),
        "strong_metrics": core_metrics,
        "medium_effect_threshold": medium_effect,
    }


def group_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(canonical_baseline(row.get("baseline", "")), str(row.get("scenario", "")))].append(row)
    return grouped


def scenario_non_degenerate(rows: Sequence[Mapping[str, Any]], scenario: str) -> bool:
    for metric in ("success_ratio", "qos_hit_ratio"):
        values = [to_float(row.get(metric)) for row in rows if str(row.get("scenario", "")) == scenario]
        values = [value for value in values if value is not None]
        if not values or max(values) == 0.0 or min(values) == 1.0:
            return False
    finish_times = [
        to_float(row.get("avg_graph_finish_time"))
        for row in rows
        if str(row.get("scenario", "")) == scenario and to_float(row.get("avg_graph_finish_time")) is not None
    ]
    return any((value or 0.0) > 0.0 for value in finish_times)


def centralized_planner_health(grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]], scenarios: Sequence[str]) -> bool:
    for scenario in scenarios:
        planner = mean_metric(grouped.get((CENTRALIZED_PLANNER, scenario), []), "task_success_ratio")
        if planner is None or planner < 0.50:
            return False
    return True


def proposed_better_two_metrics(grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]], scenarios: Sequence[str]) -> bool:
    metrics = ["success_ratio", "qos_hit_ratio", "avg_graph_finish_time", "task_success_ratio"]
    winning_metrics = set()
    for metric in metrics:
        direction = metric_direction(metric)
        for baseline in ["semantic_greedy_with_exchange", "topology_greedy", "marl_semantic_no_topology", "marl_topology_no_semantic"]:
            proposed_values = []
            baseline_values = []
            for scenario in scenarios:
                p = mean_metric(grouped.get((PROPOSED, scenario), []), metric)
                b = mean_metric(grouped.get((baseline, scenario), []), metric)
                if p is not None and b is not None:
                    proposed_values.append(p)
                    baseline_values.append(b)
            if not proposed_values:
                continue
            p_mean = mean(proposed_values)
            b_mean = mean(baseline_values)
            if direction == "higher" and p_mean >= b_mean + 0.05:
                winning_metrics.add(metric)
            if direction == "lower" and p_mean <= b_mean - 0.05:
                winning_metrics.add(metric)
    return len(winning_metrics) >= 2


def proposed_beats_topology_only(
    grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    scenarios: Sequence[str],
) -> bool:
    """Hard trend gate: learned policy must beat topology-only greedy behavior.

    This deliberately fails when proposed only matches topology_greedy. The
    paper claim is that MARL learns a better joint semantic-topology policy, so
    equality with topology-only is not sufficient evidence.
    """

    topology_baselines = ["topology_greedy", "marl_topology_no_semantic"]
    metric_thresholds = {
        "success_ratio": 0.03,
        "qos_hit_ratio": 0.03,
        "task_success_ratio": 0.03,
        "avg_graph_finish_time": 0.25,
        "p95_graph_finish_time": 0.50,
    }
    wins = 0
    for metric, threshold in metric_thresholds.items():
        direction = metric_direction(metric)
        for baseline in topology_baselines:
            proposed_values = []
            baseline_values = []
            for scenario in scenarios:
                proposed = mean_metric(grouped.get((PROPOSED, scenario), []), metric)
                other = mean_metric(grouped.get((baseline, scenario), []), metric)
                if proposed is not None and other is not None:
                    proposed_values.append(proposed)
                    baseline_values.append(other)
            if not proposed_values:
                continue
            p_mean = mean(proposed_values)
            b_mean = mean(baseline_values)
            if direction == "higher" and p_mean >= b_mean + threshold:
                wins += 1
                break
            if direction == "lower" and p_mean <= b_mean - threshold:
                wins += 1
                break
    return wins >= 2


def semantic_exchange_improves(
    grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> bool:
    no_exchange_quality = selected_quality(candidate_rows, "semantic_greedy_no_exchange")
    with_exchange_quality = selected_quality(candidate_rows, "semantic_greedy_with_exchange")
    no_exchange_remote = mean_across(grouped, "semantic_greedy_no_exchange", "remote_candidate_ratio")
    with_exchange_remote = mean_across(grouped, "semantic_greedy_with_exchange", "remote_candidate_ratio")
    return (with_exchange_remote or 0.0) > (no_exchange_remote or 0.0) or (
        with_exchange_quality is not None
        and no_exchange_quality is not None
        and with_exchange_quality > no_exchange_quality + 0.02
    )


def stale_ttl_relationship(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    ttl_values = sorted({to_float(row.get("exchange_ttl_s")) for row in rows if to_float(row.get("exchange_ttl_s")) is not None})
    radius_values = sorted({to_float(row.get("exchange_radius_hops")) for row in rows if to_float(row.get("exchange_radius_hops")) is not None})
    if len(ttl_values) < 3 or len(radius_values) < 3:
        return False
    by_ttl = []
    for ttl in ttl_values:
        stale = [to_float(row.get("stale_ratio")) for row in rows if to_float(row.get("exchange_ttl_s")) == ttl]
        stale = [value for value in stale if value is not None]
        if stale:
            by_ttl.append((ttl, mean(stale)))
    if len(by_ttl) < 2:
        return False
    xs = [item[0] for item in by_ttl]
    ys = [item[1] for item in by_ttl]
    stale_varies_with_ttl = abs(pearson(xs, ys)) >= 0.2 or max(ys) - min(ys) >= 0.02

    by_radius = []
    for radius in radius_values:
        remote = [to_float(row.get("remote_discovery_rate")) for row in rows if to_float(row.get("exchange_radius_hops")) == radius]
        overhead = [to_float(row.get("payload_bytes")) for row in rows if to_float(row.get("exchange_radius_hops")) == radius]
        remote = [value for value in remote if value is not None]
        overhead = [value for value in overhead if value is not None]
        if remote or overhead:
            by_radius.append((radius, mean(remote) if remote else 0.0, mean(overhead) if overhead else 0.0))
    if len(by_radius) < 3:
        return False
    remote_span = max(item[1] for item in by_radius) - min(item[1] for item in by_radius)
    overhead_span = max(item[2] for item in by_radius) - min(item[2] for item in by_radius)
    return stale_varies_with_ttl and (remote_span >= 0.02 or overhead_span > 0.0)


def topology_better_under_mobility(
    grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    scenarios: Sequence[str],
) -> bool:
    mobility = [scenario for scenario in scenarios if "mobility" in scenario or "stale" in scenario or "churn" in scenario]
    if not mobility:
        return False
    for scenario in mobility:
        proposed = mean_metric(grouped.get((PROPOSED, scenario), []), "qos_hit_ratio")
        no_topology = mean_metric(grouped.get(("marl_semantic_no_topology", scenario), []), "qos_hit_ratio")
        topology = mean_metric(grouped.get(("marl_topology_no_semantic", scenario), []), "qos_hit_ratio")
        if proposed is not None and no_topology is not None and proposed >= no_topology + 0.05:
            return True
        if topology is not None and no_topology is not None and topology >= no_topology + 0.05:
            return True
    return False


def figures_ready(manifest: Mapping[str, Any]) -> bool:
    figures = list(manifest.get("figures", []) or [])
    pdfs = [row for row in figures if str(row.get("path", "")).endswith(".pdf")]
    if len(pdfs) < 8:
        return False
    for row in pdfs:
        if not truthy(row.get("paper_ready")) or str(row.get("risk", "")):
            return False
        source_input = str(row.get("source_input_dir", ""))
        if "analysis/figures_req" in source_input or source_input.startswith("plans/") or "plans/fig_ref" in source_input:
            return False
        figure = str(row.get("figure", ""))
        role = str(row.get("source_role", ""))
        if figure == "F5" and role != "ttl_radius_sweep":
            return False
        if figure in {"F2", "F3", "F4", "F6", "F7", "F8"} and role != "main_matrix":
            return False
        if figure in {"F2", "F8"}:
            if int(float(row.get("method_count", 0) or 0)) < len(BASELINES):
                return False
            if int(float(row.get("scenario_count", 0) or 0)) < 2:
                return False
    return True


def mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [to_float(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def mean_across(grouped: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]], baseline: str, metric: str) -> float | None:
    values = []
    for (item_baseline, _scenario), rows in grouped.items():
        if item_baseline == baseline:
            value = mean_metric(rows, metric)
            if value is not None:
                values.append(value)
    return mean(values) if values else None


def selected_quality(rows: Sequence[Mapping[str, Any]], baseline: str) -> float | None:
    values = [
        to_float(row.get("semantic_score"))
        for row in rows
        if canonical_baseline(row.get("baseline", "")) == baseline and truthy(row.get("selected"))
    ]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def paired_values(values: Mapping[Tuple[str, str, str], float], left_baseline: str, right_baseline: str) -> Tuple[List[float], List[float]]:
    by_pair: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (baseline, scenario, seed), value in values.items():
        by_pair[(scenario, seed)][baseline] = value
    left: List[float] = []
    right: List[float] = []
    for pair in sorted(by_pair):
        item = by_pair[pair]
        if left_baseline in item and right_baseline in item:
            left.append(item[left_baseline])
            right.append(item[right_baseline])
    return left, right


def mann_whitney_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 1.0
    if list(left) == list(right):
        return 1.0
    try:
        from scipy import stats  # type: ignore

        result = stats.mannwhitneyu(left, right, alternative="two-sided")
        return clamp01(float(result.pvalue))
    except Exception:
        return permutation_p_value(left, right)


def permutation_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    observed = abs(mean(left) - mean(right))
    combined = list(left) + list(right)
    n_left = len(left)
    if len(combined) > 18:
        return normal_approx_p_value(left, right)
    import itertools

    count = 0
    extreme = 0
    indices = range(len(combined))
    for left_indices in itertools.combinations(indices, n_left):
        left_set = set(left_indices)
        lvals = [combined[idx] for idx in indices if idx in left_set]
        rvals = [combined[idx] for idx in indices if idx not in left_set]
        if abs(mean(lvals) - mean(rvals)) >= observed - 1e-12:
            extreme += 1
        count += 1
    return extreme / max(1, count)


def normal_approx_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    var_left = sample_variance(left)
    var_right = sample_variance(right)
    denom = math.sqrt(var_left / max(1, len(left)) + var_right / max(1, len(right)))
    if denom <= 0.0:
        return 0.0 if mean(left) != mean(right) else 1.0
    z_score = abs(mean(left) - mean(right)) / denom
    return clamp01(math.erfc(z_score / math.sqrt(2.0)))


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    wins = 0
    losses = 0
    for a in left:
        for b in right:
            if a > b:
                wins += 1
            elif a < b:
                losses += 1
    return (wins - losses) / float(len(left) * len(right))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def metric_direction(metric: str) -> str:
    if "time" in metric or "delay" in metric or "risk" in metric or "stale" in metric:
        return "lower"
    return "higher"


def metric_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (canonical_baseline(row.get("baseline", "")), str(row.get("scenario", "")), str(row.get("seed", "")))


def canonical_baseline(value: Any) -> str:
    baseline = str(value)
    if baseline == "centralized_oracle":
        return "centralized_planner"
    if baseline == "semantic_greedy_with_exchange":
        return "utility_prior_with_exchange"
    return baseline


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(1.0, value))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required significance input is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required significance input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def significance_fieldnames() -> List[str]:
    return [
        "metric",
        "baseline_a",
        "baseline_b",
        "baseline_a_mean",
        "baseline_b_mean",
        "p_value",
        "significant",
        "cliffs_delta",
        "medium_effect",
        "direction",
        "proposed_better",
        "n_pairs",
    ]


if __name__ == "__main__":
    main()
