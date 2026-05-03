from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether SAC fine-tuning improves over the BC topology-greedy student.")
    parser.add_argument("--train-root", required=True, help="semantic_runtime_train directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--required-sac-selected", type=int, default=3)
    args = parser.parse_args()

    train_root = Path(args.train_root)
    rows = collect_rows(train_root)
    output = Path(args.output) if args.output else train_root.parent / "bc_sac_uplift_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "seed",
        "bc_score",
        "bc_success_ratio",
        "bc_min_success_ratio",
        "best_source",
        "best_episode",
        "best_score",
        "best_raw_score",
        "best_success_ratio",
        "best_min_success_ratio",
        "sac_delta_score",
        "sac_delta_pct",
        "sac_selected",
        "validation_seed_success_non_decrease_count",
        "training_completed",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summary = []
    for variant, items in sorted(by_variant.items()):
        completed = [row for row in items if int(row["training_completed"])]
        sac_selected = sum(int(row["sac_selected"]) for row in completed)
        deltas = [float(row["sac_delta_score"]) for row in completed]
        summary.append(
            {
                "variant": variant,
                "seed_count": len(items),
                "completed_seed_count": len(completed),
                "sac_selected_count": sac_selected,
                "required_sac_selected": int(args.required_sac_selected),
                "majority_sac_selected": len(completed) == len(items) and sac_selected >= int(args.required_sac_selected),
                "mean_delta_score": mean(deltas) if deltas else 0.0,
                "min_delta_score": min(deltas) if deltas else 0.0,
                "max_delta_score": max(deltas) if deltas else 0.0,
            }
        )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(summary_path), "rows": len(rows)}, indent=2))
    return 0


def collect_rows(train_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(train_root.glob("**/ippo_seed_*")):
        if not seed_dir.is_dir():
            continue
        variant = variant_name(train_root, seed_dir)
        seed = seed_dir.name.replace("ippo_seed_", "")
        checkpoint_rows = read_csv(seed_dir / "checkpoint_selection.csv")
        if not checkpoint_rows:
            continue
        bc_row = next((row for row in checkpoint_rows if str(row.get("checkpoint_source", "")) == "bc"), checkpoint_rows[0])
        summary_path = seed_dir / "train_summary.json"
        best = {}
        training_completed = summary_path.exists() and (seed_dir / "masac_policy.pt").exists()
        if summary_path.exists():
            try:
                best = dict(json.loads(summary_path.read_text(encoding="utf-8")).get("best_selection", {}) or {})
            except Exception:
                best = {}
        if not best:
            eligible = [row for row in checkpoint_rows if str(row.get("eligible_for_best_checkpoint", "")) in {"1", "True", "true"}]
            best = max(eligible or checkpoint_rows, key=lambda row: to_float(row.get("selection_score")))
        bc_score = to_float(bc_row.get("selection_score", bc_row.get("raw_selection_score")))
        best_score = to_float(best.get("selection_score", best.get("raw_selection_score")))
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "bc_score": bc_score,
                "bc_success_ratio": to_float(bc_row.get("success_ratio")),
                "bc_min_success_ratio": to_float(bc_row.get("min_success_ratio")),
                "best_source": str(best.get("checkpoint_source", "")),
                "best_episode": best.get("episode", ""),
                "best_score": best_score,
                "best_raw_score": to_float(best.get("raw_selection_score", best.get("selection_score"))),
                "best_success_ratio": to_float(best.get("success_ratio")),
                "best_min_success_ratio": to_float(best.get("min_success_ratio")),
                "sac_delta_score": best_score - bc_score,
                "sac_delta_pct": (best_score / bc_score - 1.0) if bc_score else 0.0,
                "sac_selected": int(str(best.get("checkpoint_source", "")) == "sac"),
                "validation_seed_success_non_decrease_count": best.get("validation_seed_success_non_decrease_count", ""),
                "training_completed": int(training_completed),
            }
        )
    return rows


def variant_name(train_root: Path, seed_dir: Path) -> str:
    parent = seed_dir.parent
    if parent == train_root:
        return "proposed_semantic_topology_marl"
    return parent.name


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
