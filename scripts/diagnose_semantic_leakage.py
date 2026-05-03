#!/usr/bin/env python3
"""Diagnose why no_semantic can approach/beat Full in checkpoint selection.

Three hypotheses:
  1. Slot-position leakage: no_semantic benefits from semantic-first candidate ordering
  2. Full over-trusts semantic-high/topology-bad → more timeouts
  3. Semantic mismatch runtime cost is near zero → no real penalty for wrong semantic

Usage:
  python scripts/diagnose_semantic_leakage.py \
    --checkpoint-dir experiment_artifacts/raw_data/<root>/semantic_runtime_train \
    --variant-pair full=proposed_semantic_topology_marl,ablation=marl_no_semantic \
    --scenario semantic_runtime_contention_stress \
    --role full_hybrid \
    --seeds 0 1 2 \
    --max-steps 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "AirFogSim"))
sys.path.insert(0, str(WORKSPACE / "methods_baselines" / "lasdm"))


def load_config(config_path: str, repair_config_path: str | None) -> Dict[str, Any]:
    from run_complete_runtime_experiment import _load_semantic_config

    return _load_semantic_config(config_path, repair_config_path)


def run_diagnostic_episode(
    config: Dict[str, Any],
    scenario: Mapping[str, Any],
    role: str,
    seed: int,
    variant_name: str,
    variant_label: str,
    checkpoint_path: Path,
    max_steps: int = 80,
) -> List[Dict[str, Any]]:
    """Run one evaluation episode with candidate selection tracing."""
    from run_complete_runtime_experiment import (
        _build_semantic_runtime_env,
        _checkpoint_action_dim,
        _checkpoint_has_critic_body,
        _checkpoint_observation_dim,
        _config_with_baseline_updates,
        _ippo_policy_kwargs,
    )
    from airfogsim.lasdm.marl_policy import MASACPolicy
    from airfogsim.lasdm.graph_observation import flatten_observation

    env, air_env = _build_semantic_runtime_env(config, scenario, role, seed, variant_name, max_steps)
    observations = env.reset()
    policy_config = _config_with_baseline_updates(config, variant_name)

    # Build policy
    try:
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint_path), map_location="cpu")
    actor_state = state.get("actor", state) if isinstance(state, Mapping) else state
    obs_dim = _checkpoint_observation_dim(actor_state) or (
        max(len(flatten_observation(obs)) for obs in observations.values()) if observations else 1
    )
    max_candidates = _checkpoint_action_dim(actor_state) or int(config.get("marl", {}).get("max_candidates", 16))
    policy = MASACPolicy(
        **_ippo_policy_kwargs(policy_config, obs_dim, max_candidates, seed, observations=observations, state=actor_state),
        q_lr=float(config.get("marl", {}).get("masac_q_lr", config.get("marl", {}).get("ippo_lr", 3e-4)) or 3e-4),
        alpha=float(config.get("marl", {}).get("masac_alpha", 0.05) or 0.05),
        tau=float(config.get("marl", {}).get("masac_tau", 0.005) or 0.005),
    )
    policy.load_sac_state_dict(state, strict=_checkpoint_has_critic_body(actor_state))
    policy.model.eval()

    rows: List[Dict[str, Any]] = []
    for step in range(max_steps):
        action_observations = observations
        actions = policy.act(action_observations, deterministic=True)
        observations, rewards, done, info = env.step(actions)
        summary = info.get("summary", {})

        # Extract candidate selection diagnostics from env internals
        candidate_diag = _extract_candidate_diagnostics(action_observations, info.get("decisions", []) or [])
        for item in candidate_diag:
            item.update({
                "variant": variant_label,
                "variant_name": variant_name,
                "seed": seed,
                "step": step,
                "success_ratio": float(summary.get("success_ratio", 0) or 0),
                "timeout_ratio": float(summary.get("timeout_ratio", 0) or 0),
            })
        rows.extend(candidate_diag)

        if done:
            break

    # Also capture runtime trace diagnostics
    runtime_diag = _extract_runtime_diagnostics(env, variant_label, variant_name, seed)
    rows.extend(runtime_diag)

    if air_env is not None:
        try:
            from run_complete_runtime_experiment import _close_env
            _close_env(air_env)
        except Exception:
            pass

    return rows


def _extract_candidate_diagnostics(
    observations: Mapping[str, Mapping[str, Any]], decisions: List[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract per-selected-candidate diagnostics from env state."""
    rows = []
    for decision in decisions:
        sfc_id = str(decision.get("sfc_id", ""))
        diagnostics = dict(decision.get("diagnostics", {}) or {})
        selected_candidates = dict(diagnostics.get("selected_candidates", {}) or {})
        if not selected_candidates:
            continue

        for sfc_node_id, candidate in selected_candidates.items():
            metadata = dict(candidate.get("metadata", {}) or {})
            # Find slot index — look through all observations for this candidate
            slot_index = -1
            semantic_rank = -1
            all_candidates_for_node = []
            for obs in observations.values():
                for cs in obs.get("candidate_sets", []) or []:
                    if cs.get("sfc_id") == sfc_id and cs.get("sfc_node_id") == sfc_node_id:
                        raw = list(cs.get("raw_candidates", []) or [])
                        all_candidates_for_node = raw
                        for i, rc in enumerate(raw):
                            if rc.get("instance_id") == candidate.get("instance_id"):
                                slot_index = i
                                break
                        # Compute semantic rank (position if sorted by semantic_score desc)
                        sorted_by_sem = sorted(
                            enumerate(raw),
                            key=lambda x: float(x[1].get("semantic_score", 0)),
                            reverse=True,
                        )
                        for rank, (idx, rc) in enumerate(sorted_by_sem):
                            if rc.get("instance_id") == candidate.get("instance_id"):
                                semantic_rank = rank
                                break

            rows.append({
                "diagnostic_type": "candidate_selection",
                "sfc_id": sfc_id,
                "sfc_node_id": sfc_node_id,
                "instance_id": str(candidate.get("instance_id", "")),
                "slot_index": slot_index,
                "semantic_rank": semantic_rank,
                "num_candidates": len(all_candidates_for_node),
                "semantic_score": float(candidate.get("semantic_score", -1)),
                "semantic_group": str(metadata.get("semantic_group", "")),
                "is_decoy": str(metadata.get("is_decoy", "")),
                "is_remote": str(candidate.get("is_remote", "")),
                "staleness_s": float(candidate.get("staleness_s", 0)),
                "cold_start_s": float(metadata.get("cold_start_s", 0)),
                "route_available": str(metadata.get("route_available", "")),
                "deadline_slack_s": float(metadata.get("deadline_slack_s", 0) or 0),
                "route_hops": float(metadata.get("route_hops", -1)),
                "expected_runtime_penalty_s": float(metadata.get("expected_runtime_penalty_s", 0) or 0),
                "topology_risk": float(metadata.get("topology_risk", -1)),
                "load_ratio": float(candidate.get("load_ratio", -1)),
                "node_type": str(candidate.get("node_type", "")),
                "region_id": str(candidate.get("region_id", "")),
            })
    return rows


def _extract_runtime_diagnostics(env, variant_label: str, variant_name: str, seed: int) -> List[Dict[str, Any]]:
    """Extract runtime cost diagnostics from runtime_bridge task traces."""
    rows = []
    bridge = getattr(env, "runtime_bridge", None)
    if bridge is None:
        return rows

    task_traces = list(getattr(bridge, "function_execution_trace", None) or [])
    for task in task_traces:
        rows.append({
            "diagnostic_type": "runtime_cost",
            "variant": variant_label,
            "variant_name": variant_name,
            "seed": seed,
            "sfc_id": str(task.get("sfc_id", "")),
            "sfc_node_id": str(task.get("function_id", "")),
            "status": str(task.get("status", "")),
            "semantic_mismatch_runtime_cost_s": float(task.get("semantic_mismatch_runtime_cost_s", 0) or 0),
            "semantic_score": float(task.get("semantic_score", -1)),
            "semantic_min_score": float(task.get("semantic_min_score", -1)),
            "semantic_quality_violation": str(task.get("semantic_quality_violation", "")),
            "candidate_runtime_cost_s": float(task.get("candidate_runtime_cost_s", 0) or 0),
            "tx_delay": float(task.get("tx_delay", 0) or 0),
            "compute_delay": float(task.get("compute_delay", 0) or 0),
            "queue_delay": float(task.get("queue_delay", 0) or 0),
            "e2e_delay": float(task.get("e2e_delay", 0) or 0),
        })
    return rows


def analyze_results(all_rows: List[Dict[str, Any]], output_dir: Path):
    """Produce summary tables for the three hypotheses."""
    df = pd.DataFrame(all_rows)

    # Separate candidate selection from runtime cost rows
    sel = df[df["diagnostic_type"] == "candidate_selection"].copy()
    runtime = df[df["diagnostic_type"] == "runtime_cost"].copy()
    if sel.empty:
        print("No candidate selection data collected.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Hypothesis 1: Slot position leakage =====
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Slot-position semantic leakage")
    print("=" * 70)
    matched_sel = sel[sel["slot_index"] >= 0].copy()
    if matched_sel.empty:
        print("No selected candidates could be matched back to same-step candidate slots.")
        slot_stats = pd.DataFrame()
    else:
        match_rate = (sel["slot_index"] >= 0).groupby(sel["variant"]).mean().round(4)
        print("Same-step slot match rate:")
        print(match_rate.to_string())
        slot_stats = matched_sel.groupby("variant").agg(
        mean_slot_index=("slot_index", "mean"),
        median_slot=("slot_index", "median"),
        slot_0_ratio=("slot_index", lambda x: (x == 0).mean()),
        slot_01_ratio=("slot_index", lambda x: (x <= 1).mean()),
        mean_semantic_rank=("semantic_rank", "mean"),
        mean_semantic_score=("semantic_score", "mean"),
        ).round(4)
        print(slot_stats.to_string())
    print("\nInterpretation: If no_semantic mean_slot_index ≈ Full but mean_semantic_score still high,")
    print("  → slot-order leakage: no_semantic benefits from semantic-first ordering without seeing the score.")

    # ===== Hypothesis 2: Full over-selects semantic-high/topology-bad =====
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Semantic group selection comparison")
    print("=" * 70)
    if "semantic_group" in sel.columns:
        # Normalize semantic_group values
        sel["semantic_group_clean"] = sel["semantic_group"].fillna("real_non_decoy")
        sel.loc[sel["semantic_group_clean"].str.strip() == "", "semantic_group_clean"] = "real_non_decoy"

        group_pivot = sel.pivot_table(
            index="semantic_group_clean",
            columns="variant",
            values="instance_id",
            aggfunc="count",
            fill_value=0,
        )
        # Convert to percentages
        group_pct = group_pivot.div(group_pivot.sum(axis=0), axis=1) * 100
        print("\nSelection share per semantic group (%):")
        print(group_pct.round(2).to_string())
        print("\nInterpretation: If Full selects semantic_high_topology_bad or stale_remote at")
        print("  higher rates than no_semantic, Full may be over-trusting semantic signal.")

        # Per-group success/timeout
        if "success_ratio" in sel.columns:
            outcome_by_group = sel.groupby(["variant", "semantic_group_clean"]).agg(
                count=("instance_id", "count"),
                mean_success=("success_ratio", "mean"),
            ).round(4)
            print("\nPer-group mean success ratio:")
            print(outcome_by_group.to_string())

    # ===== Hypothesis 3: Semantic mismatch runtime cost =====
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Semantic mismatch runtime cost magnitude")
    print("=" * 70)
    if not runtime.empty:
        cost_cols = [
            "semantic_mismatch_runtime_cost_s",
            "candidate_runtime_cost_s",
            "cold_start_s",
            "tx_delay",
            "compute_delay",
            "queue_delay",
            "e2e_delay",
        ]
        existing = [c for c in cost_cols if c in runtime.columns]
        if existing:
            cost_stats = runtime.groupby("variant")[existing].agg(["mean", "max", "count"]).round(4)
            print(cost_stats.to_string())
            zero_mask = (runtime[existing] > 0).mean().round(4)
            print("\nFraction of tasks with non-zero runtime cost:")
            print(zero_mask.to_string())
    else:
        print("No runtime trace data collected. Check if traces are written during evaluation.")
        # Check if semantic cost is being applied at all
        sem_scores = sel[sel["semantic_score"] >= 0]["semantic_score"]
        if not sem_scores.empty:
            print(f"\nCandidate semantic_score stats: mean={sem_scores.mean():.4f}, "
                  f"min={sem_scores.min():.4f}, max={sem_scores.max():.4f}")
        deadline_slacks = sel[sel["deadline_slack_s"].notna() & (sel["deadline_slack_s"] > -100)]
        if not deadline_slacks.empty:
            slack = deadline_slacks["deadline_slack_s"]
            print(f"Deadline slack: mean={slack.mean():.1f}s, "
                  f"fraction_negative={(slack < 0).mean():.2%}")
            print("If slack is mostly >> 0, semantic mismatch cost can't cause timeout → Hypothesis 3 confirmed.")

    # Write raw data
    sel.to_csv(output_dir / "candidate_selection_diagnostics.csv", index=False)
    if not runtime.empty:
        runtime.to_csv(output_dir / "runtime_cost_diagnostics.csv", index=False)
    print(f"\nRaw diagnostics written to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Diagnose semantic leakage in MASAC training")
    parser.add_argument("--config", default=str(WORKSPACE / "methods_baselines/lasdm/configs/semantic_topology_marl.yaml"),
                        help="Base semantic topology YAML config")
    parser.add_argument("--repair-config", default=str(WORKSPACE / "methods_baselines/lasdm/configs/semantic_topology_runtime_repair.yaml"),
                        help="Runtime repair YAML overlay")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Training root directory (contains variant/seed subdirs)")
    parser.add_argument("--variant-pair", default="full=proposed_semantic_topology_marl,ablation=marl_no_semantic",
                        help="Format: full=<name>,ablation=<name>")
    parser.add_argument("--scenario", default="semantic_runtime_contention_stress",
                        help="Scenario to evaluate")
    parser.add_argument("--role", default="full_hybrid", help="Service role sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds to evaluate")
    parser.add_argument("--max-steps", type=int, default=80,
                        help="Max steps per episode")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for diagnostics CSV")
    args = parser.parse_args()

    # Parse variant pair
    pair = {}
    for item in args.variant_pair.split(","):
        k, v = item.strip().split("=")
        pair[k.strip()] = v.strip()
    full_name = pair.get("full", "proposed_semantic_topology_marl")
    ablation_name = pair.get("ablation", "marl_no_semantic")

    config = load_config(args.config, args.repair_config)
    checkpoint_root = Path(args.checkpoint_dir)
    from run_complete_runtime_experiment import _select_scenarios
    scenario_matches = _select_scenarios(config, [args.scenario])
    if not scenario_matches:
        raise SystemExit(f"Unknown scenario: {args.scenario}")
    scenario = dict(scenario_matches[0])

    output_dir = Path(args.output_dir or f"analysis/semantic_leakage_diag_{full_name}_vs_{ablation_name}")

    all_rows: List[Dict[str, Any]] = []

    for variant_label, variant_name in [("full", full_name), ("ablation", ablation_name)]:
        variant_subdir = checkpoint_root
        if variant_name != "proposed_semantic_topology_marl":
            variant_subdir = checkpoint_root / variant_name

        for seed in args.seeds:
            seed_dir = variant_subdir / f"ippo_seed_{seed}"
            if not seed_dir.exists():
                print(f"WARNING: {seed_dir} does not exist, skipping {variant_name} seed {seed}")
                continue

            # Find best checkpoint
            csv_path = seed_dir / "checkpoint_selection.csv"
            if not csv_path.exists():
                print(f"WARNING: No checkpoint_selection.csv in {seed_dir}, skipping")
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            best = df.sort_values("selection_score", ascending=False).iloc[0]
            best_ep = int(best["episode"])

            # Find checkpoint file
            pt_candidates = list(seed_dir.glob(f"masac_policy_best_ep{best_ep}*.pt"))
            if not pt_candidates:
                pt_candidates = list(seed_dir.glob("masac_policy*.pt"))
            if not pt_candidates:
                print(f"WARNING: No checkpoint .pt file in {seed_dir}, skipping")
                continue
            checkpoint_path = pt_candidates[0]

            print(f"Running {variant_label} ({variant_name}) seed={seed} ep={best_ep} "
                  f"checkpoint={checkpoint_path.name} ...")

            try:
                rows = run_diagnostic_episode(
                    config, scenario, args.role, seed,
                    variant_name, variant_label, checkpoint_path, args.max_steps,
                )
                all_rows.extend(rows)
                print(f"  → collected {len(rows)} diagnostic rows")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    if all_rows:
        analyze_results(all_rows, output_dir)
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
