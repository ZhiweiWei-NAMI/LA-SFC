# Claims Vs Evidence

Latest paper-runtime suite:

- Raw runtime data: `experiment_artifacts/raw_data/lasdm_paper_runtime_final/lasdm_airfogsim/20260427_164344_459807/`
- Analysis artifacts: `analysis/runtime_final/`
- Figures: `analysis/runtime_final/figures/`
- Significance table: `analysis/runtime_final/significance_summary.csv`

Scope: 3 core baselines (`nearest_edge`, `centralized_greedy`, `proposed`) x 5 scenarios (`normal`, `high_mobility`, `load_burst`, `link_fault`, `airspace_event`) x 10 seeds (`0..9`) = 150 real AirFogSim/SUMO runtime runs.

Latest Semantic-Topology MARL runtime suite:

- Main raw runtime data: `experiment_artifacts/raw_data/semantic_topology_paper_matrix_v3/semantic_runtime_eval/`
- TTL/radius sweep data: `experiment_artifacts/raw_data/semantic_topology_ttl_radius_sweep_v3/semantic_runtime_eval/`
- Paper figures: `analysis/semantic_topology_paper/figures/`
- Figure tables: `analysis/semantic_topology_paper/figure_tables/`
- Significance table: `analysis/semantic_topology_paper/significance_summary.csv`
- Gate report: `analysis/semantic_topology_paper/semantic_topology_paper_gate.json`

Scope: 8 semantic/topology baselines x 2 non-calibration stress scenarios x 10 seeds = 160 real AirFogSim/SUMO runtime runs, plus a 3-baseline TTL/radius sweep with 81 runtime runs.

## Claim 1: LASDM enters and affects AirFogSim runtime execution.

### Evidence:

- All 150 runs completed in `airfogsim` mode with real `env.step()` execution.
- `analysis/runtime_final/function_execution_trace.csv` contains 1760 function-level runtime rows with selected node, route, tx delay, compute delay, queue delay, finish time, and status.
- Proposed makes a different runtime placement decision: it balances functions across `RSU_0` and `RSU_1`, while `nearest_edge` concentrates on `RSU_0` and `centralized_greedy` concentrates on `RSU_1`.
- Runtime differences are visible in compute and tx delay: proposed vs nearest_edge has significant differences for `avg_tx_delay` (`p=2.67923e-14`) and `avg_compute_delay` (`p=6.29896e-18`).

### Missing evidence:

- The current evidence covers the core 3-baseline runtime set, not the full 9-baseline matrix.

## Claim 2: LASDM significantly improves service-chain completion and deadline satisfaction.

### Evidence:

- `proposed`: graph completion `1.0`, deadline satisfaction `1.0`, failure rate `0.0`.
- `nearest_edge`: graph completion `0.5`, deadline satisfaction `0.6`, failure rate `0.5`.
- `centralized_greedy`: graph completion `0.4`, deadline satisfaction `0.6`, failure rate `0.6`.
- Significance tests support the improvement:
  - graph completion, proposed vs nearest_edge: `p=5.28418e-11`
  - graph completion, proposed vs centralized_greedy: `p=3.78511e-15`
  - deadline satisfaction, proposed vs nearest_edge: `p=6.40212e-07`
  - deadline satisfaction, proposed vs centralized_greedy: `p=6.40212e-07`

### Missing evidence:

- Full runtime significance for the remaining 6 baselines is still a later-stage requirement.

## Claim 3: LASDM reduces effective graph delay under resource contention.

### Evidence:

- Analysis now reports penalized graph delay: failed graphs are counted at their scenario deadline instead of as zero-delay completions.
- Penalized average delay: proposed `5.0s`, nearest_edge `5.9s`, centralized_greedy `6.0s`.
- Significance tests support delay reduction:
  - proposed vs nearest_edge: `p=4.19791e-13`
  - proposed vs centralized_greedy: `p=3.21319e-17`
- Function-level delay also differs significantly for proposed against both baselines.

### Missing evidence:

- More complex multi-hop/cloud-heavy workloads are still needed before claiming general end-to-end latency dominance.

## Claim 4: The system fails for the expected mechanism.

### Evidence:

- `analysis/runtime_final/failure_trace.json` contains 220 SFC-level failures.
- All observed failures are `deadline_missed`, matching the tuned runtime mechanism: baseline policies concentrate tasks on one RSU, causing compute contention and graph deadline misses.
- `task_success_ratio` and `function_success_ratio` are not significant (`p=1`) because AirFogSim tasks complete successfully; the failure is correctly at the SFC deadline layer.

### Missing evidence:

- This suite validates deadline-missed failures. Dedicated forced cases are still needed for `no_candidate`, link disconnect, energy exhaustion, and node moved away.

## Claim 5: Runtime overhead is low and measurable.

### Evidence:

- `analysis/runtime_final/runtime_overhead.csv` records decision overhead for all three baselines.
- Average decision time remains sub-millisecond to low-millisecond scale: centralized_greedy `0.783 ms`, nearest_edge `0.793 ms`, proposed `0.856 ms`.
- P95 decision time remains below `1.8 ms` for all three core baselines.

### Missing evidence:

- Scaling overhead for larger catalogs, more concurrent SFCs, and full multi-region orchestration remains untested.

## Claim 6: Results are stable across seeds and scenario stress.

### Evidence:

- The suite uses 10 seeds across all 5 scenarios.
- Proposed achieves completion `1.0` in every scenario.
- Baseline degradation is scenario-sensitive: both `nearest_edge` and `centralized_greedy` drop to `0.0` completion in `high_mobility` and `load_burst`, while partial completion remains in `link_fault` and `airspace_event`.

### Missing evidence:

- Stability is shown for the tuned core runtime suite only; confidence intervals and full 9-baseline results should be added before final submission.

## Claim 7: LASDM can be compared against Agentic baselines.

### Evidence:

- `analysis/runtime_final/comparison_lasdm_vs_agentic.csv` is generated with the standard schema.

### Missing evidence:

- This runtime-final suite does not include matched Agentic runtime runs. A LASDM-vs-Agentic claim still requires the same seeds, scenarios, runtime mode, and baseline scope on both sides.

## Claim 8: Semantic-topology MARL is validated in real AirFogSim runtime.

### Evidence:

- The main semantic matrix completed 160 runtime runs with all 8 baselines and 2 non-calibration stress scenarios.
- `proposed_semantic_topology_marl` loaded real IPPO checkpoints in all proposed eval rows; heuristic policies are not accepted for proposed eval.
- `runtime_task_lifecycle_trace.csv` includes real task lifecycle states with `done` rows, confirming the result is not a hidden timeout-only artifact.
- `semantic_topology_paper_gate.json` passes all checklist gates: non-degenerate scenarios, oracle upper bound, checkpoint loading, semantic discovery evidence, TTL/stale relation, topology-aware mobility benefit, non-diagnostic figures, and significance evidence.
- Strong metrics in `significance_summary.csv` include `success_ratio`, `qos_hit_ratio`, `task_success_ratio`, `avg_graph_finish_time`, and `selected_topology_risk`.

### Missing evidence:

- The current semantic suite supports the stress scenarios used here. Generalization to additional AirFogSim maps, larger catalogs, and longer IPPO training horizons remains a follow-up item.

## Claim 9: The semantic exchange and topology components contribute distinct effects.

### Evidence:

- `semantic_candidate_detail_trace.csv` supports F4 top-k semantic candidate quality with per-candidate rank, semantic score, topology risk, selected flag, stale flag, and local/remote source.
- The TTL/radius sweep supports F5: TTL=2 suppresses stale candidates, while TTL=6/10 exposes stale remote candidates; radius affects message payload/coverage cost and is shown with the discovery/staleness panel.
- Topology-aware policies outperform no-topology variants under mobility/staleness stress in the gate report.
- Proposed improves over semantic-only/no-topology failure modes on success, QoS hit, task success, and topology-risk metrics.

### Missing evidence:

- Semantic exchange improves remote discovery/candidate quality, but not every exchange-enabled heuristic improves completion. The paper should phrase this as a mechanism contribution and not claim that naive exchange is always beneficial.

## Current Paper-Readiness Verdict

The LASDM runtime suite supports a core superiority claim for `proposed` over `nearest_edge` and `centralized_greedy` under tuned AirFogSim runtime resource contention. The Semantic-Topology MARL suite now also has a complete runtime main matrix, paper-ready F1-F8 figures, passing checklist gate, and significance evidence for the specified stress scenarios. Remaining gaps are broader map/catalog generalization, explicit non-deadline failure-family tests, and matched LASDM-vs-Agentic runtime comparison.
