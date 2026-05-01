# LASDM AirFogSim Experiment Protocol

## Goal

Evaluate whether a LASDM declarative SFC model plus an Agentic Orchestrator improves service composition, placement, deployment, routing, and continuity under dynamic network quality, load, mobility, resource, trust, and airspace-risk changes.

## MVP Experiment

The MVP is intentionally lightweight and reproducible. It validates the new LASDM layer without requiring SUMO.

1. Load `methods_baselines/lasdm/configs/lasdm_airfogsim.yaml`.
2. Register service instances in `ServiceInstanceDirectory`.
3. Submit one declarative SFC.
4. Run `LASDMManager.step(current_time=0.0)`.
5. Verify accepted assignment and resource reservations.
6. Mark graph completion and inspect metrics.

Command:

```bash
python methods_baselines/lasdm/examples/lasdm_minimal_orchestration.py
pytest methods_baselines/lasdm/tests/test_lasdm_extension.py
```

MVP acceptance criteria:

- SFC DAG validates as acyclic.
- All required nodes receive instance assignments.
- QoS filters reject insufficient reliability/accuracy/trust.
- Resource reservations prevent overcommit.
- No-candidate SFC enters failed terminal state.
- Summary reports submitted, succeeded, failed, timeout, QoS hit ratio, and failure reasons.

## Minimal Runtime Integration Experiment

The minimal runtime experiment is the first AirFogSim-facing check. It answers a narrow question:

> Does the LASDM runtime bridge preserve correct SFC lifecycle behavior when one declarative service chain is converted into simulator-facing tasks and advanced through the AirFogSim loop?

This experiment verifies lifecycle, metrics, and artifact integrity. It should not be used as the full paper performance study.

Runtime loop:

```text
algorithm.scheduleStep(env)
env.step()
algorithm.runtime.on_airfogsim_step_finished(env)
```

Integration assumptions:

- `scheduleStep(env)` submits or updates SFCs and applies LASDM orchestration decisions.
- The runtime bridge creates ready AirFogSim function tasks from SFC nodes.
- The scheduler or manager applies selected instance and route decisions without changing the physics order inside `AirFogSimEnv.step()`.
- `env.step()` remains the simulator authority for task, mobility, communication, and resource progression.
- Runtime sync maps AirFogSim task completion, failure, and timeout back to LASDM node and SFC status after each step.
- Final metric flush happens after the last `env.step()`.

Minimal questions:

1. Does a valid LASDM SFC complete through the runtime bridge?
2. Are unavailable candidates reported as terminal graph failures with standardized reasons?
3. Are QoS filters and resource reservations visible in summaries and optional event logs?
4. Do baseline policies produce comparable `summary.csv` rows under identical seeds?
5. Does the artifact layout contain enough information to reproduce the run and regenerate statistics?

Minimal baselines:

- `lasdm_greedy`: LASDM directory filtering plus heuristic instance scoring.
- `lasdm_static`: fixed valid instance mapping, when available.
- `lasdm_random_valid`: random valid instance selection with fixed seeds.
- `no_lasdm_task_dag`: plain AirFogSim task DAG path, only if already available in the benchmark adapter.

Minimal scenarios:

- `normal`: all required instances are available and links are stable.
- `no_candidate`: one required capability or QoS condition cannot be satisfied.
- `link_fault`: a selected route or instance becomes unavailable after submission, once runtime failure mapping exists.
- `load_burst`: selected candidate lacks remaining resource after reservations, once resource sync exists.

Default non-SUMO checks:

```bash
python methods_baselines/lasdm/run_lasdm_benchmark.py \
  --baseline lasdm_greedy \
  --scenario normal \
  --seed 0 \
  --mode offline \
  --output-root experiment_artifacts/raw_data/lasdm_runtime_smoke
```

SUMO/TraCI gated integration checks:

```bash
python methods_baselines/lasdm/run_lasdm_benchmark.py \
  --baseline proposed \
  --scenario normal \
  --seed 0 \
  --mode airfogsim \
  --output-root experiment_artifacts/raw_data/lasdm_runtime_smoke
```

For the negative candidate-path check:

```bash
python methods_baselines/lasdm/run_lasdm_benchmark.py \
  --baseline edge_only \
  --scenario normal \
  --seed 0 \
  --mode airfogsim \
  --output-root experiment_artifacts/raw_data/lasdm_runtime_smoke
```

Statistics command:

```bash
python experiment_artifacts/plotting/analyze_lasdm_results.py \
  <benchmark_output_dir>/summary.csv \
  --output-dir <benchmark_output_dir>/analysis
```

`benchmark_output_dir` is printed by the benchmark runner and has the form
`<output-root>/lasdm_airfogsim/<timestamp>/`.

Minimal acceptance criteria:

- `normal` completes at least one graph for seed `0`.
- `no_candidate` produces a failed terminal SFC with a non-empty failure reason.
- Every baseline/scenario/seed combination emits one run JSON and one `summary.csv` row.
- `manifest.json` records config, seeds, scenarios, baselines, command, and git commit when available.
- `paper_aggregate.csv` and `paper_stats.json` can be regenerated from `summary.csv`.
- `summary.csv` includes both current adapter fields and paper-facing aliases, including `status`, `graph_submitted_count`, `avg_graph_finish_time`, `p95_graph_finish_time`, `p99_graph_finish_time`, `failure_reason`, `cold_start_count`, and `deployment_cost`.

## Full Research Experiment

The full version should bridge LASDM decisions through `LASDMScheduler.scheduleStep(env)` and use the existing AirFogSim benchmark loop:

```text
lasdm_scheduler.scheduleStep(env)
env.step()
runtime_bridge.sync_from_airfogsim_tasks(env)  # final flush at loop end
```

Do not change the default physics order inside `AirFogSimEnv.step()` unless using optional instrumentation hooks.

Paper-scale matrix:

- Seeds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` exactly for the main reported tables.
- Scenarios: `normal`, `high_mobility`, `load_burst`, `link_fault`, `airspace_event`.
- Baselines: `fixed_sfc`, `deployment_only`, `nearest_edge`, `cats_style`, `centralized_greedy`, `regional_distributed_greedy`, `edge_only`, `uav_only`, `proposed`.
- Main run count: 10 seeds x 5 scenarios x 9 baselines = 450 runs before any ablation or sensitivity study.
- Every baseline must use the same seed list and scenario definitions. Missing or failed runs remain explicit rows in `summary.csv`; they are not silently dropped.

## Research Questions

1. Does declarative LASDM SFC expressiveness outperform plain task DAGs?
2. Does semantic-aware instance selection improve QoS satisfaction?
3. Does dynamic orchestration outperform static and greedy baselines?
4. Does cloud-edge collaboration outperform cloud-first and edge-only schemes?
5. How far is distributed multi-region orchestration from a centralized oracle?
6. Does MARL dynamic path selection help under high network dynamics?
7. Does the Agentic Orchestrator adapt to load, risk, and mobility changes?

## Baselines

Use existing benchmark baselines where possible:

- `fixed_sfc`: fixed graph composition and centralized greedy placement.
- `deployment_only`: predeployment baseline.
- `nearest_edge`: nearest edge serving.
- `cats_style`: communication and compute delay score.
- `centralized_greedy`: global greedy heuristic.
- `regional_distributed_greedy`: RSU-region distributed heuristic.
- `edge_only`: no UAV serving.
- `uav_only`: UAV-preferred serving.
- `proposed`: adaptive rule-based graph plus regional serving.

Add LASDM-specific baselines in the full version:

- `lasdm_static`: declarative SFC with fixed instance mapping.
- `lasdm_greedy`: LASDM directory with heuristic scoring.
- `lasdm_random_valid`: random valid instance selection.
- `lasdm_marl_centralized`: centralized placement policy.
- `lasdm_marl_regional`: per-region MARL policy.

## Experiment Matrix

| Factor | MVP values | Full values |
| --- | --- | --- |
| Scenario | normal | normal, high_mobility, load_burst, link_fault, airspace_event |
| Baseline | LASDM greedy | fixed_sfc, deployment_only, nearest_edge, cats_style, centralized_greedy, regional_distributed_greedy, edge_only, uav_only, proposed |
| Seeds | 0 | 0-9, exactly 10 seeds for paper tables |
| Mobility | static smoke | SUMO, tripinfo, UAV trajectory replay |
| Network | fixed capacity | outage, link down/up, jitter, RSU-cloud congestion |
| Resources | static instances | autoscaling, migration, resource exhaustion |
| Risk | low/high context | region risk timeline and event engine |
| Security | trust filter | auth, malicious result, interception/privacy |

## Metrics

Minimal runtime metrics:

- graph submit count
- graph complete count
- graph failed count
- graph timeout count
- graph completion ratio
- deadline satisfaction ratio
- task success ratio, when the run uses the real AirFogSim task scheduler
- QoS hit ratio
- average, P95, and P99 graph finish time
- failure reason
- candidate count before and after filters
- reject reason counts
- cold start count
- deployment cost
- cross-region forward count
- payload transmitted MB

Full service graph metrics:

- graph submit count
- graph complete count
- graph failed count
- graph timeout count
- graph completion ratio
- deadline satisfaction ratio
- task success ratio
- all-QoS hit ratio
- average, P95, P99 graph finish time
- failure reason histogram

Composition metrics:

- missing capability ratio
- extra capability ratio
- wrong composition ratio
- risk-triggered insertion count
- alternate-branch count

Placement/deployment metrics:

- candidate count before and after filters
- reject reason counts
- cold start count
- deployment cost
- service instance utilization
- autoscale and migration events

Network metrics:

- payload transmitted MB
- per-hop bytes
- wireless outage ratio
- wired queue length
- link fault count
- adaptation latency after link degradation

Mobility and region metrics:

- route switch count
- offload target churn
- cross-region forward count
- per-region load imbalance
- region risk timeline

Security/resource metrics:

- trust/auth filtered candidates
- malicious result failures
- UAV energy at assignment and completion
- storage reservation failures
- privacy/interception violations

MARL metrics:

- invalid action rate
- defer count
- action entropy
- reward components
- replay sample count

## Data Recording

MVP:

- `LASDMMetrics.summary()` for run-level metrics.
- Optional `LASDMMetrics.write_events_jsonl(path)`.
- `experiment_artifacts/plotting/analyze_lasdm_results.py` for summary statistics.

Minimal runtime:

```text
experiment_artifacts/raw_data/lasdm_runtime_smoke/
  manifest.json
  summary.csv
  aggregate.csv
  runs/
    <baseline>__<scenario>__seed_<seed>.json
  events/
    <baseline>__<scenario>__seed_<seed>.jsonl
  step_metrics/
    <baseline>__<scenario>__seed_<seed>.jsonl
  analysis/
    paper_aggregate.csv
    paper_stats.json
```

Required `summary.csv` columns:

```text
baseline,scenario,seed,status,graph_submitted_count,graph_complete_count,
graph_failed_count,graph_timeout_count,graph_completion_ratio,
deadline_satisfaction_ratio,task_success_ratio,qos_hit_ratio,avg_graph_finish_time,
p95_graph_finish_time,p99_graph_finish_time,failure_reason,
payload_tx_mb,cold_start_count,deployment_cost,cross_region_forward_count
```

Recommended extra columns:

```text
candidate_count_before_filters,candidate_count_after_filters,
reject_reason_count,resource_reservation_failures,route_switch_count,
adaptation_latency,invalid_action_rate,reward_total
```

Full AirFogSim/SUMO:

- `manifest.json`: config, seed, git commit, environment metadata.
- `runs/<baseline>__seed_<seed>.json`: raw run metrics and flattened fields.
- `summary.csv`: one row per baseline/seed/scenario.
- `aggregate.csv`: simple means and std.
- `paper_aggregate.csv`: sample std, SEM, 95 percent CI, median, IQR, min, max.
- `events.jsonl`: task, graph, network, deployment, migration events.
- `step_metrics.jsonl`: per-step deltas and snapshots.

## Statistical Analysis

Current analyzer behavior:

- Reads one `summary.csv`.
- Groups rows by `baseline`.
- Computes `n`, mean, sample standard deviation, SEM, normal-approximation 95 percent CI, median, IQR, min, and max.
- Writes `paper_aggregate.csv` and `paper_stats.json`.
- Ignores non-numeric metric values and does not currently group by scenario.

Minimum paper analysis:

1. Use identical seed sets across baselines.
2. Report mean, sample standard deviation, SEM, and 95 percent CI.
3. Use paired comparisons by seed for each metric.
4. Report effect size against `proposed` or the LASDM method.
5. Report failed-run policy explicitly.

MVP analysis command:

```bash
python experiment_artifacts/plotting/analyze_lasdm_results.py \
  experiment_artifacts/raw_data/agentic_service_orchestration/<suite>/summary.csv \
  --output-dir experiment_artifacts/raw_data/lasdm_analysis
```

Gotchas:

- The current analyzer is descriptive, not a complete paper statistics pipeline.
- Paired seed comparisons, effect sizes, bootstrap intervals, and failed-run inclusion policies still need implementation before final paper claims.
- `qos_hit_ratio` and `all_qos_hit_ratio` should be normalized before final tables.
- Failed runs must be encoded consistently. Do not silently drop failed rows from `summary.csv`.
- Do not mix MVP smoke rows, minimal runtime rows, and full SUMO rows in the same aggregate without a `suite` or `scenario` column.

## Minimal Figures And Tables

- Table 1: baseline by scenario summary with completion ratio, QoS hit ratio, finish time, and failure count.
- Figure 1: completion ratio by baseline and scenario.
- Figure 2: graph finish time distribution by baseline for successful runs.
- Figure 3: failure reason histogram.
- Optional Figure 4: per-step SFC state timeline for one seed.

## Validation Tests

Default non-SUMO checks:

```bash
python methods_baselines/lasdm/examples/lasdm_minimal_orchestration.py
pytest -q methods_baselines/lasdm/tests
python methods_baselines/lasdm/run_lasdm_benchmark.py \
  --baseline lasdm_greedy \
  --scenario normal \
  --seed 0 \
  --output-root /tmp/lasdm-runtime-smoke
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py \
  --baseline proposed \
  --seed 0 \
  --output-root /tmp/airfogsim-service-smoke
```

These checks must not require SUMO, TraCI, generated traffic files, or a live simulator process.

SUMO/TraCI gated integration checks:

```bash
RUN_LASDM_SUMO_INTEGRATION=1 \
pytest -q methods_baselines/lasdm/tests/test_lasdm_real_runtime_acceptance.py
```

The gated test command requires a working SUMO binary, importable `traci`, and valid AirFogSim SUMO map/config files. It exercises the real AirFogSim loop through:

```text
AgenticServiceAlgorithmModule.scheduleStep(env)
env.step()
```

Unit coverage:

- SFC DAG validation and cycle rejection.
- Instance capability, semantic, trust, reliability, accuracy, resource filtering.
- Reservation and release.
- No-candidate graph failure.
- MARL observation and reward helpers.
- Analysis statistics from runtime-derived CSV.

Real runtime integration coverage:

- `AgenticServiceAlgorithmModule.scheduleStep -> env.step` with service tasks.
- Graph failure from failed AirFogSim task.
- Cloud route through RSU.
- Regional remote serving under local overload.
- Link fault and recovery.

Reproducibility tests:

- Same seed, same smoke output stable fields.
- Manifest schema.
- Run JSON schema.
- Config path resolution.
- SUMO tests gated behind integration marker.

## Risk Management

- Keep LASDM MVP independent of `AirFogSimEnv.step()` to avoid simulator regressions.
- Add resource reservations with rollback to prevent partial planning leaks.
- Treat strict security/QoS filters separately from soft penalties.
- Record every failure reason before making paper claims.
- Do not compare predeploy and lazy-deploy baselines without explicit cost accounting.
- Use smoke tests for CI and SUMO integration tests for nightly/manual runs.
