# LASDM + Agentic Orchestrator Design for AirFogSim

This document consolidates the 12 architecture explorations into a concrete extension plan for a LASDM declarative service model and Agentic Orchestrator on top of AirFogSim. The implementation should keep AirFogSim's internal physical simulation order intact and use the existing `scheduleStep(env) -> env.step()` contract as the primary integration surface.

## Architecture Decision

The MVP adds a new independent package, `airfogsim/lasdm`, with:

- Declarative SFC model: `LASDMServiceChain`, `LASDMSFCNode`, `LASDMQoS`.
- Service instance directory: `ServiceInstanceDirectory`, `ServiceInstance`.
- Orchestrator: capability/QoS/resource-aware placement via `LASDMOrchestrator`.
- Lifecycle manager and metrics: `LASDMManager`, `LASDMMetrics`.
- MARL interface: `LASDMMARLInterface` for observation, action decoding, and reward shaping.
- AirFogSim-compatible facade: `LASDMScheduler.scheduleStep(env)`.

The MVP deliberately does not mutate `AirFogSimEnv.step()`. The full version can later add optional step-finalized collectors and richer hooks.

## Existing Concept Mapping

| LASDM concept | Existing AirFogSim mapping | Extension target |
| --- | --- | --- |
| SFC graph | `ServiceGraphSpec`, `TaskManager` DAG | `LASDMServiceChain` with graph status and QoS |
| Microservice | `MicroserviceSpec` | `LASDMSFCNode` and instance records |
| Service instance | `DeploymentRegistry.deployed[node] -> set(ms_id)` | `ServiceInstanceDirectory` with capacity, health, trust, region |
| Region | nearest RSU in `RegionalServiceAgent` | explicit region/context manager |
| Orchestrator | `AgenticServiceAlgorithmModule.scheduleStep` | LASDM manager plus policy/MARL bridge |
| MARL agent | DQN/MADDPG task offloading modules | service placement/deployment agents |
| QoS | `deadline_s`, `priority` on tasks | reliability, accuracy, energy, continuity |
| Dynamic network | `ChannelManagerCP`, `WiredNetworkManager` | unified `LinkState` and fault events |

## Subagent 0: Overall Architecture Audit

1. **Relevant files/classes/functions**  
   `airfogsim_env.py::AirFogSimEnv.step`, `_updateTask`, `_updateWirelessCommunication`, `_updateWiredCommunication`, `_updateComputation`; `airfogsim_algorithm.py::BaseAlgorithmModule.scheduleStep`; schedulers `TaskScheduler.setTaskOffloading`, `CommunicationScheduler.setCommunicationWithRB`, `ComputationScheduler.setComputingCallBack`; `service_orchestration/agentic_service_algorithm.py`.

2. **Current capability**  
   AirFogSim is a two-stage loop: `algorithm.scheduleStep(env)` writes decisions, then `env.step()` consumes them. Decisions are held in `env.activated_offloading_tasks_with_RB_Nos`, `env.alloc_cpu_callback`, and `env.task_return_routes`.

3. **Missing capability/gap**  
   No typed decision protocol, no structured event bus, weak step hook model, and tests are mostly stub/smoke rather than full `AirFogSimEnv.step()`.

4. **Required extension**  
   Add a typed scheduling protocol, lifecycle events, and structured metrics while preserving compatibility with the existing raw env fields.

5. **Integration point in env.step**  
   Primary: pre-step `scheduleStep(env)`. Optional full-version hook: after `task_manager.checkTasks()` and before `clearDecisions()`.

6. **Minimal implementation plan**  
   Add `LASDMScheduler.scheduleStep(env)` that writes LASDM decisions and metrics onto `env` without changing the simulator core.

7. **Full implementation plan**  
   Introduce `StepContext`, pre/post stage hooks, JSONL event logging, and core step-finalized collectors.

8. **Metrics and logs**  
   Task state transitions, RB utilization, wired queue length, CPU allocation, graph status, and failure reasons.

9. **Validation tests**  
   Unit tests for decision protocol and smoke tests for `scheduleStep -> env.step` consumption.

10. **Risks and design trade-offs**  
   Core hooks improve observability but risk destabilizing simulator semantics; keep them optional.

## Subagent 1: LASDM Model, SFC DAG, QoS, Failure

1. **Relevant files/classes/functions**  
   `task_manager.py::_generateTaskDAG`, `checkTaskDependency`, `checkTasks`; `entities/task.py::Task`; `service_spec.py::QoSProfile`, `ServiceIntent`, `ServiceGraphSpec`; `service_runtime.py::AirFogServiceRuntime`.

2. **Current capability**  
   AirFogSim has native random task DAGs and service graphs. Deadline and priority map to task fields. Failure codes include deadline, TTI, node removal, parent failure, and malicious result.

3. **Missing capability/gap**  
   `reliability_min`, `accuracy_min`, and `max_energy_j` are declared but not enforced. Service graphs have no failed terminal state.

4. **Required extension**  
   Add graph states `RUNNING/SUCCEEDED/FAILED/TIMED_OUT`, failure reasons, and QoS checking at placement and runtime.

5. **Integration point in env.step**  
   Read done and failed tasks in `runtime.on_airfogsim_step_finished(env)`; later add a post-step collector.

6. **Minimal implementation plan**  
   Add graph-level status and failure metrics in LASDM manager.

7. **Full implementation plan**  
   Add retry, explicit alternate path, partial continuation, and explicit declarative graph input independent from random task DAGs.

8. **Metrics and logs**  
   Graph failure count, timeout count, QoS violation counts by field, per-microservice retry count.

9. **Validation tests**  
   Success graph, no-candidate failure, deadline timeout, and QoS filtering tests.

10. **Risks and design trade-offs**  
   Strict QoS makes results more realistic but can reduce feasibility and change baseline comparability.

## Subagent 2: Service Instance Directory and Selection

1. **Relevant files/classes/functions**  
   `service_catalog.py::ServiceCatalog`; `service_spec.py::MicroserviceSpec`; `service_matcher.py::RuleBasedServiceMatcher`; `deployment_registry.py::DeploymentRegistry`; `service_orchestrator.py::ServiceOrchestrator`; `regional_agent.py`.

2. **Current capability**  
   Catalog entries include capabilities, allowed node types, CPU, memory, storage, trust, cold start, and deployment cost. Orchestrator supports centralized and regional heuristics.

3. **Missing capability/gap**  
   Deployments are service-id sets, not instance records. There is no instance health, load, replica state, TTL, version, or resource reservation.

4. **Required extension**  
   Add `ServiceInstanceDirectory` with instance metadata, resource capacity/used, health, reliability, accuracy, trust, region, and concurrency.

5. **Integration point in env.step**  
   Use pre-step candidate selection in `scheduleServiceOffloading`; later sync instance health after task completion.

6. **Minimal implementation plan**  
   Implement in-memory directory and capability/QoS/resource filtering.

7. **Full implementation plan**  
   Add distributed advertisements, TTL, reservation tokens, and conflict resolution.

8. **Metrics and logs**  
   Candidate counts, reject reasons, instance load, stale advertisement drops, cross-region hit rate.

9. **Validation tests**  
   Capability filter, trust filter, resource reservation/release, no-candidate graph failure.

10. **Risks and design trade-offs**  
   Instance precision increases realism but adds synchronization and rollback complexity.

## Subagent 3: SFC Lifecycle and QoS Statistics

1. **Relevant files/classes/functions**  
   `TaskManager`, `MissionManager`, `TaskScheduler`, `MissionScheduler`, `AirFogServiceRuntime`, `OrchestrationMetrics`, `benchmark_runner._derive_summary_fields`.

2. **Current capability**  
   Tasks, missions, and service graphs have partial lifecycle tracking. Benchmark outputs graph completion and deadline satisfaction.

3. **Missing capability/gap**  
   SFC failure terminal state and structured lifecycle events are missing. Some task failure views can miss non-finished failures.

4. **Required extension**  
   Add lifecycle events for task, mission, and SFC graph; add terminal failure accounting and QoS buckets.

5. **Integration point in env.step**  
   MVP: LASDM manager lifecycle state. Full: collector after AirFogSim task checks.

6. **Minimal implementation plan**  
   `LASDMMetrics` records submitted/succeeded/failed/timed_out and reason counts.

7. **Full implementation plan**  
   JSONL event bus, step metrics snapshots, unified terminal counters.

8. **Metrics and logs**  
   `events.jsonl`, `step_metrics.jsonl`, `run_summary.json`.

9. **Validation tests**  
   Failure reason, timeout, no duplicate accounting, summary correctness.

10. **Risks and design trade-offs**  
   Full event logs are useful but can be large; default to summary mode.

## Subagent 4: Mobility

1. **Relevant files/classes/functions**  
   `TrafficManager.stepSimulation`, `updateVehicleMobilityPatterns`, `updateUAVMobilityPatterns`; `TrafficScheduler`; `EntityScheduler.getNeighborNodeInfosById`; `AirFogSimEnv._updateTraffics`.

2. **Current capability**  
   SUMO or tripinfo drives vehicles; UAVs use speed/angle/phi integration. Positions update before communication rate updates.

3. **Missing capability/gap**  
   No UAV trajectory file replay, no mobility prediction, no handover cost, no route stability penalty.

4. **Required extension**  
   Add mobility-aware features: predicted distance, link lifetime, route churn, handover cost.

5. **Integration point in env.step**  
   Decisions are written before `env.step`; `_updateTraffics` then applies movement before channel updates.

6. **Minimal implementation plan**  
   Add mobility penalty to orchestrator scoring using existing position/distance queries.

7. **Full implementation plan**  
   Add trajectory providers, waypoint following, spatial index, and handover hysteresis.

8. **Metrics and logs**  
   Route switch count, offload target churn, link lifetime error, mobility-induced failure count.

9. **Validation tests**  
   Stable route under low mobility, reroute under high mobility, deterministic trajectory replay.

10. **Risks and design trade-offs**  
   Prediction improves adaptation but can overfit mobility model bias.

## Subagent 5: Communication, Network Quality, Faults

1. **Relevant files/classes/functions**  
   `ChannelManagerCP.computeRate`, `activateLink`, `getRateByChannelType`; callbacks for pathloss/shadowing/fading/outage; `WiredNetworkManager`; `CommunicationScheduler`.

2. **Current capability**  
   Wireless RB-level rates and wired RSU/cloud links are modeled. Outage can zero RB rate. Routes can include RSU/cloud hops.

3. **Missing capability/gap**  
   Wired propagation delay, wired drop/fault/jitter, unified link state, and link-failure event injection are missing.

4. **Required extension**  
   Add `LinkState`, `NetworkFaultManager`, per-hop delay/loss/jitter, and scheduler link availability queries.

5. **Integration point in env.step**  
   Wireless faults after rate computation and before execution; wired faults inside `WiredNetworkManager.step`.

6. **Minimal implementation plan**  
   Add up/down link status and KPI counters.

7. **Full implementation plan**  
   Add multi-hop wired routing, queue caps, AQM/drop, stochastic faults, replay logs.

8. **Metrics and logs**  
   Per-link bytes, delay, loss, queue, outage ratio, fault recovery time.

9. **Validation tests**  
   Link down/up behavior, wired delay, zero-rate hard failure.

10. **Risks and design trade-offs**  
   Realistic network faults increase compute cost and require careful seeding.

## Subagent 6: Heterogeneous Resources and Deployment

1. **Relevant files/classes/functions**  
   `FogNode.getFogProfile`; `Task.compute`; `TaskManager.computeTasks`; `ComputationScheduler`; `StorageManager`; `DeploymentRegistry`; `ServiceOrchestrator`.

2. **Current capability**  
   CPU is allocated per step by callback. Memory and storage exist as static profiles. Service placement checks static capacity.

3. **Missing capability/gap**  
   No resource ledger, GPU, per-instance lifecycle, autoscaling, or migration.

4. **Required extension**  
   Add resource vectors, instance lifecycle, resource reservations, autoscaling and migration models.

5. **Integration point in env.step**  
   Refresh before placement; release after computation and node removal; retain warm deployments across steps.

6. **Minimal implementation plan**  
   The MVP directory reserves CPU/memory/storage per service instance.

7. **Full implementation plan**  
   Add CPU/GPU/memory/storage manager, scale-out/in, checkpoint transfer, and cloud rerouting.

8. **Metrics and logs**  
   CPU utilization, memory/storage used ratio, placement rejection, autoscale and migration events.

9. **Validation tests**  
   Resource exhaustion, release, GPU-only rejection, migration reserve-before-release.

10. **Risks and design trade-offs**  
   Retrofitting resource accounting can double-count without clear ownership semantics.

## Subagent 7: Energy, Storage, Trust, Security, Privacy

1. **Relevant files/classes/functions**  
   `EnergyManager`, `StorageManager`, `SimpleAuthManager`, `AuthScheduler`, `BlockchainManager`, `DeploymentRegistry.can_host`, `ServiceOrchestrator._score_node`.

2. **Current capability**  
   UAV energy, LRU cache, auth/trust, malicious result validation, and blockchain transactions exist.

3. **Missing capability/gap**  
   Trust filtering does not fully exclude malicious nodes; privacy/interception model is absent; energy and cache do not feed placement.

4. **Required extension**  
   Add unified node policy view: auth, trust, energy, storage, privacy, reputation, and reason-coded decisions.

5. **Integration point in env.step**  
   Pre-step candidate filtering; enforcement during auth, communication, storage, energy, and blockchain updates.

6. **Minimal implementation plan**  
   Filter unauthenticated or low-trust nodes and add energy/storage penalties.

7. **Full implementation plan**  
   Add privacy budgets, encryption overhead, interception risk, blockchain-backed reputation.

8. **Metrics and logs**  
   Auth-filter count, malicious result failures, energy at assignment/completion, privacy violations.

9. **Validation tests**  
   Malicious node exclusion, low-energy penalty, stateful storage rejection, blockchain transaction emission.

10. **Risks and design trade-offs**  
   Hard security filtering protects safety but may reduce service continuity.

## Subagent 8: Airspace Risk, Events, Regions

1. **Relevant files/classes/functions**  
   `RegionalServiceAgent`, `RuleBasedServiceMatcher`, `ServiceOrchestrator`, `MissionManager`, `examples/task_intents.py`.

2. **Current capability**  
   Regions are nearest-RSU domains. High `risk_level` inserts `conflict_prediction`. Regional agents exchange lightweight advertisements.

3. **Missing capability/gap**  
   No explicit region polygons, event engine, weather/no-fly/coverage zones, or event-aware route scoring.

4. **Required extension**  
   Add `AirspaceContextManager` with region risk, active events, adjacency, and context enrichment.

5. **Integration point in env.step**  
   Pre-step context enrichment in `scheduleStep`; full event updates before task/mission/communication phases.

6. **Minimal implementation plan**  
   Derive region from nearest RSU and inject risk/event fields into LASDM context.

7. **Full implementation plan**  
   Add mission-to-intent coupling, event-triggered graph reclassification, rich regional advertisements.

8. **Metrics and logs**  
   Risk timeline, event duration, adaptation latency, risk-triggered insertions, cross-region reroutes.

9. **Validation tests**  
   Region assignment, risk-triggered graph changes, remote lower-risk region selection.

10. **Risks and design trade-offs**  
   Rich event adaptation improves realism but complicates running graph migration semantics.

## Subagent 9: Agentic Orchestrator and MARL Interface

1. **Relevant files/classes/functions**  
   `BaseAlgorithmModule.scheduleStep`; DQN and MADDPG benchmarks; schedulers; `AgenticServiceAlgorithmModule`; `ServiceOrchestrator`; `OrchestrationMetrics`.

2. **Current capability**  
   RL examples exist for task offloading, but not service graph placement. Agentic service orchestration is rule-based.

3. **Missing capability/gap**  
   No Gym-style obs/action/reward interface for LASDM and no regional MARL boundary.

4. **Required extension**  
   Add observation builder, action decoder, and graph-level reward function.

5. **Integration point in env.step**  
   Wrap or replace service offloading in pre-step scheduling; reward is computed after next runtime completion sync.

6. **Minimal implementation plan**  
   `LASDMMARLInterface` exposes instance/graph observation, action decoding, and reward shaping.

7. **Full implementation plan**  
   Add centralized and per-region policies, replay recording, composition-level actions, RB/CPU control heads.

8. **Metrics and logs**  
   Invalid action rate, selected node-type distribution, action entropy, reward components.

9. **Validation tests**  
   Observation content, action decode, invalid action, terminal reward.

10. **Risks and design trade-offs**  
   Placement-first learning is tractable; joint composition and placement can explode the action space.

## Subagent 10: Experiments, Baselines, Statistics

1. **Relevant files/classes/functions**  
   `methods_baselines/benchmarks/agentic_service_orchestration/benchmark_runner.py`, `OrchestrationMetrics`, benchmark configs, tests.

2. **Current capability**  
   Smoke and AirFogSim benchmark modes, 9 baselines, manifest, per-run JSON, summary CSV, aggregate CSV.

3. **Missing capability/gap**  
   No paper statistics script, confidence intervals, paired tests, effect size, event metrics, or trace timelines.

4. **Required extension**  
   Add post-processing script with sample stats, CI, paper tables, and run schema.

5. **Integration point in env.step**  
   Capture pre-decision, post-decision, post-env-step, and final flush metrics around the benchmark loop.

6. **Minimal implementation plan**  
   Add `experiment_artifacts/plotting/analyze_lasdm_results.py` for summary CSV aggregation.

7. **Full implementation plan**  
   Add long-form trace data, paired seed tests, bootstrap CI, multiple-comparison correction, figure data.

8. **Metrics and logs**  
   Completion, deadline, latency, payload, cold start, deployment cost, cross-region forwarding, adaptation latency.

9. **Validation tests**  
   Synthetic summary stats correctness and schema validation.

10. **Risks and design trade-offs**  
   Rich traces aid paper analysis but can be expensive for long AirFogSim runs.

## Subagent 11: Reproducibility, Tests, Delivery

1. **Relevant files/classes/functions**  
   `pytest.ini`, `tests/`, benchmark configs, `benchmark_runner._set_seed`, `manifest.json`.

2. **Current capability**  
   Pytest path is configured. Smoke tests avoid SUMO. Benchmark outputs include manifest and run summaries.

3. **Missing capability/gap**  
   No CI config, no test markers, no artifact schema, partial seed contract, no dependency profile split.

4. **Required extension**  
   Add reproducibility contract, smoke/integration/GPU test tiers, schema validation, and environment metadata.

5. **Integration point in env.step**  
   Reproducibility should wrap benchmark and scheduler loops, not mutate the simulator step.

6. **Minimal implementation plan**  
   Add LASDM unit tests and deterministic config examples.

7. **Full implementation plan**  
   Add CI matrix, dependency splits, SUMO validation, dirty git status, Python/platform/dependency hashes.

8. **Metrics and logs**  
   Run duration, failure tracebacks, config/catalog/intents hashes, pass/fail flags.

9. **Validation tests**  
   Same-seed reproducibility, artifact schema, no repository output side effects.

10. **Risks and design trade-offs**  
   Strict schemas improve delivery quality but can slow exploratory metric evolution.

## Delivered MVP Files

- `airfogsim/lasdm/*`: model, instance directory, orchestrator, manager, scheduler, MARL interface, metrics, YAML API.
- `methods_baselines/lasdm/configs/lasdm_airfogsim.yaml`: minimal LASDM experiment config.
- `methods_baselines/lasdm/examples/lasdm_minimal_orchestration.py`: executable MVP example.
- `experiment_artifacts/plotting/analyze_lasdm_results.py`: summary CSV aggregation for paper tables.
- `methods_baselines/lasdm/tests/test_lasdm_extension.py`: unit tests for model, directory, manager, MARL, analysis helpers.

## Full Research Roadmap

1. Bridge LASDM decisions into current `service_orchestration` runtime tasks.
2. Add graph failure consumption from AirFogSim failed task queues.
3. Add resource ledger and instance lifecycle.
4. Add region/event/network-fault managers.
5. Add MARL wrapper and regional policy hooks.
6. Add event adaptation metrics and paper statistics pipeline.
