# Runtime Function Execution Trace

This note defines the minimal trace artifact for runtime behavior explainability:

`function_execution_trace.csv`

Required columns:

`baseline,seed,scenario,sfc_id,function_id,selected_node,route,tx_delay,compute_delay,queue_delay`

## Generator

Use the offline generator on a benchmark suite directory:

```bash
python analysis/scripts/generate_function_execution_trace.py \
  experiment_artifacts/raw_data/lasdm/lasdm_airfogsim/<timestamp>
```

or on a single run JSON:

```bash
python analysis/scripts/generate_function_execution_trace.py \
  experiment_artifacts/raw_data/lasdm/lasdm_airfogsim/<timestamp>/runs/proposed__normal__seed_0.json
```

The default output path is `<suite>/function_execution_trace.csv`. Use `--output <path>` to write elsewhere.

## Current Coverage

LASDM benchmark outputs already contain enough function-level planning data in each per-run JSON under `decisions`:

- `node_mapping`: `function_id -> selected_node`
- `assignments`: `function_id -> service_instance_id`
- `routes`: `function_id -> route`

The generator combines those fields with `manifest.json` `parsed_config.service_chains` and
`parsed_config.service_instances`.

For LASDM offline runs, delay fields are deterministic proxies because no packet-level simulator timeline is emitted:

- `tx_delay`: `route_hops * payload_mb * 0.02`
- `compute_delay`: `max(0.05, cpu_mb * 0.12 + memory_mb * 0.0005)`, matching the existing LASDM benchmark latency model
- `queue_delay`: `compute_delay * current_load / max_concurrency`, capped at one compute interval

## Agentic Runtime Hook Gap

Existing Agentic `agentic_service_orchestration` per-run JSON stores graph-level metrics only. It does not persist the
function task placement selected inside `ServiceOrchestrator.select_serving_node()` or the task route handed to
`TaskScheduler.setTaskOffloading()`, so the required trace cannot be reconstructed after the run.

The smallest non-core hook would be in benchmark/analysis code around
`AgenticServiceAlgorithmModule.scheduleServiceOffloading()` or `ServiceOrchestrator.select_serving_node()`:

- after `target_node_id, route = decision`, append one trace event with `baseline`, `seed`, scenario/mode,
  `service_graph_id`, `microservice_id`, `target_node_id`, `route`
- optionally add scheduler-observed timing when available from task state; otherwise record estimated delays from
  `_estimate_comm_delay()` and `getComputeDelayByNodeId()`
- write the events under `raw_metrics.function_execution_trace` or top-level `function_execution_trace`

The generator already accepts either of those explicit trace locations for Agentic or future LASDM runtime runs.
