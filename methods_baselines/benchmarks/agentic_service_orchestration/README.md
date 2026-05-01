# Agentic Service Orchestration Benchmarks

This benchmark entrypoint reuses the new service orchestration stack and provides config-driven comparisons for:

- `fixed_sfc`
- `deployment_only`
- `nearest_edge`
- `cats_style`
- `centralized_greedy`
- `regional_distributed_greedy`
- `edge_only`
- `uav_only`
- `proposed`

The benchmark supports two modes:

- `smoke`: uses a lightweight environment from `airfogsim.service_orchestration.testing`
- `airfogsim`: uses a user-provided AirFogSim scenario config and Poisson intent generation

`edge_only` restricts serving to RSU/cloud nodes. `uav_only` prefers UAV nodes and falls back to catalog-eligible non-UAV nodes only when a service cannot legally run on UAVs.

The default config lives at `methods_baselines/benchmarks/agentic_service_orchestration/benchmark_config.yaml`.
The equivalent explicit full-smoke config lives at `methods_baselines/benchmarks/agentic_service_orchestration/configs/smoke_full.yaml`.
The local-map AirFogSim configs live at `methods_baselines/benchmarks/agentic_service_orchestration/configs/airfogsim_local_map_smoke.yaml` and `methods_baselines/benchmarks/agentic_service_orchestration/configs/airfogsim_local_map_full.yaml`.

Examples:

```bash
# Backward-compatible single smoke run
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py --baseline proposed

# Batch run using the default config
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py

# Override the output root for a run
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py \
  --baseline regional_distributed_greedy \
  --seed 0 \
  --output-root /tmp/aso-results

# Explicit full-smoke suite
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py \
  --config methods_baselines/benchmarks/agentic_service_orchestration/configs/smoke_full.yaml

# Local SUMO map, all baselines, one seed
python methods_baselines/benchmarks/agentic_service_orchestration/main_agentic_service_orchestration.py \
  --config methods_baselines/benchmarks/agentic_service_orchestration/configs/airfogsim_local_map_full.yaml
```

Each run writes results under:

- `experiment_artifacts/raw_data/agentic_service_orchestration/<experiment_name>/<timestamp>/manifest.json`
- `experiment_artifacts/raw_data/agentic_service_orchestration/<experiment_name>/<timestamp>/runs/<baseline>__seed_<seed>.json`
- `experiment_artifacts/raw_data/agentic_service_orchestration/<experiment_name>/<timestamp>/summary.csv`
- `experiment_artifacts/raw_data/agentic_service_orchestration/<experiment_name>/<timestamp>/aggregate.csv`
