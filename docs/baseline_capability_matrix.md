# Baseline Capability Matrix

`analysis/baseline_capability_matrix.csv` is the fairness audit table for current LASDM and Agentic baselines. It records candidate, resource, path, composition, region, and deployment capabilities without changing runtime code.

## Design

- `family`: `lasdm` or `agentic`.
- `baseline`: registered baseline name.
- `registered`: whether the row represents a currently registered baseline.
- `candidate_scope`: the universe searched before filters.
- `candidate_filters`: capability, semantic, node-type, trust, QoS, and resource filters applied before selection.
- `resource_capability`: whether the baseline has consumptive reservations or only static capacity gates.
- `path_capability`: route/path construction and routing information available to the baseline.
- `composition_capability`: whether service graph/SFC composition is fixed, adaptive, or an alias.
- `region_capability`: locality, regional-agent, or global-candidate behavior.
- `deployment_capability`: predeployment, lazy deployment, or pre-registered instance assumptions.
- `intentionally_limited`: `true` when the row is deliberately constrained for ablation or fairness comparison.
- `limitation_axis`: the restricted axis, such as `candidate_scope`, `composition_order`, `deployment_mode`, `region_path_bias`, `regional_information`, or `scoring_ablation`.
- `fairness_treatment`: how to interpret the row in aggregate tables.
- `source_refs`: implementation files used to derive the row.

## Current Fairness Rules

- Keep intentionally limited baselines in the matrix and in `summary.csv`; do not silently drop no-candidate or slow runs.
- Treat `edge_only` and LASDM `uav_only` no-candidate outcomes as expected restriction evidence, not runner failure.
- Treat Agentic `uav_only` as UAV-preferred rather than hard UAV-only.
- Do not double-count compatibility aliases such as `lasdm_static` and `lasdm_greedy` as independent paper methods unless the table explicitly reports aliases.
- Compare resource-sensitive baselines with `cold_start_count`, `deployment_cost`, candidate counts, and failure reason fields visible.

## Regeneration

```bash
python analysis/scripts/generate_baseline_capability_matrix.py
python analysis/scripts/generate_baseline_capability_matrix.py --check
```
