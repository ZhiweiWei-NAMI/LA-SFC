# LASDM Failure Reason Contract

This note defines the Subagent 2 failure trace contract without changing runtime core code.
`failure_trace.json` is generated from existing benchmark artifacts by
`methods_baselines/lasdm/scripts/generate_failure_trace.py`.

## Output

The trace file uses schema `lasdm_failure_trace_v1` and contains:

- `mapping_contract`: the expected mapping for each paper-facing failure family.
- `runs`: per-run failure evidence reconstructed from run JSON, `summary.csv`, decisions, and future metrics events.
- `reason_histogram` and `family_histogram`: suite-level counts.
- `coverage`: whether each required family was observed and whether native AirFogSim codes support it.

The generator is intentionally tolerant. Current benchmark outputs often contain only summary counters, while future LASDM runtime outputs may include structured `raw_metrics.events` with `airfogsim_failure_code`, task id, SFC node id, and service type.

## Required Mapping Families

| Failure family | LASDM canonical reason | Existing AirFogSim native support | Current evidence | Gap |
| --- | --- | --- | --- | --- |
| `no_candidate` | `no_candidate` | Yes: `TASK_FAIL_OUT_OF_NODE` | LASDM `decision.rejected_reason`, `failure_reason_count`, native task code 2 | Native code conflates no node, lost node, and no executor. |
| `link_disconnect` | `deadline_missed` | Partial: `TASK_FAIL_OUT_OF_TTI` | Transmission timeout or textual link fault signals | No explicit link-down task failure code; add a link-state/runtime hook to distinguish from generic TTI. |
| `energy_exhausted` | `energy_violation` | No | Textual runtime signal containing energy/battery | Add EnergyManager-to-task/SFC hook or a native `TASK_FAIL_*` code. |
| `deadline_too_small` | `deadline_missed` | Yes: `TASK_FAIL_OUT_OF_DDL`, `TASK_FAIL_OUT_OF_TTI` | Native deadline/TTI code, LASDM graph timeout | Supported today for task and graph timeout evidence. |
| `node_moved_away` | `no_candidate` | Partial: `TASK_FAIL_OUT_OF_NODE` | Out-of-node code or textual node moved/coverage hole signals | `TaskManager.removeTasksByNodeId` stores removed tasks separately without a consumed failure code; bridge does not collect `_removed_tasks`. |

## Runtime Observations

AirFogSim currently defines native task failure codes in `airfogsim/enum_const.py`:

- `TASK_FAIL_OUT_OF_DDL`: maps to `deadline_missed`.
- `TASK_FAIL_OUT_OF_TTI`: maps to `deadline_missed`; used as partial link-disconnect evidence.
- `TASK_FAIL_OUT_OF_NODE`: maps to `no_candidate`; also the closest native evidence for node movement.
- `TASK_FAIL_PARENT_FAILED`: maps to `task_failed`.
- `TASK_FAIL_MALICIOUS_RESULT`: maps to `reliability_violation`.

`runtime_bridge.sync_from_airfogsim_tasks` consumes `getRecentlyFailedTasks()` and `getOutOfDDLTasks()`. It does not currently consume `TaskManager._removed_tasks`, so node movement/removal experiments need either a native runtime trace record or a follow-up hook.

## Command

```bash
python methods_baselines/lasdm/scripts/generate_failure_trace.py \
  experiment_artifacts/raw_data/agentic_service_orchestration/aso_v1/<run_dir>
```

The command writes `<run_dir>/failure_trace.json` for directory inputs.
