# Task Requirement Templates

本文件定义七类服务 catalog 如何生成任务需求与 SFC DAG。当前只记录规则，不实现生成代码。

## Urban Monitoring

基础链：

```text
network_identification
  -> video_object_detect
  -> semantic_compress
  -> mission_event_classifier
  -> situation_fusion
  -> alert_publish
```

动态插入：

- `risk_level >= 0.7`: 插入 `service_checkpoint`。
- `congestion == true`: 偏向 edge/local serving，并记录 payload cost。
- `link_degraded == true`: 插入 `relay_select` 或 `failover_select`。

主要指标：

- deadline satisfaction
- accuracy hit ratio
- payload transmitted
- service continuity

## Logistics And Governance

基础链：

```text
network_identification
  -> geoawareness_query
  -> flight_authorization
  -> trajectory_replan
  -> conformance_monitor
  -> eta_update
```

动态插入：

- `weather_alert == true`: 插入 `weather_risk_fusion`。
- `airspace_event == true`: 插入 `local_daa_check`。
- `risk_level >= 0.8`: 提高 `reliability_min` 和 `min_trust`。

主要指标：

- governance violation count
- deadline satisfaction
- route recomputation latency
- cloud/edge placement ratio

## Target Tracking And Search

基础链：

```text
video_object_detect
  -> target_track
  -> semantic_compress
  -> situation_fusion
  -> trajectory_replan
  -> alert_publish
```

动态插入：

- `target_lost == true`: 插入 `formation_coordination` 或 reacquire service。
- `link_degraded == true`: 插入 C2 continuity services。
- `high_mobility == true`: 增加 route stability penalty。

主要指标：

- tracking continuity
- reacquisition delay
- deadline satisfaction
- migration and handoff count

## C2 Continuity And Emergency

基础链：

```text
c2_link_monitor
  -> relay_select
  -> qos_slice_guard
  -> failover_select
```

动态插入：

- `coverage_hole == true`: 插入 `service_checkpoint` 和 `service_migration`。
- `low_battery == true`: 触发 UAV-hosted service draining。
- `rsu_congestion == true`: 允许 remote region resilience。

主要指标：

- service interruption time
- failover success ratio
- continuity hit ratio
- migration cost

## Generation Constraints

- 生成出的 `service_type` 必须能在 `service_instances.service_id` 中找到候选。
- 每个必需节点至少有一个满足 capability、semantic、QoS、trust 和 resource 的候选实例。
- 动态插入节点应在 `metadata.trigger` 中记录触发原因。
- Optional/resilience 节点应显式设置 `optional` 或 `alternate_nodes`。
- 同一 seed、同一场景参数下生成结果必须可复现。

