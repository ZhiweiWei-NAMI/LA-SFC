# Seven Low-Altitude Service Classes

本文件从 `laso_airfogsim_extension.zip` 中迁移必要思想，但不迁移其独立 `la_sfc` 包。目标是把七类服务 catalog 作为数据生成层定义。

## 1. connectivity/C2

典型能力：

- C2 link monitoring
- relay selection
- QoS slice guard
- failover selection

典型服务：

- `c2_link_monitor`
- `relay_select`
- `qos_slice_guard`
- `failover_select`

典型节点：

- UAV
- RSU
- edge server

编排意义：

- 链路退化、覆盖空洞或高动态移动时优先插入。
- 影响服务连续性、控制时延和故障恢复。

## 2. airspace_governance

典型能力：

- remote ID
- geo-awareness
- flight authorization
- conformance monitoring

典型服务：

- `network_identification`
- `geoawareness_query`
- `flight_authorization`
- `conformance_monitor`

典型节点：

- RSU
- edge server
- cloud
- controller

编排意义：

- 高风险空域、航线变更、监管事件中必须出现。
- 一般不优先放在资源弱或不可信 UAV 上。

## 3. sensing/situation_awareness

典型能力：

- video object detection
- target tracking
- situation fusion
- weather/risk fusion

典型服务：

- `video_object_detect`
- `target_track`
- `situation_fusion`
- `weather_risk_fusion`

典型节点：

- UAV
- RSU
- edge server
- cloud

编排意义：

- 决定感知精度、任务语义输出和 payload 缩减机会。
- 可用于比较 edge inference、cloud verification 和 split inference。

## 4. trajectory/cooperative_control

典型能力：

- detect-and-avoid
- trajectory replanning
- formation coordination

典型服务：

- `local_daa_check`
- `trajectory_replan`
- `formation_coordination`

典型节点：

- UAV
- RSU
- edge server
- cloud

编排意义：

- 天气、目标丢失、空域事件或交通冲突时插入。
- 与安全约束、实时性和本地控制闭环强相关。

## 5. mission_oriented_application

典型能力：

- mission event classification
- alert publishing
- ETA/status update

典型服务：

- `mission_event_classifier`
- `alert_publish`
- `eta_update`

典型节点：

- UAV
- RSU
- edge server
- cloud

编排意义：

- 将感知/控制结果转成任务业务输出。
- 用于区分普通 task DAG 与声明式 SFC 的语义终点。

## 6. edge_intelligence/computing

典型能力：

- semantic compression
- split inference
- model cache update

典型服务：

- `semantic_compress`
- `model_split_inference`
- `model_cache_update`

典型节点：

- UAV
- edge server
- cloud

编排意义：

- 决定 payload 缩减、模型冷启动、缓存命中和边云协同收益。
- 可作为 `cloud-first`、`edge-only`、`proposed` baseline 的核心差异来源。

## 7. continuity/resilience

典型能力：

- checkpoint
- degraded mode
- migration

典型服务：

- `service_checkpoint`
- `failover_select`
- `service_migration`

典型节点：

- UAV
- RSU
- edge server
- cloud

编排意义：

- 链路失败、节点过载、移动出区、低电量时触发。
- 直接对应服务连续性、迁移成本和中断时间指标。

## Mapping To Current LASDM Fields

| Taxonomy concept | LASDM YAML field |
| --- | --- |
| service class | `metadata.service_class` |
| service name | `service_id` / `service_type` |
| capability | `capabilities` / `required_capabilities` |
| eligible node type | `node_type` filter or `context.allowed_node_types` |
| input semantic | `input_semantic` |
| output semantic | `output_semantic` |
| resource demand | `cpu_mb`, `memory_mb`, `storage_mb` |
| QoS demand | `qos` / `qos_override` |
| safety/trust hint | `trust_score`, `context.min_trust`, `metadata.safety_critical` |

