# AirFogSim LASDM YAML Contract

本文件定义 catalog factory 输出到 AirFogSim/LASDM 的最小数据合同。生成器只需要满足这个合同，就可以和当前 `airfogsim/lasdm` 接口解耦。

## Top-Level Sections

```yaml
schema_version: lasdm.airfogsim.v1

orchestrator:
  score_weights: {}

service_instances: []
service_chains: []

experiment: {}
```

`experiment` 是可选研究配置，当前 LASDM API 不强制读取，但 benchmark 和实验脚本可以使用。

## Service Instance Contract

每个 `service_instances` 元素映射到 `ServiceInstance.from_dict()`。

必需字段：

- `instance_id`: 全局唯一实例 ID。
- `service_id`: 服务类型 ID，必须能被 SFC 节点引用。
- `node_id`: AirFogSim 中的 UAV、vehicle、RSU、edge 或 cloud 节点 ID。
- `node_type`: 候选节点类型，例如 `uav`、`vehicle`、`rsu`、`edge_server`、`cloud_server`。
- `capabilities`: 能力标签列表。

建议字段：

- `region_id`
- `input_semantic`
- `output_semantic`
- `capacity`
- `used`
- `max_concurrency`
- `status`
- `health_score`
- `reliability_score`
- `accuracy_score`
- `trust_score`
- `cold_start_s`
- `version`
- `metadata`

约束：

- `service_id` 表达服务功能类型，不表达部署节点。
- `instance_id` 表达具体副本或容器实例。
- `node_type` 不应引入 AirFogSim 无法识别的节点类别。
- `capacity` 和 `used` 中资源名应保持 `cpu`、`memory`、`storage`。

## Service Chain Contract

每个 `service_chains` 元素映射到 `LASDMServiceChain.from_dict()`。

必需字段：

- `sfc_id`
- `source_node_id`
- `payload_semantic`
- `payload_mb`
- `qos.deadline_s`
- `nodes`

建议字段：

- `sink_node_id`
- `qos.priority`
- `qos.reliability_min`
- `qos.accuracy_min`
- `qos.max_energy_j`
- `qos.continuity_min`
- `context`
- `edges`

每个 SFC 节点需要：

- `node_id`: DAG 内部节点 ID。
- `service_type`: 要匹配的 `ServiceInstance.service_id`。
- `required_capabilities`: 所需能力标签。
- `input_semantic`
- `output_semantic`
- `cpu_mb`
- `memory_mb`
- `storage_mb`
- `optional`
- `alternate_nodes`
- `qos_override`
- `metadata`

约束：

- `edges` 只能引用 `nodes` 中存在的 `node_id`。
- 图必须是 DAG。
- `deadline_s` 必须为正。
- `payload_mb` 不能为负。
- 若某节点不是 `optional`，则无候选实例时应判为 `no_candidate`。

## Separation Rule

本目录允许生成：

- declarative catalog YAML
- service instance YAML
- task intent YAML
- service chain YAML
- experiment matrix YAML

本目录不生成：

- AirFogSim 内核代码
- `env.step()` patch
- 私有 manager/scheduler 调用
- 与论文场景强绑定的仿真平台类

