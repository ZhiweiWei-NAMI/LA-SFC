# Semantic-Topology LASDM MARL 实验设计

## 1. 研究问题

RQ1: 分布式语义交换是否能在局部目录不完整时提升 SFC 服务候选质量与成功率？

RQ2: 时空拓扑特征是否能在高移动性和链路波动场景下降低端到端延迟与失败率？

RQ3: 语义层、拓扑层、MARL 决策分别贡献多少，是否存在 overhead-performance Pareto 优势？

RQ4: UAV 作为服务节点、任务节点或双重角色时，对系统性能和稳定性的影响是什么？

## 2. UAV/车辆角色设置

本文不应把 UAV 固定为单一角色。推荐主设置：

- ground vehicles: 主要任务节点；部分车辆可作为 opportunistic service nodes。
- UAVs: sensing UAVs 可产生视频/感知任务；idle or compute-capable UAVs 可作为 mobile fog/service nodes。
- RSUs: region agents 与稳定边缘服务节点。
- cloud: 全局高容量但高延迟候选。

核心对照：

| setting | task nodes | service nodes | 目的 |
|---|---|---|---|
| infrastructure_only | vehicle + UAV | RSU + cloud | 最保守 VFC/SFC 基线 |
| plus_vehicle_services | vehicle + UAV | RSU + cloud + vehicles | 衡量地面车辆服务贡献 |
| plus_uav_services | vehicle + UAV | RSU + cloud + UAVs | 衡量 UAV 服务贡献 |
| full_hybrid | vehicle + UAV | RSU + cloud + vehicles + UAVs | 本文主方法设置 |

## 3. 场景矩阵

| scenario | 验证目标 | 关键压力 |
|---|---|---|
| distributed_service_discovery | 局部目录不完整时，remote semantic ads 提升候选可见性 | limited catalog visibility, TTL |
| semantic_ambiguity | 服务描述相似但能力/准确率不同，验证 SBERT + QoS 组合 | ambiguous service ratio |
| spatio_temporal_mobility | 高移动性下 temporal graph 是否有效 | link churn, UAV/vehicle speed |
| high_contention_partial_observability | 高请求率与服务争用下的 MARL 决策收益 | arrival rate, service supply shortage |

建议主实验：4 scenarios × 8 methods × 10 eval seeds = 320 evaluation runs。

训练实验：每个 scenario 5 train seeds，IPPO 每 seed 20 episodes 起步；用相同 10 eval seeds 评估 checkpoint。

## 4. 参数控制

避免全因子爆炸。主文固定一个 full-hybrid base setting，仅做 one-axis sensitivity：

- task demand: arrival rate 或 task-node count = low/mid/high。
- service supply: service-node count 或 supply ratio = 0.5/1.0/1.5。
- mobility: vehicle/UAV speed or topology churn = low/mid/high。
- semantic exchange: TTL = 1/3/5/10 s；compressed_dim = 16/32/64/128。
- role mix: infrastructure_only、+vehicle、+UAV、full_hybrid。

## 5. 输出指标

SFC 层：

- success ratio / acceptance ratio。
- deadline hit ratio。
- E2E latency mean/P95/P99。
- failure reason distribution: no candidate、deadline missed、task failed、reliability violation。
- resource cost: CPU reservation、wireless RB、backhaul bytes、energy proxy。
- service utilization and load balance。

语义与分布式层：

- top-1/top-k semantic score。
- semantic margin = score_top1 - score_top2。
- remote candidate discovery rate。
- stale advertisement ratio。
- message count and payload bytes。
- candidate miss rate。

拓扑层：

- edge churn rate。
- node churn rate。
- average link stability。
- mean speed / load delta。

MARL 层：

- episodic reward。
- convergence step/episode。
- invalid action ratio。
- entropy/value loss if IPPO logging is enabled。

## 6. 主文图表安排

| Figure | 内容 | 数据文件 |
|---|---|---|
| F1 | 系统架构和闭环图 | 手工绘图 |
| F2 | success/deadline hit by method × scenario | ablation_summary.csv |
| F3 | E2E latency CDF/P95 | marl_transition_trace.jsonl / manager metrics |
| F4 | semantic similarity distribution/top-k quality | semantic_candidate_trace.csv |
| F5 | remote discovery/stale ratio vs TTL/radius | semantic_candidate_trace.csv + message_overhead.csv |
| F6 | topology churn vs reward/QoS | topology_trace.jsonl + reward_curve.csv |
| F7 | IPPO reward curve | reward_curve.csv |
| F8 | message overhead vs performance Pareto | message_overhead.csv + ablation_summary.csv |

最少核心结果：F2、F3、F4、F5、F6、F7、F8 七组结果。F1 是方法图，不计实验结果。

## 7. 实现分布式语义压缩

每个 agent 维护本地完整向量，不交换完整 embedding。交换内容为压缩广告：

```text
(instance_id, service_id, node_id, region_id, capabilities,
 compressed_embedding[int8, d=64], created_at_s, ttl_s, metadata)
```

流程：

1. 对服务描述和 SFC node request 描述预计算 SBERT embedding。
2. 本地查询用 full embedding cosine similarity。
3. 远程广告用 random projection: R^(384) -> R^(64)，L2 normalize 后 int8 quantization。
4. 接收端对 request embedding 走同一 projection，与 remote compressed embedding 近似 cosine。
5. TTL 到期的广告视为 stale，不进入候选或进入但带 staleness penalty。
6. 通信开销用 JSON payload bytes 的可复现实验代理。

## 8. 实现时空特征提取

每步构建 `DynamicTopology`：

- nodes: vehicle/uav/rsu/cloud_server。
- edges: v2v/v2u/u2v/v2i/i2v/u2i/i2u/u2u/i2i/i2c/c2i。
- node features: resource、load、trust、energy、service_count、position、region。
- edge features: distance、rate、latency、reliability、wireless。

`TemporalStateBuffer(K=4)` 存最近 K 步图，并输出：node churn、edge churn、load delta、mean speed、link stability。

MARL observation = padded graph arrays + candidate semantic arrays + temporal summary。

## 9. 消融解释准则

- proposed > marl_semantic_no_topology：说明 topology/temporal features 有贡献。
- proposed > marl_topology_no_semantic：说明 semantic discovery/compression 有贡献。
- semantic_greedy_with_exchange > semantic_greedy_no_exchange：说明 distributed exchange 有贡献。
- compressed_dim=64 与 full embedding close，但 overhead 显著更低：说明 semantic compression 有价值。
- TTL 太短 remote discovery 下降；TTL 太长 stale ratio 上升：说明 TTL 存在最优区间。
