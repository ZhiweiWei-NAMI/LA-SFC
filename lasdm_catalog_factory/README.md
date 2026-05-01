# LASDM Catalog Factory

本目录是 LASDM 研究语义层的外置工作区，用于沉淀和后续生成“七类低空服务 catalog”及任务需求。它不属于 AirFogSim 仿真平台核心包，不应被 `airfogsim/` 反向依赖。

## 当前判断

七类服务 catalog 应当作为数据层能力提供，而不是融合进 AirFogSim 的仿真内核。当前 `airfogsim/lasdm/api.py` 已经提供了足够窄的接口：

- `service_instances` 进入 `ServiceInstanceDirectory`
- `service_chains` 进入 `LASDMServiceChain`
- `orchestrator.score_weights` 进入 `LASDMOrchestrator`

因此，catalog 生成逻辑可以完全放在本目录，最终只产出兼容上述接口的 YAML 或实验配置清单。AirFogSim 只负责加载、调度、仿真和记录结果。

## 解耦原则

1. 本目录可以描述服务分类、任务模板、动态上下文触发规则和生成约束。
2. 本目录未来可以增加生成脚本，但生成结果必须是普通数据文件。
3. `airfogsim/` 目录只保留仿真平台接口、运行时、调度器和指标逻辑。
4. 任何七类服务语义变化优先在本目录演化，再通过 YAML 输出接入 AirFogSim。
5. 不在 AirFogSim 内部硬编码论文场景、服务分类、任务模板或特定 baseline 语义。

## 目录结构

- `taxonomy/`: 七类低空服务分类、候选微服务、能力标签和典型部署位置。
- `templates/`: 任务需求模板、SFC 生成模板和动态上下文触发规则。
- `interface/`: 与 `airfogsim/lasdm/api.py` 对接的 YAML 合同。
- `generated/`: 未来自动生成的 catalog、service instance、service chain 和 experiment YAML。
- `notes/`: 从外部参考实现中迁移过来的设计思想和取舍记录。

## 预期数据流

```text
七类服务分类 + 任务模板 + 场景参数
    -> catalog/task generator
    -> generated/*.yaml
    -> AirFogSim/airfogsim/lasdm/api.py
    -> LASDMScheduler.scheduleStep(env)
    -> AirFogSim 仿真与统一输出
```

## 非目标

- 本目录不直接修改 `AirFogSimEnv.step()`。
- 本目录不直接调用私有 task manager 字段。
- 本目录不保存实际 Docker/container 镜像。
- 本目录不替代 `airfogsim/service_orchestration/*` 或 `airfogsim/lasdm/*`。
