# AirFogSim SFC Workspace

这个目录是 LASDM/SFC 研究工作区。`AirFogSim/` 是独立 Git 仿真平台包；其余目录与它平级，用于放研究方法、baseline、计划、七类服务生成、实验数据和绘图分析代码。

## 子目录

- `AirFogSim/`: AirFogSim 仿真平台 Git 包。保留 `airfogsim` Python 包、原有示例/benchmark，以及仿真可调用的 SFC runtime/interface。
- `lasdm_catalog_factory/`: 七类低空服务 catalog 与任务需求生成层。这里只定义 taxonomy、模板和 YAML 接口合同，不直接修改 AirFogSim。
- `methods_baselines/`: LASDM 方法、Agentic Orchestrator baseline、实验入口、方法配置和方法侧测试。
- `plans/`: 当前研究执行计划和阶段拆分。
- `docs/`: 设计文档与实验协议。
- `experiment_artifacts/`: 实验 raw data、绘图/统计脚本、生成图和日志归档。

## 边界原则

- AirFogSim 只作为仿真平台和 SFC 仿真接口存在。
- 七类服务 catalog、任务模板和生成逻辑不进入 `AirFogSim/airfogsim` 包。
- 方法、baseline 和 plans 放在工作区根目录，与 `AirFogSim/` 平级。
- 生成数据和图绘制脚本放在 `experiment_artifacts/`，不混进平台包。

