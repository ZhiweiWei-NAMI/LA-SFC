# LASO Zip Takeaways

已读取的 `laso_airfogsim_extension.zip` 可作为参考，但不应原样导入。

## 保留的思想

- 七类低空服务 catalog：
  - `connectivity/C2`
  - `airspace_governance`
  - `sensing/situation_awareness`
  - `trajectory/cooperative_control`
  - `mission_oriented_application`
  - `edge_intelligence/computing`
  - `continuity/resilience`
- 基于 situation 的动态 SFC 扩展：
  - `link_degraded`
  - `coverage_hole`
  - `congestion`
  - `target_lost`
  - `weather_alert`
  - `airspace_event`
  - `risk_level`
- 低空任务模板：
  - urban monitoring
  - logistics/governance
  - target tracking/search
  - C2 continuity/emergency
- AirFogSim adapter 的边界思想：
  - 从 env 读取节点、链路、场景状态。
  - 输出 decision/log，不修改 `env.step()`。
  - 通过 scheduleStep 前置接入。

## 不迁移的内容

- 独立 `la_sfc` 包结构。
- `__pycache__`。
- 与当前 `airfogsim/lasdm` 重复的 dataclass。
- 直接调用不稳定 `task_manager.assignTask/offloadTask` 的逻辑。
- `laso_decisions` 这套输出命名。

## 迁移后的边界

七类服务和任务模板进入 `lasdm_catalog_factory/`；仿真执行仍由 `airfogsim/lasdm` 和现有 `airfogsim/service_orchestration` 负责。

