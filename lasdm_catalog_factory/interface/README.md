# Interface Documents

这里记录 catalog factory 与 AirFogSim/LASDM 的边界。核心要求是：本目录只生成数据，AirFogSim 只消费数据。

当前唯一稳定接口是 YAML：

- `service_instances`: 由 `AirFogSim/airfogsim/lasdm/api.py::load_instances_from_yaml()` 读取。
- `service_chains`: 由 `AirFogSim/airfogsim/lasdm/api.py::load_chains_from_yaml()` 读取。
- `orchestrator.score_weights`: 由 `build_manager_from_yaml()` 传给 `LASDMOrchestrator`。

未来如果增加生成脚本，也应只依赖这些公开字段，避免绑定 AirFogSim 内部私有对象。
