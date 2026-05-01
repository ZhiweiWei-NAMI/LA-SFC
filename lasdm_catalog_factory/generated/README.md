# Generated Outputs

这里用于放置未来自动生成的数据文件。当前不提交生成结果，只定义命名约定。

建议命名：

- `seven_class_service_instances.<scenario>.yaml`
- `seven_class_service_chains.<scenario>.<seed>.yaml`
- `seven_class_experiment_matrix.yaml`
- `manifest.<timestamp>.json`

生成文件必须满足 `interface/airfogsim_lasdm_yaml_contract.md`，并可由以下接口读取：

```python
from airfogsim.lasdm.api import build_manager_from_yaml

manager = build_manager_from_yaml("lasdm_catalog_factory/generated/example.yaml")
```

运行前需要把 `AirFogSim/` 加入 `PYTHONPATH`，或从方法/benchmark 脚本中使用相同的路径注入方式。

如果生成文件要参与论文实验，应在 benchmark 输出中记录文件路径、hash、seed 和生成参数。
