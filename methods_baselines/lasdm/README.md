# LASDM Method Assets

这里放 LASDM 方法侧资产：

- `configs/lasdm_airfogsim.yaml`: LASDM smoke 配置。
- `configs/lasdm_sensitivity.yaml`: LASDM 参数敏感性扫描配置，覆盖 `deadline={tight,medium,loose}`、`network={stable,degraded,burst}`、`load={low,medium,high}`，默认 offline。
- `examples/lasdm_minimal_orchestration.py`: 不依赖 SUMO 的 LASDM 最小示例。
- `run_lasdm_benchmark.py`: 标准 benchmark 入口，输出 `manifest.json`、`summary.csv`、`aggregate.csv` 和 per-run JSON。
- `run_lasdm_sensitivity.py`: 参数敏感性入口，额外输出 `sensitivity_summary.csv`。
- `tests/`: LASDM 方法侧测试，包括 runtime bridge、task adapter、failure mapping、event logger、benchmark adapter、region rerouting、instance lifecycle 和 MARL wrapper。

运行时需要让 Python 能看到 `../../AirFogSim`，示例脚本和测试文件会自动把该路径加入 `sys.path`。

常用验收命令：

```bash
python examples/lasdm_minimal_orchestration.py
pytest -q tests
python run_lasdm_benchmark.py --baseline lasdm_greedy --scenario normal --seed 0 --output-root /tmp/lasdm-runtime-smoke
python run_lasdm_sensitivity.py --baseline proposed --seed 0 --output-root /tmp/lasdm-sensitivity-smoke
```
