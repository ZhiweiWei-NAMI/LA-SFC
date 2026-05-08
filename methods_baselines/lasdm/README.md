# LASDM Semantic-Topology Assets

这里保留 v46 Semantic-Topology MARL 主线资产。

- `configs/semantic_topology_marl.yaml`: base semantic-topology config.
- `configs/semantic_topology_runtime_figures_aligned.yaml`: v46 Poisson runtime/training override.
- `configs/semantic/`: service type、request type、implementation profile 和 schema catalog.
- `train_semantic_topology_marl.py`: offline/runtime env 构建与训练入口。
- `run_complete_runtime_experiment.py`: runtime training/evaluation orchestration.
- `plot_semantic_topology_marl.py`: F1-F8 figure generation.
- `tests/`: runtime bridge、task adapter、failure mapping、event logger、instance lifecycle、MARL wrapper、semantic matrix 与 topology metrics 测试。

旧 LASDM benchmark/sensitivity 入口和默认 repair eval 已移除；v46 训练和评估应显式使用 `semantic_topology_runtime_figures_aligned.yaml`。
