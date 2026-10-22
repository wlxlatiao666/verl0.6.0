# 脚本入口

- `math_grpo/`：当前统一的 MATH 实验，包含 pure GRPO、多根 tree rollout、
  多根 TreePR；参数、运行方式见该目录 README。
- `legacy/root/`：原 `tree/run_*.sh`，已弃用。
- `legacy/scripts/`：原训练脚本和旧 Qwen 对照脚本，已弃用。
- 本目录中的 `build_*`、`run_probe_*`、`run_high_entropy_probe.sh`、
  `generate_math_datasets.py`：保留的数据构建和 probe 分析工具。

legacy 文件保留历史配置供查阅，入口直接退出，禁止误启动旧实验。
历史 probe 文档中引用的旧训练命令也已弃用。
评估和数据准备工具仍在 `tree/eval_qwen2.5-math7b.sh`、
`tree/prepare_eval_data.*`；评估新实验时显式指定其 `CHECKPOINT_DIR`。
