# Probe A 分叉的 tree process reward RL

启动脚本：`tree/scripts/train_qwen2.5_math7b-treepr-probeA.sh`。
以远端 `train_qwen2.5_math7b-treepr.sh` 为基准，保留 GRPO、segment process reward、
数据、模型、学习率、batch、保存及验证频率。另用 `probe` 替换分叉触发策略。

默认配置：

- 模型 Qwen2.5-Math-7B；训练 DAPO math 17k，验证 AIME 2024。
- 8 GPU、TP=1、train batch=96、PPO mini batch=16、micro batch/GPU=2。
- 学习率 1e-6、1 epoch；每 2 步验证、每 20 步保存。
- Probe checkpoint：`/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-sample-abcd-seed42/abcd/A.pt`。
- 自动读取 checkpoint 原始 logit 阈值，目前为 `0.026098472997546196`。
- 每个 token 计算 probe 分数，达到阈值且 segment 长度至少 10 时允许分叉。
- `branch_sampling=sample`、温度 1、k=4、最大深度 3。
- 沿用原训练脚本补齐到每题 **64** 个答案；前面的在线推理实验是 16 个答案。
- `tree_process_reward=True`；probe 参数冻结，actor 正常进行 RL 更新。
- Probe 模式不使用 entropy/WAAD gate，并跳过其动态阈值统计前向。
- 使用 V0、eager、同步输出处理；WandB 默认 offline，无脚本内置凭据。

在远端执行：

```bash
cd /inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0

# 仅检查配置，不创建训练任务或占用 GPU
DRY_RUN=1 bash tree/scripts/train_qwen2.5_math7b-treepr-probeA.sh

# 正式训练（默认 8 卡）
bash tree/scripts/train_qwen2.5_math7b-treepr-probeA.sh

# 先做 20 步短程试验
EXPERIMENT_NAME=qwen2.5_math7b_probeA_pilot \
  bash tree/scripts/train_qwen2.5_math7b-treepr-probeA.sh \
  trainer.total_training_steps=20
```

支持环境变量：`MODEL_PATH`、`TRAIN_FILE`、`TEST_FILE`、`PROBE_PATH`、
`PROBE_THRESHOLD`、`N_GPUS_PER_NODE`、`OUTPUT_ROOT`、`PROJECT_NAME`、
`EXPERIMENT_NAME`、`LOG_DIR`。末尾可继续传 Hydra overrides。
默认实验名包含时间戳，默认输出根目录保留原训练脚本的
`/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/verl_data`；可用
`OUTPUT_ROOT` 更改。日志在用户工作目录的 `verl_logs`。

固定 probe 会读取 RL 更新后的 actor hidden state，分数分布可能随训练变化；
本脚本不自动重训 probe 或重估阈值。默认训练数据仍是原始 DAPO 文件；此前
从该文件抽出的 100 题不能继续作为本轮 RL 的独立留出评估集。正式效果评估
使用原脚本的 AIME 2024，或另行排除训练题的数据集。

需要同时部署本次修改的 verl 配置、rollout 参数连接、trainer 校准条件，以及
先前已部署的自定义 vLLM probe 支持；单独复制 shell 脚本到旧环境不足以运行。
