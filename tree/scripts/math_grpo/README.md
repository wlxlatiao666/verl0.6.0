# MATH：统一 GRPO 对照实验

从仓库根目录执行，默认 Qwen2.5-Math-7B、单机 8 GPU：

```bash
bash tree/scripts/math_grpo/grpo.sh
bash tree/scripts/math_grpo/treerollout.sh
bash tree/scripts/math_grpo/treepr.sh
```

三个入口共享 `common.sh`，建议分开运行，避免同时争用 GPU。
训练数据默认来自服务器
`/inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0/data/math/train.parquet`，
验证使用同目录 `test.parquet`，分别有 7,500 和 5,000 条。
只过滤超过 2,048 token 的 prompt；不按奖励过滤问题、不动态补充训练 batch。

| 配置 | grpo | treerollout | treepr |
| --- | --- | --- | --- |
| 树生成 | 关闭 | 8 个独立根 | 8 个独立根 |
| rollout.n | 64 | 1 | 1 |
| 每题回答数 | 64 | 8 × 2³ = 64 | 8 × 2³ = 64 |
| 优势 | 标准 GRPO | 标准 GRPO | 现有 TreePR 局部/全局优势 |
| 过程奖励及共享 token 权重 | 关闭 | 关闭 | 开启 |

## Pure GRPO 的定义

- 使用 `verl.trainer.main_ppo`，组内 outcome reward 标准差归一化。
- vanilla PPO policy loss，对称裁剪 0.2，按序列求 token 均值再求 batch 均值
  (`seq-mean-token-mean`)，不使用 DAPO 的 token-mean 聚合。
- 启用 reference KL loss，系数 0.001、`low_var_kl`；不在 reward 再扣 KL。
- `clip_ratio_c=1e9` 关闭实际可达范围内的 dual clipping：
  当前 vanilla 实现将 log ratio 限制在 [-20, 20]，ratio 最大 exp(20) < 1e9。
  使用有限值可避免零优势乘无穷产生 NaN。
- `naive` reward manager；MATH 数据源 `DigitalLearningGmbH/MATH-lighteval`
  分派到 `math_reward` 的答案正确性奖励。没有 DAPO overlong penalty、
  Clip-Higher、动态奖励过滤或额外 entropy bonus。

两组树实验复用上述优化设置。TreePR 沿用仓库既有算法：
`proc_agg_mode=raw`、local/global 权重各 0.5，替换标准 outcome-only
优势并启用 inverse-sharing 权重。因此 TreePR 是对 GRPO 的算法扩展。
两组树均使用 entropy 分叉、最小段长 64、P80 阈值校准（初始及每 10 步）、
有放回分支采样；不足叶子从原问题独立采样补齐。所有根共享原题 UID。
每题回答预算相同，但树组额外有阈值校准成本，并非等 FLOPs 实验。

## 配置与输出

```bash
DRY_RUN=1 bash tree/scripts/math_grpo/grpo.sh
WANDB_MODE=online bash tree/scripts/math_grpo/treepr.sh
EXPERIMENT_NAME=treepr_smoke bash tree/scripts/math_grpo/treepr.sh \
    trainer.total_training_steps=1 trainer.test_freq=-1 \
    trainer.val_before_train=False
```

`DRY_RUN=1` 只解析配置，不启动 Ray、加载模型或创建训练输出。
依赖应使用安装了当前本地 veRL 和自定义 vLLM 的服务器 Python 环境。
支持 `PYTHON_BIN`、`TASK_ROOT`、`DATA_DIR`、`MODEL_PATH`、`OUTPUT_ROOT`、
`PROJECT_NAME`、`EXPERIMENT_NAME`、`WANDB_MODE`、`WANDB_DIR` 环境变量。
其他 Hydra 参数可作为命令行参数传入，覆盖后应重新检查是否仍满足对照要求。
尤其不要单独改树的根数/深度/分支数而不调整 baseline 的 `rollout.n`。

默认输出为 `$TASK_ROOT/verl_data/math_pure_grpo/qwen2.5_math7b_<模式>/`，
包含 checkpoints、rollouts、validation、logs、wandb。
W&B 默认离线；在线凭据使用环境变量或已有登录，不在脚本中保存。
完整验证会遍历 5,000 条测试数据；短检查可关闭训练前验证并设置 `trainer.test_freq=-1`。
启用 reference KL 会增加显存占用，GPU 配置需要实际训练验证。
