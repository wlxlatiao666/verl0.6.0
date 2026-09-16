# Qwen2.5-Math-7B 分支收益 probe

本目录实现离线数据标注、probe 训练及题目级留出验证。模型与数据路径、chat template、
2048 prompt/response 长度、temperature=1、top_p=1 和本地 DAPO verifier，
以 `tree/scripts/train_qwen2.5_math7b-treepr.sh` 及其默认配置为基准。
不修改原有 GRPO/tree decoding，不运行 WAAD；当前产物还没有接入在线分叉。

默认候选策略显式选择 **topk**，对应本次讨论的 deterministic-k。
注意：当前原训练代码的 `branch_sampling` 默认是 **sample**。
若要对齐该默认行为，准备数据时设置 `BRANCH_SAMPLING=sample`，并使用新的 WORK_DIR。
sample 使用 Gumbel top-k 不放回抽样，不能视为 k 个独立的原策略样本。

## 环境与先决条件

在服务器已有的 **verl + 本地定制 vLLM** Python 环境执行；需要 CUDA、torch、transformers、
numpy、pyarrow。不要为了这个脚本升级/重装 vLLM。脚本没有引入新的第三方训练框架。
将本地新增的 `tree/probe/` 和 `tree/scripts/train_qwen2.5_math7b-probe.sh` 同步到服务器 verl 仓库。

```bash
cd /实际路径/verl0.6.0
python3 -c 'import torch, transformers, pyarrow, numpy, vllm; print(torch.__version__, vllm.__file__); assert torch.cuda.is_available()'
export MODEL_PATH=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B
export TRAIN_FILE=/inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0/data/dapo-math-17k.parquet
export WORK_DIR=/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/qwen2.5-math7b-topk-pilot
```

默认 feature extraction 在一个 GPU 上加载 BF16 7B 模型，需容纳模型和最长 4096 token 的激活，
建议使用服务器上的 40/80GB GPU。该阶段不支持 TP；vLLM 两个生成阶段支持 `TP_SIZE`。
各阶段独立进程退出释放显存；probe head 默认在 CPU 训练，不需再次加载 7B。

## 先跑 100 题 pilot

```bash
CUDA_VISIBLE_DEVICES=0 NUM_QUESTIONS=100 \
  bash tree/scripts/train_qwen2.5_math7b-probe.sh all
```

80/10/10 题分别用于 train/val/test；每题 2 条轨迹，每条最多抽 6 个位置，k=4。
训练位置每候选续写 4 次，val/test 每候选续写 8 次。
100 题最多 23,040 次标注续写，另有 200 条初始轨迹；EOS 候选和短轨迹会减少实际生成量。
这不是便宜的 smoke test：如只需确认环境，用新的 WORK_DIR、`NUM_QUESTIONS=10 TRAJECTORIES=1 POSITIONS=2`。
10 题结果没有统计意义。

中断后执行同一命令：GPU 阶段按题目跳过已完成且配置一致的产物。
未完成的题目重新计算；训练 head 中断则从头重新训练，已成功完成的 head 会跳过。
修改模型、数据、采样配置或标注次数必须使用新 WORK_DIR（或新的 label replica），不会静默混用缓存。
同一 shard 不要同时启动两个进程。

## 多 GPU 正式标注

例如 1000 题，保持 val/test=100 题，最大标注量 230,400 次续写。
下面每卡一份 7B 副本，TP_SIZE=1；8 卡用题目分片并行。先启动 prepare 一次，
每个 GPU 阶段结束并检查所有退出码，再进入下一阶段。

```bash
export WORK_DIR=/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/qwen2.5-math7b-topk-1000
NUM_QUESTIONS=1000 bash tree/scripts/train_qwen2.5_math7b-probe.sh prepare

# 在 bash 中执行以下循环。请按实际可用 GPU 数修改。
for stage in rollout features label; do
  pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$gpu" NUM_SHARDS=8 SHARD_INDEX="$gpu" TP_SIZE=1 \
      bash tree/scripts/train_qwen2.5_math7b-probe.sh "$stage" \
      > "$WORK_DIR/${stage}-${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" != 0 ]]; then echo "Failed stage: $stage; inspect logs"; break; fi
done

# 确认三个阶段均成功后执行。缺少任一题目的产物时 train 会报错。
bash tree/scripts/train_qwen2.5_math7b-probe.sh train
bash tree/scripts/train_qwen2.5_math7b-probe.sh evaluate
```

也可逐阶段单卡执行：

```bash
bash tree/scripts/train_qwen2.5_math7b-probe.sh prepare
CUDA_VISIBLE_DEVICES=0 bash tree/scripts/train_qwen2.5_math7b-probe.sh rollout
CUDA_VISIBLE_DEVICES=0 bash tree/scripts/train_qwen2.5_math7b-probe.sh features
CUDA_VISIBLE_DEVICES=0 bash tree/scripts/train_qwen2.5_math7b-probe.sh label
bash tree/scripts/train_qwen2.5_math7b-probe.sh train
bash tree/scripts/train_qwen2.5_math7b-probe.sh evaluate
```

## 数据、标签与 token 对齐

- 读取 parquet 的 `prompt` chat messages、`reward_model.ground_truth`、`data_source`。
  prompt 是别的列名时，prepare 使用 `--prompt-key`。错误格式直接报错。
- 按规范化 prompt（保留大小写，只合并空白）做精确去重后划分题目。不是语义近重复去重；
  需要人工或上游工具排除同模板改数字等近重复。验证/测试题不接触 AIME。
- 导出 `probe_{train,val,test}.parquet`，并从完整原始题库排除所有留出题目的精确重复，
  生成 `grpo_train_without_probe_holdout.parquet`。后续 GRPO 若要保留此验证隔离，可将 TRAIN_FILE 指向它。
- 生成轨迹不使用树搜索。按 response 长度分层均匀选位置，不以 entropy/WAAD/最终正确性筛位置。
- `position=t` 表示前缀已有 t 个 response tokens。取模型 final norm 后的
  `hidden[prompt_length + t - 1]`，预测尚未出现的 `response[t]`。
  完整轨迹 teacher forcing 使用 causal attention，不会读到未来信息。
- 对每个候选固定前缀 `prompt + response[:t] + [candidate]`，独立随机续写 m 次。
  每次最大新增长度为 `2048 - t - 1`，不是重新获得 2048 token。EOS 候选直接评分。
- 复用仓库 `verl/utils/reward_score/math_dapo.py`，与原训练的 DAPO data_source 分派一致。
  verifier 输出 score=±1，标签使用其二元 `acc`；不增加 overlong penalty。
  该 verifier 的答案判定能力与原实验相同，并非符号数学等价判定器。
- 保存每次 seed、accuracy、reward、提取答案、结束原因、生成长度。
  label 可加 `--save-text` 保存完整文本（磁盘占用明显增加）。

对 k 个候选的 m 次二元结果，令 q_i 为成功率样本均值：

```text
U_naive = mean_i (q_i - mean(q))²
noise   = (k-1)/k² * sum_i q_i*(1-q_i)/(m-1)
U_raw   = U_naive - noise
```

同时保存非负截断版本，但**训练和评估使用未截断 U_raw**，保持方差修正的无偏性，
避免将负估计全归零造成正偏。单样本标签可以为负，真实期望仍非负。
训练目标放大 4 倍改善数值尺度，模型输出除以 4 才是 utility 单位。
这是“候选成功率差异”的代理，不等于固定预算下分叉对最终 GRPO 的因果收益。

## 训练、消融与指标

默认 head：仅用 hidden state，逐维标准化（只用训练集统计量），
`Linear(d,128) → GELU → Linear(128,1)`。
冻结大模型，MSE 回归修正标签；val MSE 早停；val 分数 80% 分位数校准阈值。
隐藏状态来自 HF BF16 forward，标注由本地 vLLM 完成；部署前应在相同前缀上验证两种实现的数值对齐。

```bash
# 线性 hidden probe
bash tree/scripts/train_qwen2.5_math7b-probe.sh train \
  --name linear_hidden --architecture linear

# entropy 小 MLP 对照；evaluate 还会自动提供直接 entropy 排序对照
bash tree/scripts/train_qwen2.5_math7b-probe.sh train \
  --name mlp_entropy --inputs entropy

# hidden + entropy 消融
bash tree/scripts/train_qwen2.5_math7b-probe.sh train \
  --name mlp_hidden_entropy --inputs hidden_entropy

# 应根据各自 validation.json 选择方案，再进行最终 test，避免反复用 test 调参
bash tree/scripts/train_qwen2.5_math7b-probe.sh evaluate --name mlp_hidden
```

产物位于 `$WORK_DIR/models/<name>/`：

| 文件 | 内容 |
|---|---|
| `probe.pt` | head 权重、归一化参数、模型/数据配置、目标缩放、验证集校准阈值 |
| `history.json` | 各 epoch 训练/验证 MSE |
| `validation.json` | 选型与调参指标 |
| `test_main.json` | 留出测试集指标 |
| `predictions_main.csv` | 测试位置 ID、标签、entropy、probe 分数和阈值选择结果 |

重点比较 `retrospective_matched_budget` 的 probe/entropy/random 被选位置平均 `utility_raw`，
及 `probe_minus_entropy_question_mean.bootstrap_95_ci`（按题目 bootstrap）。
同时报告 zero predictor MSE、整体/同题 Spearman、负标签比例、截断比例。
`calibrated_threshold` 使用固定 val 阈值在 test 上选择，报告实际选择比例；大量并列分数可能造成比例偏离目标。
matched-budget 是每题离线选相同比例位置的回顾性指标，不能当成在线树搜索效果。
这里没有运行 GRPO 或验证实际树搜索加速/准确率增益。

## 独立重复标注审计

在相同测试前缀、相同候选上换 seed 重新标注，避免只在一次 noisy label 上比较。
不会重新训练或重新校准阈值：

```bash
CUDA_VISIBLE_DEVICES=0 bash tree/scripts/train_qwen2.5_math7b-probe.sh label \
  --replica audit --split test --repeats-override 16
bash tree/scripts/train_qwen2.5_math7b-probe.sh evaluate --replica audit
```

输出 `test_audit.json` 与 `predictions_audit.csv`。检查 probe 相对 entropy 的优势是否保持。
小样本置信区间宽/包含零时应增加独立题目数，不能单凭训练 loss 下降宣称有效。

## 本地检查

```bash
python3 -m pytest -q tree/probe/tests --confcutdir=tree/probe/tests
bash -n tree/scripts/train_qwen2.5_math7b-probe.sh
```

测试覆盖方差修正期望、题目划分、token 前缀和剩余预算、EOS、断点缓存、
signed reward 转 accuracy、head 训练/加载/评估及训练统计不泄漏验证集。
GPU/vLLM 真正端到端运行需在存有模型与 parquet 的服务器完成。
