# Tree Decoding：vLLM 与 verl 修改总览

> 更新日期：2026-08-19
> verl：`/Users/bytedance/codes/verl0.6.0`，分支 `wlx/tree_0811`，提交 `e002f94`  
> vLLM：`/Users/bytedance/codes/vllm`，分支 `wlx/tree_0811`，提交 `5eb1383c4`  
> 本文依据两个仓库的完整分支历史、当前源码、配置和实验脚本整理，不只覆盖最后一次修改。

## 1. 目标与整体数据流

这套实现的目标是把 GRPO 中同一 prompt 的多次独立 rollout，替换为一次能够主动分叉的 tree decoding，并进一步利用共享前缀形成的 segment 结构进行奖励分配和梯度优化。

完整链路如下：

```text
prompt
  │
  ▼
vLLM V0 tree decoding
  ├─ 训练路径只按 entropy 选择分叉位置
  ├─ 分叉动作使用当前分布的 deterministic top-k token
  └─ 返回内部节点、叶节点、父子关系和各节点局部 token
  │
  ▼
verl vLLM rollout adapter
  ├─ 从叶到根重建完整 response
  ├─ 建立 unique_segments 与 leaf_segment_indices
  ├─ 不足 B^D 个叶子时用普通采样补齐
  └─ 将 prompt tensor 按叶子路由展开
  │
  ▼
trainer（actor old-logprob 重算后）
  ├─ 用 sibling branch log-prob 重建 conditional top-k actor mass
  ├─ leaf 使用路径概率质量，shared segment 使用 descendant mass 之和
  └─ advantage、PG、entropy 与 KL 使用同一加权测度
  │
  ├──────── treerollout ────────► actor-mass-weighted GRPO / leaf loss
  │
  └──────── tree_process_reward
              ├─ 叶奖励按 actor mass 自底向上传播
              ├─ 计算 local + global segment advantage
              ├─ leaf batching：每条 leaf 使用 actor path mass
              └─ tree_segment：每个唯一 segment 使用 actor reach mass
```

三个训练层级的核心区别是：

| Setting | Tree rollout | Tree process advantage | Segment loss | Segment batching |
|---|---:|---:|---:|---:|
| GRPO | 否 | 否 | 否 | 否 |
| treerollout | 是 | 否 | 否 | 否 |
| treepr | 是 | 是 | 否 | 否 |
| treesr | 是 | 是 | 是 | 是 |

## 2. vLLM 修改

### 2.1 使用约束

- 实现位于 vLLM V0 路径，训练脚本需要 `VLLM_USE_V1=0`。
- Tree search 与 `SamplingParams.n > 1` 不能同时用于同一个原始请求；tree 请求通常设置 `n=1`，由树自己扩展。
- 分叉产生的 child 被重新作为带前缀 prompt 的普通子请求加入 engine；子请求自身关闭 `enable_tree_search`，但父级 `ParallelSampleSequenceGroup` 保留原始 tree 配置并统一驱动后续分叉。

### 2.2 `TreeSearchParams`

定义位置：`vllm/sampling_params.py`。

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `enable_tree_search` | `False` | Tree decoding 总开关 |
| `entropy_threshold` | `1.0` | entropy 大于该值才允许 entropy 类策略分叉 |
| `branching_factor` | `3` | 每个分叉点最多产生的 child 数 |
| `max_tree_depth` | `3` | 最大分叉层数；root segment 的 `tree_depth=0` |
| `tau_importance` | `None` | WAAD 阈值；旧接口下非 `None` 表示启用 entropy+WAAD |
| `has_pending_branch` | `False` | entropy+WAAD 两阶段状态机的内部状态 |
| `min_seg_length` | `128` | 当前节点达到该输出长度后才允许再次分叉 |
| `branch_trigger_mode` | `None` | 显式模式：`random`、`entropy`、`entropy_waad` |
| `random_branch_probability` | `0.2` | random 模式中每个 eligible node/step 的分叉概率 |
| `max_num_leaves` | `None` | 完整叶子数上限；`None` 保持旧行为 |
| `collect_importance_stats` | `True` | threshold-stats 的兼容默认；verl 训练显式传 `False` 关闭 WAAD 计算 |

兼容解析规则：

```python
branch_trigger_mode is not None  -> 使用显式模式
branch_trigger_mode is None and tau_importance is None -> entropy
branch_trigger_mode is None and tau_importance is not None -> entropy_waad
```

因此，`tau_importance=0.0` **不是关闭 WAAD**，而是启用 entropy+WAAD，并使用 `WAAD > 0` 作为初始门槛。

新增参数被追加在 dataclass 尾部，旧调用方不传它们时保持原有行为。构造阶段会校验 mode、random probability 和 leaf cap；显式 `entropy_waad` 必须提供数值型 `tau_importance`。

### 2.3 分叉位置策略

vLLM fork 仍保留 random 和 entropy+WAAD，供独立 decoding comparison 使用；verl 训练 adapter 显式锁定 `branch_trigger_mode="entropy"`。

所有模式首先要求：

```text
tree_depth < max_tree_depth
output_length >= min_seg_length
```

#### Random

```text
u(node, step, seed) < random_branch_probability
```

随机数不是从模型 sampling RNG 中抽取，而是用 seed、完整 token path 的 rolling FNV hash、output length 和 tree depth，经 SplitMix64 finalizer 得到。这样启用 random gate 不会消耗或移动 actor sampling 的随机流，同一 seed 与路径可复现。

Random 仅随机决定“是否分叉”；一旦分叉，child token 仍是 deterministic top-k，不是随机抽取。

#### Entropy-only

对 engine 收到的 token log-probability 行计算：

```math
H_t=-\sum_a p_t(a)\log p_t(a)
```

实现会先把非有限 log-probability 的贡献置零，避免 `0 * -inf` 形成 NaN。当 `H_t > entropy_threshold` 时立即分叉。

#### Entropy + WAAD

这是一个延迟一步的两阶段流程：

1. Phase A：位置 `t1` 的 entropy 超过阈值时，保存该位置的 top-k token，但暂不分叉。
2. 正常再生成一个 probe token，使 attention 层能够取得 `t1` 的 query/对应 KV 状态。
3. Phase B：若计算出的 `WAAD > tau_importance`，回退触发 token 与 probe token，在 `t1` 位置用之前保存的 top-k token 建立 child；否则清除 pending 状态并继续生成。

### 2.4 WAAD 定义与实现（仅 legacy/comparison）

WAAD 在最后一个 attention layer 上计算。对当前 query 到所有历史 token 的 attention，在 head 维取平均后，按距离加权：

```math
WAAD_t=\sum_{i=0}^{t-1}\bar{A}_{t,i}\min(t-i,W),\qquad W=10
```

其中：

- `Ā` 是所有 query heads 的平均 attention；
- 当前 token 自身不纳入求和；
- GQA/MQA 的 KV heads 会 repeat 到 query-head 数量；
- 支持常见 4D/5D KV-cache layout；
- `W=10` 当前是函数内默认值，不是可配置参数；
- ALiBi 模型会将 slope bias 加到手动重算的 attention logits 上。

为降低常规生成开销，attention layer 只缓存 decode-only batch 的最后层 query；只有以下情况才触发额外计算：

- 显式/兼容解析后的 `entropy_waad` 请求正处于 pending Phase B；
- 同时设置 `collect_threshold_stats=True` 与 `collect_importance_stats=True` 的显式 WAAD 标定请求。

训练侧使用 `collect_importance_stats=False`，因此 entropy 阈值标定会在读取 attention cache 前返回，不执行 WAAD 重算。

### 2.5 分叉动作与叶子预算

通过 `torch.topk(row_logprobs, branching_factor)` 选择分叉 token。候选会先根据 base vocab/LoRA vocab 做越界过滤。

`max_num_leaves=N` 时，在每次实际分叉前计算当前叶子数 `L`。把一个叶子替换成 `C` 个 children 后，叶子数变化为 `C-1`，因此：

```text
C <= N - L + 1
```

不足以产生至少两个 child 时跳过分叉。最后一次分叉可以少于 `branching_factor`，所以该机制适用于任意 `N/B/depth`，不要求 `N=B^D`。

训练 pipeline 当前没有传 `max_num_leaves`，训练侧仍以 `B^D` 为目标并在 verl 中补齐；该 cap 主要由 `tree_decoding_comparison` 使用。

### 2.6 Tree request 的状态与生命周期

`Sequence` 新增的主要字段：

| 字段 | 含义 |
|---|---|
| `tree_depth` | 节点深度 |
| `parent_req_id` / `parent_seq_id` | 父节点定位 |
| `is_leaf` | 当前节点是否为叶子 |
| `new_branch_token_id` | child 替换进入的分叉 token |
| `old_branch_token_id` | parent 被替换掉的最后一个 token |
| `old_branch_token_id_extra` | deferred 模式中需要额外去掉的 probe token |
| `pending_branch_token_ids` | Phase A 保存的 top-k 候选 |
| `entropy_list` / `importance_list` | 标定模式下逐 token 统计 |

分叉时：

1. parent 标记为非叶节点并结束；
2. immediate 模式从 parent token path 去掉最后一个 token，deferred 模式去掉最后两个 token；
3. 每个 top-k token 与回退后的 prefix 组成 child prompt；
4. child 深度加一、记录父 ID，并作为新叶子加入 assembled group；
5. 所有 child/leaf 完成后才组装最终 `RequestOutput`。

### 2.7 输出协议与 segment 修正

`CompletionOutput` 增加：

```text
tree_depth
parent_req_id
parent_seq_id
seq_id
is_leaf
tree_text
tree_ids
entropy_list
importance_list
```

普通 `text/token_ids` 是该 vLLM 子请求的输出，可能不包含作为 child prompt 最后一个 token 的 branch token。为让上层能够正确拼树：

- child 的 `tree_text/tree_ids` 会补回 `new_branch_token_id`；
- parent 的 `tree_text/tree_ids` 会去掉已被替换的 trigger token；
- deferred parent 还会去掉 probe token；
- detokenizer 记录每个 token 实际贡献的字符数，避免按字符串长度猜测 token 边界。

注意：`RequestOutput.outputs` 同时包含内部节点和叶节点；只有 `is_leaf=True` 的 root-to-leaf 拼接结果才是完整候选答案。

### 2.8 阈值统计模式

`SamplingParams.collect_threshold_stats=True` 时不分叉。entropy 始终收集；只有 `collect_importance_stats=True` 才重算 WAAD 并填充 `importance_list`。verl 训练显式传 `False`，且阈值 pre-pass clone 完整 rollout distribution 与 LoRA，只修改 sample count、最大长度和 tree/stats 开关；独立 comparison 可显式传 `True` 标定 WAAD。

### 2.9 vLLM 主要修改文件

| 文件 | 修改内容 |
|---|---|
| `vllm/sampling_params.py` | Tree 参数、显式 trigger mode、random probability、leaf cap、统计开关 |
| `vllm/sequence.py` | Tree 节点状态、ParallelSampleSequenceGroup、分叉建 child、组装生命周期 |
| `vllm/engine/llm_engine.py` | Tree hook、三种 trigger、entropy、deferred WAAD、top-k、leaf cap |
| `vllm/attention/layer.py` | 缓存最后层 decode query、从 KV cache 重算 WAAD |
| `vllm/worker/model_runner.py` | 按需计算 importance 并写入 sampler output |
| `vllm/model_executor/layers/sampler.py` | sampler output 携带 importance scores |
| `vllm/outputs.py` | 输出 tree metadata、修正后的局部 segment text/token IDs |
| `vllm/transformers_utils/detokenizer.py` | child branch token 解码与 parent trim 长度记录 |
| `benchmark_tree_decoding*.py` | Tree decoding 性能/规模测试 |
| `tests/engine/*tree*`、`tests/test_sampling_params.py` | Tree 生成、阈值统计、trigger 和 leaf cap 测试 |

## 3. verl 修改

### 3.1 Rollout 配置

配置类位于 `verl/workers/config/rollout.py`，Hydra 默认值位于 `verl/trainer/config/rollout/rollout.yaml`。

| verl 参数 | 默认值 | 作用 |
|---|---:|---|
| `tree_search.enable` | `False` | 训练 rollout 是否启用 tree decoding |
| `tree_search.entropy_threshold` | `1.0` | 初始 entropy 阈值 |
| `tree_search.branching_factor` | `3` | 分支数 B |
| `tree_search.max_tree_depth` | `3` | 最大深度 D |
| `tree_search.topup_leaves_to_target` | `True` | 是否补到 `B^D` 个 response/prompt |
| `tree_search.tree_process_reward` | `False` | 是否输出 segment tree 并使用 process advantage |
| `tree_search.threshold_stats_n` | `1` | 每 prompt 的阈值统计样本数 |
| `tree_search.threshold_stats_interval` | dataclass `10` | 每多少个训练 step 更新一次阈值；但当前 Hydra YAML 未声明此键，driver 的实际 fallback 是 `1` |
| `tree_search.threshold_stats_max_tokens` | dataclass `64` | 阈值统计 rollout 长度；当前 worker dataclass 转换后通常为 64，`0` 表示使用正常 response length |

当前 verl adapter 向 vLLM 传 `enable_tree_search`、entropy、B、D，并强制设置 `branch_trigger_mode="entropy"`、`tau_importance=None`。它没有传 `min_seg_length` 或 `max_num_leaves`。因此当前训练行为是：

- `min_seg_length` 隐式使用所安装 vLLM 的默认值 128；
- 所有训练 tree 都在 entropy 超阈值后立即分叉，不进入 pending/deferred WAAD 状态；
- threshold pre-pass 只统计 entropy，不计算 attention importance；
- random tree 目前只接入独立 decoding comparison，未接入训练 Hydra 配置。

### 3.2 Tree rollout 与完整叶子重建

实现位置：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`。

对每个 vLLM output：

1. 用 `seq_id` 建立节点表；
2. 对每个 `is_leaf=True` 节点沿 `parent_seq_id` 回溯；
3. 将路径反转为 root-to-leaf；
4. 拼接每个节点的 `tree_ids`，形成完整 response；
5. 记录该 response 属于哪个原始 prompt。

Tree rollout 会写入 routing metadata：

| 字段 | 形态 | 含义 |
|---|---|---|
| `tree_prompt_indices` | 每 response 一个整数 | worker 内原 prompt index |
| `tree_num_leaves` | 每 response 重复存储 | 当前 worker response 总数 |
| `tree_num_prompts` | 每 response 重复存储 | 当前 worker prompt 总数 |

trainer 根据这些字段把 prompt batch 按实际叶子路由展开，而不是使用固定 `repeat(n)`。跨 worker concat 后会把 local prompt index 转成 global prompt index，并校验每个 prompt 的 response 数。

训练/验证隔离：`do_sample=False`、validation（包括 sampled validation）和 REMAX baseline 都显式传 `tree_search_params=None`，避免验证 batch 被意外扩成树。

### 3.3 固定规模 top-up

开启 `topup_leaves_to_target=True` 后，目标为：

```math
K=B^D
```

若 prompt 的真实 tree 只有 `L<K` 个叶子，adapter 使用与 rollout 相同的 temperature/top-p/top-k/min-p/penalty/max-tokens，再进行 `K-L` 次普通、非 tree 采样。

- tree leaves 与 fillers 使用同一个 prompt uid，仍属于同一 GRPO group；
- filler 被表示为一个独立 synthetic root segment；
- filler 的 `token_share_weights=1`；
- 一个展开 tree root 与每个普通 top-up root 被视为等权 estimator strata；展开 tree 内部再按 actor 条件概率分配质量；
- top-up 后严格检查每个 prompt 恰有 K 个 response，否则报错，不静默 trim/复制。

### 3.4 阈值动态标定

trainer 在配置的 interval 上先执行短 rollout：

1. 每个 actor rollout worker 收集 entropy 列表；
2. worker 内计算 entropy p80；
3. driver 对 worker 的 p80 再取平均；
4. 动态更新 rollout engine 的 `entropy_threshold`；
5. 记录 `tree/entropy_threshold` 和耗时。

这里存在配置层差异：`TreeSearchConfig` dataclass 把 interval 声明为 10，但当前 `rollout.yaml` 没有列出该键，driver 对缺失字段使用 `.get(..., 1)`。因此按当前主配置，entropy 阈值实际上**每个训练 step 都会标定并覆盖**脚本中的初始值。若显式把 interval 配成 10，才会在第 10、20、30… step 更新。

### 3.5 Tree topology 与 actor occupancy 权重

所有启用 tree search 的 rollout 都保存 topology；`token_share_weights` 仅作为旧 TreePR 诊断字段保留：

| 字段 | 存储位置 | 含义 |
|---|---|---|
| `unique_segments` | non-tensor | 按 `seq_id` 去重后的所有树节点 token 列表 |
| `leaf_segment_indices` | non-tensor、每 leaf | root-to-leaf 路径上的 segment index |
| `token_share_weights` | tensor、每 token | legacy `1 / descendant_count`，仅 process 数据携带，不再参与 loss |
| `worker_segments_offsets` | non-tensor metadata | concat 前各 worker segment 边界 |
| `worker_leaves_offsets` | non-tensor metadata | concat 前各 worker leaf 边界 |
| `tree_leaf_masses` | tensor、每 leaf | prompt 内归一的 conditional top-k actor path mass |
| `tree_loss_scales` | tensor、每 leaf | 针对实际 `loss_agg_mode` 一次性全局归一后的固定 loss scale |

设 parent `s` 实际展开的 children 集合为 `C(s)`。训练用重算后的旧 actor log-prob 计算：

```math
p(c\mid s,C)=\frac{\pi_{old}(a_c\mid s)}{\sum_{j\in C(s)}\pi_{old}(a_j\mid s)}
```

leaf mass 是路径上这些条件概率的乘积；segment reach mass 是其所有 descendant leaf mass 之和。于是共享 prefix 即使在多个 leaf response 中物理重复，其总贡献也恰好等于 actor 到达该 segment 的概率质量，而不是 descendant 数。普通 treerollout、TreePR 和 TreeSR 都使用该权重。

这些概率只在**已展开的 top-k support 内条件化**。代码同时记录每个 branch 的 `sum(exp(old_logprob_child))` 作为 coverage；未展开的 tail 没有样本，不能由有限权重恢复。因此这是 conditional top-k actor objective，不是完整 actor iid objective。

固定 `tree_loss_scales` 在完整 batch/全 DP ranks 上只归一一次；micro-batch 只按原 reducer 的 token/sequence denominator 合并，不再局部 self-normalize。这样即使一个 micro-batch 只有一条 leaf，0.9/0.1 的质量也不会退化成 1/1。

skip-rollout cache 也做了双向检查：

- treepr 缺少 process 字段时直接报错；
- treerollout 读取旧 treepr cache 时会清除 process 字段；
- routing 字段与当前树规模不匹配时直接拒绝。

### 3.6 Tree process advantage

实现位置：`compute_tree_process_advantage()`。

#### 3.6.1 Node score

叶节点分数等于完整叶子的 token reward 总和：

```math
Q(leaf)=\sum_t r_t
```

内部节点自底向上按 child 的条件 actor mass 加权：

```math
Q(s)=\sum_{c\in C(s)}p(c\mid s,C)Q(c)
```

#### 3.6.2 Local advantage

对同一 parent 下的 siblings：

```math
A_{local}(s)=\frac{Q(s)-Q(parent(s))}{std(\{Q(c):c\in C(parent(s))\})+10^{-6}}
```

root 没有 parent，因此 local advantage 为 0。

#### 3.6.3 Global advantage

以 prompt 为组，用该 prompt 所有 leaf score 的 actor-mass 加权均值和总体标准差归一化所有节点：

```math
A_{global}(s)=\frac{Q(s)-mean(Q_{leaves})}{std(Q_{leaves})+10^{-6}}
```

最终：

```math
A(s)=\lambda_{local}A_{local}(s)+\lambda_{global}A_{global}(s)
```

trainer 实际读取的配置键是：

```text
algorithm.local_adv_weight   # 默认 0.5
algorithm.global_adv_weight  # 默认 0.5
```

随后将 segment advantage 填到每条叶子中属于该 segment 的所有 token 上。

`algorithm.proc_agg_mode` 当前实际支持：

- `raw`：segment 内每个 token 都使用原始 `A(s)`；
- `length_balanced`：每 token 使用 `A(s) / segment_length * mean_segment_length`。

配置注释中曾写成 mean/min/max，但当前代码并不实现这三种模式，应以 `raw/length_balanced` 为准。

### 3.7 `loss_mode=tree_segment`

`build_segment_tensors()` 将 leaf tensor 重新映射为 unique segment tensor：

1. 每个 segment 选择第一条包含它的 canonical leaf；
2. 记录 `(canonical_leaf, token_offset)`；
3. 从该 leaf 提取 old log-prob、mask、IS weight；
4. segment 内 advantage 取第一个有效 token 的值并铺满；
5. padding 到当前 batch 最大 segment length。

`compute_policy_loss_tree_segment()` 在这些 segment tensor 上执行与 vanilla PPO 相同的 clipped objective、dual clip、KL metric 和可选 rollout IS weighting。区别不在 PPO 公式，而在输入的基本样本已经从 leaf 变成 unique segment。

每个 unique segment 的 loss weight 是其 reach mass：

```math
M(s)=\sum_{y:\,s\in path(y)}M(y)
```

PG、entropy 和显式 KL 都提取同一个 canonical segment slice 并使用同一 `M(s)`。跨 rank 的 scale normalization 使用 all-reduce sufficient statistics，避免每个 rank 各自归一后扭曲 occupancy。

### 3.8 两种 tree-segment batching 策略

#### `tree_segment_batch_strategy=leaf`

仍先按 leaf 切 mini/micro batch，再筛选该 batch 涉及的 segments。为避免不同 rank 的叶子数、micro-batch 数不同造成 FSDP/NCCL collective 错位，短的 rank 用 zero-scale dummy forward/backward 对齐。

#### `tree_segment_batch_strategy=segment`

只有以下条件同时满足时启用：

```text
policy_loss.loss_mode == tree_segment
tree_segment_batch_strategy == segment
tree segment metadata 存在
use_dynamic_bsz == False
```

每个 rank 处理自己叶子引用的全部 local unique segments，不跨 rank 搬运 segment。流程为：

1. 计算与 leaf/base GRPO 相同的目标 optimizer step 数；
2. 将全部 local segments 均衡分入这些 optimizer groups；
3. 每个 group 再按 `ppo_micro_batch_segments` 切 micro-batch；
4. all-gather 每个 rank 每个 group 的 micro-batch 数；
5. 短 rank 插入 zero-scale dummy micro-batch；
6. 每个 group 边界统一执行一次 optimizer step；
7. 末尾断言实际 step 数与目标一致。

### 3.9 Optimizer step 对齐

普通 GRPO 的 response multiplicity 是 `rollout.n`；固定 tree rollout 的逻辑 multiplicity 是：

```math
K=B^D
```

FSDP 初始化现在使用独立 helper 计算 multiplier：

```math
local\_ppo\_mini=\frac{ppo\_mini\_batch\_size\times K}{DP}
```

每个 rank 的 local response 数为：

```math
local\_batch=\frac{prompt\_batch\times K}{DP}
```

所以每个 PPO epoch 的 optimizer steps 为：

```math
\frac{local\_batch}{local\_ppo\_mini}
=\frac{prompt\_batch}{ppo\_mini\_batch\_size}
```

在当前 Math-7B 主实验中：

```text
prompt batch P = 96
K = 2^6 = 64
DP = 8
ppo mini M = 16

local batch = 96*64/8 = 768
local mini  = 16*64/8 = 128
optimizer steps = 768/128 = 6 / PPO epoch
```

因此 GRPO、treerollout、treepr 与 treesr 现在都以 6 个逻辑 optimizer steps 为目标。`tree_segment + segment` 会把 segment micro-batches 分成 6 个 optimizer groups，而不是把所有 segments 累积成一次更新。

### 3.10 分布式 `DataProto` 支持

`unique_segments` 的长度不是 leaf batch size，因此不能沿用普通 per-sample ndarray 规则。`verl/protocol.py` 增加了：

- metadata key 的长度校验豁免；
- concat 时合并各 worker segments，并平移 `leaf_segment_indices`；
- 保存 worker leaf/segment offsets；
- chunk 时按原 worker 边界恢复 local segment 范围；
- 没有 offsets 时，按当前 chunk 使用到的 segment 重建 local segment table；
- select/slice/repeat/reorder 时不错误索引全局 segment metadata。

### 3.11 监控、评测与数据工具

#### Tree metrics

rollout 输出包括：

```text
tree/total_nodes
tree/leaf_nodes
tree/branch_points
tree/branching_rate
tree/avg_max_depth
tree/global_max_depth
tree/avg_tree_leaves_per_prompt
tree/avg_leaves_per_prompt
tree/min_leaves_per_prompt
tree/max_leaves_per_prompt
tree/topup_samples
tree/prompts_need_topup
tree/expansion_ratio
```

#### Logging

- 将 W&B、TensorBoard、`VERL_LOGGING_LEVEL`、`VLLM_USE_V1` 等环境变量传入 Ray workers；
- W&B adapter 支持从环境变量显式 login；
- controller/actor/rollout 中加入了大量带时间戳和 rank 的 debug 输出；
- `actor.tree_process_loss_log_interval` 默认是 10。仅当 actor 收到
  `tree_process_reward=True` 对应的完整 process 数据时，rank 0 会在第一个、随后每第 N 个
  actor update 抽样打印一行 `[TREE_PROCESS_LOSS]`。TreePR 输出 sequence 的 process PG loss、
  actor-mass-weighted effective PG loss，以及按 token 位置压缩的 `advantage_runs`；legacy
  `token_share_weights` 只用于优先挑选含共享 prefix 的诊断样本，不再修改 advantage。TreeSR
  输出 standalone segment PG loss、canonical sequence/offset 和 `advantage_runs`。RLE 使用
  包含首尾的位置区间，最多展示 16 段，超出部分会折叠；单条样本值仅作 token-mean 诊断。
  设为 0 可关闭；
- scripts 支持 console、W&B offline 和 TensorBoard。

#### Reward/eval

- `math_dapo` reward 新增最后一个 `\\boxed{...}` 的 fallback 判分；
- reward router 新增 `math500`、`amc`、`olympiad_bench`、AIME 和 `math_dapo_*` 数据源；
- `tree/prepare_eval_data.py` 统一生成评测 parquet；
- `tree/eval_qwen2.5-math7b.sh` 支持批量评测 checkpoints。

#### 独立 pass@k 对比工具

`tree_decoding_comparison/` 对比：

1. iid GRPO `n` 次；
2. random tree；
3. entropy-only tree；
4. entropy+WAAD tree。

每种方法严格返回 `n` 个完整候选。Tree 通过 vLLM `max_num_leaves=n` 防止超预算，不足部分用 fresh iid sampling 补齐。重复/空 completion 仍占候选预算；pass@k 使用与候选顺序无关的组合形式：

```math
pass@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
```

输出 JSON 带 schema、run ID、candidate source 和严格预算校验，分析脚本会拒绝不同运行或不同 schema 的文件混用。

### 3.12 verl 主要修改文件

| 文件 | 修改内容 |
|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | entropy-only params/stats、叶子重建、同分布/同 LoRA top-up、routing、topology、metrics |
| `verl/trainer/ppo/tree_weighting.py` | conditional top-k leaf mass、segment reach mass、固定 reducer scale 与校验 |
| `verl/trainer/ppo/ray_trainer.py` | 动态 entropy 阈值、actor-mass advantage、tree routing/cache、权重与数据生命周期 |
| `verl/trainer/ppo/core_algos.py` | weighted GRPO/reducer、segment tensor 重建、tree-segment PPO loss |
| `verl/workers/actor/dp_actor.py` | leaf/segment batching、reach weights、跨 micro/rank reducer、dummy collective、step 计数 |
| `verl/workers/fsdp_workers.py` | 固定 tree multiplicity 的 PPO mini-batch 归一化 |
| `verl/utils/tree_training.py` | multiplier、process gate、均衡 optimizer group helper |
| `verl/protocol.py` | 非 per-leaf tree metadata 的 concat/chunk/select/slice/repeat |
| `verl/workers/config/{rollout,actor}.py` | TreeSearchConfig 与 segment batching 配置 |
| `verl/trainer/config/**` | Hydra 默认配置 |
| `verl/utils/reward_score/**` | 数学数据源与 boxed fallback |
| `verl/utils/tracking.py`、`constants_ppo.py` | W&B/TensorBoard/Ray 环境传播 |
| `tree/` | 训练、数据准备、checkpoint eval、实验脚本和 debug 记录 |
| `tree_decoding_comparison/` | 四路 decoding pass@k 对比与分析 |

## 4. 实验 setting 汇总

### 4.1 主 Math-7B：B=2，D=6，K=64

公共配置：Qwen2.5-Math-7B，DAPO Math 17K，prompt batch 96，prompt/response length 2048，8 GPUs，PPO mini 16，micro/GPU 2，LR `1e-6`，一轮 PPO epoch 对应 6 optimizer steps。

| 脚本 | `rollout.n` | Tree | Process reward | Loss mode | Batch unit |
|---|---:|---:|---:|---|---|
| `tree/run_qwen2.5-math7b-grpo.sh` | 64 | 否 | 否 | vanilla | leaf/sequence |
| `tree/run_qwen2.5-math7b-treerollout.sh` | 1 | B2D6 + top-up64 | 否 | vanilla | leaf/sequence |
| `tree/run_qwen2.5-math7b-treepr.sh` | 1 | B2D6 + top-up64 | 是 | vanilla | leaf/sequence + share weight |
| `tree/run_qwen2.5-math7b-treesr.sh` | 1 | B2D6 + top-up64 | 是 | tree_segment | segment，micro=2 segments |

这些脚本使用 entropy-only 分叉；初始阈值为 `0.8`，threshold pre-pass 会按 interval 动态更新 entropy 阈值。

### 4.2 Base-7B：B=2，D=3，K=8

| 脚本 | 候选规模 | Feature |
|---|---:|---|
| `tree/run_qwen2.5-base7b-grpo.sh` | `n=8` | baseline |
| `tree/run_qwen2.5-base7b-treerollout.sh` | B2D3 | tree rollout |
| `tree/run_qwen2.5-base7b-treepr.sh` | B2D3 | tree rollout + process reward |

这组使用 4 GPUs、prompt batch 96、PPO mini 16、micro/GPU 2。

### 4.3 B=3，D=3，K=27

| 脚本 | 实际 setting |
|---|---|
| `tree/scripts/run_qwen2.5-math7b-grpo-n27.sh` | GRPO `n=27` |
| `tree/scripts/run_qwen2.5-math7b-tree-rollout-b3d3.sh` | B3D3 tree rollout |
| `tree/scripts/run_qwen2.5-math7b-tree-segment-b3d3.sh` | B3D3 + process reward + `tree_segment` loss/segment batching |

`tree-segment-b3d3.sh` 已显式设置 `policy_loss.loss_mode=tree_segment`、`tree_segment_batch_strategy=segment` 和 segment micro-batch size，因此会运行完整 TreeSR，而不是悄悄退化成 TreePR。

### 4.4 Qwen2.5-3B ablation

GSM8K 与 MATH 各有三组：pure GRPO、treerollout、treepr。

公共设置：prompt batch 64、B4D3、目标 K=64、PPO mini 16、micro/GPU 2、4 GPUs、`loss_agg_mode=seq-mean-token-mean`。GSM8K response length 1024，MATH response length 2048。

两个 3B treepr 脚本现在把 lambda 写到 trainer 实际读取的顶层键：

```text
algorithm.local_adv_weight
algorithm.global_adv_weight
```

因此 `TREE_PR_LAMBDA` 会正确控制 local/global process advantage 的混合比例。

### 4.5 独立 decoding comparison

典型命令：

```bash
cd /Users/bytedance/codes/verl0.6.0/tree_decoding_comparison

./run_comparison.sh \
  --model-path /path/to/Qwen2.5-Math-7B \
  --dataset-path /path/to/dapo-math-17k.parquet \
  --branching-factor 3 \
  --max-tree-depth 3 \
  --n 27 \
  --random-branch-probability 0.2
```

这里 `n` 是参与 pass@k 的完整候选数，不是内部节点数，也不要求等于 `B^D`。

## 5. 当前实现的关键语义与已知限制

这些条目对解释实验结果非常重要。

### 5.1 Tree rollout 不是 actor policy 的 iid rollout

分叉 token 是 deterministic top-k，不能把 emitted leaves 等权当作 actor iid 样本。当前实现不再这样做：它用重算后的 `old_log_probs` 在每组实际展开的 siblings 内恢复 actor 条件概率，并将路径 mass 用于 leaf，将 descendant mass 之和用于 shared segment。

一个 branch ratio 虽然只来自分叉 token，却缩放该分支的整条后续路径；它不是只缩放该 token。当前 reducer 正是按这条语义传播概率质量。

边界也必须说清楚：deterministic top-k 没有给 omitted tail 生成样本，因此当前修正得到的是“actor 在已展开 children 上的条件分布”。coverage 小的 branch 仍有 support bias。通用 `rollout_is` 是另一层用于 rollout/trainer 数值 mismatch 的机制，不能补回未展开动作；而且当前 vLLM forced branch token 没有与 continuation logprob 对齐的完整 proposal 协议，因此 tree + `rollout_is_threshold` 现在会显式报错，防止静默叠加错误比值。

### 5.2 WAAD 仅保留在独立 comparison

- importance 只在 decode-only batch 缓存 query；混入 prefill 时可能没有 importance；
- child 分叉会产生新的 prefill 请求，因此 scheduler/chunked prefill 会影响 WAAD 是否可得；
- 独立 comparison 已关闭 chunked prefill；训练 pipeline 不再进入 WAAD 路径；
- TP>1 时当前 WAAD 没有聚合所有 tensor-parallel ranks 的 heads，因此该限制只影响显式启用 WAAD 的 comparison/legacy 调用。

### 5.3 阈值标定不是严格条件分布标定

当前统计取所有位置 entropy 的 marginal p80，再对 worker p80 求平均；它不是“满足 position eligibility 的位置”的条件分位数。训练 stats max tokens 默认 64，而 vLLM `min_seg_length` 默认 128，这两者的语义也并不完全一致。

### 5.4 Process reward 的统计性质

- 内部节点、sibling moments 和 prompt-level leaf moments现在使用同一 actor mass；这消除了均匀树统计与加权 loss 的测度错配；
- B=2 时 weighted sibling-std normalization 仍会使 local advantage 的幅度主要由符号和 branch probability 决定，reward 差的绝对尺度会被归一化；
- top-up 样本是单 root segment，local advantage 恒为 0；默认 0.5/0.5 混合时它只保留一半 global 信号；
- 很短 segment 虽定位精细，但有效 token/梯度贡献小；很长 segment 又会把分叉后的不同决策混入同一 advantage，这正是当前方法的核心 trade-off。

### 5.5 Segment batching 不等于 segment 等权

`tree_segment_batch_strategy=segment` 决定的是调度单位；概率权重始终是 actor reach mass。`loss_agg_mode` 决定在该概率测度下按 token 还是按 segment 聚合：

- `token-mean`：目标为 reach-mass-weighted token mean，长 segment 按有效 token 数贡献；
- `seq-mean-token-mean`：目标为 reach-mass-weighted segment mean。

主 treesr 脚本没有覆盖 `loss_agg_mode`，因此沿用默认 `token-mean`。

所有带 `tree_loss_scales` 的 Tree、TreePR 和 TreeSR 训练当前都要求 FSDP data-parallel actor；Megatron 的 pipeline schedule 会先对各 micro-batch reducer 等权平均，再对 DP ranks 等权平均，无法保持这里的全局 actor mass，因此会在切 batch 前显式报错。TreeSR 的 unique-segment 重建还要求 `ulysses_sequence_parallel_size=1`，避免 sequence-parallel ranks 使用不同 segment shuffle。分布式 TreeSR 暂不接受 `seq-mean-token-sum-norm`，因为该模式以 rank-local padded segment width 为 denominator；推荐 `token-mean` 或 `seq-mean-token-mean`。

### 5.6 计算预算并未严格对齐

Top-up 后完整候选数和 optimizer step 数已对齐，但 tree 会共享一部分 decode prefix，分叉 child 又产生额外 prefill。因此候选数相同不代表 token 数、prefill 次数、FLOPs 或 wall time 相同。

### 5.7 EOS/stop 与 deferred probe

Branch candidates 当前只检查 vocab 范围，没有专门过滤 EOS/stop token；作为 child prompt token 后可能继续生成。训练使用 immediate entropy 分叉，不依赖 deferred probe，但仍应增加 EOS、stop strings 和 max-token 边界测试。

### 5.8 实验脚本中的凭据

部分 `tree/*.sh` 当前包含非空的 inline W&B API key fallback。该值不应进入共享仓库或本文。建议立即 revoke/rotate，并从脚本与 Git 历史中清理，统一使用环境变量或私有 key file。

## 6. 建议的配置方式

### 6.1 当前训练中的四组公平对照

```text
GRPO:       rollout.n=K, tree.enable=False
treerollout rollout.n=1, tree.enable=True, topup=True, process=False
treepr:     rollout.n=1, tree.enable=True, topup=True, process=True
treesr:     treepr + loss_mode=tree_segment
                    + tree_segment_batch_strategy=segment
                    + 明确选择 loss_agg_mode
```

所有组保持相同 prompt batch、`ppo_mini_batch_size`、PPO epochs、LR、response length 和 `K`。固定 tree 必须保持 `rollout.n=1`。

### 6.2 Entropy-only 与 legacy WAAD 的边界

verl 训练链路固定为 entropy-only，不再暴露 `tau_importance`。如需研究 entropy+WAAD，应使用 `tree_decoding_comparison/` 中显式设置 `branch_trigger_mode="entropy_waad"` 与 `collect_importance_stats=True` 的独立工具，不应复用训练配置。

### 6.3 建议记录的有效配置

每个 run 至少记录：

```text
effective branch_trigger_mode
effective entropy threshold
min_seg_length
B / D / target K
tree-only leaves / top-up count
branch points / depth
prompt batch / local leaf batch
normalized local PPO mini-batch
optimizer steps per PPO epoch
loss_mode / batch_strategy / loss_agg_mode
local/global advantage weights / proc_agg_mode
branch coverage / leaf-mass ESS / max leaf mass
chunked_prefill / TP size
```

## 7. 版本演进摘要

### vLLM

1. 增加 TreeSearchParams、Sequence tree metadata 和基础 tree branching；
2. 使用 ParallelSampleSequenceGroup 管理动态 child requests；
3. 修正 request/sequence ID、scheduler、log-prob、token/text 拼接与输出树结构；
4. 接入 attention importance，形成 entropy + deferred WAAD 两阶段分叉；
5. 增加 threshold stats、NaN 处理和 `min_seg_length`；
6. 增加 random trigger、显式 mode、任意候选预算 leaf cap 与测试。

### verl

1. 接入 vLLM tree rollout，并按叶子扩展 prompt/data batch；
2. 增加 tree metrics、动态 entropy p80 和评测工具；
3. 保存 unique segments/path，增加 bottom-up tree process advantage；
4. 增加 local/global advantage、长度聚合和 legacy inverse-sharing diagnostics；
5. 增加 tree_segment loss 与 leaf/segment 两种 batching；
6. 扩展 DataProto 的跨 worker segment concat/chunk，加入 dummy collective 防死锁；
7. 固定 top-up 到 `B^D`，并修复 optimizer step 数与 GRPO 不一致；
8. 将 token share weight 限定到 treepr，强化 routing/cache/process 数据契约；
9. 增加 random/entropy/WAAD 四路 pass@k 对比工具；
10. 用 conditional top-k actor leaf/reach mass 统一加权 advantage、PG、entropy 与 KL，并固定跨 micro-batch/DP-rank normalization；
11. 训练分叉强制 entropy-only，WAAD 仅由 comparison 显式 opt-in。

## 8. 相关文档与入口

- `README_tree.md`：早期 tree training 设计说明；
- `tree/DEBUG_SUMMARY.md`：tree rollout 调试记录；
- `tree_decoding_comparison/README.md`：四路 pass@k 工具使用说明；
- `tree_decoding_comparison/TREE_DECODING_COMPARISON.md`：comparison 预算与统计语义；
- `tests/utils/test_tree_training_on_cpu.py`：optimizer multiplier、process gate 和 batching helper 测试；
- `/Users/bytedance/codes/vllm/tests/engine/test_tree_decoding_helpers.py`：random trigger 与 leaf cap CPU 测试。

## 9. 辅助、数据与非核心变更

除上述运行主链路外，当前两个分支还包含以下配套修改。它们有助于复现实验或排查问题，但不应被误认为 tree decoding/训练目标本身的一部分。

| 类别 | 主要文件或目录 | 用途 |
|---|---|---|
| 数据构造 | `tree/scripts/generate_math_datasets.py`、`data/` 下的数学数据与说明 | 将 GSM8K/MATH 等数据整理成 verl 可读取的数据文件，并保存少量示例/评测数据 |
| 数值诊断 | `debug_std.py`、`debug_std2.py`、`test_fix.py` | 检查 tree process reward 的 sibling/global 标准化、零方差和 bottom-up 聚合行为 |
| verl CPU 测试 | `tests/utils/test_tree_training_on_cpu.py` | 在不启动完整分布式训练的情况下检查 tree multiplier、feature gate 和 segment 调度 helper |
| actor-mass CPU 测试 | `tests/trainer/ppo/test_tree_weighting_on_cpu.py` | 检查 branch 概率递归、top-up strata、segment reach、加权 advantage 以及 micro-batch/rank normalization |
| comparison 测试 | `tree_decoding_comparison/tests/` | 检查 exact-N candidate budget、组合式 pass@k、结果 schema、tree path 重建与参数校验 |
| vLLM benchmark/test | `/Users/bytedance/codes/vllm/benchmarks/` 和 `/Users/bytedance/codes/vllm/tests/` 中的 tree 相关文件 | 测试分叉、输出重建、WAAD/entropy 统计、随机 trigger、leaf cap 和延迟；部分早期脚本仅用于调试，不代表严格 equal-budget benchmark |
| 实验产物 | `tree/` 下的 TensorBoard event、日志和结果文件 | 历史运行记录；不参与代码执行，比较实验时应以对应脚本和有效 Hydra 配置为准 |
| 仓库清理 | `.gitignore` | 忽略 core dump、event/log 与 comparison 生成结果，避免把大体积运行产物继续提交到仓库 |

这些辅助文件中可能保留早期实验假设或一次性调试参数；判断当前功能语义时，应以第 2、3 节列出的实际运行源码为准。

本文描述的是当前分支的最终行为。历史提交中存在已经被后续修正的实现（例如 treerollout 也携带 share weights、tree 与 GRPO optimizer steps 不一致、segment 模式每 epoch 只更新一次等），不应再用旧提交的行为解释当前实验。
