# Tree Decoding & Segment-Level PPO 文档

本文档介绍在 verl 官方源码基础上实现的 tree decoding 替换传统 rollout，以及 segment-level reward 计算和 micro batch 更新机制。

## 目录
1. [Rollout → Tree Decoding 替换与格式对齐](#1-rollout--tree-decoding-替换与格式对齐)
2. [Entropy-only 分叉与阈值标定](#2-entropy-only-分叉与阈值标定)
3. [Tree Process Reward: Local Advantage 与 Global Advantage](#3-tree-process-reward-local-advantage-与-global-advantage)
4. [Loss Mode = Tree Segment: 从 Sequence 到 Segment 的切分](#4-loss-mode--tree-segment-从-sequence-到-segment-的切分)
5. [Tree Segment Batch Strategy = Segment: Micro Batch 更新机制](#5-tree-segment-batch-strategy--segment-micro-batch-更新机制)

---

## 1. Rollout → Tree Decoding 替换与格式对齐

### 修改位置
- 主要调用入口: `verl/trainer/ppo/ray_trainer.py` (L1288-L1291)
- Worker 实现: `verl/workers/actor/dp_actor.py` (generate_sequences, 不在本 diff 内)

### 修改思路

传统 rollout 每次为每个 prompt 生成 `n` 个独立响应序列，而 tree decoding 生成一个搜索树，树的每个叶节点是一个完整响应序列。

**格式对齐设计**：

```python
# Tree search 返回的 batch 包含以下额外字段:
batch.non_tensor_batch["tree_prompt_indices"]  # 每个叶节点对应的 prompt 索引
batch.non_tensor_batch["tree_num_leaves"]       # 每个 worker 的叶节点数量
batch.non_tensor_batch["tree_num_prompts"]      # 每个 worker 的 prompt 数量
```

**叶节点格式对齐** (`ray_trainer.py` L1370-L1387):

```python
# Tree search: 每个 prompt 可能有可变数量的叶节点响应
# 从每个 worker 的局部索引重建全局 prompt 索引
if "tree_prompt_indices" in gen_batch_output.non_tensor_batch:
    local_idx = gen_batch_output.non_tensor_batch["tree_prompt_indices"].astype(int)
    num_leaves = gen_batch_output.non_tensor_batch["tree_num_leaves"].astype(int)
    num_prompts = gen_batch_output.non_tensor_batch["tree_num_prompts"].astype(int)
    global_idx = np.empty_like(local_idx)
    prompt_offset = 0
    i = 0
    while i < len(local_idx):
        chunk_leaves = int(num_leaves[i])
        chunk_prompts = int(num_prompts[i])
        for j in range(i, i + chunk_leaves):
            global_idx[j] = local_idx[j] + prompt_offset
        prompt_offset += chunk_prompts
        i += chunk_leaves
    batch = batch[global_idx]
else:
    # 传统方式: 简单重复
    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
```

这样，后续代码（reward 计算、advantage 计算等）可以无缝接收叶节点序列，无需感知底层是传统 rollout 还是 tree decoding。

---

## 2. Entropy-only 分叉与阈值标定

### 修改位置
- 主逻辑: `verl/trainer/ppo/ray_trainer.py` (L1257-L1282)

### 修改思路

训练侧显式设置 `branch_trigger_mode="entropy"`、`tau_importance=None`，只用 entropy threshold 决定是否立即分叉。阈值预采样设置 `collect_importance_stats=False`，不会进入 attention/WAAD 计算；WAAD 仅保留给独立 comparison 显式启用。

**收集 entropy 阈值统计** (`ray_trainer.py`):

```python
# 在 rollout 前收集 entropy 阈值统计；训练路径不计算 WAAD
_tree_cfg = self.config.actor_rollout_ref.rollout.get("tree_search", None)
if _tree_cfg is not None and _tree_cfg.get("enable", False):
    _interval = int(_tree_cfg.get("threshold_stats_interval", 1))
    _should_collect = (_interval <= 1) or (self.global_steps % _interval == 0)
    if _should_collect:
        with marked_timer("tree_threshold_stats", timing_raw, color="yellow"):
            stats_output = self.actor_rollout_wg.collect_threshold_stats(gen_batch)
```

**处理跨 worker 的统计** (`ray_trainer.py` L1267-L1282):

```python
# ONE_TO_ALL 返回列表（每个 worker 一个 DataProto）；跨 worker 平均
if isinstance(stats_output, list):
    _entropy_vals = [s.meta_info["entropy_p80"] for s in stats_output if "entropy_p80" in s.meta_info]
    _entropy_p80 = float(sum(_entropy_vals) / len(_entropy_vals)) if _entropy_vals else 1.0
else:
    _entropy_p80 = stats_output.meta_info.get("entropy_p80", 1.0)

# 更新 worker 的阈值
self.actor_rollout_wg.update_entropy_threshold(_entropy_p80)
```

**统计日志记录**:
- `tree/entropy_threshold`: 当前使用的熵阈值

### Tree rollout 的 actor occupancy 权重

Tree leaves 不是 iid actor samples，不能等权训练。trainer 在重算 `old_log_probs` 后，对每个实际展开的 sibling 集合计算：

```math
p(c\mid s,C)=\frac{\pi_{old}(a_c\mid s)}{\sum_{j\in C(s)}\pi_{old}(a_j\mid s)}
```

- leaf mass 是所属 root 的 leaf-slot prior，乘以 root-to-leaf 路径上的条件分叉概率；若 root 有 `d` 个 descendant leaves、prompt 共 `N` 条 emitted leaves，则 root prior 为 `d/N`；
- unique segment 的 reach mass 是所有 descendant leaf mass 之和；
- 若 tree 有 `m` 个叶子、另有 `k` 条普通 top-up，则展开 tree root 获得 `m/(m+k)` 总质量、每条 top-up 获得 `1/(m+k)`；因此 top-up 质量恰好等于 tree leaf 的平均质量，之后再按 prompt 归一；top-up 完整继承原 rollout 的采样参数与 LoRA；
- advantage、policy loss、entropy 和 KL 使用同一概率质量；
- scale 在完整 batch/所有 DP ranks 上一次性归一，micro-batch 不做局部 self-normalization。

这只恢复 actor 在已展开 top-k support 上的条件分布。未展开 tail 无样本，不能声称已经恢复完整 actor iid GRPO。通用 `rollout_is` 与 tree forced-token proposal 尚无正确对齐协议，二者同时启用会显式报错。

当前 actor-mass 的 micro-batch/DP 精确归约只在 FSDP actor 实现；Tree、TreePR 或 TreeSR 若使用 Megatron actor 会在切 batch 前显式报错，避免 Megatron 的 micro-batch/rank 等权平均悄悄改变概率质量。

---

## 3. Tree Process Reward: Local Advantage 与 Global Advantage

### 修改位置
- 核心函数: `verl/trainer/ppo/ray_trainer.py` (L263-L475) - `compute_tree_process_advantage`

### 修改思路

设置 `tree_process_reward=True` 时，不再使用传统 token-level advantage，而是从叶节点向上传播奖励，为树中每个 segment 计算 combined advantage。

#### 整体流程

```
步骤 1: Bottom-up 传播奖励 → 每个节点有自己的 score
步骤 2: 计算 Local & Global Advantage → 两者加权结合
步骤 3: 组装 Token-Level Advantage → 每个 token 填所属 segment 的 advantage
```

#### Step 1: Bottom-up 分数传播 (`ray_trainer.py` L291-L392)

```python
# 叶节点分数 = token-level rewards 之和
leaf_scores = data.batch["token_level_rewards"].sum(-1).float()

# 从叶节点分数反向传播到所有内部节点
# 内部节点分数 = 按 conditional actor mass 加权的子节点分数
for i in internal_nodes_sorted:
    children = children_of[i]
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    child_scores = node_scores[children_t]
    child_weights = segment_masses[children_t] / segment_masses[i]
    node_scores[i] = (child_weights * child_scores).sum()
```

#### Step 2: Local Advantage + Global Advantage (`ray_trainer.py` L394-L443)

**Local Advantage (局部优势)**:
- 相对于父节点和兄弟节点的优势
- 公式: `(score(s) - score(parent(s))) / (weighted_std(siblings) + eps)`
- 根节点 local advantage = 0

```python
for p, children in children_of.items():
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    sibling_scores = node_scores[children_t]
    sibling_weights = segment_masses[children_t] / segment_masses[p]
    sib_std = weighted_std(sibling_scores, sibling_weights)
    parent_score = node_scores[p]
    seg_local_advantages[children_t] = (sibling_scores - parent_score) / (sib_std + 1e-6)
```

**Global Advantage (全局优势)**:
- 相对于同一 prompt 所有叶节点的优势
- 按 Prompt 分组计算（类似 GRPO）
- 公式: `(score(s) - weighted_mean(leaf_scores)) / (weighted_std(leaf_scores) + eps)`

```python
# 按 prompt 分组计算
for prompt_id, prompt_leaf_segs in prompt_id_to_leaf_segs.items():
    prompt_leaf_segs_t = torch.tensor(prompt_leaf_segs, dtype=torch.long, device=device)
    prompt_leaf_node_scores = node_scores[prompt_leaf_segs_t]
    prompt_weights = tree_leaf_masses[prompt_rows]
    mean_leaf_score = weighted_mean(prompt_leaf_node_scores, prompt_weights)
    std_leaf_score = weighted_std(prompt_leaf_node_scores, prompt_weights)
    
    # 计算该 prompt 下所有 segment 的 global advantage
    prompt_all_segs = [i for i in range(n_unique) if seg_to_prompt_id.get(i, None) == prompt_id]
    prompt_all_segs_t = torch.tensor(prompt_all_segs, dtype=torch.long, device=device)
    seg_global_advantages[prompt_all_segs_t] = (node_scores[prompt_all_segs_t] - mean_leaf_score) / (std_leaf_score + 1e-6)
```

**Combined Advantage (加权结合)**:
```python
local_adv_weight = self.config.algorithm.get("local_adv_weight", 0.5)
global_adv_weight = self.config.algorithm.get("global_adv_weight", 0.5)
seg_advantages = local_adv_weight * seg_local_advantages + global_adv_weight * seg_global_advantages
```

#### Step 3: 组装 Token-Level Advantage (`ray_trainer.py` L445-L473)

每个叶节点的响应是其路径上 segment 的拼接，将每个 token 位置填为对应 segment 的 advantage：

```python
token_advantages = torch.zeros(n_leaves, resp_len, dtype=torch.float32, device=device)
for j, path in enumerate(leaf_segment_indices):
    pos = 0
    for seg_idx in path:
        seg_len = len(unique_segments[seg_idx])
        end = min(pos + seg_len, resp_len)
        # 两种聚合模式
        if proc_agg_mode == "raw":
            token_advantages[j, pos:end] = seg_advantages[seg_idx]
        elif proc_agg_mode == "length_balanced":
            valid_seg_len = max(end - pos, 1)
            token_advantages[j, pos:end] = seg_advantages[seg_idx] / valid_seg_len * mean_seg_len
        pos += seg_len
```

---

## 4. Loss Mode = Tree Segment: 从 Sequence 到 Segment 的切分

### 修改位置
- 切分函数: `verl/trainer/ppo/core_algos.py` (L1340-L1447) - `build_segment_tensors`
- Policy Loss: `verl/trainer/ppo/core_algos.py` (L1442-L1497) - `compute_policy_loss_tree_segment`
- Worker 调用: `verl/workers/actor/dp_actor.py` (L478-L683, L980-L1087)

### 修改思路

#### Tree Segment Data 结构

Tree decoding 返回的额外数据：
```python
data.non_tensor_batch["unique_segments"]        # 所有唯一 segment 列表 [n_unique]
data.non_tensor_batch["leaf_segment_indices"]   # 每个叶节点路径上的 segment 索引 [n_leaves]
data.non_tensor_batch["unique_segment_seq_ids"] # 每个 segment 的 token IDs（可选）
```

#### 重建 Segment Tensor (`core_algos.py` L1340-L1447)

**原理**: 每个 segment 至少在一个叶节点中完整出现，找到该叶节点并提取对应 token 范围。

```python
# Step 1: 为每个 segment 找到 canonical 叶节点和 token 偏移
seg_canonical: list[tuple[int, int]] = [(-1, -1)] * n_unique
for j, path in enumerate(leaf_segment_indices):
    offset = 0
    for seg_idx in path:
        if seg_canonical[seg_idx][0] == -1:
            seg_canonical[seg_idx] = (j, offset)  # (叶节点索引, token 偏移)
        offset += len(unique_segments[seg_idx])

# Step 2: 从 canonical 叶节点提取各 tensor
for seg_idx in range(n_unique):
    leaf_j, tok_offset = seg_canonical[seg_idx]
    seg_len = seg_lens[seg_idx]
    end = min(tok_offset + seg_len, resp_len)
    actual_len = end - tok_offset
    
    if log_prob is not None:
        seg_log_prob[seg_idx, :actual_len] = log_prob[leaf_j, tok_offset:end]
    seg_old_log_prob[seg_idx, :actual_len] = old_log_prob[leaf_j, tok_offset:end]
    
    # Advantage: 同 segment 所有位置值相同，取第一个有效值
    if actual_len > 0:
        first_pos = -1
        for pos in range(tok_offset, end):
            if response_mask[leaf_j, pos] > 0:
                first_pos = pos
                break
        if first_pos >= 0:
            seg_advantage_val = advantages[leaf_j, first_pos]
            seg_advantages[seg_idx, :actual_len] = seg_advantage_val
```

#### Worker 端的 Local 重建 (`dp_actor.py` L478-L568)

由于数据在 worker 间分片，每个 worker 只重建自己分到的叶节点涉及的 segments：

```python
# 从 global unique_segments 筛选出该 worker 叶节点用到的 segments
used_global_seg_indices = set()
for path in global_leaf_segment_indices:
    for seg_idx in path:
        used_global_seg_indices.add(seg_idx)
used_global_seg_indices = sorted(used_global_seg_indices)

# 建立 global → local 索引映射
global_to_local_seg_idx_map = {
    global_seg_idx: local_seg_idx
    for local_seg_idx, global_seg_idx in enumerate(used_global_seg_indices)
}

# 重建 local unique_segments 和 leaf_segment_indices
local_unique_segments = [global_unique_segments[i] for i in used_global_seg_indices]
local_leaf_segment_indices = [
    [global_to_local_seg_idx_map[i] for i in path]
    for path in global_leaf_segment_indices
]
```

#### Tree Segment Policy Loss (`core_algos.py` L1442-L1497)

结构与 vanilla PPO 相同，但输入是 segment-level tensors：

```python
@register_policy_loss("tree_segment")
def compute_policy_loss_tree_segment(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode, config, rollout_is_weights=None
):
    # 标准 PPO clip loss，但在 segment 维度计算
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.max(pg_losses1, pg_losses2)
    
    # Dual clip
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # 应用 IS weights（如有）
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

---

## 5. Tree Segment Batch Strategy = Segment: Micro Batch 更新机制

### 修改位置
- 主逻辑: `verl/workers/actor/dp_actor.py` (L570-L883)

### 修改思路

传统策略按 leaf 分 batch，`tree_segment_batch_strategy=segment` 改为按 segment 分 batch，实现更细粒度的更新。

#### 整体设计原则

1. **Worker 数据隔离**: 每个 worker 处理本地叶节点和 segments；仅用 all-reduce 汇总 weight/reducer 的两个标量统计
2. **Micro Batch 对齐**: 通过 dummy micro-batch 确保所有 worker 的 micro batch 数量一致，避免 FSDP 死锁
3. **Gradient Accumulation**: 用实际 reducer units 合并 micro-batch，保持与未切分 loss 严格等价

#### Micro Batch 对齐机制 (`dp_actor.py` L640-L683, L367-L417)

```python
# 收集所有 worker 的 micro batch 数量，取最大值
max_micro_batches = num_micro_batches
if dist.is_initialized() and world_size > 1:
    count_tensor = torch.tensor([num_micro_batches], dtype=torch.int64, device=get_device_id())
    gathered_counts = [torch.tensor([0], dtype=torch.int64, device=get_device_id()) for _ in range(world_size)]
    dist.all_gather(gathered_counts, count_tensor)
    max_micro_batches = max([cnt.item() for cnt in gathered_counts])

# 处理到 max_micro_batches，超出部分用 dummy
for m in range(max_micro_batches):
    is_dummy = m >= num_micro_batches
    if is_dummy:
        # Dummy micro-batch: 用真实数据跑一次，但 loss scale = 0
        if num_micro_batches > 0 and local_batch_size > 0:
            first_seg_indices = segment_micro_batches[0]
            # ... 构建 dummy batch ...
            # 关键: loss_scale_factor = 0，不更新梯度
            loss_scale_factor = 0.0
            # ... 正常 forward/backward ...
            loss = policy_loss * loss_scale_factor
            loss.backward()
        continue
    
    # Real micro-batch；mean-style reducer 使用全 DP optimizer group denominator
    loss_scale_factor = micro_reducer_units * world_size / global_group_reducer_units
    # ... 正常更新 ...
```

#### Segment-Based Batch 流程 (`dp_actor.py` L570-L883)

```python
# 1. 配置检查
tree_segment_batch_strategy = getattr(self.config, "tree_segment_batch_strategy", "leaf")
use_segment_batching = (
    loss_mode == "tree_segment" and
    tree_segment_batch_strategy == "segment" and
    local_tree_seg_targets is not None
)

if use_segment_batching:
    # 2. 获取 micro batch 大小
    ppo_micro_batch_segments = getattr(self.config, "ppo_micro_batch_segments", None)
    total_segments = len(local_tree_seg_targets["seg_lens"])
    
    if ppo_micro_batch_segments is None:
        # 默认: 按 leaf/segment 比例估算
        if self.config.ppo_micro_batch_size_per_gpu is not None:
            avg_segments_per_leaf = num_assigned_segments / max(local_batch_size, 1)
            ppo_micro_batch_segments = max(8, int(self.config.ppo_micro_batch_size_per_gpu * avg_segments_per_leaf))
        else:
            ppo_micro_batch_segments = max(8, num_assigned_segments // 10)
    
    target_optimizer_steps = max(1, ceil(local_leaf_count / ppo_mini_batch_size))
    for epoch in range(self.config.ppo_epochs):
        # 3. 先形成与 base GRPO 相同数量的 optimizer groups，再在组内切 micro-batch
        segment_indices = assigned_local_seg_indices.copy()
        random.shuffle(segment_indices)
        optimizer_groups = build_optimizer_micro_batches(
            segment_indices,
            num_optimizer_steps=target_optimizer_steps,
            micro_batch_size=ppo_micro_batch_segments,
        )

        for group in optimizer_groups:
            # 4. 汇总该 optimizer group 在所有 DP ranks 上的 reducer units
            global_group_reducer_units = all_reduce_sum(local_group_reducer_units)
            aligned_group = align_with_zero_scale_dummies(group)

            self.actor_optimizer.zero_grad()
            for seg_indices in aligned_group:
                if seg_indices is None:
                    run_dummy_forward_backward(loss_scale_factor=0.0)
                    continue

                # 5. 只 forward canonical leaves，并提取 PG/entropy/KL 的相同 segment slice
                required_leaves = canonical_leaves(seg_indices)
                entropy, log_prob = self._forward_micro_batch(data[required_leaves], ...)
                segment_tensors = gather_canonical_slices(
                    log_prob,
                    entropy,
                    ref_log_prob,
                    seg_indices,
                )
                policy_loss = compute_weighted_segment_loss(segment_tensors)

                # 6. 用固定的全组 denominator 合并 micro-batch；不在 micro 内重归一权重
                loss_scale_factor = (
                    micro_reducer_units * world_size / global_group_reducer_units
                )
                (policy_loss * loss_scale_factor).backward()

            # 每个 group 恰好对应一次 base-GRPO optimizer step
            self._optimizer_step()
```

#### Leaf-Based Batch 兼容路径 (`dp_actor.py` L980-L1087)

仍支持 `tree_segment_batch_strategy=leaf`（默认），此时在 leaf batch 内筛选 segments：

```python
# 选择 canonical leaf 在当前 micro-batch 的 segments
if local_to_global_seg_idx_map is None:
    # 原始行为
    seg_indices = [i for i, (leaf_j, _) in enumerate(seg_canonical) if leaf_j in present_leaves]
else:
    # DataProto.chunk 已完成 local segment 重映射；禁止再次按 rank 取模过滤
    seg_indices = [i for i, (leaf_j, _) in enumerate(seg_canonical) if leaf_j in present_leaves]
```

---

## 内存优化细节

在 `ray_trainer.py` L1416-L1427 实现了临时移除大数据以减少内存传输：

```python
# 优化: 暂时移除 tree segment data 以减少内存传输
# Tree data 只在 compute_tree_process_advantage 需要
# 注意: 保留 worker_*_offsets，因为 chunk() 需要！
tree_data_cache = {}
tree_segment_large_keys = ["unique_segments", "unique_segment_seq_ids", "leaf_segment_indices"]
has_tree_data = any(key in batch.non_tensor_batch for key in tree_segment_large_keys)
if has_tree_data:
    for key in tree_segment_large_keys:
        if key in batch.non_tensor_batch:
            tree_data_cache[key] = batch.non_tensor_batch.pop(key)

# [ compute_log_prob, compute_ref_log_prob, compute_values ... ]

# Restore tree data for compute_tree_process_advantage
if tree_data_cache:
    for key, value in tree_data_cache.items():
        batch.non_tensor_batch[key] = value
```
