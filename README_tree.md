# Tree Decoding & Segment-Level PPO 文档

本文档介绍在 verl 官方源码基础上实现的 tree decoding 替换传统 rollout，以及 segment-level reward 计算和 micro batch 更新机制。

## 目录
1. [Rollout → Tree Decoding 替换与格式对齐](#1-rollout--tree-decoding-替换与格式对齐)
2. [Entropy Threshold 与 Tau Importance 采样逻辑](#2-entropy-threshold-与-tau-importance-采样逻辑)
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

## 2. Entropy Threshold 与 Tau Importance 采样逻辑

### 修改位置
- 主逻辑: `verl/trainer/ppo/ray_trainer.py` (L1257-L1282)

### 修改思路

为 tree decoding 动态调整搜索空间，使用 entropy threshold 控制探索深度，使用 tau importance 控制采样重要性。

**收集阈值统计** (`ray_trainer.py` L1260-L1266):

```python
# 在 rollout 前收集熵/重要性阈值统计
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
    _imp_vals = [s.meta_info["importance_p80"] for s in stats_output if s.meta_info.get("importance_p80") is not None]
    _importance_p80 = float(sum(_imp_vals) / len(_imp_vals)) if _imp_vals else None
else:
    _entropy_p80 = stats_output.meta_info.get("entropy_p80", 1.0)
    _importance_p80 = stats_output.meta_info.get("importance_p80", None)

# 更新 worker 的阈值
self.actor_rollout_wg.update_entropy_threshold(_entropy_p80)
if _importance_p80 is not None:
    self.actor_rollout_wg.update_tau_importance(_importance_p80)
```

**统计日志记录**:
- `tree/entropy_threshold`: 当前使用的熵阈值
- `tree/tau_importance`: 当前使用的重要性采样参数

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
# 内部节点分数 = 子节点分数的平均
for i in internal_nodes_sorted:
    children = children_of[i]
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    child_scores = node_scores[children_t]
    node_scores[i] = child_scores.mean()
```

#### Step 2: Local Advantage + Global Advantage (`ray_trainer.py` L394-L443)

**Local Advantage (局部优势)**:
- 相对于父节点和兄弟节点的优势
- 公式: `(score(s) - score(parent(s))) / (std(siblings) + eps)`
- 根节点 local advantage = 0

```python
for p, children in children_of.items():
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    sibling_scores = node_scores[children_t]
    sib_std = sibling_scores.std() if len(children) > 1 else torch.tensor(0.0, device=device)
    parent_score = node_scores[p]
    seg_local_advantages[children_t] = (sibling_scores - parent_score) / (sib_std + 1e-6)
```

**Global Advantage (全局优势)**:
- 相对于同一 prompt 所有叶节点的优势
- 按 Prompt 分组计算（类似 GRPO）
- 公式: `(score(s) - mean(leaf_scores)) / (std(leaf_scores) + eps)`

```python
# 按 prompt 分组计算
for prompt_id, prompt_leaf_segs in prompt_id_to_leaf_segs.items():
    prompt_leaf_segs_t = torch.tensor(prompt_leaf_segs, dtype=torch.long, device=device)
    prompt_leaf_node_scores = node_scores[prompt_leaf_segs_t]
    mean_leaf_score = prompt_leaf_node_scores.mean()
    std_leaf_score = prompt_leaf_node_scores.std() if len(prompt_leaf_node_scores) > 1 else torch.tensor(0.0, device=device)
    
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

1. **Worker 数据隔离**: 每个 worker 只处理自己 rollout 产生的叶节点和 segments，无需跨 worker 通信
2. **Micro Batch 对齐**: 通过 dummy micro-batch 确保所有 worker 的 micro batch 数量一致，避免 FSDP 死锁
3. **Gradient Accumulation**: 正确调整 gradient accumulation 步数

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
    
    # Real micro-batch
    loss_scale_factor = 1 / self.gradient_accumulation
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
    
    for epoch in range(self.config.ppo_epochs):
        # 3. Shuffle segments
        segment_indices = assigned_local_seg_indices.copy()
        random.shuffle(segment_indices)
        
        # 4. Split into micro batches
        segment_micro_batches = [
            segment_indices[i:i + ppo_micro_batch_segments]
            for i in range(0, len(segment_indices), ppo_micro_batch_segments)
        ]
        num_micro_batches = len(segment_micro_batches)
        
        # 5. 重要: gradient_accumulation = micro_batch 数量
        self.gradient_accumulation = num_micro_batches
        
        # 6. Micro batch 对齐（见上）
        # ... [align + dummy logic] ...
        
        self.actor_optimizer.zero_grad()
        for m in range(max_micro_batches):
            if is_dummy:
                # ... dummy ...
            else:
                seg_indices = segment_micro_batches[m]
                
                # 7. 收集这些 segment 涉及的叶节点
                seg_canonical = local_tree_seg_targets["seg_canonical"]
                required_leaves = list({seg_canonical[i][0] for i in seg_indices})
                
                # 8. Forward pass on required leaves
                mini_batch = data[required_leaves]
                model_inputs = {**mini_batch.batch, **mini_batch.non_tensor_batch}
                entropy, log_prob = self._forward_micro_batch(...)
                
                # 9. 构建 segment log_prob
                leaf_inverse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(required_leaves)}
                max_seg_len = local_tree_seg_targets["old_log_prob"].shape[1]
                log_prob_pieces = []
                for seg_idx in seg_indices:
                    leaf_j, tok_offset = seg_canonical[seg_idx]
                    local_leaf_j = leaf_inverse_map[leaf_j]
                    seg_len = seg_lens[seg_idx]
                    end = min(tok_offset + seg_len, log_prob.shape[1])
                    piece = log_prob[local_leaf_j, tok_offset:end]
                    if piece.shape[0] < max_seg_len:
                        piece = torch.nn.functional.pad(piece, (0, max_seg_len - piece.shape[0]))
                    log_prob_pieces.append(piece)
                seg_log_prob_local = torch.stack(log_prob_pieces)
                
                # 10. 获取其他 segment tensors
                seg_old_log_prob_local = local_tree_seg_targets["old_log_prob"][seg_indices]
                seg_advantages_local = local_tree_seg_targets["advantages"][seg_indices]
                seg_response_mask_local = local_tree_seg_targets["response_mask"][seg_indices]
                
                # 11. Compute loss & backward
                policy_loss_fn = get_policy_loss_fn(loss_mode)
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    old_log_prob=seg_old_log_prob_local,
                    log_prob=seg_log_prob_local,
                    advantages=seg_advantages_local,
                    response_mask=seg_response_mask_local,
                    ...
                )
                loss_scale_factor = 1 / self.gradient_accumulation
                loss = policy_loss * loss_scale_factor
                loss.backward()
        
        # Optimizer step after full gradient accumulation
        grad_norm = self._optimizer_step()
```

#### Leaf-Based Batch 兼容路径 (`dp_actor.py` L980-L1087)

仍支持 `tree_segment_batch_strategy=leaf`（默认），此时在 leaf batch 内筛选 segments：

```python
# 选择 canonical leaf 在当前 micro-batch 的 segments
if local_to_global_seg_idx_map is None:
    # 原始行为
    seg_indices = [i for i, (leaf_j, _) in enumerate(seg_canonical) if leaf_j in present_leaves]
else:
    # 新: 同时确保该 segment 归属于该 worker（通过 global_seg_idx % world_size == rank）
    seg_indices = []
    for i, (leaf_j, _) in enumerate(seg_canonical):
        if leaf_j in present_leaves:
            global_seg_idx = local_to_global_seg_idx_map[i]
            if global_seg_idx % world_size == rank:
                seg_indices.append(i)
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
