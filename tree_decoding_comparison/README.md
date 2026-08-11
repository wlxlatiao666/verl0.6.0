# Tree Decoding pass@k 对比

该目录在相同的每题候选预算 `n` 下比较四种生成方式：

1. `base_grpo`：从 actor policy 独立采样 `n` 次；
2. `random_tree`：每个满足 depth / `min_seg_length` 的位置，以概率 0.2 分叉，不计算 entropy 或 WAAD；
3. `entropy_only_tree`：仅在 entropy 超过阈值时分叉；
4. `entropy_waad_tree`：entropy 与 WAAD 均超过阈值时分叉。

三种 tree 的分支 token 都使用相同的 deterministic top-k。Tree 最多产生 `n` 个完整叶子；不足的 slot 用普通 iid GRPO 采样补齐。内部节点不是完整答案，不参加 pass@k。

## 运行

```bash
export RAY_DATA_HOME=/path/to/data-root
export VLLM_SOURCE_PATH=/Users/bytedance/codes/vllm

./run_comparison.sh \
  --model-path /path/to/model \
  --n 8 \
  --branching-factor 2 \
  --max-tree-depth 3 \
  --random-branch-probability 0.2
```

wrapper 默认自动用 p80 标定 entropy 和 WAAD。若要使用手工阈值：

```bash
./run_comparison.sh \
  --model-path /path/to/model \
  --no-auto-calibrate-thresholds \
  --entropy-threshold 1.0 \
  --tau-importance 0.1
```

任意 `n / branching_factor / depth` 组合都可使用；预算不再依赖 `branching_factor ** depth`。例如 `n=10, B=4, depth=3` 时，引擎最多保留 10 个完整叶子，若实际只有 7 个，则另采 3 个 iid filler。

## 指标与输出

pass@k 使用与候选顺序无关的组合估计：

```text
1 - C(n-c, k) / C(n, k)
```

其中 `c` 是该题 `n` 个候选中的正确数。对相关的 tree leaves，它表示“从当前候选池均匀选择 k 个，至少一个正确”的概率。

输出目录包含：

- `generations.json`：四种方法的文本、正确性，以及每条候选的 `tree` / `grpo_filler` 来源；
- `results_summary.json`：pass@k、耗时、平均 tree leaf/filler 数和实际配置；
- `results_summary.csv`：四种方法及其相对 base 的差值；
- `sampled_examples.json`：本次题目与答案。

分析已有结果：

```bash
python3 analyze_results.py --results-dir ./tree_decoding_results
```

CPU 单测：

```bash
pytest -q test_comparison_helpers.py test_analyze_results.py
python3 test_evaluation.py
```

注意：这里对齐的是“完整候选数”，不是 generated-token 数、内部节点数或 FLOPs。不同树会共享不同数量的 prefix，因此若研究计算效率，应同时参考耗时或另行统计 token/KV-cache 开销。
