# Tree Decoding vs Base GRPO Comparison

此目录包含 Tree Decoding 与 Base GRPO 采样的对比实验代码。

## 文件说明

| 文件 | 说明 |
|------|------|
| `tree_decoding_comparison.py | 主实验脚本 |
| `run_comparison.sh` | 便捷运行脚本 |
| `analyze_results.py` | 结果分析脚本 |
| `test_evaluation.py` | 评估逻辑测试 |
| `TREE_DECODING_COMPARISON.md` | 完整文档 |

## 快速开始

```bash
# 设置数据路径
export RAY_DATA_HOME=/path/to/data

# 使用 wrapper 脚本运行
./run_comparison.sh --model-path /path/to/model

# 或直接运行 Python 脚本
python tree_decoding_comparison.py --model-path /path/to/model
```

## 实验配置

- 数据集：dapo-math (500条唯一查询)
- Base GRPO：n=8，独立采样
- Tree Decoding：branching_factor=2，max_tree_depth=3，不足8条时用baseline补齐
- 评估指标：pass@1 至 pass@8

## 结果分析

运行完实验后，可以运行分析脚本：

```bash
python analyze_results.py --results-dir ./tree_decoding_results
```
