# Tree Decoding vs Base GRPO Comparison Experiment

This directory contains code to run an end-to-end comparison experiment between
**Tree Decoding** (high-entropy branching strategy) and **Base GRPO** (independent sampling)
on the dapo-math dataset.

## Overview

The experiment:
1. Loads the dapo-math dataset and samples 500 unique queries
2. Uses both sampling methods to generate 8 responses per query
3. Evaluates pass@1 through pass@8 for both methods
4. Outputs a detailed comparison

**Tree Decoding**: Uses vllm's tree search with branching_factor=2, max_tree_depth=3
(2^3 = 8 potential leaves). If the tree produces fewer than 8 leaves, the remaining
are filled with base sampling.

**Base GRPO**: Uses independent sampling with n=8.

## Quick Start

### Prerequisites

- vllm (available at `/Users/weilongxuan/codes/vllm`)
- Python 3.8+
- PyTorch
- pandas, numpy, pyarrow
- sympy (for math evaluation)

### Environment Setup

```bash
# Set RAY_DATA_HOME to where dapo-math-17k.parquet is located
export RAY_DATA_HOME=/path/to/your/data

# Ensure vllm is in PYTHONPATH (handled automatically by the script)
```

### Run with the wrapper script

```bash
# Basic usage
./run_comparison.sh --model-path /path/to/your/model

# With custom parameters
./run_comparison.sh \
    --model-path /path/to/model \
    --num-samples 500 \
    --n 8 \
    --branching-factor 2 \
    --max-tree-depth 3 \
    --temperature 0.7 \
    --max-tokens 1024 \
    --output-dir ./results
```

### Run directly with Python

```bash
# Basic usage
python tree_decoding_comparison.py --model-path /path/to/your/model

# Full options
python tree_decoding_comparison.py \
    --model-path /path/to/model \
    --dataset-path ${RAY_DATA_HOME}/data/dapo-math-17k.parquet \
    --num-samples 500 \
    --n 8 \
    --branching-factor 2 \
    --max-tree-depth 3 \
    --entropy-threshold 0.8 \
    --temperature 0.7 \
    --max-tokens 1024 \
    --output-dir ./tree_decoding_results \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

## Command Line Arguments

### Dataset

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | `${RAY_DATA_HOME}/data/dapo-math-17k.parquet` | Path to dapo-math dataset |
| `--num-samples` | 500 | Number of unique queries to sample |

### Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to model checkpoint |
| `--tensor-parallel-size` | 1 | Tensor parallel size |
| `--dtype` | float16 | Data type |
| `--gpu-memory-utilization` | 0.9 | GPU memory utilization |
| `--max-model-len` | 4096 | Max model length |

### Sampling

| Argument | Default | Description |
|----------|---------|-------------|
| `--n` | 8 | Number of sequences per prompt |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | 1024 | Max tokens to generate |

### Tree Decoding

| Argument | Default | Description |
|----------|---------|-------------|
| `--branching-factor` | 2 | Tree branching factor |
| `--max-tree-depth` | 3 | Max tree depth |
| `--entropy-threshold` | 0.8 | Entropy threshold for branching |
| `--tau-importance` | None | Tau for importance sampling (optional) |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./tree_decoding_results` | Output directory |

## Output Files

The experiment produces the following files in the output directory:

| File | Description |
|------|-------------|
| `sampled_examples.json` | The 500 sampled problems with ground truth |
| `generations.json` | All generations from both methods + correctness labels |
| `results_summary.json` | Pass@k results + timing + config |
| `results_summary.csv` | Pass@k results in CSV format for easy analysis |

## Example Output

```
================================================================================
COMPARISON SUMMARY
================================================================================

Metric          Base GRPO       Tree Decoding   Improvement
------------------------------------------------------------
pass@1          0.4520          0.5240          +0.0720 (+15.9%)
pass@2          0.5480          0.6120          +0.0640 (+11.7%)
pass@3          0.6040          0.6760          +0.0720 (+11.9%)
pass@4          0.6480          0.7120          +0.0640 (+9.9%)
pass@5          0.6840          0.7440          +0.0600 (+8.8%)
pass@6          0.7080          0.7640          +0.0560 (+7.9%)
pass@7          0.7200          0.7760          +0.0560 (+7.8%)
pass@8          0.7320          0.7840          +0.0520 (+7.1%)
------------------------------------------------------------
Time (s)        120.45          135.20          +14.75
```

## Evaluation Methodology

### Answer Extraction

Answers are extracted from model generations by looking for `\boxed{...}` or `\fbox{...}`.
If no boxed answer is found, the response is marked as incorrect.

### Grading

Two answers are considered matching if:
1. Normalized string match (case-insensitive, ignoring common units/formatting)
2. Numerical match (within 1e-4 tolerance)
3. Symbolic match (using sympy, as a fallback)

### Pass@k Calculation

For each problem, we have 8 ordered responses. pass@k is the fraction of problems
where at least one of the first k responses is correct.

## Directory Structure

```
.
├── tree_decoding_comparison.py  # Main experiment script
├── run_comparison.sh            # Convenience wrapper
└── TREE_DECODING_COMPARISON.md  # This file
```

## Notes

1. **vllm Integration**: The script adds `/Users/weilongxuan/codes/vllm` to sys.path.
   Modify this path if your vllm is located elsewhere.

2. **Tree Leaf Handling**: Tree Decoding may produce fewer than 8 leaves. The script
   automatically fills the remaining with base sampling (slightly higher temperature
   for diversity).

3. **Determinism**: For reproducibility, the dataset sampling uses random_state=42.
   Generation sampling is not deterministic by default (add seed to SamplingParams
   if needed).

4. **Memory Usage**: Adjust `--gpu-memory-utilization` if you encounter OOM errors.

## Troubleshooting

### vllm not found

Make sure vllm is at `/Users/weilongxuan/codes/vllm`, or modify the sys.path in
`tree_decoding_comparison.py`.

### Dataset not found

Set `RAY_DATA_HOME` environment variable, or provide the full path with `--dataset-path`.

### OOM Errors

- Reduce `--tensor-parallel-size` (if larger than 1)
- Reduce `--gpu-memory-utilization`
- Reduce `--max-model-len`
- Reduce `--max-tokens`

### Slow generation

Try increasing `--tensor-parallel-size` (if you have multiple GPUs available).

## Extending

### Custom Evaluation Metrics

Add new evaluation metrics in the `compute_all_pass_k` function or create a
new function that operates on the results.

### Custom Dataset

Modify `load_dapo_math_dataset` and `extract_unique_queries` to work with
your dataset format.

### Different Branching Strategies

Modify `generate_tree_decoding` to experiment with different branching
strategies or ways of filling missing leaves.
