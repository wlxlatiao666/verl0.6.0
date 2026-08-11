# Threshold Auto-calibration Feature

This script now supports automatic calibration of `entropy_threshold` and `tau_importance` using vllm's `collect_threshold_stats` mode.

## How it works

1. Before the main experiment, the script runs a calibration pass on the first 10 prompts
2. For each prompt, it generates `n=5` sequences with shorter max tokens (200 by default)
3. It collects `entropy_list` and `importance_list` from all tokens across all sequences
4. It computes the 80th percentile (configurable) of the collected values
5. Uses those quantiles as `entropy_threshold` and `tau_importance` for the tree decoding run

## Usage

### Enable auto-calibration

```bash
# Using the shell script
./run_comparison.sh --model-path /path/to/model --auto-calibrate-thresholds

# Using Python directly
python3 tree_decoding_comparison.py --model-path /path/to/model --auto-calibrate-thresholds
```

### Customize calibration parameters

```bash
# Customize quantile and other calibration settings
./run_comparison.sh \
    --model-path /path/to/model \
    --auto-calibrate-thresholds \
    --calibration-quantile 0.85 \
    --calibration-n 3 \
    --calibration-max-tokens 100
```

## New Arguments

### Python script

| Argument | Default | Description |
|----------|---------|-------------|
| `--auto-calibrate-thresholds` | `False` | Enable auto-calibration mode |
| `--calibration-n` | `5` | Number of sequences per prompt for calibration |
| `--calibration-max-tokens` | `200` | Max tokens per sequence for calibration |
| `--calibration-quantile` | `0.8` | Quantile for threshold (0.8 = 80th percentile) |

### Shell script

| Argument | Default | Description |
|----------|---------|-------------|
| `--auto-calibrate-thresholds` | enabled | Enable auto-calibration mode |
| `--no-auto-calibrate-thresholds` | — | Use manual entropy/WAAD thresholds |
| `--calibration-n` | `5` | Number of sequences per prompt for calibration |
| `--calibration-max-tokens` | `200` | Max tokens per sequence for calibration |
| `--calibration-quantile` | `0.8` | Quantile for threshold |

## Output

When auto-calibration is enabled, you'll see additional output:

```
2.5. Auto-calibrating thresholds (using 80th percentile)...
Collecting threshold stats...
  Entropy stats: 25647 tokens
  Entropy range: [0.1234, 2.3456]
  Entropy 80th percentile: 1.5678
  Importance stats: 25647 tokens
  Importance range: [0.0123, 0.5678]
  Importance 80th percentile: 0.3456

Calibrated thresholds:
  entropy_threshold: 1.5678
  tau_importance: 0.3456
```

The calibrated thresholds are also saved in `results_summary.json`.

## Notes

- Calibration runs before the main experiment
- If calibration cannot collect both entropy and WAAD data, the run fails explicitly; it never silently substitutes `tau_importance=0`
- The main experiment will use the calibrated thresholds instead of the manually specified ones
- The calibrated entropy threshold is shared by entropy-only and entropy+WAAD; only the latter uses the WAAD threshold
