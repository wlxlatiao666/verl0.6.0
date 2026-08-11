# Four-way Tree Decoding Comparison

## Method semantics

| Result key | Branch-position trigger | WAAD | Branch actions |
|---|---|---|---|
| `base_grpo` | none; `n` iid rollouts | no | policy sample |
| `random_tree` | Bernoulli(`p`) at each eligible token | no | deterministic policy top-k |
| `entropy_only_tree` | `entropy > entropy_threshold` | no | deterministic policy top-k |
| `entropy_waad_tree` | entropy gate followed by deferred `WAAD > tau_importance` | yes | deterministic policy top-k |

An eligible tree token must also satisfy `tree_depth < max_tree_depth` and `segment_length >= min_seg_length`.

`branch_trigger_mode=None` remains available inside vLLM for backward compatibility: `tau_importance=None` resolves to entropy-only and a numeric tau resolves to entropy+WAAD. Existing verl training code does not pass the new comparison fields, so its behavior is unchanged.

## Candidate budget

For every prompt and method:

```text
complete tree leaves + fresh iid GRPO fillers = n
```

vLLM receives `max_num_leaves=n`, so a split is capped before it can overrun the budget. A final split may use fewer than `branching_factor` children when only part of the budget remains. If depth, EOS, or trigger conditions leave fewer than `n` leaves, the comparison runner performs one variable-`n` conventional generation call per affected prompt.

Empty completions and duplicate texts remain valid sampling outcomes and each consume one slot. The runner never deduplicates, truncates an over-budget tree, copies an existing answer, or manufactures an empty filler. Any count mismatch is a hard error.

Internal parent nodes are saved by vLLM for path reconstruction but are not complete answer candidates. Candidate-count equality therefore does not imply equal token/FLOP budgets.

## pass@k

The output list has no meaningful common order: tree traversal order, top-k rank, and filler placement differ by mode. The runner therefore computes, per problem with `n` candidates and `c` correct candidates:

```text
pass@k = 1 - C(n-c, k) / C(n, k)
```

For iid GRPO this is the standard estimator. For correlated tree leaves, interpret it as the success probability of a uniformly selected k-subset of the observed candidate pool.

## Threshold calibration

With `--auto-calibrate-thresholds`, a short observation-only rollout gathers entropy and WAAD statistics and applies the requested quantile (p80 by default). The resulting entropy threshold is shared by both entropy methods; the calibrated WAAD threshold is used only by `entropy_waad_tree`.

The comparison LLM disables chunked prefill because this custom WAAD implementation only computes importance for decode-only batches. Missing entropy or WAAD statistics causes a clear failure instead of silently falling back to zero.

## Reproducibility and artifacts

Use `--seed` for model sampling and the stateless random-branch gate. Set `VLLM_SOURCE_PATH`, or pass `--vllm-source-path` to the shell wrapper, to select the patched checkout. At startup, the runner prints and records the actual imported `vllm.__file__`.

Each tree method records per prompt:

- `tree_leaf_count`;
- `tree_internal_node_count` and `tree_output_node_count`;
- `filler_count` and strict `total_count`;
- `candidate_sources` aligned with the candidate text list.

Wall-clock time for a tree method includes its own iid filler generation.
