# High-entropy conditional probe pilot

This experiment retrains the setting-A classifier on high-entropy positions only.
It does not start RL. Existing sample-without-replacement candidate outcomes are
reused; no deterministic-top-k labels are substituted.

- Fit a fixed entropy P80 cutoff on saved **training** positions, without labels.
- Retain H >= cutoff in train, validation and test; report categories/counts.
- Fit RMS normalization + PCA (up to 32 dimensions) and balanced L2 logistic
  classification on eligible training positions, with A's original label rules.
- Fit the probe logit P80 cutoff on eligible validation positions. Freeze both
  cutoffs before reading test labels. Test includes uncertain/background points.
- Report within-high-entropy ranking/U against entropy and exact expected random
  selection. These metrics are conditional and cannot be directly compared with
  original A's all-position metrics.
- Online `probe_entropy` checks H >= cutoff first and computes probe scores only
  for eligible sampler rows; branching additionally requires score >= cutoff,
  minimum segment length and depth/budget eligibility.
- Evaluate 100 existing fresh-question trial questions, seed42, 16 responses per
  question, against entropy-only and random gates using the same model/budget.
  These questions have been used in a previous diagnostic evaluation, so this
  is a paired comparison set, not an untouched final holdout. Do not tune on it.
- Branch candidates retain Gumbel-top-k weighted sampling without replacement.

This pilot filters previously saved sparse positions; P80 is **not** calibrated
on all tokens of full trajectories. Filtering reduces the original ~940 training
positions to ~188 (actual counts are logged). Sparse eligible validation/test
points and small question counts limit conclusions. If positive counts are low,
collect additional high-entropy positions/questions rather than selecting test
thresholds. PCA32 is an additional change from A's PCA128; this pilot alone does
not isolate the causal contribution of entropy conditioning.

After synchronizing the modified vLLM files and verl files to the server:

```bash
cd /inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0
CUDA_VISIBLE_DEVICES=0 nohup bash tree/scripts/run_high_entropy_probe.sh \
  > /inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/high-entropy-probe-launch.log 2>&1 &
```

`WORK_DIR`, `OUTPUT_DIR`, `QUESTIONS_FILE` can override defaults. Output:
`fit/screening.json`, `fit/validation.json`, `fit/test.json`, `fit/A.pt`,
`online/comparison.json`, mode logs and `status.txt`. Completed question outputs
resume under an identical manifest. An interrupted fit after checkpoint freezing
requires a new output directory (it intentionally cannot silently overwrite).

Local syntax and engine boundary checks passed. Tensor tests in
`tests/engine/test_branch_probe.py` require the server's PyTorch/vLLM environment.
At implementation time SSH initially connected, then subsequent connections
repeatedly timed out during banner exchange; no files were deployed and no
remote training was launched.
