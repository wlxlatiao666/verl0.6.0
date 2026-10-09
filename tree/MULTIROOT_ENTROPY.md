# Multi-root shallow entropy rollout

Implementation: 2026-10-09.

Each original question starts 8 independent vLLM requests, each with at most
8 leaves (branching factor 2, depth 3). rollout.n remains 1. A branch requires
segment output length >= 64, depth < 3, remaining leaf capacity, and entropy
strictly above the current threshold. No probe or WAAD is used.

Candidate tokens use sample_with_replacement: two independent categorical
draws from softmax(logprobs / branch_temperature), temperature 1. Repeated
tokens are retained as separate children. This is not top-k or Gumbel-top-k.
Each seeded root receives a reproducible distinct seed; seeded children also
receive separate streams. Mutable tree parameters are deep-copied per root.

Threshold calibration reuses the reference pre-pass and its sample/length
settings, but selects entropy quantile 0.8 (P80), matching the reference. Calibration runs
before the first rollout and every 10 training steps (threshold_stats_interval=10 in both launchers). The trainer
averages worker quantiles: this is an approximation, not the exact pooled
global P80. entropy_threshold=0.8 remains the initial fallback setting.
The threshold pre-pass is ordinary actor sampling, without answer rewards.

Each root's deficit to 8 leaves is filled with ordinary independent samples
from the original question. Therefore every question has 64 responses. All
roots and top-ups map back to the same original UID, including across worker
chunks. PPO response multiplicity remains 64. Metrics per_prompt refer to
original questions; avg_leaves_per_root additionally reports root-level usage.

treePR=True preserves existing segment sharing weights and local/global
advantage computation. Nodes of independent roots are not connected as
siblings; global statistics include all responses of the original question.
treePR=False emits no process-reward segments or inverse-sharing weights and
uses ordinary GRPO with the tree-generated responses.

## Run on the training instance

cd /inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0

# treePR
bash tree/scripts/train_qwen2.5_math7b-multiroot-entropy-treepr.sh

# rollout-only + GRPO
bash tree/scripts/train_qwen2.5_math7b-multiroot-entropy-grpo.sh

Both use project verl_grpo_tree_0722 and distinct experiment names.
Defaults preserve the reference's 8 GPUs, batch 96, PPO mini-batch 16,
micro-batch 2/GPU, learning rate 1e-6, prompt/response length 2048, model,
data, reward configuration, validation and checkpoint intervals.
Hydra-resolved comparisons found only the intended tree settings, experiment
name and derived output directories differ from train_qwen2.5_math7b-treepr.sh.
The two new scripts differ only by tree_process_reward and run identity.

WANDB_MODE defaults to offline, matching the reference; export WANDB_MODE=online
to upload live with an existing wandb login or WANDB_API_KEY. No key is embedded.
DRY_RUN=1 resolves Hydra config without starting training. Additional Hydra
overrides may be passed as script arguments. Do not set rollout.n=8.

## Verification

CPU/helper and mocked-engine tests cover root isolation, original UID routing,
top-ups, per-question multiplicity, single-root compatibility, treePR global
statistics across roots with separate local siblings, and quantile calibration.
Existing vLLM sampling/gating tests cover duplicate categorical samples and
branch budgets.

A real Qwen2.5-Math-7B smoke test ran two questions, 8 roots each, response
length 256, static entropy threshold 0.8. It produced 128 responses, 64 per
question. The same decoded outputs were assembled with treePR off and on;
synthetic rewards validated finite advantages. This is an integration check,
not an accuracy benchmark or an 8-GPU RL training run.

Reproduce the smoke test:
python3 tests/utils/multiroot_decode_smoke.py \
  --model /inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B \
  --output /tmp/multiroot-decode.json

Tests:
VLLM_USE_V1=0 python3 -m pytest --noconftest -q \
  tests/utils/test_multiroot_rollout.py tests/utils/test_tree_training_on_cpu.py

Production-budget proportion test: tests/utils/measure_multiroot.py uses 20 held-out
questions, 2048-token responses, batches of 2, and P80 updates at batches 1 and 10.
The actor is fixed; this measures decode composition, not RL performance.
