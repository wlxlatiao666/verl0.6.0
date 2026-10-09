#!/usr/bin/env bash
# Frozen Probe A gates tree branches; retain the reference tree-process-reward RL settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK_ROOT=${TASK_ROOT:-/inspire/hdd/global_user/weilongxuan-253108120168}
MODEL_PATH=${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B}
TRAIN_FILE=${TRAIN_FILE:-$TASK_ROOT/verl0.6.0/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-$TASK_ROOT/verl0.6.0/data/aime-2024.parquet}
PROBE_PATH=${PROBE_PATH:-$TASK_ROOT/probe_runs/math7b-sample-abcd-seed42/abcd/A.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/inspire/qb-ilm2/project/neosmosis/weilongxuan-253108120168/verl_data}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_tree_probe}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_math7b_treepr_probeA_$(date +%Y%m%d_%H%M%S)}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
LOG_DIR=${LOG_DIR:-$TASK_ROOT/verl_logs}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

export PYTHONUNBUFFERED=1 VLLM_USE_V1=0 RAY_DEDUP_LOGS=0 NCCL_SHM_DISABLE=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_DEBUG_LOG_PATH=${VERL_DEBUG_LOG_PATH:-$TASK_ROOT}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$TASK_ROOT/wandb_offline}
mkdir -p "$WANDB_DIR"
# Offline wandb needs no embedded API key. Supply credentials externally for online logging.
for input_path in "$MODEL_PATH" "$TRAIN_FILE" "$TEST_FILE" "$PROBE_PATH"; do
    [[ -e "$input_path" ]] || { echo "Missing input: $input_path" >&2; exit 1; }
done
CHECKPOINT_THRESHOLD=$(python3 - "$PROBE_PATH" <<'PY'
import math
import sys
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if c.get("group") != "A":
    raise ValueError("Expected an ABCD setting-A checkpoint")
t = float(c["threshold"])
if not math.isfinite(t):
    raise ValueError("Invalid checkpoint threshold")
print(repr(t))
PY
)
PROBE_THRESHOLD=${PROBE_THRESHOLD:-$CHECKPOINT_THRESHOLD}
echo "Probe: $PROBE_PATH; raw-logit threshold: $PROBE_THRESHOLD; GPUs: $N_GPUS_PER_NODE"
echo "Log: $LOG_FILE"

command_args=(python3 -m verl.trainer.main_ppo)
# Compose the full Hydra config without allocating GPUs or starting Ray/training.
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    command_args+=(--cfg job --resolve)
fi
"${command_args[@]}" \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=96 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_async_output_proc=True \
    actor_rollout_ref.rollout.tree_search.enable=True \
    actor_rollout_ref.rollout.tree_search.branch_trigger_mode=probe \
    actor_rollout_ref.rollout.tree_search.branch_probe_path="$PROBE_PATH" \
    actor_rollout_ref.rollout.tree_search.branch_probe_threshold="$PROBE_THRESHOLD" \
    actor_rollout_ref.rollout.tree_search.tau_importance=null \
    actor_rollout_ref.rollout.tree_search.min_seg_length=10 \
    actor_rollout_ref.rollout.tree_search.branching_factor=4 \
    actor_rollout_ref.rollout.tree_search.max_tree_depth=3 \
    actor_rollout_ref.rollout.tree_search.branch_sampling=sample \
    actor_rollout_ref.rollout.tree_search.branch_temperature=1.0 \
    actor_rollout_ref.rollout.tree_search.topup_leaves_to_target=True \
    actor_rollout_ref.rollout.tree_search.tree_process_reward=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=4096 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb","tensorboard"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    +ray_kwargs.ray_init.log_to_driver=True \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUTPUT_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.rollout_data_dir="$OUTPUT_ROOT/rollout_data/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.validation_data_dir="$OUTPUT_ROOT/validation_data/$PROJECT_NAME/$EXPERIMENT_NAME" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    "$@" 2>&1 | tee "$LOG_FILE"
