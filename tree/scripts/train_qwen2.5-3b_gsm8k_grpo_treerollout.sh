#!/usr/bin/env bash
# =============================================================================
#  Ablation: Qwen2.5-3B-Instruct + GSM8K — GRPO + Tree Rollout
#  - 4 GPUs / node
#  - Tree: bf=4, depth=3, entropy_threshold=0.8, no tree_process_reward
#  - 统一 GRPO setting (seq-mean-token-mean, clip=0.2, kl_loss_coef=0.01)
#  - train_batch_size=64 prompts, rollout.n=1 → 64 trees / step
#    理论最多 64*64=4096 leaves (与 pure-GRPO 的 64*64=4096 对齐)
# =============================================================================
set -x
_ORIG_HOME="${HOME}"

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_DEBUG_LOG_PATH="${HOME}"
export RAY_DEDUP_LOGS=0
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO

HOME="${HOME:-/home/user}"
verl_dir="${verl_dir:-${HOME}/verl_data}"
project_name="verl_ablation_qwen25_3b"
experiment_name="gsm8k_grpo_treerollout"

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl0.6.0}"
DATA_DIR="${DATA_DIR:-${RAY_DATA_HOME}/data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/gsm8k_train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/gsm8k_val.parquet}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen/Qwen2.5-3B-Instruct}"

LOG_DIR="${HOME}/verl_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training started at $(date) ===" | tee -a "${LOG_FILE}"
echo "TRAIN_FILE=${TRAIN_FILE}"  | tee -a "${LOG_FILE}"
echo "TEST_FILE=${TEST_FILE}"    | tee -a "${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}"  | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}"     | tee -a "${LOG_FILE}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for _wandb_keyfile in "${_ORIG_HOME}/.wandb_api_key" "${HOME}/.wandb_api_key"; do
    if [[ -f "${_wandb_keyfile}" ]]; then
      export WANDB_API_KEY="$(tr -d ' \n\r\t' < "${_wandb_keyfile}")"
      break
    fi
  done
fi
_WANDB_API_KEY_INLINE=""
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -n "${_WANDB_API_KEY_INLINE}" ]]; then
  export WANDB_API_KEY="${_WANDB_API_KEY_INLINE}"
fi
export WANDB_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${HOME}/wandb_offline"
mkdir -p "${WANDB_DIR}"

# ===== 与 pure-GRPO 保持一致的超参 =====
TRAIN_BATCH_SIZE=64
ROLLOUT_N=1
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_PER_GPU=2
LR=1e-6
CLIP_RATIO=0.2
KL_COEF=0.01
MAX_RESP_LEN=1024
LOSS_AGG_MODE=seq-mean-token-mean
TREE_BF=4
TREE_DEPTH=3
TREE_ENTROPY_THRESHOLD=0.8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_RESP_LEN} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tree_search.enable=True \
    actor_rollout_ref.rollout.tree_search.entropy_threshold=${TREE_ENTROPY_THRESHOLD} \
    actor_rollout_ref.rollout.tree_search.branching_factor=${TREE_BF} \
    actor_rollout_ref.rollout.tree_search.max_tree_depth=${TREE_DEPTH} \
    actor_rollout_ref.rollout.tree_search.tree_process_reward=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$((MAX_RESP_LEN * 2)) \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +ray_kwargs.ray_init.log_to_driver=True \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${verl_dir}/checkpoints/${project_name}/${experiment_name}" \
    trainer.rollout_data_dir="${verl_dir}/rollout_data/${project_name}/${experiment_name}" \
    trainer.validation_data_dir="${verl_dir}/validation_data/${project_name}/${experiment_name}" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    $@ 2>&1 | tee -a "${LOG_FILE}"
