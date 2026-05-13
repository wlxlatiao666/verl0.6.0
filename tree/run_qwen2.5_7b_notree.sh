set -x
# Save login home before HOME is overridden (otherwise ~/.wandb_api_key would be wrong).
_ORIG_HOME="${HOME}"

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_DEBUG_LOG_PATH=/inspire/hdd/project/project-public/zhangshenao-CZXS25250096
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
HOME=/inspire/hdd/project/project-public/zhangshenao-CZXS25250096
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl0.6.0"}

TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime2026.parquet"}

# Real-time log file: each line is written immediately; data is not lost if the job is killed
LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training started at $(date) ===" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"


# WANDB: must be set before Ray (no TTY). Order matters — do not `exit 1` before loading key.
# 1) already exported in shell  2) ${_ORIG_HOME}/.wandb_api_key (e.g. /root/.wandb_api_key)  3) ${HOME}/.wandb_api_key (project dir)
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for _wandb_keyfile in "${_ORIG_HOME}/.wandb_api_key" "${HOME}/.wandb_api_key"; do
    if [[ -f "${_wandb_keyfile}" ]]; then
      export WANDB_API_KEY="$(tr -d ' \n\r\t' < "${_wandb_keyfile}")"
      break
    fi
  done
fi
# Local testing only: put your key here if you do not use env / ~/.wandb_api_key.
# Priority: shell export > key files above > this line (empty = skip).
# Do not commit real keys to shared repos.
_WANDB_API_KEY_INLINE="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"
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -n "${_WANDB_API_KEY_INLINE}" ]]; then
  export WANDB_API_KEY="${_WANDB_API_KEY_INLINE}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is empty. trainer.logger includes wandb but Ray has no TTY for wandb login." >&2
  echo "Fix: set _WANDB_API_KEY_INLINE in this script, OR export WANDB_API_KEY=..., OR use ${_ORIG_HOME}/.wandb_api_key / ${HOME}/.wandb_api_key" >&2
  exit 1
fi
export WANDB_KEY="${WANDB_API_KEY}"

# wandb offline mode: saves every wandb.log() call to disk immediately in real-time.
# Sync later: wandb sync ${HOME}/wandb_offline/wandb/run-*
export WANDB_MODE=offline
export WANDB_DIR="${HOME}/wandb_offline"
mkdir -p "${WANDB_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=96 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.path=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
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
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name='verl_grpo_treerollout' \
    trainer.experiment_name='qwen2.5_7b_instruct_no_tree' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    # trainer.rollout_data_dir="${HOME}/rollout_data" \
    # trainer.validation_data_dir="${HOME}/validation_data" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False\
    $@ 2>&1 | tee -a "${LOG_FILE}"
