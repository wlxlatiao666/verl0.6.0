#!/usr/bin/env bash
# Evaluate one or more checkpoints on MATH-500 / AMC / OlympiadBench by running
# verl's PPO trainer in val_only mode. For each global_step_N folder under
# ${CHECKPOINT_DIR} this script:
#   1. points trainer.resume_from_path at that step,
#   2. sets trainer.val_only=True so we skip any optimizer updates,
#   3. points data.val_files at the three parquet files produced by
#      prepare_eval_data.sh (data_source="math_dapo").
#
# The metric reported is val-core/math_dapo/acc/mean@1, identical to what
# run_qwen3-8b.sh uses during training validation.
#
# Typical usage:
#   bash eval_qwen3-8b.sh                        # loop over every ckpt in CHECKPOINT_DIR
#   CHECKPOINT_DIR=/path/to/exp bash eval_qwen3-8b.sh
#   CHECKPOINT_STEP=120 bash eval_qwen3-8b.sh    # only evaluate global_step_120
#   RESUME_FROM_PATH=/abs/path/global_step_80 bash eval_qwen3-8b.sh
#
# All defaults mirror run_qwen3-8b.sh so the validation setup is comparable.

set -uxo pipefail

# ---- paths / env ------------------------------------------------------------

_ORIG_HOME="${HOME}"

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_DEBUG_LOG_PATH=/inspire/hdd/global_user/weilongxuan-253108120168
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
HOME=/inspire/hdd/global_user/weilongxuan-253108120168

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl0.6.0"}

# ---- datasets (produced by prepare_eval_data.sh) ----------------------------

EVAL_DATA_DIR=${EVAL_DATA_DIR:-"${RAY_DATA_HOME}/data/eval"}
MATH500_FILE=${MATH500_FILE:-"${EVAL_DATA_DIR}/math500_test.parquet"}
AMC_FILE=${AMC_FILE:-"${EVAL_DATA_DIR}/amc_test.parquet"}
OLYMPIAD_FILE=${OLYMPIAD_FILE:-"${EVAL_DATA_DIR}/olympiad_bench_test.parquet"}

for f in "${MATH500_FILE}" "${AMC_FILE}" "${OLYMPIAD_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: eval parquet not found: ${f}" >&2
    echo "Run: bash $(dirname "${BASH_SOURCE[0]}")/prepare_eval_data.sh" >&2
    exit 1
  fi
done

# Hydra list literal with single-quoted paths so commas inside paths are fine.
VAL_FILES="['${MATH500_FILE}','${AMC_FILE}','${OLYMPIAD_FILE}']"

# ---- checkpoint discovery ---------------------------------------------------

# Default checkpoint root matches verl's default:
#   checkpoints/<project_name>/<experiment_name>/global_step_N/actor/...
PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_tree_latest"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_math7b_grpo"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RAY_DATA_HOME}/tree/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

# Build the list of global_step_* directories to evaluate.
declare -a CKPT_PATHS
if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
  CKPT_PATHS=("${RESUME_FROM_PATH}")
elif [[ -n "${CHECKPOINT_STEP:-}" ]]; then
  CKPT_PATHS=("${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}")
else
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
  # sort numerically by the step suffix
  mapfile -t CKPT_PATHS < <(
    find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' \
      | awk -F'global_step_' '{printf "%d\t%s\n", $2, $0}' \
      | sort -n \
      | cut -f2
  )
  if [[ "${#CKPT_PATHS[@]}" -eq 0 ]]; then
    echo "ERROR: no global_step_* folders under ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
fi

# ---- logging ----------------------------------------------------------------

LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${RUN_STAMP}.log"
SUMMARY_FILE="${LOG_DIR}/eval_${RUN_STAMP}_summary.txt"

echo "=== Evaluation started at $(date) ===" | tee -a "${LOG_FILE}"
echo "Checkpoint root: ${CHECKPOINT_DIR}"     | tee -a "${LOG_FILE}"
echo "Checkpoints to eval:"                   | tee -a "${LOG_FILE}"
for p in "${CKPT_PATHS[@]}"; do echo "  - ${p}" | tee -a "${LOG_FILE}"; done
echo "Val files: ${VAL_FILES}"                | tee -a "${LOG_FILE}"
echo "Log file : ${LOG_FILE}"                 | tee -a "${LOG_FILE}"
echo "Summary  : ${SUMMARY_FILE}"             | tee -a "${LOG_FILE}"

# ---- WANDB (mirror run_qwen3-8b.sh) ----------------------------------------
# Validation-only runs still emit metrics; keep wandb offline for consistency.

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for _wandb_keyfile in "${_ORIG_HOME}/.wandb_api_key" "${HOME}/.wandb_api_key"; do
    if [[ -f "${_wandb_keyfile}" ]]; then
      export WANDB_API_KEY="$(tr -d ' \n\r\t' < "${_wandb_keyfile}")"
      break
    fi
  done
fi
_WANDB_API_KEY_INLINE="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"
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -n "${_WANDB_API_KEY_INLINE}" ]]; then
  export WANDB_API_KEY="${_WANDB_API_KEY_INLINE}"
fi
export WANDB_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=offline
export WANDB_DIR="${HOME}/wandb_offline"
mkdir -p "${WANDB_DIR}"

# ---- eval loop --------------------------------------------------------------

printf "checkpoint\tmath_dapo_acc_mean@1\n" > "${SUMMARY_FILE}"

for CKPT in "${CKPT_PATHS[@]}"; do
  if [[ ! -d "${CKPT}" ]]; then
    echo "skip: ${CKPT} is not a directory" | tee -a "${LOG_FILE}"
    continue
  fi

  STEP_NAME="$(basename "${CKPT}")"
  STEP_LOG="${LOG_DIR}/eval_${RUN_STAMP}_${STEP_NAME}.log"
  echo "=== [${STEP_NAME}] evaluating ${CKPT} at $(date) ===" | tee -a "${LOG_FILE}"

  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="$MATH500_FILE" \
      data.val_files="$VAL_FILES" \
      data.train_batch_size=96 \
      data.max_prompt_length=2048 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=False \
      data.truncation='error' \
      actor_rollout_ref.actor.clip_ratio_low=0.2 \
      actor_rollout_ref.actor.clip_ratio_high=0.28 \
      actor_rollout_ref.model.path=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B \
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
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.tree_search.enable=False \
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
      trainer.project_name="${PROJECT_NAME}" \
      trainer.experiment_name="${EXPERIMENT_NAME}_eval_${STEP_NAME}" \
      trainer.n_gpus_per_node=4 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=1 \
      trainer.total_epochs=1 \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.resume_mode=resume_path \
      trainer.resume_from_path="${CKPT}" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      "$@" 2>&1 | tee -a "${STEP_LOG}" | tee -a "${LOG_FILE}"

  # Extract val-core/math_dapo/acc/mean@1 from the step log. verl prints the
  # validation metric dict via pprint, so we grep-match both common spellings.
  ACC=$(grep -Eo "val-core/math_dapo/acc/mean@1[^,}]*" "${STEP_LOG}" | tail -n1 | sed -E "s/.*: *([0-9.eE+-]+).*/\1/") || true
  if [[ -z "${ACC}" ]]; then
    ACC=$(grep -Eo "'val-core/math_dapo/acc/mean@1': *[0-9.eE+-]+" "${STEP_LOG}" | tail -n1 | sed -E "s/.*: *([0-9.eE+-]+).*/\1/") || true
  fi
  printf "%s\t%s\n" "${STEP_NAME}" "${ACC:-NA}" | tee -a "${SUMMARY_FILE}"
done

echo "=== Evaluation finished at $(date) ===" | tee -a "${LOG_FILE}"
echo "Summary:"
cat "${SUMMARY_FILE}"
