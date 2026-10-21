#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Expected grpo, treerollout, or treepr}"
shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

case "$MODE" in
    grpo) TREE_ENABLED=False; TREE_PROCESS_REWARD=False; ROLLOUT_N=64 ;;
    treerollout) TREE_ENABLED=True; TREE_PROCESS_REWARD=False; ROLLOUT_N=1 ;;
    treepr) TREE_ENABLED=True; TREE_PROCESS_REWARD=True; ROLLOUT_N=1 ;;
    *) echo "Unknown experiment: $MODE" >&2; exit 2 ;;
esac

TASK_ROOT="${TASK_ROOT:-/inspire/hdd/global_user/weilongxuan-253108120168}"
DATA_DIR="${DATA_DIR:-$TASK_ROOT/verl0.6.0/data/math}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TASK_ROOT/verl_data}"
PROJECT_NAME="${PROJECT_NAME:-math_pure_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5_math7b_${MODE}}"
RUN_DIR="$OUTPUT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1 VLLM_USE_V1=0 RAY_DEDUP_LOGS=0 NCCL_SHM_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"

command=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.proc_agg_mode=raw
    +algorithm.local_adv_weight=0.5
    +algorithm.global_adv_weight=0.5
    "data.train_files=$DATA_DIR/train.parquet"
    "data.val_files=$DATA_DIR/test.parquet"
    data.train_batch_size=96
    data.max_prompt_length=2048
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation=error
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.clip_ratio_c=1e9
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.do_sample=True
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.tree_search.enable=$TREE_ENABLED"
    "actor_rollout_ref.rollout.tree_search.tree_process_reward=$TREE_PROCESS_REWARD"
    actor_rollout_ref.rollout.tree_search.num_roots=8
    actor_rollout_ref.rollout.tree_search.branching_factor=2
    actor_rollout_ref.rollout.tree_search.max_tree_depth=3
    actor_rollout_ref.rollout.tree_search.branch_trigger_mode=entropy
    actor_rollout_ref.rollout.tree_search.branch_sampling=sample_with_replacement
    actor_rollout_ref.rollout.tree_search.branch_temperature=1.0
    actor_rollout_ref.rollout.tree_search.min_seg_length=64
    actor_rollout_ref.rollout.tree_search.entropy_threshold=0.8
    actor_rollout_ref.rollout.tree_search.threshold_stats_quantile=0.8
    +actor_rollout_ref.rollout.tree_search.threshold_stats_interval=10
    actor_rollout_ref.rollout.tree_search.topup_leaves_to_target=True
    actor_rollout_ref.rollout.tree_search.tau_importance=null
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    reward_model.reward_manager=naive
    trainer.critic_warmup=0
    'trainer.logger=["console","wandb","tensorboard"]'
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=2
    trainer.total_epochs=1
    "trainer.default_local_dir=$RUN_DIR/checkpoints"
    "trainer.rollout_data_dir=$RUN_DIR/rollouts"
    "trainer.validation_data_dir=$RUN_DIR/validation"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exec "${command[@]}" --cfg job --resolve "$@"
fi

for data_file in "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet"; do
    if [[ ! -f "$data_file" ]]; then
        echo "Missing MATH dataset: $data_file" >&2
        exit 2
    fi
done
mkdir -p "$RUN_DIR/logs" "$WANDB_DIR"
LOG_FILE="$RUN_DIR/logs/$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Experiment: %s\nLog: %s\n' "$EXPERIMENT_NAME" "$LOG_FILE"
"${command[@]}" "$@" 2>&1 | tee "$LOG_FILE"
