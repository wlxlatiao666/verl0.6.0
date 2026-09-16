#!/usr/bin/env bash
# Run in the remote verl/vLLM environment. No GRPO launch and no WAAD computation.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B}"
RAY_DATA_HOME="${RAY_DATA_HOME:-/inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/dapo-math-17k.parquet}"
WORK_DIR="${WORK_DIR:-${RAY_DATA_HOME}/probe_runs/qwen2.5-math7b-topk-pilot}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
TP_SIZE="${TP_SIZE:-1}"
STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

prepare() {
    "$PYTHON_BIN" tree/probe/run.py prepare --work-dir "$WORK_DIR" \
        --model "$MODEL_PATH" --data "$TRAIN_FILE" \
        --num-questions "${NUM_QUESTIONS:-100}" \
        --branch-sampling "${BRANCH_SAMPLING:-topk}" \
        --trajectories "${TRAJECTORIES:-2}" --positions "${POSITIONS:-6}" \
        --repeats "${REPEATS:-4}" --eval-repeats "${EVAL_REPEATS:-8}" "$@"
}
gpu_stage() {
    local stage="$1"
    shift
    local extra=()
    if [[ "$stage" != features ]]; then
        extra+=(--tensor-parallel-size "$TP_SIZE" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.7}")
    fi
    "$PYTHON_BIN" tree/probe/run.py "$stage" --work-dir "$WORK_DIR" \
        --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" "${extra[@]}" "$@"
}
case "$STAGE" in
    prepare) prepare "$@" ;;
    rollout|features|label) gpu_stage "$STAGE" "$@" ;;
    train|evaluate) "$PYTHON_BIN" tree/probe/run.py "$STAGE" --work-dir "$WORK_DIR" "$@" ;;
    all)
        if [[ "$NUM_SHARDS" != 1 ]]; then
            echo "For multiple shards run prepare once, GPU stages with a barrier, then train/evaluate once." >&2
            exit 2
        fi
        prepare "$@"
        gpu_stage rollout
        gpu_stage features
        gpu_stage label
        "$PYTHON_BIN" tree/probe/run.py train --work-dir "$WORK_DIR"
        "$PYTHON_BIN" tree/probe/run.py evaluate --work-dir "$WORK_DIR"
        ;;
    *) echo "Usage: bash $0 {all|prepare|rollout|features|label|train|evaluate} [stage arguments]" >&2; exit 2 ;;
esac
