#!/usr/bin/env bash
# Compare iid GRPO with random, entropy-only, and entropy+WAAD trees.

set -euo pipefail

export VLLM_USE_V1=0

TREE_DECODING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_WORK_ROOT="${USER_WORK_ROOT:-/inspire/hdd/global_user/weilongxuan-253108120168}"
RAY_DATA_HOME="${RAY_DATA_HOME:-${USER_WORK_ROOT}/verl0.6.0}"
VLLM_SOURCE_PATH="${VLLM_SOURCE_PATH:-/Users/bytedance/codes/vllm}"
export RAY_DATA_HOME VLLM_SOURCE_PATH

MODEL_PATH=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B
DATASET_PATH=""
OUTPUT_DIR="${TREE_DECODING_DIR}/tree_decoding_results"
NUM_SAMPLES=500
N=8
BRANCHING_FACTOR=2
MAX_TREE_DEPTH=3
MIN_SEG_LENGTH=128
RANDOM_BRANCH_PROBABILITY=0.2
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
MAX_TOKENS=4096
ENTROPY_THRESHOLD=1.0
TAU_IMPORTANCE=0.0
SEED=42
AUTO_CALIBRATE_THRESHOLDS=1
CALIBRATION_N=5
CALIBRATION_MAX_TOKENS=200
CALIBRATION_QUANTILE=0.8
TENSOR_PARALLEL_SIZE=1
DTYPE=float16
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
QUICK_TEST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --vllm-source-path) VLLM_SOURCE_PATH="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --branching-factor) BRANCHING_FACTOR="$2"; shift 2 ;;
        --max-tree-depth) MAX_TREE_DEPTH="$2"; shift 2 ;;
        --min-seg-length) MIN_SEG_LENGTH="$2"; shift 2 ;;
        --random-branch-probability)
            RANDOM_BRANCH_PROBABILITY="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --entropy-threshold) ENTROPY_THRESHOLD="$2"; shift 2 ;;
        --tau-importance) TAU_IMPORTANCE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --auto-calibrate-thresholds)
            AUTO_CALIBRATE_THRESHOLDS=1; shift ;;
        --no-auto-calibrate-thresholds)
            AUTO_CALIBRATE_THRESHOLDS=0; shift ;;
        --calibration-n) CALIBRATION_N="$2"; shift 2 ;;
        --calibration-max-tokens) CALIBRATION_MAX_TOKENS="$2"; shift 2 ;;
        --calibration-quantile) CALIBRATION_QUANTILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --quick-test) QUICK_TEST=1; shift ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Export only after CLI parsing so the selected checkout is always first.
export VLLM_SOURCE_PATH
export PYTHONPATH="${VLLM_SOURCE_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -d "${VLLM_SOURCE_PATH}/vllm" ]]; then
    echo "Invalid vLLM source checkout: ${VLLM_SOURCE_PATH}" >&2
    exit 1
fi

if [[ -z "${DATASET_PATH}" ]]; then
    DATASET_PATH="${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
fi

COMMAND=(
    python3 "${TREE_DECODING_DIR}/tree_decoding_comparison.py"
    --model-path "${MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --num-samples "${NUM_SAMPLES}"
    --n "${N}"
    --branching-factor "${BRANCHING_FACTOR}"
    --max-tree-depth "${MAX_TREE_DEPTH}"
    --min-seg-length "${MIN_SEG_LENGTH}"
    --random-branch-probability "${RANDOM_BRANCH_PROBABILITY}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --max-tokens "${MAX_TOKENS}"
    --entropy-threshold "${ENTROPY_THRESHOLD}"
    --tau-importance "${TAU_IMPORTANCE}"
    --seed "${SEED}"
    --calibration-n "${CALIBRATION_N}"
    --calibration-max-tokens "${CALIBRATION_MAX_TOKENS}"
    --calibration-quantile "${CALIBRATION_QUANTILE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ ${AUTO_CALIBRATE_THRESHOLDS} -eq 1 ]]; then
    COMMAND+=(--auto-calibrate-thresholds)
fi
if [[ ${QUICK_TEST} -eq 1 ]]; then
    COMMAND+=(--quick-test)
fi

echo "=========================================="
echo "GRPO vs three tree-decoding policies"
echo "=========================================="
echo "Model:                     ${MODEL_PATH}"
echo "Dataset:                   ${DATASET_PATH}"
echo "vLLM source:               ${VLLM_SOURCE_PATH}"
echo "Prompts / candidates:      ${NUM_SAMPLES} / ${N}"
echo "Tree B / depth / min seg:  ${BRANCHING_FACTOR} / ${MAX_TREE_DEPTH} / ${MIN_SEG_LENGTH}"
echo "Random branch probability: ${RANDOM_BRANCH_PROBABILITY}"
echo "Entropy / WAAD threshold:  ${ENTROPY_THRESHOLD} / ${TAU_IMPORTANCE}"
echo "Auto calibration:          ${AUTO_CALIBRATE_THRESHOLDS} (q=${CALIBRATION_QUANTILE})"
echo "Sampling T / top-p / top-k:${TEMPERATURE} / ${TOP_P} / ${TOP_K}"
echo "Max tokens / seed:         ${MAX_TOKENS} / ${SEED}"
echo "TP / dtype / GPU memory:   ${TENSOR_PARALLEL_SIZE} / ${DTYPE} / ${GPU_MEMORY_UTILIZATION}"
echo "Max model length:          ${MAX_MODEL_LEN}"
echo "Output:                    ${OUTPUT_DIR}"
echo
echo "Methods: base_grpo, random_tree, entropy_only_tree, entropy_waad_tree"
echo "Budget: every method returns exactly N complete candidates per prompt."
echo

if ! python3 -c "import vllm; print('Imported vLLM:', vllm.__file__)"; then
    echo "Failed to import vLLM from ${VLLM_SOURCE_PATH}" >&2
    exit 1
fi

exec "${COMMAND[@]}"
