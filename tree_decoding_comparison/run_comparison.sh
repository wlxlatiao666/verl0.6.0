#!/bin/bash
# Run Tree Decoding vs Base GRPO Comparison Experiment
#
# Usage:
#   ./run_comparison.sh --model-path /path/to/model [other options]
#
# Example:
#   export RAY_DATA_HOME=/path/to/data
#   ./run_comparison.sh --model-path /path/to/Qwen2.5-7B-Instruct
#
export VLLM_USE_V1=0
HOME=/inspire/hdd/global_user/weilongxuan-253108120168
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl0.6.0"}
# Set defaults
TREE_DECODING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${TREE_DECODING_DIR}/tree_decoding_results"
NUM_SAMPLES=500
N=8
BRANCHING_FACTOR=2
MAX_TREE_DEPTH=3
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
MAX_TOKENS=4096
ENTROPY_THRESHOLD=1.0
TAU_IMPORTANCE=0.0
AUTO_CALIBRATE_THRESHOLDS=0
CALIBRATION_N=5
CALIBRATION_MAX_TOKENS=200
CALIBRATION_QUANTILE=0.8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --branching-factor)
            BRANCHING_FACTOR="$2"
            shift 2
            ;;
        --max-tree-depth)
            MAX_TREE_DEPTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --entropy-threshold)
            ENTROPY_THRESHOLD="$2"
            shift 2
            ;;
        --tau-importance)
            TAU_IMPORTANCE="$2"
            shift 2
            ;;
        --auto-calibrate-thresholds)
            AUTO_CALIBRATE_THRESHOLDS=1
            shift 1
            ;;
        --calibration-n)
            CALIBRATION_N="$2"
            shift 2
            ;;
        --calibration-max-tokens)
            CALIBRATION_MAX_TOKENS="$2"
            shift 2
            ;;
        --calibration-quantile)
            CALIBRATION_QUANTILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check model path
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required"
    echo
    echo "Usage: $0 --model-path /path/to/model [other options]"
    exit 1
fi

# Set dataset path if not provided
if [[ -z "$DATASET_PATH" ]]; then
    if [[ -n "$RAY_DATA_HOME" ]]; then
        DATASET_PATH="${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
    else
        echo "Warning: RAY_DATA_HOME not set, trying default path"
        DATASET_PATH="./data/dapo-math-17k.parquet"
    fi
fi

# Build command
CMD="python ${TREE_DECODING_DIR}/tree_decoding_comparison.py"
CMD="$CMD --model-path ${MODEL_PATH}"
CMD="$CMD --dataset-path ${DATASET_PATH}"
CMD="$CMD --num-samples ${NUM_SAMPLES}"
CMD="$CMD --n ${N}"
CMD="$CMD --branching-factor ${BRANCHING_FACTOR}"
CMD="$CMD --max-tree-depth ${MAX_TREE_DEPTH}"
CMD="$CMD --temperature ${TEMPERATURE}"
CMD="$CMD --top-p ${TOP_P}"
CMD="$CMD --top-k ${TOP_K}"
CMD="$CMD --max-tokens ${MAX_TOKENS}"
CMD="$CMD --entropy-threshold ${ENTROPY_THRESHOLD}"
CMD="$CMD --tau-importance ${TAU_IMPORTANCE}"

if [[ ${AUTO_CALIBRATE_THRESHOLDS} -eq 1 ]]; then
    CMD="$CMD --auto-calibrate-thresholds"
    CMD="$CMD --calibration-n ${CALIBRATION_N}"
    CMD="$CMD --calibration-max-tokens ${CALIBRATION_MAX_TOKENS}"
    CMD="$CMD --calibration-quantile ${CALIBRATION_QUANTILE}"
fi

CMD="$CMD --output-dir ${OUTPUT_DIR}"

if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
    CMD="$CMD --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
fi

# Print configuration
echo "=========================================="
echo "Tree Decoding vs Base GRPO Comparison"
echo "=========================================="
echo
echo "Configuration:"
echo "  Model path:           ${MODEL_PATH}"
echo "  Dataset path:         ${DATASET_PATH}"
echo "  Number of samples:    ${NUM_SAMPLES}"
echo "  Sequences per query:  ${N}"
echo "  Branching factor:     ${BRANCHING_FACTOR}"
echo "  Max tree depth:       ${MAX_TREE_DEPTH}"
echo "  Temperature:          ${TEMPERATURE}"
echo "  Top-p:                ${TOP_P}"
echo "  Top-k:                ${TOP_K}"
echo "  Max tokens:           ${MAX_TOKENS}"
echo "  Entropy threshold:    ${ENTROPY_THRESHOLD}"
echo "  Tau importance:       ${TAU_IMPORTANCE}"

if [[ ${AUTO_CALIBRATE_THRESHOLDS} -eq 1 ]]; then
    echo
    echo "Threshold Auto-calibration:"
    echo "  Enabled:              Yes"
    echo "  Calibration n:        ${CALIBRATION_N}"
    echo "  Calibration tokens:   ${CALIBRATION_MAX_TOKENS}"
    echo "  Calibration quantile: ${CALIBRATION_QUANTILE}"
fi

echo "  Output directory:     ${OUTPUT_DIR}"
echo
echo "=========================================="
echo

# Check if vllm is available
python -c "import sys; sys.path.insert(0, '/Users/weilongxuan/codes/vllm'); import vllm; print('vllm available: OK')" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "Warning: vllm import failed. Make sure vllm is available at /Users/weilongxuan/codes/vllm"
fi

# Run command
echo "Running experiment..."
echo
eval "$CMD"
