#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/inspire/hdd/global_user/weilongxuan-253108120168
WORK_DIR=${WORK_DIR:-$ROOT/probe_runs/math7b-sample-abcd-seed42}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/probe_runs/online-A-seed42}
export VLLM_USE_V1=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
mkdir -p "$OUTPUT_DIR"
extra_args=()
if [[ -n "${QUESTIONS_FILE:-}" ]]; then
    extra_args+=(--questions-file "$QUESTIONS_FILE")
fi
trap 'echo failed > "$OUTPUT_DIR/status.txt"' ERR
for mode in probe entropy random; do
    echo "running $mode" > "$OUTPUT_DIR/status.txt"
    python3 tree/probe/online_tree_trial.py --work-dir "$WORK_DIR" \
        --output-dir "$OUTPUT_DIR" --mode "$mode" \
        --split test --leaves 16 --max-response 2048 \
        "${extra_args[@]}" \
        > "$OUTPUT_DIR/$mode.log" 2>&1
done
python3 tree/probe/summarize_online_trial.py --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/comparison.log" 2>&1
echo complete > "$OUTPUT_DIR/status.txt"
