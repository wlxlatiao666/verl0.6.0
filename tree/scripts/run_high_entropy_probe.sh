#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/inspire/hdd/global_user/weilongxuan-253108120168
WORK_DIR=${WORK_DIR:-$ROOT/probe_runs/math7b-sample-abcd-seed42}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/probe_runs/high-entropy-probe-seed42}
QUESTIONS_FILE=${QUESTIONS_FILE:-$ROOT/probe_runs/online-A-fresh100-seed42/questions.json}
export VLLM_USE_V1=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p "$OUTPUT_DIR"
trap 'echo failed > "$OUTPUT_DIR/status.txt"' ERR
if [[ ! -f "$OUTPUT_DIR/fit/test.json" ]]; then
    echo training > "$OUTPUT_DIR/status.txt"
    python3 tree/probe/high_entropy_probe.py --work-dir "$WORK_DIR" \
        --output-dir "$OUTPUT_DIR/fit" --entropy-quantile .8 --probe-quantile .8 --pca-dim 32 \
        > "$OUTPUT_DIR/train.log" 2>&1
fi
for mode in probe_entropy entropy random; do
    echo "evaluating $mode" > "$OUTPUT_DIR/status.txt"
    python3 tree/probe/online_tree_trial.py --work-dir "$WORK_DIR" \
        --output-dir "$OUTPUT_DIR/online" --probe-checkpoint "$OUTPUT_DIR/fit/A.pt" \
        --mode "$mode" --questions-file "$QUESTIONS_FILE" --limit 100 \
        --leaves 16 --max-response 2048 > "$OUTPUT_DIR/$mode.log" 2>&1
done
python3 tree/probe/summarize_online_trial.py --output-dir "$OUTPUT_DIR/online" \
    --probe-mode probe_entropy > "$OUTPUT_DIR/comparison.log" 2>&1
echo complete > "$OUTPUT_DIR/status.txt"
