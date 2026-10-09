#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/inspire/hdd/global_user/weilongxuan-253108120168
WORK_DIR=${WORK_DIR:-$ROOT/probe_runs/high-entropy-data700-seed42}
mkdir -p "$WORK_DIR/training"
exec 9>"$WORK_DIR/training/runner.lock"
flock -n 9 || exit 1
trap 'echo failed > "$WORK_DIR/training/status.txt"' ERR
echo waiting_for_labels > "$WORK_DIR/training/status.txt"
while true; do
    state=$(cat "$WORK_DIR/status.txt")
    if [[ "$state" == complete ]]; then break; fi
    if [[ "$state" == failed ]]; then
        echo 'Dataset construction failed; training not started' >&2
        exit 1
    fi
    sleep 30
done
echo training > "$WORK_DIR/training/status.txt"
python3 tree/probe/train_high_entropy_data.py --work-dir "$WORK_DIR" \
    --output-dir "$WORK_DIR/training/fit" --pca-dim 32 --probe-quantile .8 \
    > "$WORK_DIR/training/train.log" 2>&1
echo complete > "$WORK_DIR/training/status.txt"
