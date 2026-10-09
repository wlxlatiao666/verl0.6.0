#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/inspire/hdd/global_user/weilongxuan-253108120168
WORK_DIR=${WORK_DIR:-$ROOT/probe_runs/high-entropy-data700-seed42}
export VLLM_USE_V1=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p "$WORK_DIR"
exec 9>"$WORK_DIR/runner.lock"
flock -n 9 || { echo 'Dataset runner already active'; exit 1; }
trap 'echo failed > "$WORK_DIR/status.txt"' ERR
# Queue behind the existing evaluation runner, including its between-mode gaps.
echo waiting_for_existing_evaluation > "$WORK_DIR/status.txt"
python3 - <<'PY'
from pathlib import Path
import time
while True:
    active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=p.read_bytes().split(b'\0')
            if any(a.endswith(b'/run_high_entropy_probe.sh') for a in argv):
                active.append(p.parent.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            pass
    if not active:
        break
    print('Waiting for existing evaluation runner:',active,flush=True)
    time.sleep(30)
PY
for stage in rollout scan calibrate features; do
    echo "$stage" > "$WORK_DIR/status.txt"
    python3 tree/probe/build_high_entropy_data.py "$stage" --work-dir "$WORK_DIR" \
        > "$WORK_DIR/$stage.log" 2>&1
done
echo labeling_pilot > "$WORK_DIR/status.txt"
python3 tree/probe/build_high_entropy_data.py label --work-dir "$WORK_DIR" --pilot-only \
    > "$WORK_DIR/label_pilot.log" 2>&1
python3 tree/probe/build_high_entropy_data.py summarize --work-dir "$WORK_DIR" \
    --summary-name pilot_summary.json > "$WORK_DIR/pilot_summary.log" 2>&1
echo labeling_all > "$WORK_DIR/status.txt"
python3 tree/probe/build_high_entropy_data.py label --work-dir "$WORK_DIR" \
    > "$WORK_DIR/label_all.log" 2>&1
python3 tree/probe/build_high_entropy_data.py summarize --work-dir "$WORK_DIR" \
    > "$WORK_DIR/summary.log" 2>&1
echo complete > "$WORK_DIR/status.txt"
