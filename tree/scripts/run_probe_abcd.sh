#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export VLLM_USE_V1=0 PYTHONUNBUFFERED=1
WORK_DIR="${WORK_DIR:-/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-sample-abcd-seed42}"
SOURCE_DIR="${SOURCE_DIR:-/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/math7b-topk-pilot}"
mkdir -p "$WORK_DIR"
exec 9>"$WORK_DIR/run.lock"
flock -n 9 || { echo 'ABCD run already active'; exit 1; }
stage=initialize
status() {
    python3 - "$WORK_DIR" "$stage" "$1" <<'PY'
import datetime,json,os,sys
from pathlib import Path
root,stage,status=sys.argv[1:]
p=Path(root)/'status.json'
t=p.with_suffix('.tmp')
t.write_text(json.dumps(dict(stage=stage,status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
os.replace(t,p)
PY
}
trap 'status failed' ERR
status running
python3 -u tree/probe/abcd.py initialize --work-dir "$WORK_DIR" --source-dir "$SOURCE_DIR"
stage=initial_labels
status running
python3 -u tree/probe/abcd.py annotate --work-dir "$WORK_DIR" --phase initial
stage=independent_confirmation
status running
python3 -u tree/probe/abcd.py annotate --work-dir "$WORK_DIR" --phase confirm
stage=train_abcd
status running
python3 -u tree/probe/abcd.py train --work-dir "$WORK_DIR"
stage=evaluate_abcd
status running
python3 -u tree/probe/abcd.py evaluate --work-dir "$WORK_DIR"
stage=complete
status complete
