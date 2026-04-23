#!/usr/bin/env bash
# Download & preprocess the three eval datasets (MATH-500, AMC, OlympiadBench)
# into verl-style parquet files. The data_source field is fixed to "math_dapo"
# so that validation emits val-core/math_dapo/acc/mean@1.
#
# Override defaults via env vars, e.g.:
#   HOME=/inspire/hdd/global_user/weilongxuan-253108120168 \
#   EVAL_DATA_DIR=$HOME/verl0.6.0/data/eval \
#   bash prepare_eval_data.sh
#
# Re-run with OVERWRITE=1 to force redownload / reprocess.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to the same layout as run_qwen3-8b.sh
: "${RAY_DATA_HOME:=${HOME}/verl0.6.0}"
: "${EVAL_DATA_DIR:=${RAY_DATA_HOME}/data/eval}"
: "${OVERWRITE:=0}"
: "${DATA_SOURCE:=math_dapo}"

# Optional HuggingFace mirror (useful inside China). Unset if you don't need it.
#   export HF_ENDPOINT=https://hf-mirror.com
: "${HF_ENDPOINT:=}"
if [[ -n "${HF_ENDPOINT}" ]]; then
  export HF_ENDPOINT
fi

mkdir -p "${EVAL_DATA_DIR}"

need_build=0
for name in math500 amc olympiad_bench; do
  if [[ ! -f "${EVAL_DATA_DIR}/${name}_test.parquet" ]]; then
    need_build=1
  fi
done

if [[ "${need_build}" -eq 0 && "${OVERWRITE}" -ne 1 ]]; then
  echo "[prepare_eval_data] all parquet files already exist under ${EVAL_DATA_DIR}"
  echo "[prepare_eval_data] set OVERWRITE=1 to re-download."
  exit 0
fi

python3 "${SCRIPT_DIR}/prepare_eval_data.py" \
  --local_save_dir "${EVAL_DATA_DIR}" \
  --data_source "${DATA_SOURCE}" \
  --datasets math500 amc olympiad_bench

echo "[prepare_eval_data] done. Files written to: ${EVAL_DATA_DIR}"
ls -lh "${EVAL_DATA_DIR}"
