#!/usr/bin/env bash
echo "Deprecated script. Use tree/scripts/math_grpo/{grpo,treerollout,treepr}.sh; see its README." >&2
exit 2
set -euo pipefail
export TREE_PROCESS_REWARD=False
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_qwen2.5_math7b-multiroot-entropy-treepr.sh" "$@"
