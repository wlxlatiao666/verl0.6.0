#!/usr/bin/env bash
set -euo pipefail
export TREE_PROCESS_REWARD=False
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_qwen2.5_math7b-multiroot-entropy-treepr.sh" "$@"
