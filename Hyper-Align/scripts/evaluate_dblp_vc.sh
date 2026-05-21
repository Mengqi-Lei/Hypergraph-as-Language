#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_NAME="nc"
export DATASET_NAME="dblp"
export DEFAULT_HYPER_DATA_ROOT="../HyperAlign-Bench/dataset/dblp"

exec "${SCRIPT_DIR}/_eval_hyperalign_task.sh" "$@"
