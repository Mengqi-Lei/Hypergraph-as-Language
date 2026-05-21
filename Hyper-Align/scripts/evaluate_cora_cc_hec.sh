#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_NAME="hecls"
export DATASET_NAME="cora_cc"
export DEFAULT_HYPER_DATA_ROOT="../HyperAlign-Bench/dataset/cora_cc"

exec "${SCRIPT_DIR}/_eval_hyperalign_task.sh" "$@"
