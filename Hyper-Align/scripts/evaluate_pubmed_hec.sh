#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_NAME="hecls"
export DATASET_NAME="pubmed"
export DEFAULT_HYPER_DATA_ROOT="../HyperAlign-Bench/dataset/pubmed"

exec "${SCRIPT_DIR}/_eval_hyperalign_task.sh" "$@"
