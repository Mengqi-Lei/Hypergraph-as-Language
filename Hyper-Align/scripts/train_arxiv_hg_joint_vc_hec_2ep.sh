#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-8B}"
export CONV_MODE="${CONV_MODE:-qwen3}"
export USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
export DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/zero2.json}"
export MASTER_PORT="${MASTER_PORT:-61000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-300}"
export REPORT_TO="${REPORT_TO:-none}"

export TASK_NAME="nc-hecls"
export DATASET_NAME="arxiv_hg"
export DEFAULT_HYPER_DATA_ROOT="../HyperAlign-Bench/dataset/arxiv_hg"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
OUTPUT_TAG="${OUTPUT_TAG:-qwen3emb_joint_${NUM_EPOCHS}ep}"
EMB_TYPE="${EMB_TYPE:-qwen3emb_0.6b}"

echo "============================================================"
echo "  Hyper-Align joint training: Arxiv-HG VC + HEC"
echo "  epochs      : ${NUM_EPOCHS}"
echo "  batch_size  : ${BATCH_SIZE}"
echo "  grad_accum  : ${GRAD_ACCUM}"
echo "  emb_type    : ${EMB_TYPE}"
echo "  data_root   : ${HYPER_DATA_ROOT:-${DEFAULT_HYPER_DATA_ROOT}}"
echo "  output_tag  : ${OUTPUT_TAG}"
echo "============================================================"

exec "${SCRIPT_DIR}/_train_hyperalign_task.sh" \
    "${NUM_EPOCHS}" \
    "${BATCH_SIZE}" \
    "${GRAD_ACCUM}" \
    "${OUTPUT_TAG}" \
    "${EMB_TYPE}"
