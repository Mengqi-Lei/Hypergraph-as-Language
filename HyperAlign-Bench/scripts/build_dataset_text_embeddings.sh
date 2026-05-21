#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-dataset/arxiv_hg}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
HF_ENDPOINT="${HF_ENDPOINT:-}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_text_embeddings.py"
  --input-dir "${INPUT_DIR}"
  --model-name "${MODEL_NAME}"
)

if [[ -n "${HF_ENDPOINT}" ]]; then
  CMD+=(--hf-endpoint "${HF_ENDPOINT}")
fi

exec "${CMD[@]}" "$@"
