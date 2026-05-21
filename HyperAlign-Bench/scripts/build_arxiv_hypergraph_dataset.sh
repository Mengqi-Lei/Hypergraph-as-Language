#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-data/raw/ogbn_arxiv}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/arxiv_hg}"
TITLEABS_PATH="${TITLEABS_PATH:-}"
MAX_CHILD_HYPEREDGES="${MAX_CHILD_HYPEREDGES:-1}"
FORMAL_HIDT_DEPTH="${FORMAL_HIDT_DEPTH:-3}"
OVERVIEW_HOPS="${OVERVIEW_HOPS:-2}"
OVERVIEW_ORDER_BUCKETS="${OVERVIEW_ORDER_BUCKETS:-4}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_arxiv_hypergraph_dataset.py"
  --raw-root "${RAW_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --max-child-hyperedges "${MAX_CHILD_HYPEREDGES}"
  --formal-hidt-depth "${FORMAL_HIDT_DEPTH}"
  --overview-hops "${OVERVIEW_HOPS}"
  --overview-order-buckets "${OVERVIEW_ORDER_BUCKETS}"
)

if [[ -n "${TITLEABS_PATH}" ]]; then
  CMD+=(--titleabs-path "${TITLEABS_PATH}")
fi

exec "${CMD[@]}" "$@"
