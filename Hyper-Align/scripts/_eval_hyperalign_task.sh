#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

task="${TASK_NAME:?TASK_NAME must be set by the wrapper script}"
dataset="${DATASET_NAME:?DATASET_NAME must be set by the wrapper script}"
default_hyper_data_root="${DEFAULT_HYPER_DATA_ROOT:?DEFAULT_HYPER_DATA_ROOT must be set by the wrapper script}"

normalize_split() {
  local raw="${1:-}"
  case "${raw,,}" in
    val|valid|validation)
      printf '%s\n' "valid"
      ;;
    test)
      printf '%s\n' "test"
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_split_or_fail() {
  local raw="${1:-}"
  local normalized
  if ! normalized="$(normalize_split "${raw}")"; then
    echo "Unsupported eval split: ${raw}. Use valid/val/test." >&2
    exit 1
  fi
  printf '%s\n' "${normalized}"
}

model_path="${1:-/path/to/projector}"
start_idx="${2:-0}"
end_idx="${3:-100}"
output_tag="${4:-${task}_eval}"
default_model_base="${BASE_MODEL:?BASE_MODEL must be set (path or HF id of the base LLM)}"
arg5="${5:-}"
arg6="${6:-}"

if [[ -n "${arg5}" ]] && cli_split="$(normalize_split "${arg5}")"; then
  model_base="${default_model_base}"
else
  model_base="${arg5:-${default_model_base}}"
  cli_split=""
fi

if [[ -n "${arg6}" ]]; then
  cli_split="$(resolve_split_or_fail "${arg6}")"
fi

if [[ -n "${EVAL_SPLIT:-}" ]]; then
  eval_split="$(resolve_split_or_fail "${EVAL_SPLIT}")"
elif [[ -n "${cli_split}" ]]; then
  eval_split="${cli_split}"
else
  eval_split="test"
fi

mode="${CONV_MODE:-qwen3}"
emb="${PRETRAINED_EMBEDDING_TYPE:-qwen3emb_0.6b}"
template="${HYPER_TEMPLATE:-HIDT_O}"
hyper_data_root="${HYPER_DATA_ROOT:-${default_hyper_data_root}}"
max_hypergraph_tokens="${MAX_HYPERGRAPH_TOKENS:-160}"
max_incident_hyperedges="${MAX_INCIDENT_HYPEREDGES:-8}"
max_members_per_hyperedge="${MAX_MEMBERS_PER_HYPEREDGE:-8}"
max_child_hyperedges="${MAX_CHILD_HYPEREDGES:-1}"
formal_hidt_depth="${FORMAL_HIDT_DEPTH:-3}"
overview_hops="${OVERVIEW_HOPS:-2}"
overview_order_buckets="${OVERVIEW_ORDER_BUCKETS:-4}"
projector_incidence_mode="${PROJECTOR_INCIDENCE_MODE:-sample_real}"
temperature="${TEMPERATURE:-0}"
top_p="${TOP_P:-1.0}"
num_beams="${NUM_BEAMS:-1}"
omp_threads="${OMP_THREADS:-8}"
eval_batch_size="${EVAL_BATCH_SIZE:-1}"
eval_num_workers="${EVAL_NUM_WORKERS:-0}"
eval_max_new_tokens="${EVAL_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS:-}}"
flush_every="${EVAL_FLUSH_EVERY:-${FLUSH_EVERY:-1}}"
random_sample_size="${EVAL_RANDOM_SAMPLE_SIZE:-${RANDOM_SAMPLE_SIZE:-}}"
random_sample_seed="${EVAL_RANDOM_SAMPLE_SEED:-${RANDOM_SAMPLE_SEED:-42}}"
num_eval_gpus="${NUM_EVAL_GPUS:-0}"
python_bin="${PYTHON_BIN:-python}"
cache_dir="${CACHE_DIR:-../../checkpoint}"
eval_dtype="${EVAL_DTYPE:-bf16}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${task}_${output_tag}"
eval_dir="./results/${dataset}/${timestamp}_${run_name}"

if [[ -e "${eval_dir}" ]]; then
  suffix=1
  while [[ -e "${eval_dir}_${suffix}" ]]; do
    suffix=$((suffix + 1))
  done
  eval_dir="${eval_dir}_${suffix}"
fi

answers_file="${ANSWERS_FILE:-${eval_dir}/answers_${eval_split}_${start_idx}_${end_idx}.jsonl}"
log_file="${eval_dir}/eval.log"

mkdir -p "${eval_dir}"
mkdir -p "$(dirname "${answers_file}")"

{
  echo "=== Hyper-Align Eval Log ==="
  echo "Started : $(date)"
  echo "Command : $0 $*"
  echo "Full cmd: BASE_MODEL=${default_model_base} CONV_MODE=${mode:-unset} $0 $*"
  echo "=== Configuration ==="
  echo "  eval_dir     : ${eval_dir}"
  echo "  log_file     : ${log_file}"
  echo "  model_path   : ${model_path}"
  echo "  model_base   : ${model_base}"
  echo "  conv_mode    : ${mode}"
  echo "  task         : ${task}"
  echo "  dataset      : ${dataset}"
  echo "  split        : ${eval_split}"
  echo "  emb          : ${emb}"
  echo "  answers_file : ${answers_file}"
  echo "  start        : ${start_idx}"
  echo "  end          : ${end_idx}"
  echo "  eval_bs      : ${eval_batch_size}"
  echo "  eval_workers : ${eval_num_workers}"
  echo "  flush_every  : ${flush_every}"
  echo "  num_gpus     : ${num_eval_gpus}"
  echo "  temperature  : ${temperature}"
  echo "  top_p        : ${top_p}"
  echo "  num_beams    : ${num_beams}"
  echo "  inc_mode     : ${projector_incidence_mode}"
  echo "  eval_dtype   : ${eval_dtype}"
  echo "======================="
  echo ""
} > "${log_file}"

echo "model_path  : ${model_path}"
echo "task        : ${task}"
echo "dataset     : ${dataset}"
echo "split       : ${eval_split}"
echo "eval_dir    : ${eval_dir}"
echo "start       : ${start_idx}"
echo "end         : ${end_idx}"
echo "answers     : ${answers_file}"
echo "eval_bs     : ${eval_batch_size}"
echo "eval_workers: ${eval_num_workers}"
echo "flush_every : ${flush_every}"
echo "max_new_tok : ${eval_max_new_tokens:-default}"
echo "rand_sample : ${random_sample_size:-all}"
echo "rand_seed   : ${random_sample_seed}"
echo "num_gpus    : ${num_eval_gpus} (0=auto)"
echo "temperature : ${temperature}"
echo "inc_mode    : ${projector_incidence_mode}"
echo "eval_dtype  : ${eval_dtype}"

export OMP_NUM_THREADS="${omp_threads}"

cd "${REPO_ROOT}"

eval_args=(
  --eval-batch-size "${eval_batch_size}"
  --eval-num-workers "${eval_num_workers}"
  --flush-every "${flush_every}"
)

if [[ -n "${eval_max_new_tokens}" ]]; then
  eval_args+=(--max-new-tokens "${eval_max_new_tokens}")
fi

if [[ -n "${random_sample_size}" ]]; then
  eval_args+=(--random-sample-size "${random_sample_size}" --random-sample-seed "${random_sample_seed}")
fi

"${python_bin}" eval/eval_pretrain.py \
  --model_path "${model_path}" \
  --model_base "${model_base}" \
  --conv_mode "${mode}" \
  --dataset "${dataset}" \
  --pretrained_embedding_type "${emb}" \
  --answers_file "${answers_file}" \
  --task "${task}" \
  --cache_dir "${cache_dir}" \
  --hyper_template "${template}" \
  --hyper_data_root "${hyper_data_root}" \
  --max_hypergraph_tokens "${max_hypergraph_tokens}" \
  --max_incident_hyperedges "${max_incident_hyperedges}" \
  --max_members_per_hyperedge "${max_members_per_hyperedge}" \
  --max_child_hyperedges "${max_child_hyperedges}" \
  --formal_hidt_depth "${formal_hidt_depth}" \
  --overview_hops "${overview_hops}" \
  --overview_order_buckets "${overview_order_buckets}" \
  --projector_incidence_mode "${projector_incidence_mode}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --num_beams "${num_beams}" \
  --split "${eval_split}" \
  --start "${start_idx}" \
  --end "${end_idx}" \
  --num-gpus "${num_eval_gpus}" \
  --eval-dtype "${eval_dtype}" \
  "${eval_args[@]}" 2>&1 | tee -a "${log_file}"

"${python_bin}" eval/eval_res.py \
  --dataset "${dataset}" \
  --task "${task}" \
  --res_path "${answers_file}" \
  --hyper_data_root "${hyper_data_root}" 2>&1 | tee -a "${log_file}"
