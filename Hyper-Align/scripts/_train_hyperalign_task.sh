#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

task="${TASK_NAME:?TASK_NAME must be set by the wrapper script}"
dataset="${DATASET_NAME:?DATASET_NAME must be set by the wrapper script}"
default_hyper_data_root="${DEFAULT_HYPER_DATA_ROOT:?DEFAULT_HYPER_DATA_ROOT must be set by the wrapper script}"

num_epochs="${1:-1}"
batch_size="${2:-16}"
grad_accum="${3:-1}"
output_tag="${4:-${task}_run}"
emb="${5:-sbert}"

model_base="${MODEL_BASE:?MODEL_BASE must be set (path or HF id of the base LLM)}"
mode="${CONV_MODE:-qwen3}"
max_len="${MODEL_MAX_LENGTH:-4096}"

model_short_name="$(basename "${model_base}")"
projector_type="${PROJECTOR_TYPE:-htp}"
htp_semantic_core_dim="${HTP_SEMANTIC_CORE_DIM:-384}"
htp_structure_sidecar_dim="${HTP_STRUCTURE_SIDECAR_DIM:-64}"
htp_num_layers="${HTP_NUM_LAYERS:-1}"
projector_incidence_mode="${PROJECTOR_INCIDENCE_MODE:-sample_real}"
prefix="hyperalign-${model_short_name}-${emb}-hidt-${projector_type}"

template="${HYPER_TEMPLATE:-HIDT_O}"
hyper_data_root="${HYPER_DATA_ROOT:-${default_hyper_data_root}}"
max_hypergraph_tokens="${MAX_HYPERGRAPH_TOKENS:-160}"
max_incident_hyperedges="${MAX_INCIDENT_HYPEREDGES:-8}"
max_members_per_hyperedge="${MAX_MEMBERS_PER_HYPEREDGE:-8}"
max_child_hyperedges="${MAX_CHILD_HYPEREDGES:-1}"
formal_hidt_depth="${FORMAL_HIDT_DEPTH:-3}"
overview_hops="${OVERVIEW_HOPS:-2}"
overview_order_buckets="${OVERVIEW_ORDER_BUCKETS:-4}"
learning_rate="${LEARNING_RATE:-2e-3}"
warmup_ratio="${WARMUP_RATIO:-0.03}"
weight_decay="${WEIGHT_DECAY:-0.0}"
logging_steps="${LOGGING_STEPS:-10}"
save_strategy="${SAVE_STRATEGY:-epoch}"
save_steps="${SAVE_STEPS:-200}"
report_to="${REPORT_TO:-none}"
omp_threads="${OMP_THREADS:-8}"
python_bin="${PYTHON_BIN:-python}"
cache_dir="${CACHE_DIR:-../../checkpoint}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-True}"
lambda_ord="${LAMBDA_ORD:-0.0}"
lambda_rel="${LAMBDA_REL:-0.0}"
consistency_start_step="${CONSISTENCY_START_STEP:-0}"
consistency_warmup_steps="${CONSISTENCY_WARMUP_STEPS:-0}"
use_deepspeed="${USE_DEEPSPEED:-0}"
deepspeed_config="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero2.json}"
deepspeed_include="${DEEPSPEED_INCLUDE:-localhost:0}"
master_port="${MASTER_PORT:-61000}"
max_steps_override="${MAX_STEPS:-}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${prefix}_${task}_${output_tag}"
output_dir="./checkpoints/${dataset}/${timestamp}_${run_name}"

if [[ -e "${output_dir}" ]]; then
  suffix=1
  while [[ -e "${output_dir}_${suffix}" ]]; do
    suffix=$((suffix + 1))
  done
  output_dir="${output_dir}_${suffix}"
fi

log_file="${output_dir}/train.log"
mkdir -p "${output_dir}"

{
  echo "=== Hyper-Align Training Log ==="
  echo "Started : $(date)"
  echo "Command : $0 $*"
  echo "Full cmd: MODEL_BASE=${model_base} CONV_MODE=${mode} $0 $*"
  echo "=== Configuration ==="
  echo "  output_dir  : ${output_dir}"
  echo "  model_base  : ${model_base}"
  echo "  conv_mode   : ${mode}"
  echo "  task        : ${task}"
  echo "  dataset     : ${dataset}"
  echo "  emb         : ${emb}"
  echo "  epochs      : ${num_epochs}"
  echo "  batch_size  : ${batch_size}"
  echo "  grad_accum  : ${grad_accum}"
  echo "  effective_bs: $(( batch_size * grad_accum ))"
  echo "  data_root   : ${hyper_data_root}"
  echo "  num_workers : ${dataloader_num_workers}"
  echo "  grad_ckpt   : ${gradient_checkpointing}"
  echo "  deepspeed   : ${use_deepspeed}"
  echo "  lr          : ${learning_rate}"
  echo "  template    : ${template}"
  echo "  projector   : ${projector_type}"
  if [[ "${projector_type}" == "htp" ]]; then
    echo "  sem_dim     : ${htp_semantic_core_dim}"
    echo "  str_dim     : ${htp_structure_sidecar_dim}"
    echo "  layers      : ${htp_num_layers}"
    echo "  inc_mode    : ${projector_incidence_mode}"
  fi
  if [[ -n "${max_steps_override}" ]]; then
    echo "  max_steps   : ${max_steps_override} (override)"
  fi
  echo "==========================="
  echo ""
} > "${log_file}"

echo "model_base  : ${model_base}"
echo "conv_mode   : ${mode}"
echo "task        : ${task}"
echo "dataset     : ${dataset}"
echo "epochs      : ${num_epochs}"
echo "batch_size  : ${batch_size}"
echo "grad_accum  : ${grad_accum}"
echo "effective_bs: $(( batch_size * grad_accum ))"
echo "output_dir  : ${output_dir}"
echo "data_root   : ${hyper_data_root}"
echo "num_workers : ${dataloader_num_workers}"
echo "grad_ckpt   : ${gradient_checkpointing}"
echo "deepspeed   : ${use_deepspeed}"
if [[ -n "${max_steps_override}" ]]; then
  echo "max_steps   : ${max_steps_override} (override)"
fi

export OMP_NUM_THREADS="${omp_threads}"
wandb offline >/dev/null 2>&1 || true

common_args=(
  --model_name_or_path "${model_base}"
  --version "${mode}"
  --cache_dir "${cache_dir}"
  --pretrained_embedding_type "${emb}"
  --tune_mm_mlp_adapter True
  --mm_use_graph_start_end False
  --mm_use_graph_patch_token False
  --bf16 True
  --output_dir "${output_dir}"
  --overwrite_output_dir True
  --num_train_epochs "${num_epochs}"
  --per_device_train_batch_size "${batch_size}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${grad_accum}"
  --dataloader_num_workers "${dataloader_num_workers}"
  --eval_strategy "no"
  --save_strategy "${save_strategy}"
  --learning_rate "${learning_rate}"
  --weight_decay "${weight_decay}"
  --warmup_ratio "${warmup_ratio}"
  --lr_scheduler_type "cosine"
  --logging_steps "${logging_steps}"
  --tf32 True
  --model_max_length "${max_len}"
  --gradient_checkpointing "${gradient_checkpointing}"
  --lazy_preprocess True
  --report_to "${report_to}"
  --mm_projector_type "${projector_type}"
  --lambda_ord "${lambda_ord}"
  --lambda_rel "${lambda_rel}"
  --htp_semantic_core_dim "${htp_semantic_core_dim}"
  --htp_structure_sidecar_dim "${htp_structure_sidecar_dim}"
  --htp_num_layers "${htp_num_layers}"
  --projector_incidence_mode "${projector_incidence_mode}"
  --use_task "${task}"
  --use_dataset "${dataset}"
  --template "${template}"
  --hyper_template "${template}"
  --hyper_data_root "${hyper_data_root}"
  --max_hypergraph_tokens "${max_hypergraph_tokens}"
  --max_incident_hyperedges "${max_incident_hyperedges}"
  --max_members_per_hyperedge "${max_members_per_hyperedge}"
  --max_child_hyperedges "${max_child_hyperedges}"
  --formal_hidt_depth "${formal_hidt_depth}"
  --overview_hops "${overview_hops}"
  --overview_order_buckets "${overview_order_buckets}"
  --consistency_start_step "${consistency_start_step}"
  --consistency_warmup_steps "${consistency_warmup_steps}"
)

if [[ "${save_strategy}" == "steps" ]]; then
  common_args+=(--save_steps "${save_steps}")
fi

if [[ -n "${max_steps_override}" ]]; then
  common_args+=(--max_steps "${max_steps_override}")
fi

cd "${REPO_ROOT}"

if [[ "${use_deepspeed}" == "1" ]]; then
  deepspeed --include "${deepspeed_include}" --master_port "${master_port}" train/train_mem.py \
    --deepspeed "${deepspeed_config}" \
    "${common_args[@]}" 2>&1 | tee -a "${log_file}"
else
  "${python_bin}" train/train_mem.py "${common_args[@]}" 2>&1 | tee -a "${log_file}"
fi
