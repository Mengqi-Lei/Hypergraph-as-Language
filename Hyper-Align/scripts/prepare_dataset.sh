#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# prepare_dataset.sh
#
# One-shot data preparation for Hyper-Align:
# turn a single hypergraph dataset (already shipped in HyperAlign-Bench format)
# into release-ready prebaked JSONL + overview semantics PT, in one command.
#
# This is the recommended entry point for **new datasets** (e.g. when you fork
# the project and add your own data). It runs the canonical Hyper-Align pipeline:
#
#     [check upstream contract]
#         -> [validate release prompt_templates]
#         -> [auto-discover available (task, split) combinations]
#         -> [run scripts/prebake_data.py for each combination]
#         -> [print summary + next-step training command]
#
# Upstream contract (must be satisfied before running this script):
#
#     <root>/processed_data.pt
#         contains a `prompt_templates: {task: template_string}` dict where
#         every template carries the release placeholders {details} + {labels}
#         and a "Question:" marker.
#
#     <root>/samples/<prefix>_task_hg_8_8_<split>.jsonl
#         raw HIDT samples; <prefix> in {node, he}; <split> in {train, valid, test}.
#         At least one (prefix, split) must be present.
#
#     <root>/embeddings/<emb-type>/*.pt
#         pretrained semantic embeddings for nodes / hyperedges.
#
#     <root>/overview/<emb-type>/   (will be created if missing)
#         output directory for overview semantics. May or may not exist.
#
# Usage:
#
#     bash scripts/prepare_dataset.sh <dataset_path_or_name> [options]
#
# Examples:
#
#     # Project-internal dataset by short name (resolves to ../HyperAlign-Bench/dataset/cora_cc)
#     bash scripts/prepare_dataset.sh cora_cc
#
#     # External dataset by absolute path, qwen3 0.6B embedding
#     bash scripts/prepare_dataset.sh /data/my_new_hg --emb-type qwen3emb_0.6b
#
#     # Only VC/nc (skip HEC/hecls)
#     bash scripts/prepare_dataset.sh my_new_hg --task nc
#
#     # Dry-run: print plan and exit (no files written)
#     bash scripts/prepare_dataset.sh my_new_hg --dry-run
#
# Options:
#     --task {all|nc|hecls}        which tasks to prebake (default: all)
#     --emb-type TYPE              pretrained embedding type
#                                  (default: qwen3emb_0.6b; other: sbert / qwen3emb_4b)
#     --num-workers N              parallel workers for prebake (default: 16)
#     --no-text-context            disable {details} text-context injection
#                                  (default: ON, recommended)
#     --data-root-base PATH        prefix dir when <dataset> is a short name
#                                  (default: ../HyperAlign-Bench/dataset)
#     --dry-run                    plan only, do not call prebake_data.py
#     -h, --help                   show this help
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------- defaults --------------------------------------
TASK_FILTER="all"
EMB_TYPE="qwen3emb_0.6b"
NUM_WORKERS=16
ENABLE_TEXT_CONTEXT=1
DATA_ROOT_BASE="${DATA_ROOT_BASE:-../HyperAlign-Bench/dataset}"
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ARG=""

# ---------------------------- argparse --------------------------------------
print_help() {
    awk '/^# ----/{p=!p; if (p) {print; next} else {print; exit}} p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_FILTER="$2"; shift 2 ;;
        --emb-type)
            EMB_TYPE="$2"; shift 2 ;;
        --num-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        --no-text-context)
            ENABLE_TEXT_CONTEXT=0; shift ;;
        --data-root-base)
            DATA_ROOT_BASE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            print_help; exit 0 ;;
        --)
            shift; break ;;
        -*)
            echo "[error] unknown option: $1" >&2
            echo "        run with -h for usage." >&2
            exit 2 ;;
        *)
            if [[ -z "${DATASET_ARG}" ]]; then
                DATASET_ARG="$1"; shift
            else
                echo "[error] unexpected positional argument: $1" >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "${DATASET_ARG}" ]]; then
    echo "[error] missing required positional argument: <dataset_path_or_name>" >&2
    echo "        run with -h for usage." >&2
    exit 2
fi

case "${TASK_FILTER}" in
    all|nc|hecls) ;;
    *)
        echo "[error] --task must be one of: all | nc | hecls (got: ${TASK_FILTER})" >&2
        exit 2 ;;
esac

_alternate_dataset_dir() {
    case "$1" in
        arxiv_hg) printf '%s\n' "ogbn-arxiv-hg" ;;
        cora_cc) printf '%s\n' "cora_co_hg" ;;
        pubmed) printf '%s\n' "pubmed_hg" ;;
        dblp) printf '%s\n' "dblp_a_hg" ;;
        imdb) printf '%s\n' "imdb_hg" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# ---------------------------- resolve dataset path ---------------------------
if [[ "${DATASET_ARG}" == */* || -d "${DATASET_ARG}" ]]; then
    # contains a slash, or is an existing dir -> treat as path
    DATASET_PATH="${DATASET_ARG}"
else
    DATASET_PATH="${DATA_ROOT_BASE}/${DATASET_ARG}"
    ALTERNATE_DATASET_PATH="${DATA_ROOT_BASE}/$(_alternate_dataset_dir "${DATASET_ARG}")"
    if [[ ! -d "${DATASET_PATH}" && -d "${ALTERNATE_DATASET_PATH}" ]]; then
        DATASET_PATH="${ALTERNATE_DATASET_PATH}"
    fi
fi

# Make absolute for clearer logs (don't fail if not yet existent: the next check will)
if [[ -d "${DATASET_PATH}" ]]; then
    DATASET_PATH="$(cd "${DATASET_PATH}" && pwd)"
fi

DATASET_NAME="$(basename "${DATASET_PATH}")"

echo "================================================================"
echo "  prepare_dataset.sh"
echo "----------------------------------------------------------------"
echo "  dataset            : ${DATASET_NAME}"
echo "  resolved path      : ${DATASET_PATH}"
echo "  task filter        : ${TASK_FILTER}"
echo "  embedding type     : ${EMB_TYPE}"
echo "  num workers        : ${NUM_WORKERS}"
echo "  text-context inj.  : ${ENABLE_TEXT_CONTEXT}"
echo "  dry-run            : ${DRY_RUN}"
echo "================================================================"

# ---------------------------- precheck #1: directory exists ------------------
if [[ ! -d "${DATASET_PATH}" ]]; then
    cat >&2 <<EOF
[error] dataset directory does not exist:
        ${DATASET_PATH}

If you used a short name, the script tried to resolve it as
\${DATA_ROOT_BASE}/<name>, where DATA_ROOT_BASE = ${DATA_ROOT_BASE}.

Either:
  - pass an absolute or relative path containing '/'
  - or set --data-root-base /your/path
  - or place the dataset under ${DATA_ROOT_BASE}/${DATASET_NAME}
EOF
    exit 1
fi

# ---------------------------- precheck #2: processed_data.pt -----------------
PROCESSED_PT="${DATASET_PATH}/processed_data.pt"
if [[ ! -f "${PROCESSED_PT}" ]]; then
    cat >&2 <<EOF
[error] missing required file: ${PROCESSED_PT}

This is produced by your upstream HyperAlign-Bench preprocessing pipeline. It
must contain at minimum:
  - 'nodes', 'hyperedges' (entity tables)
  - 'splits' (per-task train/valid/test indices)
  - 'prompt_templates': dict[str, str]
       e.g. {'nc': '... {details} ... Question: ... {labels} ...',
             'hecls': '...'}

Released HyperAlign-Bench packages already follow this contract. For custom
datasets, mirror the prompt_templates format from those packages.
EOF
    exit 1
fi

# ---------------------------- precheck #3: prompt_templates release format ---
"${PYTHON_BIN}" - "${PROCESSED_PT}" "${TASK_FILTER}" <<'PYEOF'
import sys
import torch

path, task_filter = sys.argv[1], sys.argv[2]
data = torch.load(path, map_location="cpu", weights_only=False)
templates = data.get("prompt_templates")

if not isinstance(templates, dict) or not templates:
    print(f"[error] processed_data.pt has no usable 'prompt_templates' dict", file=sys.stderr)
    sys.exit(3)

required_tasks = ("nc", "hecls") if task_filter == "all" else (task_filter,)
missing = [t for t in required_tasks if t not in templates]
if missing:
    print(f"[error] processed_data.pt is missing prompt_templates for: {missing}", file=sys.stderr)
    print(f"        existing keys: {sorted(templates.keys())}", file=sys.stderr)
    sys.exit(3)

problems = []
for task in required_tasks:
    tmpl = templates[task]
    if not isinstance(tmpl, str):
        problems.append(f"  - [{task}] template is not a string (got {type(tmpl).__name__})")
        continue
    if "{details}" not in tmpl:
        problems.append(f"  - [{task}] missing {{details}} placeholder")
    if tmpl.count("{details}") > 1:
        problems.append(f"  - [{task}] more than one {{details}} placeholder")
    if "{labels}" not in tmpl:
        problems.append(f"  - [{task}] missing {{labels}} placeholder")
    if "Question:" not in tmpl:
        problems.append(f"  - [{task}] missing 'Question:' marker")

if problems:
    print("[error] prompt_templates do not match the Hyper-Align three-stage format:", file=sys.stderr)
    for p in problems:
        print(p, file=sys.stderr)
    print("", file=sys.stderr)
    print("Template skeleton (Background -> Details -> Question):", file=sys.stderr)
    print("", file=sys.stderr)
    print("    Given a node-centered hypergraph: <hypergraph>, where ...", file=sys.stderr)
    print("    {details}", file=sys.stderr)
    print("", file=sys.stderr)
    print("    Question: ... The N classes are: {labels}. Directly output the class name.", file=sys.stderr)
    print("", file=sys.stderr)
    sys.exit(3)

print("[ok] prompt_templates pass release validation for tasks: " + ", ".join(required_tasks))
PYEOF

# ---------------------------- precheck #4: embeddings dir --------------------
EMB_DIR="${DATASET_PATH}/embeddings/${EMB_TYPE}"
if [[ ! -d "${EMB_DIR}" ]] || [[ -z "$(ls -A "${EMB_DIR}" 2>/dev/null || true)" ]]; then
    cat >&2 <<EOF
[error] missing or empty pretrained embedding directory:
        ${EMB_DIR}

Generate it with your upstream HyperAlign-Bench embedding script for
emb-type='${EMB_TYPE}', or pass --emb-type with a different type that you
already have under ${DATASET_PATH}/embeddings/.
EOF
    if [[ -d "${DATASET_PATH}/embeddings" ]]; then
        echo "        embeddings/ currently contains:" >&2
        ls -1 "${DATASET_PATH}/embeddings/" >&2 | sed 's/^/          /'
    fi
    exit 1
fi

# ---------------------------- precheck #5: at least one raw jsonl ------------
SAMPLES_DIR="${DATASET_PATH}/samples"
if [[ ! -d "${SAMPLES_DIR}" ]]; then
    echo "[error] missing samples directory: ${SAMPLES_DIR}" >&2
    exit 1
fi

# ---------------------------- discover (task, split) combos ------------------
declare -a JOBS=()    # each entry: "<task>|<split>|<center>"

discover() {
    local task=$1 center=$2 prefix
    if [[ "$center" == "vertex" ]]; then prefix="node"; else prefix="he"; fi
    for split in train valid test; do
        local raw="${SAMPLES_DIR}/${prefix}_task_hg_8_8_${split}.jsonl"
        if [[ -f "${raw}" ]]; then
            JOBS+=("${task}|${split}|${center}")
        fi
    done
}

if [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "nc"    ]]; then discover nc    vertex   ; fi
if [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "hecls" ]]; then discover hecls hyperedge; fi

if [[ "${#JOBS[@]}" -eq 0 ]]; then
    cat >&2 <<EOF
[error] no raw HIDT JSONL files found under ${SAMPLES_DIR}.

Expected at least one of:
  node_task_hg_8_8_{train,valid,test}.jsonl   (for VC/nc)
  he_task_hg_8_8_{train,valid,test}.jsonl     (for HEC/hecls)
EOF
    exit 1
fi

echo ""
echo "[plan] will prebake ${#JOBS[@]} (task, split) combination(s):"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r t s c <<<"${entry}"
    printf '         - task=%-6s split=%-5s center=%s\n' "$t" "$s" "$c"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "[dry-run] no files written. Exit."
    exit 0
fi

# ---------------------------- run prebake ------------------------------------
TEXT_FLAG=""
if [[ "${ENABLE_TEXT_CONTEXT}" == "1" ]]; then
    TEXT_FLAG="--enable-text-context"
fi

cd "${REPO_ROOT}"

declare -a PRODUCED=()

for entry in "${JOBS[@]}"; do
    IFS='|' read -r task split center <<<"${entry}"
    echo ""
    echo "================================================================"
    echo "[prebake] dataset=${DATASET_NAME}  task=${task}  split=${split}  center=${center}"
    echo "================================================================"

    "${PYTHON_BIN}" scripts/prebake_data.py \
        --hyper-data-root "${DATASET_PATH}" \
        --task "${task}" \
        --split "${split}" \
        --center-kind "${center}" \
        --pretrained-embedding-type "${EMB_TYPE}" \
        --num-workers "${NUM_WORKERS}" \
        ${TEXT_FLAG}

    if [[ "${center}" == "vertex" ]]; then
        prefix="node"
    else
        prefix="he"
    fi
    PRODUCED+=("${SAMPLES_DIR}/${prefix}_task_${task}_hg_8_8_${split}_prebaked.jsonl")
done

# ---------------------------- summary ----------------------------------------
echo ""
echo "================================================================"
echo "  All prebake jobs finished for dataset: ${DATASET_NAME}"
echo "================================================================"
echo ""
echo "  Prebaked JSONL files (embedding-agnostic):"
for f in "${PRODUCED[@]}"; do
    if [[ -f "${f}" ]]; then
        size=$(du -h "${f}" | cut -f1)
        printf '    [ok]   %s  (%s)\n' "${f}" "${size}"
    else
        printf '    [miss] %s\n' "${f}"
    fi
done

OV_PT_DIR="${DATASET_PATH}/overview/${EMB_TYPE}"
echo ""
echo "  Overview semantics (embedding-specific, emb-type=${EMB_TYPE}):"
if [[ -d "${OV_PT_DIR}" ]]; then
    for f in "${OV_PT_DIR}"/*.pt; do
        [[ -e "${f}" ]] || continue
        size=$(du -h "${f}" | cut -f1)
        printf '    [ok]   %s  (%s)\n' "${f}" "${size}"
    done
else
    echo "    (overview dir not found: ${OV_PT_DIR})"
fi

cat <<EOF

  Next step: train the model on this dataset.
  Below is a copy-pasteable training command (VC/nc, single-card smoke test):

      cd ${REPO_ROOT}
      HYPER_DATA_ROOT=${DATASET_PATH} \\
      USE_DEEPSPEED=0 CUDA_VISIBLE_DEVICES=0 \\
          bash scripts/_train_hyperalign_task.sh 1 1 1 ${DATASET_NAME}_smoke ${EMB_TYPE}

  For the main public training recipe, use:

      MODEL_BASE=Qwen/Qwen3-8B \\
      HYPER_DATA_ROOT=${DATASET_PATH} \\
          bash scripts/train_arxiv_hg_joint_vc_hec_2ep.sh

EOF
