"""Prebake hypergraph data: convert raw JSONL to parallel-array JSONL + overview semantics PT.

Usage (full mode - first run)::

    python scripts/prebake_data.py \
        --hyper-data-root ../HyperAlign-Bench/dataset/arxiv_hg \
        --task nc --split train --center-kind vertex \
        --pretrained-embedding-type sbert --num-workers 16

Usage (overview-only - switching to a new embedding type)::

    python scripts/prebake_data.py \
        --hyper-data-root ../HyperAlign-Bench/dataset/arxiv_hg \
        --task nc --split train --center-kind vertex \
        --pretrained-embedding-type qwen3emb_0.6b --overview-only --num-workers 16

Full mode produces:
  1. ``samples/*_prebaked.jsonl`` - parallel int arrays (embedding-independent)
  2. ``overview/{emb_type}/*.pt`` - overview token embedding dict ``{entity_id -> tensor(N,dim)}``

Overview-only mode skips (1) and only regenerates (2) for a new embedding type.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Prevent OpenMP/MKL deadlocks when forking with PyTorch loaded.
# Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from tqdm import tqdm

torch.set_num_threads(1)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.final_hidt import (
    FINAL_HIDT_ORDER_BUCKETS,
    FinalHIDTConfig,
    TorchFinalHIDTAccessor,
    build_final_hidt_instance,
    order_bucket_id,
)
from utils.hypergraph_features import load_hypergraph_semantic_embeddings
from utils.hypergraph_overview import (
    OVERVIEW_TASK_OPEN,
    OverviewConfig,
    OverviewTaskSpec,
    TaskConditionedOverviewGraph,
    _order_bucket_embedding,
    build_hyperedge_overview_payload,
    build_hyperedge_shells,
    build_vertex_overview_payload,
)
from utils.hypergraph_task_text import build_hypergraph_task_conversations
from utils.hypergraph_templates import normalize_hyper_template
from utils.hypergraph_text_context import build_text_context_block

# ---------------------------------------------------------------------------
# Module-level shared state populated before fork(), read-only in workers.
# On Linux fork gives copy-on-write sharing of the large tensors for free.
# ---------------------------------------------------------------------------
_G: Dict = {}


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_parallel_arrays(tokens: List[Dict]) -> Dict[str, List[int]]:
    """Extract the 6 minimal parallel arrays from a token list."""
    token_node_ids: List[int] = []
    token_he_ids: List[int] = []
    token_type_ids: List[int] = []
    token_depths: List[int] = []
    token_slot_ids: List[int] = []
    token_parent_slot_ids: List[int] = []

    for t in tokens:
        token_node_ids.append(int(t.get("node_id", -1)))
        token_he_ids.append(int(t.get("hyperedge_id", -1)))
        token_type_ids.append(int(t["type_id"]))
        token_depths.append(min(int(t.get("depth", 0)), 3))
        token_slot_ids.append(int(t.get("slot_id", -1)))
        token_parent_slot_ids.append(int(t.get("parent_slot_id", -1)))

    return {
        "token_node_ids": token_node_ids,
        "token_he_ids": token_he_ids,
        "token_type_ids": token_type_ids,
        "token_depths": token_depths,
        "token_slot_ids": token_slot_ids,
        "token_parent_slot_ids": token_parent_slot_ids,
    }


def _extract_overview_semantics(ov_tokens: List[Dict], semantic_dim: int) -> torch.Tensor:
    """Stack overview token semantic overrides into a (num_ov, dim) tensor."""
    semantics = []
    for t in ov_tokens:
        if "semantic_override" in t:
            semantics.append(torch.tensor(t["semantic_override"], dtype=torch.float16))
        else:
            semantics.append(torch.zeros(semantic_dim, dtype=torch.float16))
    if not semantics:
        return torch.zeros(0, semantic_dim, dtype=torch.float16)
    return torch.stack(semantics, dim=0)


# ---------------------------------------------------------------------------
# Vectorised overview pre-computation (sparse matmul, replaces per-entity
# recursive aggregation that was 1000x slower)
# ---------------------------------------------------------------------------

def _precompute_overview_states(
    processed_data: Dict,
    node_embeddings: torch.Tensor,
    hyperedge_embeddings: torch.Tensor | None,
    config: OverviewConfig,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Pre-compute hyperedge states for all hops using sparse matmul.

    Returns ``(he_states_by_hop, he_sizes)`` where
    ``he_states_by_hop[hop]`` is ``(num_he, dim)`` and
    ``he_sizes`` is ``(num_he,)`` with integer hyperedge sizes.
    """
    he_ptr = processed_data["he_ptr"]
    node_ptr = processed_data["node_ptr"]
    he_node_index = processed_data["he_node_index"]

    he_ptr_t = he_ptr.long() if isinstance(he_ptr, torch.Tensor) else torch.tensor(he_ptr, dtype=torch.long)
    he_ni_t = he_node_index.long() if isinstance(he_node_index, torch.Tensor) else torch.tensor(he_node_index, dtype=torch.long)
    node_ptr_t = node_ptr.long() if isinstance(node_ptr, torch.Tensor) else torch.tensor(node_ptr, dtype=torch.long)

    num_he = len(he_ptr_t) - 1
    num_nodes = node_embeddings.shape[0]
    dim = node_embeddings.shape[-1]
    total_incidences = int(he_ptr_t[-1])

    # Sparse incidence matrix H (num_nodes x num_he).
    he_lengths = he_ptr_t[1:] - he_ptr_t[:-1]
    cols = torch.arange(num_he, dtype=torch.long).repeat_interleave(he_lengths)
    rows = he_ni_t[:total_incidences]
    H = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(total_incidences, dtype=torch.float32),
        size=(num_nodes, num_he),
    ).coalesce()
    H_t = H.t().coalesce()

    he_sizes = he_lengths.float()
    he_deg_inv = (1.0 / he_sizes.clamp(min=1)).unsqueeze(-1)

    v_degree = (node_ptr_t[1:num_nodes + 1] - node_ptr_t[:num_nodes]).float()
    v_deg_inv = (1.0 / v_degree.clamp(min=1)).unsqueeze(-1)

    # ---- per-hyperedge bucket adjustment ----
    num_buckets = len(FINAL_HIDT_ORDER_BUCKETS)
    bucket_embs = torch.stack([
        _order_bucket_embedding(bid, dim, torch.float32)
        for bid in range(num_buckets)
    ])
    bucket_ids = torch.full((num_he,), num_buckets - 1, dtype=torch.long)
    for bid, (_, lower, upper) in enumerate(FINAL_HIDT_ORDER_BUCKETS):
        mask = he_sizes >= lower
        if upper is not None:
            mask = mask & (he_sizes <= upper)
        bucket_ids[mask] = bid
    bucket_adj = bucket_embs[bucket_ids]  # (num_he, dim)

    # ---- iterative message passing ----
    v_state = node_embeddings.float()
    he_states: Dict[int, torch.Tensor] = {}

    print(f"  Pre-computing overview states (dim={dim}, nodes={num_nodes}, he={num_he}) ...")
    for step in range(1, config.hops + 1):
        he_state = torch.sparse.mm(H_t, v_state) * he_deg_inv
        he_states[step] = he_state
        print(f"    hop {step} he_state done  shape={he_state.shape}")

        if step < config.hops:
            adjusted = he_state + bucket_adj
            v_state = torch.sparse.mm(H, adjusted) * v_deg_inv
            print(f"    hop {step} v_state  done  shape={v_state.shape}")

    return he_states, he_sizes


def _build_overview_from_precomputed(
    entity_id: int,
    center_kind: str,
    processed_data: Dict,
    he_states: Dict[int, torch.Tensor],
    he_sizes: torch.Tensor,
    config: OverviewConfig,
    semantic_dim: int,
) -> torch.Tensor:
    """Build overview semantics for one entity using pre-computed global states."""
    task_spec = OverviewTaskSpec(
        task_view=OVERVIEW_TASK_OPEN,
        center_kind=center_kind,
        center_id=entity_id,
    )
    task_graph = TaskConditionedOverviewGraph(processed_data, task_spec)
    shells = build_hyperedge_shells(task_graph, config)

    semantics: List[torch.Tensor] = []
    for hop_id in range(1, config.hops + 1):
        shell_hes = shells[hop_id]
        for bucket_id in range(config.order_bucket_count):
            bucket_hes = [
                he_id for he_id in shell_hes
                if order_bucket_id(int(he_sizes[he_id])) == bucket_id
            ]
            if bucket_hes:
                indices = torch.tensor(bucket_hes, dtype=torch.long)
                sem = he_states[hop_id][indices].mean(dim=0)
            else:
                sem = torch.zeros(semantic_dim, dtype=torch.float32)
            semantics.append(sem)

    return torch.stack(semantics).to(torch.float16)


# ---------------------------------------------------------------------------
# Worker function; runs in forked child process and reads from _G.
# ---------------------------------------------------------------------------

def _build_one_entity(entity_id: int) -> Tuple[int, Dict, List[int], torch.Tensor | None]:
    """Build HIDT + overview for a single entity. Returns (id, arrays, ov_bucket_ids, ov_semantic)."""
    accessor = _G["accessor"]
    hidt_config = _G["hidt_config"]
    center_kind = _G["center_kind"]
    use_overview = _G["use_overview"]

    hidt_payload = build_final_hidt_instance(
        center_kind=center_kind,
        center_id=entity_id,
        accessor=accessor,
        config=hidt_config,
    )
    hidt_tokens = hidt_payload["tokens"]

    ov_support_bucket_ids: List[int] = []
    ov_semantic: torch.Tensor | None = None

    if use_overview:
        ov_config = _G["ov_config"]
        processed_data = _G["processed_data"]
        node_embeddings = _G["node_embeddings"]
        hyperedge_embeddings = _G["hyperedge_embeddings"]
        semantic_dim = _G["semantic_dim"]

        if center_kind == "vertex":
            ov_payload = build_vertex_overview_payload(
                data_view=processed_data,
                node_id=entity_id,
                node_embeddings=node_embeddings,
                hyperedge_embeddings=hyperedge_embeddings,
                config=ov_config,
                source="vc_hidt",
            )
        else:
            ov_payload = build_hyperedge_overview_payload(
                data_view=processed_data,
                hyperedge_id=entity_id,
                node_embeddings=node_embeddings,
                hyperedge_embeddings=hyperedge_embeddings,
                config=ov_config,
                source="ec_hidt",
            )
        ov_tokens = ov_payload["tokens"]
        ov_semantic = _extract_overview_semantics(ov_tokens, semantic_dim)
        ov_support_bucket_ids = [int(t.get("degree_bucket_id", -1)) for t in ov_tokens]
        all_tokens = hidt_tokens + ov_tokens
    else:
        all_tokens = hidt_tokens

    arrays = _extract_parallel_arrays(all_tokens)
    arrays["overview_support_bucket_ids"] = ov_support_bucket_ids
    return entity_id, arrays, ov_support_bucket_ids, ov_semantic


_TOKEN_ARRAY_KEYS = (
    "token_node_ids",
    "token_he_ids",
    "token_type_ids",
    "token_depths",
    "token_slot_ids",
    "token_parent_slot_ids",
)


def _task_center_kind(task: str) -> str:
    """Return ``'vertex'`` for the nc task, ``'hyperedge'`` for hecls."""
    if task == "nc":
        return "vertex"
    if task == "hecls":
        return "hyperedge"
    raise ValueError(f"Unknown task: {task!r}; supported tasks are 'nc' and 'hecls'.")


def _build_conversations_for_row(
    task: str,
    row: Dict,
    processed_data: Dict,
    entity_arrays: Dict[str, List[int]],
    enable_text_context: bool,
) -> List[Dict]:
    """Build [human, gpt] turns for one row, optionally injecting text context.

    ``entity_arrays`` must contain the 6 token_* parallel arrays, either
    pulled from ``entity_cache`` (full mode) or read back from the existing
    prebaked JSONL (rebuild-conversations-only mode).
    """
    text_context: str | None = None
    if enable_text_context:
        text_context = build_text_context_block(
            task=task,
            entity_id=int(row["id"]),
            arrays=entity_arrays,
            processed_data=processed_data,
        )
    conversations = [
        {"from": c["from"], "value": c["value"]}
        for c in build_hypergraph_task_conversations(
            task, row, processed_data, text_context=text_context,
        )
    ]
    return conversations


def _prebaked_jsonl_name(center_kind: str, task: str, mih: int, mmh: int, split: str) -> str:
    prefix = "node" if center_kind == "vertex" else "he"
    return f"{prefix}_task_{task}_hg_{mih}_{mmh}_{split}_prebaked.jsonl"


def _overview_pt_path(center_kind: str, mih: int, mmh: int, split: str,
                      emb_type: str = "sbert") -> str:
    prefix = "node" if center_kind == "vertex" else "he"
    return str(Path("overview") / emb_type / f"{prefix}_{mih}_{mmh}_{split}.pt")


def _rebuild_conversations_only(args: argparse.Namespace) -> None:
    """Incremental mode: rewrite only the ``conversations`` field in existing
    prebaked JSONL files. Skips HIDT sampling, overview computation, and
    semantic embedding loading; completes in seconds per split.

    Preserves every other field on each row (token_* arrays,
    overview_support_bucket_ids, etc.) so the resulting file remains
    byte-compatible with the training pipeline.
    """
    hyper_data_root = Path(args.hyper_data_root)
    processed_data = torch.load(
        hyper_data_root / "processed_data.pt", map_location="cpu", weights_only=False
    )

    mih = args.max_incident_hyperedges
    mmh = args.max_members_per_hyperedge
    tasks = [t.strip() for t in args.task.split("-") if t.strip()]

    samples_dir = hyper_data_root / "samples"
    for task in tasks:
        center_kind = _task_center_kind(task)
        jsonl_path = samples_dir / _prebaked_jsonl_name(
            center_kind, task, mih, mmh, args.split,
        )
        if not jsonl_path.exists():
            print(f"[SKIP] {jsonl_path} not found")
            continue

        print(f"[rebuild] reading {jsonl_path} ...")
        rows = _read_jsonl(jsonl_path)
        print(f"  {len(rows)} rows")

        missing_arrays = [k for k in _TOKEN_ARRAY_KEYS if k not in rows[0]]
        if missing_arrays:
            raise RuntimeError(
                f"{jsonl_path} does not look like a prebaked file "
                f"(missing arrays: {missing_arrays}). "
                "Run full prebake first, then rebuild-conversations-only."
            )

        new_rows: List[Dict] = []
        for row in tqdm(rows, desc=f"conv-rebuild ({task})"):
            entity_arrays = {k: row[k] for k in _TOKEN_ARRAY_KEYS}
            conversations = _build_conversations_for_row(
                task, row, processed_data, entity_arrays,
                enable_text_context=args.enable_text_context,
            )
            new_row = dict(row)
            new_row["conversations"] = conversations
            new_rows.append(new_row)

        tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        print(f"[rebuild] writing {tmp_path} ...")
        _write_jsonl(tmp_path, new_rows)
        os.replace(tmp_path, jsonl_path)
        print(f"[rebuild] {jsonl_path} updated ({len(new_rows)} rows)")

    print("Done (rebuild-conversations-only).")


def prebake(args: argparse.Namespace) -> None:
    if getattr(args, "rebuild_conversations_only", False):
        _rebuild_conversations_only(args)
        return

    hyper_data_root = Path(args.hyper_data_root)
    overview_only = getattr(args, "overview_only", False)
    processed_data = torch.load(
        hyper_data_root / "processed_data.pt", map_location="cpu", weights_only=False
    )
    node_embeddings, hyperedge_embeddings = load_hypergraph_semantic_embeddings(
        hyper_data_root=str(hyper_data_root),
        pretrained_embedding_type=args.pretrained_embedding_type,
    )
    semantic_dim = int(node_embeddings.shape[-1])

    ov_config = OverviewConfig(hops=args.overview_hops, order_bucket_count=args.overview_order_buckets)

    mih = args.max_incident_hyperedges
    mmh = args.max_members_per_hyperedge

    if args.center_kind == "vertex":
        old_jsonl_name = f"node_task_hg_{mih}_{mmh}_{args.split}.jsonl"
    else:
        old_jsonl_name = f"he_task_hg_{mih}_{mmh}_{args.split}.jsonl"

    samples_dir = hyper_data_root / "samples"
    old_jsonl = samples_dir / old_jsonl_name
    if not old_jsonl.exists():
        old_jsonl = hyper_data_root / old_jsonl_name
    if not old_jsonl.exists():
        print(f"[SKIP] {old_jsonl} not found")
        return

    print(f"Reading {old_jsonl} ...")
    old_rows = _read_jsonl(old_jsonl)
    print(f"  {len(old_rows)} rows")

    entity_ids = list(dict.fromkeys(int(row["id"]) for row in old_rows))
    print(f"  {len(entity_ids)} unique entities")

    num_workers = args.num_workers

    # ---- Populate shared state before fork ----
    _G["processed_data"] = processed_data
    _G["node_embeddings"] = node_embeddings
    _G["hyperedge_embeddings"] = hyperedge_embeddings
    _G["ov_config"] = ov_config
    _G["center_kind"] = args.center_kind
    _G["semantic_dim"] = semantic_dim

    # ================================================================
    # Overview-only mode: skip HIDT construction and JSONL generation,
    # only produce overview/*.pt for a new embedding type.
    #
    # Strategy: pre-compute ALL hyperedge states globally via sparse
    # matmul (seconds), then per-entity just BFS + index (fast).
    # ================================================================
    if overview_only:
        print(f"[overview-only] Generating overview PT for "
              f"emb_type={args.pretrained_embedding_type}, dim={semantic_dim}")

        he_states, he_sizes = _precompute_overview_states(
            processed_data, node_embeddings, hyperedge_embeddings, ov_config,
        )

        overview_dict: Dict[int, torch.Tensor] = {}
        for eid in tqdm(entity_ids, desc="overview (indexed)"):
            ov_sem = _build_overview_from_precomputed(
                eid, args.center_kind, processed_data,
                he_states, he_sizes, ov_config, semantic_dim,
            )
            overview_dict[eid] = ov_sem

        out_pt = hyper_data_root / _overview_pt_path(
            args.center_kind, mih, mmh, args.split,
            emb_type=args.pretrained_embedding_type,
        )
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {out_pt} ({len(overview_dict)} entries) ...")
        torch.save(overview_dict, out_pt)
        print("Done (overview-only).")
        return

    # ================================================================
    # Full mode: HIDT + overview + prebaked JSONL
    # ================================================================
    accessor = TorchFinalHIDTAccessor(processed_data)
    hidt_config = FinalHIDTConfig(
        max_depth=args.formal_hidt_depth,
        max_incident_hyperedges=args.max_incident_hyperedges,
        max_members_per_hyperedge=args.max_members_per_hyperedge,
        max_child_hyperedges=args.max_child_hyperedges,
    )
    hyper_template = normalize_hyper_template(args.hyper_template)
    use_overview = hyper_template == "HIDT_O"

    _G["accessor"] = accessor
    _G["hidt_config"] = hidt_config
    _G["use_overview"] = use_overview

    tasks = [t.strip() for t in args.task.split("-") if t.strip()]

    # ---- Build hypergraph structures (parallelised) ----
    entity_cache: Dict[int, Dict] = {}
    overview_dict_full: Dict[int, torch.Tensor] = {}

    if num_workers <= 1:
        print("Building hypergraph structures (single-process) ...")
        for eid in tqdm(entity_ids, desc="hypergraph"):
            eid, arrays, _, ov_sem = _build_one_entity(eid)
            entity_cache[eid] = arrays
            if ov_sem is not None:
                overview_dict_full[eid] = ov_sem
    else:
        print(f"Building hypergraph structures ({num_workers} workers) ...")
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            for eid, arrays, _, ov_sem in tqdm(
                pool.imap_unordered(_build_one_entity, entity_ids, chunksize=1),
                total=len(entity_ids),
                desc="hypergraph",
            ):
                entity_cache[eid] = arrays
                if ov_sem is not None:
                    overview_dict_full[eid] = ov_sem

    # ---- Build per-task conversations & write JSONL ----
    for task in tasks:
        task_rows: List[Dict] = []
        for row in tqdm(old_rows, desc=f"conversations ({task})"):
            entity_id = int(row["id"])
            entity_arrays = entity_cache[entity_id]
            conversations = _build_conversations_for_row(
                task, row, processed_data, entity_arrays,
                enable_text_context=args.enable_text_context,
            )
            task_rows.append({
                "id": entity_id,
                "conversations": conversations,
                **entity_arrays,
            })

        out_jsonl = samples_dir / _prebaked_jsonl_name(
            args.center_kind, task, mih, mmh, args.split,
        )
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {out_jsonl} ({len(task_rows)} rows) ...")
        _write_jsonl(out_jsonl, task_rows)

    if use_overview and overview_dict_full:
        out_pt = hyper_data_root / _overview_pt_path(args.center_kind, mih, mmh, args.split,
                                                     emb_type=args.pretrained_embedding_type)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {out_pt} ({len(overview_dict_full)} entries) ...")
        torch.save(overview_dict_full, out_pt)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebake hypergraph data for fast training/eval.")
    parser.add_argument("--hyper-data-root", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        help="Task name: 'nc' or 'hecls' "
                             "(multi-task via dash like 'nc-hecls' is also accepted).")
    parser.add_argument("--split", type=str, required=True,
                        help="Data split: train / valid / test.")
    parser.add_argument("--center-kind", type=str, required=True,
                        choices=["vertex", "hyperedge"],
                        help="'vertex' for nc, 'hyperedge' for hecls.")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel worker processes (default: 16).")
    parser.add_argument("--pretrained-embedding-type", type=str, default="sbert")
    parser.add_argument("--overview-only", action="store_true",
                        help="Only regenerate overview/*.pt for a new embedding type; "
                             "skip HIDT construction and prebaked JSONL generation.")
    parser.add_argument("--rebuild-conversations-only", action="store_true",
                        help="Incremental mode: rewrite only the conversations field in "
                             "existing prebaked JSONL files; skip HIDT + overview. Requires "
                             "a prior full prebake.")
    parser.add_argument("--enable-text-context", action="store_true",
                        help="Inject a natural-language description of the HIDT subgraph "
                             "(titles / abstracts / member lists) into the human prompt "
                             "via the {details} placeholder. Requires datasets that carry "
                             "node-level title/abstract fields in processed_data.pt.")
    parser.add_argument("--hyper-template", type=str, default="HIDT_O")
    parser.add_argument("--max-incident-hyperedges", type=int, default=8)
    parser.add_argument("--max-members-per-hyperedge", type=int, default=8)
    parser.add_argument("--max-child-hyperedges", type=int, default=1)
    parser.add_argument("--formal-hidt-depth", type=int, default=3)
    parser.add_argument("--overview-hops", type=int, default=2)
    parser.add_argument("--overview-order-buckets", type=int, default=4)
    prebake(parser.parse_args())


if __name__ == "__main__":
    main()
