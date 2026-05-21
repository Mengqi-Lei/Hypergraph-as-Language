from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch

from utils.constants import DEFAULT_GRAPH_PAD_ID
from utils.final_hidt import (
    FINAL_HIDT_POSITION_DIM,
    build_final_hidt_template,
)

HYPERGRAPH_STRUCTURE_DIM = FINAL_HIDT_POSITION_DIM + 4 + 4 + 4 + 4


SIMTEG_COMPONENTS = ("simteg_sbert", "simteg_roberta", "simteg_e5")


def resolve_embedding_paths(
    hyper_data_root: str,
    node_embedding_path: str | None = None,
    hyperedge_embedding_path: str | None = None,
    pretrained_embedding_type: str = "sbert",
) -> Tuple[Path, Path]:
    root = Path(hyper_data_root)
    emb_dir = root / "embeddings" / pretrained_embedding_type
    node_path = Path(node_embedding_path) if node_embedding_path else emb_dir / "node_x.pt"
    hyperedge_path = (
        Path(hyperedge_embedding_path)
        if hyperedge_embedding_path
        else emb_dir / "hyperedge_x.pt"
    )
    return node_path, hyperedge_path


def _load_single_embedding(path: Path, dtype: torch.dtype) -> torch.Tensor:
    return torch.load(path, map_location="cpu").to(dtype=dtype)


def _load_simteg_concat(
    root: Path,
    entity_prefix: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load and concatenate the three SimTEG component embeddings."""
    parts = []
    for component in SIMTEG_COMPONENTS:
        path = root / "embeddings" / component / f"{entity_prefix}_x.pt"
        parts.append(_load_single_embedding(path, dtype))
    return torch.cat(parts, dim=-1)


def load_hypergraph_semantic_embeddings(
    hyper_data_root: str,
    node_embedding_path: str | None = None,
    hyperedge_embedding_path: str | None = None,
    pretrained_embedding_type: str = "sbert",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    root = Path(hyper_data_root)
    if pretrained_embedding_type == "simteg" and node_embedding_path is None:
        node_embeddings = _load_simteg_concat(root, "node", dtype)
        he_embeddings = _load_simteg_concat(root, "hyperedge", dtype)
        return node_embeddings, he_embeddings
    node_path, hyperedge_path = resolve_embedding_paths(
        hyper_data_root=hyper_data_root,
        node_embedding_path=node_embedding_path,
        hyperedge_embedding_path=hyperedge_embedding_path,
        pretrained_embedding_type=pretrained_embedding_type,
    )
    node_embeddings = _load_single_embedding(node_path, dtype)
    hyperedge_embeddings = _load_single_embedding(hyperedge_path, dtype)
    return node_embeddings, hyperedge_embeddings


def infer_semantic_dim(
    hyper_data_root: str,
    pretrained_embedding_type: str = "sbert",
    node_embedding_path: str | None = None,
) -> int:
    """Infer the semantic embedding dimension from the actual .pt file on disk."""
    root = Path(hyper_data_root)
    if pretrained_embedding_type == "simteg" and node_embedding_path is None:
        total = 0
        for component in SIMTEG_COMPONENTS:
            path = root / "embeddings" / component / "node_x.pt"
            t = torch.load(path, map_location="cpu")
            total += t.shape[-1]
            del t
        return total
    node_path, _ = resolve_embedding_paths(
        hyper_data_root=hyper_data_root,
        node_embedding_path=node_embedding_path,
        pretrained_embedding_type=pretrained_embedding_type,
    )
    t = torch.load(node_path, map_location="cpu")
    dim = t.shape[-1]
    del t
    return dim


# ---------------------------------------------------------------------------
# Vectorised bucket helpers
# ---------------------------------------------------------------------------

def order_bucket_id_vec(sizes: torch.Tensor) -> torch.Tensor:
    """Vectorised order_bucket_id; exact mirror of scalar ``order_bucket_id``.

    FINAL_HIDT_ORDER_BUCKETS: r2=[2,2], r3_4=[3,4], r5_8=[5,8], r9p=[9,+inf)
    Sizes below 2 fall through to bucket 3 (same as the scalar version).
    """
    result = torch.full_like(sizes, 3, dtype=torch.long)
    result[(sizes >= 5) & (sizes <= 8)] = 2
    result[(sizes >= 3) & (sizes <= 4)] = 1
    result[sizes == 2] = 0
    return result


def degree_bucket_id_vec(degrees: torch.Tensor) -> torch.Tensor:
    """Vectorised degree_bucket_id; exact mirror of scalar ``degree_bucket_id``.

    FINAL_HIDT_DEGREE_BUCKETS: d0=[0,0], d1_2=[1,2], d3_7=[3,7], d8p=[8,+inf)
    Negative degrees fall through to bucket 3 (same as the scalar version).
    """
    result = torch.full_like(degrees, 3, dtype=torch.long)
    result[(degrees >= 3) & (degrees <= 7)] = 2
    result[(degrees >= 1) & (degrees <= 2)] = 1
    result[degrees == 0] = 0
    return result


# ---------------------------------------------------------------------------
# Laplacian position code cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _cached_laplacian_position_codes(
    center_kind: str,
    max_depth: int,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
    position_dim: int,
) -> torch.Tensor:
    """Return the template's Laplacian position codes (num_slots, position_dim)."""
    template = build_final_hidt_template(
        center_kind=center_kind,
        max_depth=max_depth,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
        position_dim=position_dim,
    )
    return template["laplacian_position_codes"]


# ---------------------------------------------------------------------------
# Core: build_graph_tensors  (vectorised, parallel-array interface)
# ---------------------------------------------------------------------------

def build_graph_tensors(
    token_node_ids: torch.LongTensor,
    token_he_ids: torch.LongTensor,
    token_type_ids: torch.LongTensor,
    token_depths: torch.LongTensor,
    token_slot_ids: torch.LongTensor,
    overview_semantics: torch.Tensor | None,
    node_embeddings: torch.Tensor,
    hyperedge_embeddings: torch.Tensor,
    processed_data: Dict,
    max_tokens: int,
    overview_support_bucket_ids: torch.LongTensor | None = None,
    hidt_center_kind: str = "vertex",
    hidt_max_depth: int = 3,
    hidt_max_incident_hyperedges: int = 8,
    hidt_max_members_per_hyperedge: int = 8,
    hidt_max_child_hyperedges: int = 1,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Build graph + graph_emb tensors from pre-computed parallel arrays.

    All heavy Python-loop work is replaced by bulk tensor indexing.
    """
    n = len(token_type_ids)
    semantic_dim = node_embeddings.shape[-1]
    dtype = node_embeddings.dtype

    graph = torch.full((1, max_tokens), DEFAULT_GRAPH_PAD_ID, dtype=torch.long)
    graph[0, :n] = torch.arange(n)

    # --- derived masks ---
    is_pad = token_type_ids >= 2
    overview_mask = (token_slot_ids < 0) & ~is_pad
    vertex_mask = (token_type_ids == 0) & ~is_pad & ~overview_mask
    he_mask = (token_type_ids == 1) & ~is_pad & ~overview_mask

    # --- semantic embeddings (3 bulk gathers) ---
    all_semantic = torch.zeros(n, semantic_dim, dtype=dtype)
    if vertex_mask.any():
        all_semantic[vertex_mask] = node_embeddings[token_node_ids[vertex_mask]]
    if he_mask.any():
        all_semantic[he_mask] = hyperedge_embeddings[token_he_ids[he_mask]]
    if overview_mask.any():
        num_ov = int(overview_mask.sum())
        if overview_semantics is not None and overview_semantics.numel() > 0:
            all_semantic[overview_mask] = overview_semantics[:num_ov].to(dtype)

    # --- structure features ---
    all_structure = torch.zeros(n, HYPERGRAPH_STRUCTURE_DIM, dtype=torch.float32)
    arange_n = torch.arange(n)

    # Laplacian position codes for HIDT tokens
    hidt_mask = (token_slot_ids >= 0)
    if hidt_mask.any():
        lap_codes = _cached_laplacian_position_codes(
            center_kind=hidt_center_kind,
            max_depth=hidt_max_depth,
            max_incident_hyperedges=hidt_max_incident_hyperedges,
            max_members_per_hyperedge=hidt_max_members_per_hyperedge,
            max_child_hyperedges=hidt_max_child_hyperedges,
            position_dim=FINAL_HIDT_POSITION_DIM,
        )
        slot_indices = token_slot_ids[hidt_mask].clamp(0, lap_codes.shape[0] - 1)
        all_structure[hidt_mask, :FINAL_HIDT_POSITION_DIM] = lap_codes[slot_indices]

    if overview_mask.any():
        from utils.hypergraph_overview import build_overview_position_codes
        ov_codes = build_overview_position_codes()
        num_ov = int(overview_mask.sum())
        usable = min(num_ov, ov_codes.shape[0])
        ov_indices = torch.where(overview_mask)[0]
        all_structure[ov_indices[:usable], :FINAL_HIDT_POSITION_DIM] = ov_codes[:usable]

    # type one-hot (4 bits at offset POSITION_DIM)
    all_structure[arange_n, FINAL_HIDT_POSITION_DIM + token_type_ids.clamp(0, 3)] = 1.0

    # depth one-hot (4 bits at offset POSITION_DIM + 4)
    all_structure[arange_n, FINAL_HIDT_POSITION_DIM + 4 + token_depths.clamp(0, 3)] = 1.0

    # order bucket one-hot (4 bits at offset POSITION_DIM + 8), hyperedges + overview
    order_ids = torch.full((n,), -1, dtype=torch.long)
    real_he = he_mask & (token_he_ids >= 0)
    if real_he.any():
        he_sizes = processed_data["hyperedge_size"][token_he_ids[real_he]]
        order_ids[real_he] = order_bucket_id_vec(he_sizes)
    if overview_mask.any():
        num_ov = int(overview_mask.sum())
        order_ids[overview_mask] = torch.arange(num_ov, dtype=torch.long) % 4
    order_valid = (order_ids >= 0) & (order_ids < 4)
    if order_valid.any():
        all_structure[arange_n[order_valid], FINAL_HIDT_POSITION_DIM + 8 + order_ids[order_valid]] = 1.0

    # degree bucket one-hot (4 bits at offset POSITION_DIM + 12), vertices + overview
    degree_ids = torch.full((n,), -1, dtype=torch.long)
    real_v = vertex_mask & (token_node_ids >= 0)
    if real_v.any():
        node_ptr = processed_data["node_ptr"]
        nids = token_node_ids[real_v]
        degrees = node_ptr[nids + 1] - node_ptr[nids]
        degree_ids[real_v] = degree_bucket_id_vec(degrees)
    if overview_mask.any() and overview_support_bucket_ids is not None:
        num_ov = int(overview_mask.sum())
        usable = min(num_ov, len(overview_support_bucket_ids))
        ov_idx = torch.where(overview_mask)[0]
        degree_ids[ov_idx[:usable]] = overview_support_bucket_ids[:usable]
    degree_valid = (degree_ids >= 0) & (degree_ids < 4)
    if degree_valid.any():
        all_structure[arange_n[degree_valid], FINAL_HIDT_POSITION_DIM + 12 + degree_ids[degree_valid]] = 1.0

    # --- assemble ---
    graph_emb = torch.zeros(1, max_tokens, semantic_dim + HYPERGRAPH_STRUCTURE_DIM, dtype=dtype)
    graph_emb[0, :n, :semantic_dim] = all_semantic
    graph_emb[0, :n, semantic_dim:] = all_structure.to(dtype)

    return graph, graph_emb
