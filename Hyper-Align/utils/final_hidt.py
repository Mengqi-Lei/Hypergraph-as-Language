from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import random
from typing import Dict, List, Sequence

import torch

FINAL_HIDT_POSITION_DIM = 8
FINAL_HIDT_ORDER_BUCKETS = (
    ("r2", 2, 2),
    ("r3_4", 3, 4),
    ("r5_8", 5, 8),
    ("r9p", 9, None),
)
FINAL_HIDT_DEGREE_BUCKETS = (
    ("d0", 0, 0),
    ("d1_2", 1, 2),
    ("d3_7", 3, 7),
    ("d8p", 8, None),
)
FINAL_HIDT_TYPE_LABELS = ("vertex", "hyperedge", "v_pad", "e_pad")
FINAL_HIDT_VARIANTS = {"vertex": "VC_HIDT", "hyperedge": "EC_HIDT"}
FINAL_HIDT_SERIALIZATION = "level_order"
FINAL_HIDT_CONTRACT_FIELDS = (
    "vc_hidt",
    "ec_hidt",
    "type_label",
    "order_bucket_id",
    "degree_bucket_id",
    "slot_id",
    "parent_slot_id",
    "incidence_pairs",
    "co_member_groups",
    "unrelated_sampling_pool",
)
FINAL_HIDT_SPEC_SOURCES = (
    "utils/final_hidt.py",
    "utils/hypergraph_templates.py",
    "utils/hypergraph_features.py",
    "train/hyper_dataset.py",
    "HyperAlign-Bench/hyperalign_bench/ogbn_arxiv_hg.py",
)
FINAL_HIDT_RELATION_LABELS = ("unrelated", "incidence", "co_member")
FINAL_HIDT_RELATION_TO_ID = {label: idx for idx, label in enumerate(FINAL_HIDT_RELATION_LABELS)}
FINAL_HIDT_PAD_TARGET = -100


@dataclass(frozen=True)
class FinalHIDTConfig:
    max_depth: int
    max_incident_hyperedges: int
    max_members_per_hyperedge: int
    max_child_hyperedges: int
    position_dim: int = FINAL_HIDT_POSITION_DIM


@dataclass(frozen=True)
class TemplateSlot:
    slot_id: int
    expected_kind: str
    depth: int
    parent_slot_id: int
    child_slot_ids: tuple[int, ...]
    branch_index: int
    relation_to_parent: str


class TorchFinalHIDTAccessor:
    def __init__(self, processed_data: Dict):
        self.processed_data = processed_data

    def get_incident_hyperedges(self, node_id: int, exclude_hyperedge_id: int | None = None) -> List[int]:
        node_ptr = self.processed_data["node_ptr"]
        node_hyperedge_index = self.processed_data["node_hyperedge_index"]
        start = int(node_ptr[node_id].item())
        end = int(node_ptr[node_id + 1].item())
        hyperedge_ids = [int(value) for value in node_hyperedge_index[start:end].tolist()]
        if exclude_hyperedge_id is None:
            return hyperedge_ids
        return [hyperedge_id for hyperedge_id in hyperedge_ids if hyperedge_id != int(exclude_hyperedge_id)]

    def get_hyperedge_members(self, hyperedge_id: int, exclude_node_id: int | None = None) -> List[int]:
        he_ptr = self.processed_data["he_ptr"]
        he_node_index = self.processed_data["he_node_index"]
        start = int(he_ptr[hyperedge_id].item())
        end = int(he_ptr[hyperedge_id + 1].item())
        node_ids = [int(value) for value in he_node_index[start:end].tolist()]
        if exclude_node_id is None:
            return node_ids
        return [node_id for node_id in node_ids if node_id != int(exclude_node_id)]

    def get_hyperedge_size(self, hyperedge_id: int) -> int:
        return int(self.processed_data["hyperedge_size"][hyperedge_id].item())

    def get_node_degree(self, node_id: int) -> int:
        node_ptr = self.processed_data["node_ptr"]
        return int((node_ptr[node_id + 1] - node_ptr[node_id]).item())


class NumpyFinalHIDTAccessor:
    def __init__(self, views: Dict):
        self.views = views

    def get_incident_hyperedges(self, node_id: int, exclude_hyperedge_id: int | None = None) -> List[int]:
        node_ptr = self.views["node_ptr"]
        node_hyperedge_index = self.views["node_hyperedge_index"]
        start = int(node_ptr[node_id])
        end = int(node_ptr[node_id + 1])
        hyperedge_ids = [int(value) for value in node_hyperedge_index[start:end].tolist()]
        if exclude_hyperedge_id is None:
            return hyperedge_ids
        return [hyperedge_id for hyperedge_id in hyperedge_ids if hyperedge_id != int(exclude_hyperedge_id)]

    def get_hyperedge_members(self, hyperedge_id: int, exclude_node_id: int | None = None) -> List[int]:
        he_ptr = self.views["he_ptr"]
        he_node_index = self.views["he_node_index"]
        start = int(he_ptr[hyperedge_id])
        end = int(he_ptr[hyperedge_id + 1])
        node_ids = [int(value) for value in he_node_index[start:end].tolist()]
        if exclude_node_id is None:
            return node_ids
        return [node_id for node_id in node_ids if node_id != int(exclude_node_id)]

    def get_hyperedge_size(self, hyperedge_id: int) -> int:
        return int(self.views["hyperedge_size"][hyperedge_id])

    def get_node_degree(self, node_id: int) -> int:
        node_ptr = self.views["node_ptr"]
        return int(node_ptr[node_id + 1] - node_ptr[node_id])


def order_bucket_id(size: int) -> int:
    for bucket_id, (_, lower, upper) in enumerate(FINAL_HIDT_ORDER_BUCKETS):
        if size >= lower and (upper is None or size <= upper):
            return bucket_id
    return len(FINAL_HIDT_ORDER_BUCKETS) - 1


def degree_bucket_id(degree: int) -> int:
    for bucket_id, (_, lower, upper) in enumerate(FINAL_HIDT_DEGREE_BUCKETS):
        if degree >= lower and (upper is None or degree <= upper):
            return bucket_id
    return len(FINAL_HIDT_DEGREE_BUCKETS) - 1


def _fix_eigenvector_sign(eigenvectors: torch.Tensor) -> torch.Tensor:
    """Fix eigenvector signs deterministically."""
    fixed = eigenvectors.clone()
    for column_idx in range(fixed.shape[1]):
        column = fixed[:, column_idx]
        nonzero = torch.where(column.abs() > 1e-8)[0]
        if len(nonzero) == 0:
            continue
        first_index = int(nonzero[0].item())
        if column[first_index] < 0:
            fixed[:, column_idx] = -column
    return fixed


@lru_cache(maxsize=None)
def build_final_hidt_template(
    center_kind: str,
    max_depth: int,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
    position_dim: int = FINAL_HIDT_POSITION_DIM,
) -> Dict:
    """Build the fixed HIDT slot template."""
    normalized_center_kind = center_kind.lower()
    if normalized_center_kind not in FINAL_HIDT_VARIANTS:
        raise ValueError(f"Unsupported final HIDT center kind: {center_kind}")

    slots: List[TemplateSlot] = []
    frontier = [0]
    slots.append(
        TemplateSlot(
            slot_id=0,
            expected_kind=normalized_center_kind,
            depth=0,
            parent_slot_id=-1,
            child_slot_ids=tuple(),
            branch_index=0,
            relation_to_parent="root",
        )
    )
    current_index = 0
    while current_index < len(frontier):
        slot_id = frontier[current_index]
        slot = slots[slot_id]
        current_index += 1
        if slot.depth >= max_depth:
            continue
        if slot.expected_kind == "vertex":
            child_kind = "hyperedge"
            branch_factor = max_incident_hyperedges if slot.depth == 0 and normalized_center_kind == "vertex" else max_child_hyperedges
            relation = "incidence"
        else:
            child_kind = "vertex"
            branch_factor = max_members_per_hyperedge
            relation = "member"
        child_slot_ids: List[int] = []
        for branch_index in range(branch_factor):
            child_slot_id = len(slots)
            slots.append(
                TemplateSlot(
                    slot_id=child_slot_id,
                    expected_kind=child_kind,
                    depth=slot.depth + 1,
                    parent_slot_id=slot.slot_id,
                    child_slot_ids=tuple(),
                    branch_index=branch_index,
                    relation_to_parent=relation,
                )
            )
            frontier.append(child_slot_id)
            child_slot_ids.append(child_slot_id)
        slots[slot.slot_id] = TemplateSlot(
            slot_id=slot.slot_id,
            expected_kind=slot.expected_kind,
            depth=slot.depth,
            parent_slot_id=slot.parent_slot_id,
            child_slot_ids=tuple(child_slot_ids),
            branch_index=slot.branch_index,
            relation_to_parent=slot.relation_to_parent,
        )

    # laplacian position codes
    size = len(slots)
    adjacency = torch.zeros((size, size), dtype=torch.float32)
    for slot in slots:
        for child_slot_id in slot.child_slot_ids:
            adjacency[slot.slot_id, child_slot_id] = 1.0
            adjacency[child_slot_id, slot.slot_id] = 1.0
    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask].pow(-0.5)
    laplacian = torch.eye(size, dtype=torch.float32) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    usable_dims = min(position_dim, max(size - 1, 0))
    if usable_dims > 0:
        position_codes = _fix_eigenvector_sign(eigenvectors[:, 1 : 1 + usable_dims])
    else:
        position_codes = torch.zeros((size, 0), dtype=torch.float32)
    if usable_dims < position_dim:
        padding = torch.zeros((size, position_dim - usable_dims), dtype=torch.float32)
        position_codes = torch.cat([position_codes, padding], dim=1)
    return {
        "center_kind": normalized_center_kind,
        "variant": FINAL_HIDT_VARIANTS[normalized_center_kind],
        "serialization": FINAL_HIDT_SERIALIZATION,
        "slots": slots,
        "laplacian_position_codes": position_codes,
    }


def estimate_final_hidt_token_count(
    center_kind: str,
    max_depth: int,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
) -> int:
    template = build_final_hidt_template(
        center_kind=center_kind,
        max_depth=max_depth,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
    )
    return len(template["slots"])


def _type_label_from_assignment(expected_kind: str, is_pad: bool) -> str:
    if expected_kind == "vertex":
        return "v_pad" if is_pad else "vertex"
    return "e_pad" if is_pad else "hyperedge"


def _build_token_from_assignment(
    slot: TemplateSlot,
    expected_kind: str,
    is_pad: bool,
    center_kind: str,
    entity_id: int | None,
    parent_entity_id: int | None,
    relation_to_parent: str,
    accessor,
) -> Dict:
    type_label = _type_label_from_assignment(expected_kind, is_pad)
    token = {
        "slot_id": int(slot.slot_id),
        "type_label": type_label,
        "type_id": FINAL_HIDT_TYPE_LABELS.index(type_label),
        "depth": int(slot.depth),
        "depth_label": f"depth_{slot.depth}",
        "slot_index": int(slot.branch_index),
        "parent_slot_id": int(slot.parent_slot_id),
        "center_kind": FINAL_HIDT_VARIANTS[center_kind],
        "serialization": FINAL_HIDT_SERIALIZATION,
        "relation_to_parent": relation_to_parent,
        "is_pad": int(is_pad),
        "is_root": int(slot.parent_slot_id < 0),
        "is_target": int(slot.depth == 0),
    }
    if expected_kind == "vertex":
        token["kind"] = type_label
        token["node_id"] = -1 if entity_id is None else int(entity_id)
        token["parent_hyperedge_id"] = -1 if parent_entity_id is None else int(parent_entity_id)
        token["hyperedge_id"] = -1
        if is_pad:
            token["degree_bucket_id"] = -1
            token["degree_bucket_label"] = "none"
            token["order_bucket_id"] = -1
            token["order_bucket_label"] = "none"
        else:
            node_degree = accessor.get_node_degree(int(entity_id))
            token["degree_bucket_id"] = degree_bucket_id(node_degree)
            token["degree_bucket_label"] = FINAL_HIDT_DEGREE_BUCKETS[token["degree_bucket_id"]][0]
            token["order_bucket_id"] = -1
            token["order_bucket_label"] = "none"
        token["co_member_group_id"] = int(slot.parent_slot_id) if slot.parent_slot_id >= 0 else -1
    else:
        token["kind"] = type_label
        token["hyperedge_id"] = -1 if entity_id is None else int(entity_id)
        token["parent_node_id"] = -1 if parent_entity_id is None else int(parent_entity_id)
        token["node_id"] = -1
        token["co_member_group_id"] = -1
        if is_pad:
            token["order_bucket_id"] = -1
            token["order_bucket_label"] = "none"
            token["degree_bucket_id"] = -1
            token["degree_bucket_label"] = "none"
        else:
            size = accessor.get_hyperedge_size(int(entity_id))
            token["order_bucket_id"] = order_bucket_id(size)
            token["order_bucket_label"] = FINAL_HIDT_ORDER_BUCKETS[token["order_bucket_id"]][0]
            token["degree_bucket_id"] = -1
            token["degree_bucket_label"] = "none"
    return token


def _derive_relation_metadata(tokens: Sequence[Dict]) -> Dict:
    valid_slot_ids = [int(token["slot_id"]) for token in tokens if not token["is_pad"]]
    incidence_pairs: List[List[int]] = []
    co_member_groups: List[List[int]] = []
    grouped_children: Dict[int, List[int]] = {}

    for token in tokens:
        if token["is_pad"]:
            continue
        parent_slot_id = int(token["parent_slot_id"])
        if parent_slot_id >= 0:
            incidence_pairs.append([parent_slot_id, int(token["slot_id"])])
        group_id = int(token.get("co_member_group_id", -1))
        if group_id >= 0 and token["type_label"] == "vertex":
            grouped_children.setdefault(group_id, []).append(int(token["slot_id"]))

    for slot_ids in grouped_children.values():
        if len(slot_ids) >= 2:
            co_member_groups.append(slot_ids)

    return {
        "valid_slot_ids": valid_slot_ids,
        "unrelated_sampling_pool": valid_slot_ids,
        "incidence_pairs": incidence_pairs,
        "co_member_groups": co_member_groups,
    }


derive_relation_metadata = _derive_relation_metadata


def derive_relation_metadata_from_arrays(
    token_type_ids: torch.LongTensor,
    token_slot_ids: torch.LongTensor,
    token_parent_slot_ids: torch.LongTensor,
    is_pad: torch.BoolTensor,
) -> Dict:
    """Equivalent to derive_relation_metadata(tokens) but operates on parallel arrays."""
    n = len(token_type_ids)
    valid_slot_ids = token_slot_ids[~is_pad].tolist()
    incidence_pairs: List[List[int]] = []
    grouped_children: Dict[int, List[int]] = {}

    for i in range(n):
        if is_pad[i]:
            continue
        psid = int(token_parent_slot_ids[i])
        sid = int(token_slot_ids[i])
        if psid >= 0:
            incidence_pairs.append([psid, sid])
        if int(token_type_ids[i]) == 0 and psid >= 0:
            grouped_children.setdefault(psid, []).append(sid)

    co_member_groups = [g for g in grouped_children.values() if len(g) >= 2]
    return {
        "valid_slot_ids": valid_slot_ids,
        "unrelated_sampling_pool": valid_slot_ids,
        "incidence_pairs": incidence_pairs,
        "co_member_groups": co_member_groups,
    }


def build_final_hidt_instance(
    center_kind: str,
    center_id: int,
    accessor,
    config: FinalHIDTConfig,
    forced_member_node_ids: Sequence[int] | None = None,
) -> Dict:
    """Build a final HIDT instance for a center vertex or hyperedge."""
    normalized_center_kind = center_kind.lower()
    template = build_final_hidt_template(
        center_kind=normalized_center_kind,
        max_depth=config.max_depth,
        max_incident_hyperedges=config.max_incident_hyperedges,
        max_members_per_hyperedge=config.max_members_per_hyperedge,
        max_child_hyperedges=config.max_child_hyperedges,
        position_dim=config.position_dim,
    )
    slots: Sequence[TemplateSlot] = template["slots"]
    assignments: Dict[int, Dict] = {}
    tokens: List[Dict] = [None] * len(slots)  # type: ignore[list-item]

    root_slot = slots[0]
    if (
        forced_member_node_ids is not None
        and normalized_center_kind == "hyperedge"
    ):
        root_forced_members: List[int] | None = [
            int(node_id) for node_id in forced_member_node_ids
        ]
    else:
        root_forced_members = None
    assignments[root_slot.slot_id] = {
        "entity_id": int(center_id),
        "is_pad": False,
        "parent_entity_id": None,
        "relation_to_parent": "root",
        "forced_member_node_ids": root_forced_members,
    }

    for slot in slots:
        assignment = assignments[slot.slot_id]
        expected_kind = slot.expected_kind
        token = _build_token_from_assignment(
            slot=slot,
            expected_kind=expected_kind,
            is_pad=bool(assignment["is_pad"]),
            center_kind=normalized_center_kind,
            entity_id=assignment["entity_id"],
            parent_entity_id=assignment["parent_entity_id"],
            relation_to_parent=assignment["relation_to_parent"],
            accessor=accessor,
        )
        token["hidt_max_depth"] = int(config.max_depth)
        token["hidt_max_incident_hyperedges"] = int(config.max_incident_hyperedges)
        token["hidt_max_members_per_hyperedge"] = int(config.max_members_per_hyperedge)
        token["hidt_max_child_hyperedges"] = int(config.max_child_hyperedges)
        token["hidt_position_dim"] = int(config.position_dim)
        tokens[slot.slot_id] = token

        if slot.depth >= config.max_depth:
            continue

        child_slots = [slots[child_slot_id] for child_slot_id in slot.child_slot_ids]
        if expected_kind == "vertex":
            if assignment["is_pad"]:
                child_entities: List[int] = []
            else:
                child_entities = accessor.get_incident_hyperedges(
                    int(assignment["entity_id"]),
                    exclude_hyperedge_id=assignment["parent_entity_id"],
                )[: len(child_slots)]
            for index, child_slot in enumerate(child_slots):
                if index < len(child_entities):
                    assignments[child_slot.slot_id] = {
                        "entity_id": int(child_entities[index]),
                        "is_pad": False,
                        "parent_entity_id": int(assignment["entity_id"]),
                        "relation_to_parent": "incidence",
                        "forced_member_node_ids": None,
                    }
                else:
                    assignments[child_slot.slot_id] = {
                        "entity_id": None,
                        "is_pad": True,
                        "parent_entity_id": None,
                        "relation_to_parent": "incidence",
                        "forced_member_node_ids": None,
                    }
            continue

        if assignment["is_pad"]:
            child_entities = []
        elif assignment.get("forced_member_node_ids") is not None:
            child_entities = [int(node_id) for node_id in assignment["forced_member_node_ids"][: len(child_slots)]]
        else:
            child_entities = accessor.get_hyperedge_members(
                int(assignment["entity_id"]),
                exclude_node_id=assignment["parent_entity_id"],
            )[: len(child_slots)]
        for index, child_slot in enumerate(child_slots):
            if index < len(child_entities):
                assignments[child_slot.slot_id] = {
                    "entity_id": int(child_entities[index]),
                    "is_pad": False,
                    "parent_entity_id": int(assignment["entity_id"]),
                    "relation_to_parent": "member",
                    "forced_member_node_ids": None,
                }
            else:
                assignments[child_slot.slot_id] = {
                    "entity_id": None,
                    "is_pad": True,
                    "parent_entity_id": None,
                    "relation_to_parent": "member",
                    "forced_member_node_ids": None,
                }

    return {
        "variant": FINAL_HIDT_VARIANTS[normalized_center_kind],
        "serialization": FINAL_HIDT_SERIALIZATION,
        "config": {
            "max_depth": int(config.max_depth),
            "max_incident_hyperedges": int(config.max_incident_hyperedges),
            "max_members_per_hyperedge": int(config.max_members_per_hyperedge),
            "max_child_hyperedges": int(config.max_child_hyperedges),
            "position_dim": int(config.position_dim),
        },
        "tokens": tokens,
        "token_type_ids": [int(token["type_id"]) for token in tokens],
        "token_order_bucket_ids": [int(token["order_bucket_id"]) for token in tokens],
        "token_degree_bucket_ids": [int(token["degree_bucket_id"]) for token in tokens],
        "relation_metadata": _derive_relation_metadata(tokens),
    }


def resolve_laplacian_position_codes(tokens: Sequence[Dict]) -> torch.Tensor:
    if not tokens:
        return torch.zeros((0, FINAL_HIDT_POSITION_DIM), dtype=torch.float32)
    first = tokens[0]
    variant = first["center_kind"]
    center_kind = "vertex" if variant == "VC_HIDT" else "hyperedge"
    template = build_final_hidt_template(
        center_kind=center_kind,
        max_depth=int(first["hidt_max_depth"]),
        max_incident_hyperedges=int(first["hidt_max_incident_hyperedges"]),
        max_members_per_hyperedge=int(first["hidt_max_members_per_hyperedge"]),
        max_child_hyperedges=int(first["hidt_max_child_hyperedges"]),
        position_dim=int(first["hidt_position_dim"]),
    )
    return template["laplacian_position_codes"]


def structure_labels_from_token(token: Dict) -> Dict:
    return {
        "type_id": int(token["type_id"]),
        "depth": int(token["depth"]),
        "order_bucket_id": int(token["order_bucket_id"]),
        "degree_bucket_id": int(token["degree_bucket_id"]),
    }


def coarse_type_group_id(type_label: str) -> int:
    if type_label == "vertex":
        return 0
    if type_label == "hyperedge":
        return 1
    return 2


FINAL_HIDT_PROJECTOR_ROLE_VERTEX = 0
FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE = 1
FINAL_HIDT_PROJECTOR_ROLE_OVERVIEW = 2
FINAL_HIDT_PROJECTOR_ROLE_PAD = 3


def _hyperedge_hop_from_hidt_depth(depth: int, root_role_id: int) -> int:
    depth = int(depth)
    if root_role_id == FINAL_HIDT_PROJECTOR_ROLE_VERTEX:
        if depth <= 0 or depth % 2 == 0:
            return -1
        return (depth + 1) // 2
    if depth <= 0 or depth % 2 != 0:
        return -1
    return depth // 2


def _build_hidt_tree_incidence_mask(
    role_ids: torch.Tensor,
    token_slot_ids: torch.Tensor,
    token_parent_slot_ids: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    projector_incidence_mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool)
    hidt_valid_positions = torch.where(
        (token_slot_ids >= 0) & (role_ids != FINAL_HIDT_PROJECTOR_ROLE_PAD)
    )[0]
    slot_to_position = {
        int(token_slot_ids[pos]): int(pos)
        for pos in hidt_valid_positions.tolist()
    }
    for position in hidt_valid_positions.tolist():
        parent_slot_id = int(token_parent_slot_ids[position].item())
        if parent_slot_id < 0:
            continue
        parent_position = slot_to_position.get(parent_slot_id)
        if parent_position is None:
            continue
        if role_ids[parent_position] == role_ids[position]:
            continue
        projector_incidence_mask[position, parent_position] = True
        projector_incidence_mask[parent_position, position] = True
    return projector_incidence_mask


def _build_sample_real_incidence_mask(
    role_ids: torch.Tensor,
    token_node_ids: torch.Tensor,
    token_he_ids: torch.Tensor,
    processed_data: Dict,
    max_tokens: int,
) -> torch.Tensor:
    projector_incidence_mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool)
    vertex_positions = torch.where(role_ids == FINAL_HIDT_PROJECTOR_ROLE_VERTEX)[0]
    hyperedge_positions = torch.where(role_ids == FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE)[0]
    if vertex_positions.numel() == 0 or hyperedge_positions.numel() == 0:
        return projector_incidence_mask

    node_ptr = processed_data["node_ptr"]
    node_hyperedge_index = processed_data["node_hyperedge_index"]
    hyperedge_id_to_positions: Dict[int, list[int]] = {}
    for position in hyperedge_positions.tolist():
        hyperedge_id = int(token_he_ids[position].item())
        if hyperedge_id < 0:
            continue
        hyperedge_id_to_positions.setdefault(hyperedge_id, []).append(int(position))

    for vertex_position in vertex_positions.tolist():
        node_id = int(token_node_ids[vertex_position].item())
        if node_id < 0:
            continue
        start = int(node_ptr[node_id].item())
        end = int(node_ptr[node_id + 1].item())
        for hyperedge_id in node_hyperedge_index[start:end].tolist():
            for hyperedge_position in hyperedge_id_to_positions.get(int(hyperedge_id), []):
                projector_incidence_mask[vertex_position, hyperedge_position] = True
                projector_incidence_mask[hyperedge_position, vertex_position] = True
    return projector_incidence_mask


def build_projector_metadata_from_arrays(
    token_type_ids: torch.LongTensor,
    token_depths: torch.LongTensor,
    token_slot_ids: torch.LongTensor,
    token_parent_slot_ids: torch.LongTensor,
    token_node_ids: torch.LongTensor,
    token_he_ids: torch.LongTensor,
    processed_data: Dict,
    max_tokens: int,
    incidence_mode: str = "sample_real",
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Build fixed-shape projector metadata from existing HIDT+O parallel arrays.

    The projector message passing uses vertex-hyperedge bidirectional
    incidence (Step A/B) built from either HIDT tree edges or real
    sample-internal incidence.
    """

    n = min(len(token_type_ids), max_tokens)

    projector_role_ids = torch.full(
        (max_tokens,),
        FINAL_HIDT_PROJECTOR_ROLE_PAD,
        dtype=torch.long,
    )
    projector_valid_mask = torch.zeros((max_tokens,), dtype=torch.bool)
    projector_incidence_mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool)

    if n <= 0:
        return {
            "projector_role_ids": projector_role_ids,
            "projector_valid_mask": projector_valid_mask,
            "projector_incidence_mask": projector_incidence_mask,
        }

    local_type_ids = token_type_ids[:n]
    local_slot_ids = token_slot_ids[:n]
    local_parent_slot_ids = token_parent_slot_ids[:n]
    local_node_ids = token_node_ids[:n]
    local_he_ids = token_he_ids[:n]

    local_is_pad = local_type_ids >= 2
    local_is_overview = (local_slot_ids < 0) & ~local_is_pad
    local_is_vertex = (local_type_ids == 0) & ~local_is_pad & ~local_is_overview
    local_is_hyperedge = (local_type_ids == 1) & ~local_is_pad & ~local_is_overview

    local_role_ids = torch.full((n,), FINAL_HIDT_PROJECTOR_ROLE_PAD, dtype=torch.long)
    local_role_ids[local_is_vertex] = FINAL_HIDT_PROJECTOR_ROLE_VERTEX
    local_role_ids[local_is_hyperedge] = FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE
    local_role_ids[local_is_overview] = FINAL_HIDT_PROJECTOR_ROLE_OVERVIEW

    projector_role_ids[:n] = local_role_ids
    projector_valid_mask[:n] = local_role_ids != FINAL_HIDT_PROJECTOR_ROLE_PAD

    resolved_incidence_mode = str(incidence_mode).strip().lower()
    if resolved_incidence_mode == "hidt_tree":
        projector_incidence_mask = _build_hidt_tree_incidence_mask(
            role_ids=local_role_ids,
            token_slot_ids=local_slot_ids,
            token_parent_slot_ids=local_parent_slot_ids,
            max_tokens=max_tokens,
        )
    elif resolved_incidence_mode == "sample_real":
        projector_incidence_mask = _build_sample_real_incidence_mask(
            role_ids=local_role_ids,
            token_node_ids=local_node_ids,
            token_he_ids=local_he_ids,
            processed_data=processed_data,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported projector incidence mode: {incidence_mode}. "
            "Expected one of: hidt_tree, sample_real."
        )

    return {
        "projector_role_ids": projector_role_ids,
        "projector_valid_mask": projector_valid_mask,
        "projector_incidence_mask": projector_incidence_mask,
    }


def build_final_hidt_contract_summary() -> Dict:
    return {
        "contract_fields": list(FINAL_HIDT_CONTRACT_FIELDS),
        "spec_sources": list(FINAL_HIDT_SPEC_SOURCES),
        "structure_dim": FINAL_HIDT_POSITION_DIM + 4 + 4 + 4 + 4,
        "position_dim": FINAL_HIDT_POSITION_DIM,
        "serialization": FINAL_HIDT_SERIALIZATION,
        "type_labels": list(FINAL_HIDT_TYPE_LABELS),
        "order_bucket_labels": [bucket[0] for bucket in FINAL_HIDT_ORDER_BUCKETS],
        "degree_bucket_labels": [bucket[0] for bucket in FINAL_HIDT_DEGREE_BUCKETS],
        "relation_labels": list(FINAL_HIDT_RELATION_LABELS),
    }


def _sample_limited_pairs(pairs: Sequence[tuple[int, int]], sample_size: int, rng: random.Random) -> List[tuple[int, int]]:
    if sample_size <= 0 or not pairs:
        return []
    pairs = list(pairs)
    rng.shuffle(pairs)
    return list(pairs[:sample_size])


def sample_relation_pairs(
    relation_metadata: Dict,
    seed: int,
    max_incidence_pairs: int,
    max_co_member_pairs: int,
    max_unrelated_pairs: int,
) -> tuple[List[List[int]], List[int]]:
    rng = random.Random(seed)
    incidence_pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in relation_metadata.get("incidence_pairs", [])
    }
    co_member_pairs = set()
    for group in relation_metadata.get("co_member_groups", []):
        group = sorted(int(slot_id) for slot_id in group)
        for left, right in itertools.combinations(group, 2):
            co_member_pairs.add((left, right))

    selected_pairs: List[List[int]] = []
    selected_labels: List[int] = []

    for left, right in _sample_limited_pairs(sorted(incidence_pairs), max_incidence_pairs, rng):
        selected_pairs.append([left, right])
        selected_labels.append(FINAL_HIDT_RELATION_TO_ID["incidence"])
    for left, right in _sample_limited_pairs(sorted(co_member_pairs), max_co_member_pairs, rng):
        selected_pairs.append([left, right])
        selected_labels.append(FINAL_HIDT_RELATION_TO_ID["co_member"])

    valid_slot_ids = sorted(int(slot_id) for slot_id in relation_metadata.get("unrelated_sampling_pool", []))
    unrelated_candidates = []
    positive_pairs = incidence_pairs | co_member_pairs
    for left, right in itertools.combinations(valid_slot_ids, 2):
        pair = (left, right)
        if pair in positive_pairs:
            continue
        unrelated_candidates.append(pair)
    for left, right in _sample_limited_pairs(unrelated_candidates, max_unrelated_pairs, rng):
        selected_pairs.append([left, right])
        selected_labels.append(FINAL_HIDT_RELATION_TO_ID["unrelated"])
    return selected_pairs, selected_labels


def build_final_hidt_training_targets(
    payload: Dict,
    max_tokens: int,
    seed: int,
    max_incidence_pairs: int,
    max_co_member_pairs: int,
    max_unrelated_pairs: int,
) -> Dict[str, torch.Tensor]:
    """Build auxiliary order and relation prediction targets."""
    tokens = payload["tokens"]
    type_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    type_group_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    order_bucket_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    degree_bucket_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    slot_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    parent_slot_ids = torch.full((max_tokens,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)

    for position, token in enumerate(tokens[:max_tokens]):
        type_ids[position] = int(token["type_id"])
        type_group_ids[position] = coarse_type_group_id(token["type_label"])
        order_bucket_ids[position] = int(token["order_bucket_id"])
        degree_bucket_ids[position] = int(token["degree_bucket_id"])
        slot_ids[position] = int(token["slot_id"])
        parent_slot_ids[position] = int(token["parent_slot_id"])

    pair_indices, pair_labels = sample_relation_pairs(
        relation_metadata=payload["relation_metadata"],
        seed=seed,
        max_incidence_pairs=max_incidence_pairs,
        max_co_member_pairs=max_co_member_pairs,
        max_unrelated_pairs=max_unrelated_pairs,
    )
    max_pairs = max_incidence_pairs + max_co_member_pairs + max_unrelated_pairs
    relation_pair_indices = torch.full((max_pairs, 2), -1, dtype=torch.long)
    relation_pair_labels = torch.full((max_pairs,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    for index, (pair, label) in enumerate(zip(pair_indices[:max_pairs], pair_labels[:max_pairs])):
        relation_pair_indices[index] = torch.tensor(pair, dtype=torch.long)
        relation_pair_labels[index] = int(label)

    return {
        "type_ids": type_ids,
        "type_group_ids": type_group_ids,
        "order_bucket_ids": order_bucket_ids,
        "degree_bucket_ids": degree_bucket_ids,
        "slot_ids": slot_ids,
        "parent_slot_ids": parent_slot_ids,
        "relation_pair_indices": relation_pair_indices,
        "relation_pair_labels": relation_pair_labels,
    }


def build_training_targets_from_arrays(
    token_type_ids: torch.LongTensor,
    token_slot_ids: torch.LongTensor,
    token_parent_slot_ids: torch.LongTensor,
    token_node_ids: torch.LongTensor,
    token_he_ids: torch.LongTensor,
    processed_data: Dict,
    max_tokens: int,
    seed: int,
    max_incidence_pairs: int,
    max_co_member_pairs: int,
    max_unrelated_pairs: int,
) -> Dict[str, torch.Tensor]:
    """Build auxiliary training targets directly from prebaked parallel arrays."""
    from utils.hypergraph_features import order_bucket_id_vec, degree_bucket_id_vec

    n = min(len(token_type_ids), max_tokens)

    def _pad(src: torch.Tensor, pad_val: int = FINAL_HIDT_PAD_TARGET) -> torch.Tensor:
        out = torch.full((max_tokens,), pad_val, dtype=torch.long)
        out[:n] = src[:n]
        return out

    is_pad = token_type_ids[:n] >= 2

    type_ids_out = _pad(token_type_ids)
    slot_ids_out = _pad(token_slot_ids)
    parent_slot_ids_out = _pad(token_parent_slot_ids)

    tg_raw = torch.where(token_type_ids[:n] <= 1, token_type_ids[:n], torch.tensor(2, dtype=torch.long))
    type_group_ids_out = _pad(tg_raw)

    order_raw = torch.full((n,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    he_mask = (token_he_ids[:n] >= 0) & ~is_pad
    if he_mask.any():
        sizes = processed_data["hyperedge_size"][token_he_ids[:n][he_mask]]
        order_raw[he_mask] = order_bucket_id_vec(sizes)
    order_bucket_ids_out = _pad(order_raw)

    degree_raw = torch.full((n,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    v_mask = (token_node_ids[:n] >= 0) & ~is_pad
    if v_mask.any():
        node_ptr = processed_data["node_ptr"]
        nids = token_node_ids[:n][v_mask]
        degrees = node_ptr[nids + 1] - node_ptr[nids]
        degree_raw[v_mask] = degree_bucket_id_vec(degrees)
    degree_bucket_ids_out = _pad(degree_raw)

    relation_metadata = derive_relation_metadata_from_arrays(
        token_type_ids[:n], token_slot_ids[:n], token_parent_slot_ids[:n], is_pad,
    )

    pair_indices, pair_labels = sample_relation_pairs(
        relation_metadata=relation_metadata,
        seed=seed,
        max_incidence_pairs=max_incidence_pairs,
        max_co_member_pairs=max_co_member_pairs,
        max_unrelated_pairs=max_unrelated_pairs,
    )
    max_pairs = max_incidence_pairs + max_co_member_pairs + max_unrelated_pairs
    relation_pair_indices = torch.full((max_pairs, 2), -1, dtype=torch.long)
    relation_pair_labels = torch.full((max_pairs,), FINAL_HIDT_PAD_TARGET, dtype=torch.long)
    for i, (pair, label) in enumerate(zip(pair_indices[:max_pairs], pair_labels[:max_pairs])):
        relation_pair_indices[i] = torch.tensor(pair, dtype=torch.long)
        relation_pair_labels[i] = int(label)

    return {
        "type_ids": type_ids_out,
        "type_group_ids": type_group_ids_out,
        "order_bucket_ids": order_bucket_ids_out,
        "degree_bucket_ids": degree_bucket_ids_out,
        "slot_ids": slot_ids_out,
        "parent_slot_ids": parent_slot_ids_out,
        "relation_pair_indices": relation_pair_indices,
        "relation_pair_labels": relation_pair_labels,
    }
