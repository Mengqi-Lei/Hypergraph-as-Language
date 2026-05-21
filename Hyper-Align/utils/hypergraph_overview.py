from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

import torch

from utils.final_hidt import FINAL_HIDT_ORDER_BUCKETS, FINAL_HIDT_POSITION_DIM, _fix_eigenvector_sign, order_bucket_id

HIDT_O_TEMPLATE_NAMES = {"HIDT_O", "HIDTO", "HIDT+O", "HIDT_O_V1"}
OVERVIEW_VERSION = "HIDT_O_v1"
OVERVIEW_TASK_OPEN = "open"
OVERVIEW_CENTER_VERTEX = "vertex"
OVERVIEW_CENTER_HYPEREDGE = "hyperedge"
OVERVIEW_HOPS = 2
OVERVIEW_ORDER_BUCKET_COUNT = len(FINAL_HIDT_ORDER_BUCKETS)
OVERVIEW_SUFFIX_LENGTH = OVERVIEW_HOPS * OVERVIEW_ORDER_BUCKET_COUNT
OVERVIEW_SUPPORT_BUCKETS = (
    ("s0", 0, 0),
    ("s1_2", 1, 2),
    ("s3_7", 3, 7),
    ("s8p", 8, None),
)


@dataclass(frozen=True)
class OverviewConfig:
    hops: int = OVERVIEW_HOPS
    order_bucket_count: int = OVERVIEW_ORDER_BUCKET_COUNT
    position_dim: int = FINAL_HIDT_POSITION_DIM


@dataclass(frozen=True)
class OverviewTaskSpec:
    task_view: str
    center_kind: str
    center_id: int


def is_hidto_template(template: str | None) -> bool:
    if template is None:
        return False
    return template.strip().upper() in HIDT_O_TEMPLATE_NAMES


def support_bucket_id(count: int) -> int:
    for bucket_id, (_, lower, upper) in enumerate(OVERVIEW_SUPPORT_BUCKETS):
        if count >= lower and (upper is None or count <= upper):
            return bucket_id
    return len(OVERVIEW_SUPPORT_BUCKETS) - 1


def support_bucket_label(bucket_id: int) -> str:
    return OVERVIEW_SUPPORT_BUCKETS[int(bucket_id)][0]


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _slice_as_int_list(values: Any, start: int, end: int) -> List[int]:
    sliced = values[start:end]
    if isinstance(sliced, torch.Tensor):
        return [int(v) for v in sliced.tolist()]
    return [int(v) for v in list(sliced)]


def _get_num_hyperedges(data_view: Dict[str, Any]) -> int:
    return len(data_view["hyperedge_source"])


def _get_real_incident_hyperedges(data_view: Dict[str, Any], node_id: int) -> List[int]:
    node_ptr = data_view["node_ptr"]
    node_hyperedge_index = data_view["node_hyperedge_index"]
    start = _to_int(node_ptr[node_id])
    end = _to_int(node_ptr[node_id + 1])
    return _slice_as_int_list(node_hyperedge_index, start, end)


def _get_real_hyperedge_members(data_view: Dict[str, Any], hyperedge_id: int) -> List[int]:
    he_ptr = data_view["he_ptr"]
    he_node_index = data_view["he_node_index"]
    start = _to_int(he_ptr[hyperedge_id])
    end = _to_int(he_ptr[hyperedge_id + 1])
    return _slice_as_int_list(he_node_index, start, end)


class TaskConditionedOverviewGraph:
    def __init__(self, data_view: Dict[str, Any], task_spec: OverviewTaskSpec) -> None:
        self.data_view = data_view
        self.task_spec = task_spec

    def get_incident_hyperedges(self, node_id: int) -> List[int]:
        return _get_real_incident_hyperedges(self.data_view, int(node_id))

    def get_hyperedge_members(self, hyperedge_id: int) -> List[int]:
        return _get_real_hyperedge_members(self.data_view, int(hyperedge_id))

    def get_hyperedge_size(self, hyperedge_id: int) -> int:
        return len(self.get_hyperedge_members(hyperedge_id))


def _neighbor_objects(graph: TaskConditionedOverviewGraph, kind: str, object_id: int) -> Iterable[tuple[str, int]]:
    if kind == "vertex":
        for hyperedge_id in graph.get_incident_hyperedges(int(object_id)):
            yield ("hyperedge", int(hyperedge_id))
    else:
        for node_id in graph.get_hyperedge_members(int(object_id)):
            yield ("vertex", int(node_id))


def build_hyperedge_shells(task_graph: TaskConditionedOverviewGraph, config: OverviewConfig) -> Dict[int, List[int]]:
    if task_graph.task_spec.center_kind == OVERVIEW_CENTER_VERTEX:
        start_kind = "vertex"
    else:
        start_kind = "hyperedge"
    start = (start_kind, int(task_graph.task_spec.center_id))

    max_distance = 2 * int(config.hops)
    if start_kind == "vertex":
        max_distance -= 1

    distances: Dict[tuple[str, int], int] = {start: 0}
    queue = deque([start])
    while queue:
        kind, object_id = queue.popleft()
        current_distance = distances[(kind, object_id)]
        if current_distance >= max_distance:
            continue
        for neighbor in _neighbor_objects(task_graph, kind, object_id):
            if neighbor in distances:
                continue
            distances[neighbor] = current_distance + 1
            queue.append(neighbor)

    shells: Dict[int, List[int]] = {hop_id: [] for hop_id in range(1, int(config.hops) + 1)}
    for (kind, object_id), distance in distances.items():
        if kind != "hyperedge":
            continue
        if start_kind == "vertex":
            if distance <= 0 or distance % 2 == 0:
                continue
            hop_id = (distance + 1) // 2
        else:
            if distance <= 0 or distance % 2 != 0:
                continue
            hop_id = distance // 2
        if 1 <= hop_id <= int(config.hops):
            shells[hop_id].append(int(object_id))

    def hyperedge_sort_key(hyperedge_id: int) -> tuple[int, int]:
        return (0, int(hyperedge_id))

    for hop_id in shells:
        shells[hop_id] = sorted(shells[hop_id], key=hyperedge_sort_key)
    return shells


def _order_bucket_embedding(bucket_id: int, semantic_dim: int, dtype: torch.dtype) -> torch.Tensor:
    embedding = torch.zeros(semantic_dim, dtype=dtype)
    if semantic_dim <= 0:
        return embedding
    anchor = min(int(bucket_id), semantic_dim - 1)
    embedding[anchor] = 1.0
    if semantic_dim > 4:
        tail_anchor = semantic_dim - 1 - min(int(bucket_id), semantic_dim - 1)
        embedding[tail_anchor] = -0.5
    return embedding


class ParameterFreeOverviewAggregator:
    def __init__(
        self,
        task_graph: TaskConditionedOverviewGraph,
        node_embeddings: torch.Tensor,
        hyperedge_embeddings: torch.Tensor | None,
        config: OverviewConfig,
    ) -> None:
        self.task_graph = task_graph
        self.node_embeddings = node_embeddings
        self.hyperedge_embeddings = hyperedge_embeddings
        self.config = config
        self.semantic_dim = int(node_embeddings.shape[-1])
        self.dtype = node_embeddings.dtype
        self.device = node_embeddings.device
        self.vertex_cache: Dict[tuple[int, int], torch.Tensor] = {}
        self.hyperedge_cache: Dict[tuple[int, int], torch.Tensor] = {}
        self.base_hyperedge_cache: Dict[int, torch.Tensor] = {}
        self.bucket_embedding_cache = {
            bucket_id: _order_bucket_embedding(bucket_id, self.semantic_dim, self.dtype).to(self.device)
            for bucket_id in range(int(config.order_bucket_count))
        }

    def vertex_state(self, node_id: int, step: int) -> torch.Tensor:
        key = (int(node_id), int(step))
        if key in self.vertex_cache:
            return self.vertex_cache[key]
        if step <= 0:
            state = self.node_embeddings[int(node_id)].to(self.device)
        else:
            incident = self.task_graph.get_incident_hyperedges(int(node_id))
            if not incident:
                state = self.vertex_state(int(node_id), step - 1)
            else:
                collected = []
                for hyperedge_id in incident:
                    hyperedge_state = self.hyperedge_state(int(hyperedge_id), step)
                    bucket_id = order_bucket_id(self.task_graph.get_hyperedge_size(int(hyperedge_id)))
                    collected.append(hyperedge_state + self.bucket_embedding_cache[bucket_id])
                state = torch.stack(collected, dim=0).mean(dim=0)
        self.vertex_cache[key] = state
        return state

    def base_hyperedge_state(self, hyperedge_id: int) -> torch.Tensor:
        hyperedge_id = int(hyperedge_id)
        if hyperedge_id in self.base_hyperedge_cache:
            return self.base_hyperedge_cache[hyperedge_id]
        if hyperedge_id >= 0 and self.hyperedge_embeddings is not None:
            state = self.hyperedge_embeddings[hyperedge_id].to(self.device)
        else:
            members = self.task_graph.get_hyperedge_members(hyperedge_id)
            if not members:
                state = torch.zeros(self.semantic_dim, dtype=self.dtype, device=self.device)
            else:
                member_states = self.node_embeddings[torch.tensor(members, dtype=torch.long)].to(self.device)
                state = member_states.mean(dim=0)
        self.base_hyperedge_cache[hyperedge_id] = state
        return state

    def hyperedge_state(self, hyperedge_id: int, step: int) -> torch.Tensor:
        key = (int(hyperedge_id), int(step))
        if key in self.hyperedge_cache:
            return self.hyperedge_cache[key]
        if step <= 0:
            state = self.base_hyperedge_state(int(hyperedge_id))
        else:
            members = self.task_graph.get_hyperedge_members(int(hyperedge_id))
            if not members:
                state = self.base_hyperedge_state(int(hyperedge_id))
            else:
                states = [self.vertex_state(int(node_id), step - 1) for node_id in members]
                state = torch.stack(states, dim=0).mean(dim=0)
        self.hyperedge_cache[key] = state
        return state


@lru_cache(maxsize=None)
def build_overview_position_codes(
    hops: int = OVERVIEW_HOPS,
    order_bucket_count: int = OVERVIEW_ORDER_BUCKET_COUNT,
    position_dim: int = FINAL_HIDT_POSITION_DIM,
) -> torch.Tensor:
    num_nodes = int(hops) * int(order_bucket_count)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    def slot_index(hop_id: int, bucket_id: int) -> int:
        return (hop_id - 1) * int(order_bucket_count) + int(bucket_id)

    for hop_id in range(1, int(hops) + 1):
        for bucket_id in range(int(order_bucket_count)):
            current = slot_index(hop_id, bucket_id)
            if bucket_id + 1 < int(order_bucket_count):
                right = slot_index(hop_id, bucket_id + 1)
                adjacency[current, right] = 1.0
                adjacency[right, current] = 1.0
            if hop_id + 1 <= int(hops):
                down = slot_index(hop_id + 1, bucket_id)
                adjacency[current, down] = 1.0
                adjacency[down, current] = 1.0

    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask].pow(-0.5)
    laplacian = torch.eye(num_nodes, dtype=torch.float32) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    usable_dims = min(int(position_dim), max(num_nodes - 1, 0))
    if usable_dims > 0:
        position_codes = _fix_eigenvector_sign(eigenvectors[:, 1 : 1 + usable_dims])
    else:
        position_codes = torch.zeros((num_nodes, 0), dtype=torch.float32)
    if usable_dims < int(position_dim):
        padding = torch.zeros((num_nodes, int(position_dim) - usable_dims), dtype=torch.float32)
        position_codes = torch.cat([position_codes, padding], dim=1)
    return position_codes


def _build_support_refs(hyperedge_ids: Sequence[int]) -> List[Dict[str, Any]]:
    refs = []
    for hyperedge_id in hyperedge_ids:
        refs.append(
            {
                "kind": "real",
                "hyperedge_id": int(hyperedge_id),
            }
        )
    return refs


def build_overview_payload(
    data_view: Dict[str, Any],
    task_spec: OverviewTaskSpec,
    node_embeddings: torch.Tensor | None = None,
    hyperedge_embeddings: torch.Tensor | None = None,
    config: OverviewConfig | None = None,
    source: str | None = None,
) -> Dict[str, Any]:
    resolved_config = config or OverviewConfig()
    task_graph = TaskConditionedOverviewGraph(data_view, task_spec)
    shells = build_hyperedge_shells(task_graph, resolved_config)
    position_codes = build_overview_position_codes(
        hops=resolved_config.hops,
        order_bucket_count=resolved_config.order_bucket_count,
        position_dim=resolved_config.position_dim,
    )

    aggregator = None
    if node_embeddings is not None:
        aggregator = ParameterFreeOverviewAggregator(
            task_graph=task_graph,
            node_embeddings=node_embeddings,
            hyperedge_embeddings=hyperedge_embeddings,
            config=resolved_config,
        )

    tokens = []
    for hop_id in range(1, int(resolved_config.hops) + 1):
        shell_hyperedges = shells[hop_id]
        for bucket_id in range(int(resolved_config.order_bucket_count)):
            bucket_hyperedges = [
                int(hyperedge_id)
                for hyperedge_id in shell_hyperedges
                if order_bucket_id(task_graph.get_hyperedge_size(int(hyperedge_id))) == bucket_id
            ]
            semantic_override = None
            if aggregator is not None:
                if bucket_hyperedges:
                    semantic_override = (
                        torch.stack(
                            [aggregator.hyperedge_state(int(hyperedge_id), hop_id) for hyperedge_id in bucket_hyperedges],
                            dim=0,
                        )
                        .mean(dim=0)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                else:
                    semantic_override = (
                        torch.zeros(
                            aggregator.semantic_dim,
                            dtype=aggregator.dtype,
                            device=aggregator.device,
                        )
                        .cpu()
                        .tolist()
                    )
            overview_index = (hop_id - 1) * int(resolved_config.order_bucket_count) + bucket_id
            support_count = len(bucket_hyperedges)
            support_bucket = support_bucket_id(support_count)
            token = {
                "kind": "overview_token",
                "type_label": "hyperedge",
                "type_id": 1,
                "depth": int(hop_id),
                "depth_label": f"hop_{hop_id}",
                "slot_id": -1,
                "parent_slot_id": -1,
                "relation_to_parent": "overview",
                "center_kind": task_spec.center_kind,
                "center_id": int(task_spec.center_id),
                "is_pad": 0,
                "is_root": 0,
                "is_target": 0,
                "is_overview": 1,
                "overview_kind": "bucket_summary",
                "overview_version": OVERVIEW_VERSION,
                "task_view": task_spec.task_view,
                "hop_id": int(hop_id),
                "order_bucket_id": int(bucket_id),
                "order_bucket_label": FINAL_HIDT_ORDER_BUCKETS[bucket_id][0],
                "degree_bucket_id": int(support_bucket),
                "degree_bucket_label": support_bucket_label(support_bucket),
                "support_count": int(support_count),
                "support_bucket_id": int(support_bucket),
                "support_bucket_label": support_bucket_label(support_bucket),
                "is_empty": int(support_count == 0),
                "position_code_override": position_codes[overview_index].tolist(),
                "support_hyperedge_refs": _build_support_refs(bucket_hyperedges),
            }
            if semantic_override is not None:
                token["semantic_override"] = semantic_override
            tokens.append(token)

    return {
        "version": OVERVIEW_VERSION,
        "source": source or task_spec.center_kind,
        "task_view": task_spec.task_view,
        "center_kind": task_spec.center_kind,
        "center_id": int(task_spec.center_id),
        "aggregation_hops": int(resolved_config.hops),
        "order_bucket_labels": [bucket[0] for bucket in FINAL_HIDT_ORDER_BUCKETS[: resolved_config.order_bucket_count]],
        "support_bucket_labels": [bucket[0] for bucket in OVERVIEW_SUPPORT_BUCKETS],
        "position_dim": int(resolved_config.position_dim),
        "tokens": tokens,
    }


def build_vertex_overview_payload(
    data_view: Dict[str, Any],
    node_id: int,
    node_embeddings: torch.Tensor | None = None,
    hyperedge_embeddings: torch.Tensor | None = None,
    config: OverviewConfig | None = None,
    source: str = "vc_hidt",
) -> Dict[str, Any]:
    return build_overview_payload(
        data_view=data_view,
        task_spec=OverviewTaskSpec(
            task_view=OVERVIEW_TASK_OPEN,
            center_kind=OVERVIEW_CENTER_VERTEX,
            center_id=int(node_id),
        ),
        node_embeddings=node_embeddings,
        hyperedge_embeddings=hyperedge_embeddings,
        config=config,
        source=source,
    )


def build_hyperedge_overview_payload(
    data_view: Dict[str, Any],
    hyperedge_id: int,
    node_embeddings: torch.Tensor | None = None,
    hyperedge_embeddings: torch.Tensor | None = None,
    config: OverviewConfig | None = None,
    source: str = "ec_hidt",
) -> Dict[str, Any]:
    return build_overview_payload(
        data_view=data_view,
        task_spec=OverviewTaskSpec(
            task_view=OVERVIEW_TASK_OPEN,
            center_kind=OVERVIEW_CENTER_HYPEREDGE,
            center_id=int(hyperedge_id),
        ),
        node_embeddings=node_embeddings,
        hyperedge_embeddings=hyperedge_embeddings,
        config=config,
        source=source,
    )


def can_use_overview_payload(payload: Dict[str, Any] | None) -> bool:
    if not payload:
        return False
    tokens = payload.get("tokens", [])
    if not tokens:
        return False
    return "support_hyperedge_refs" in tokens[0]


def resolve_overview_token_semantic(
    token: Dict[str, Any],
    data_view: Dict[str, Any],
    node_embeddings: torch.Tensor,
    hyperedge_embeddings: torch.Tensor | None,
) -> torch.Tensor:
    support_refs = token.get("support_hyperedge_refs", [])
    if not support_refs:
        return torch.zeros(node_embeddings.shape[-1], dtype=node_embeddings.dtype)

    task_spec = OverviewTaskSpec(
        task_view=str(token.get("task_view", OVERVIEW_TASK_OPEN)),
        center_kind=str(token.get("center_kind", OVERVIEW_CENTER_VERTEX)),
        center_id=int(token.get("center_id", 0)),
    )
    aggregator = ParameterFreeOverviewAggregator(
        task_graph=TaskConditionedOverviewGraph(data_view, task_spec),
        node_embeddings=node_embeddings,
        hyperedge_embeddings=hyperedge_embeddings,
        config=OverviewConfig(
            hops=max(int(token.get("hop_id", 1)), OVERVIEW_HOPS),
            order_bucket_count=OVERVIEW_ORDER_BUCKET_COUNT,
        ),
    )
    hop_id = int(token.get("hop_id", 1))
    states = [
        aggregator.hyperedge_state(int(ref["hyperedge_id"]), hop_id)
        for ref in support_refs
    ]
    return torch.stack(states, dim=0).mean(dim=0)
