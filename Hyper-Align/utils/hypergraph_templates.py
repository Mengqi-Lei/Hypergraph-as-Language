from __future__ import annotations

from typing import Dict, List, Sequence

from utils.final_hidt import (
    FinalHIDTConfig,
    TorchFinalHIDTAccessor,
    build_final_hidt_instance,
    estimate_final_hidt_token_count,
)
from utils.hypergraph_overview import (
    HIDT_O_TEMPLATE_NAMES,
    OverviewConfig,
    build_hyperedge_overview_payload,
    build_vertex_overview_payload,
    can_use_overview_payload,
)

FORMAL_HIDT_NAMES = {"HIDT", "FORMAL_HIDT", "HIDT_FORMAL"}


def normalize_hyper_template(template: str | None) -> str:
    template_name = (template or "HIDT").strip().upper()
    if template_name in HIDT_O_TEMPLATE_NAMES:
        return "HIDT_O"
    return "HIDT"


def is_formal_hidt(template: str | None) -> bool:
    return normalize_hyper_template(template) in {"HIDT", "HIDT_O"}


def uses_hidto_overview(template: str | None) -> bool:
    return normalize_hyper_template(template) == "HIDT_O"


def estimate_formal_hidt_token_count(
    root_kind: str,
    max_depth: int,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
) -> int:
    normalized_root_kind = "vertex" if root_kind == "node" else "hyperedge"
    return estimate_final_hidt_token_count(
        center_kind=normalized_root_kind,
        max_depth=max_depth,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
    )


def build_formal_node_hidt_tokens(
    node_id: int,
    processed_data: Dict,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
    max_depth: int,
) -> List[Dict]:
    config = FinalHIDTConfig(
        max_depth=max_depth,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
    )
    accessor = TorchFinalHIDTAccessor(processed_data)
    return build_final_hidt_instance(
        center_kind="vertex",
        center_id=int(node_id),
        accessor=accessor,
        config=config,
    )["tokens"]


def build_formal_hyperedge_hidt_tokens(
    hyperedge_id: int,
    processed_data: Dict,
    max_incident_hyperedges: int,
    max_members_per_hyperedge: int,
    max_child_hyperedges: int,
    max_depth: int,
    forced_member_node_ids: Sequence[int] | None = None,
) -> List[Dict]:
    config = FinalHIDTConfig(
        max_depth=max_depth,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
    )
    accessor = TorchFinalHIDTAccessor(processed_data)
    return build_final_hidt_instance(
        center_kind="hyperedge",
        center_id=int(hyperedge_id),
        accessor=accessor,
        config=config,
        forced_member_node_ids=forced_member_node_ids,
    )["tokens"]


def build_node_hidt_tokens(
    sample: Dict,
    processed_data: Dict | None = None,
    max_incident_hyperedges: int = 8,
    max_members_per_hyperedge: int = 8,
    template: str | None = "HIDT",
    formal_hidt_depth: int = 3,
    max_child_hyperedges: int = 1,
    node_embeddings=None,
    hyperedge_embeddings=None,
    overview_hops: int = 2,
    overview_order_buckets: int = 4,
) -> List[Dict]:
    normalized_template = normalize_hyper_template(template)
    if normalized_template == "HIDT_O":
        return build_node_hidto_tokens(
            sample=sample,
            processed_data=processed_data,
            max_incident_hyperedges=max_incident_hyperedges,
            max_members_per_hyperedge=max_members_per_hyperedge,
            formal_hidt_depth=formal_hidt_depth,
            max_child_hyperedges=max_child_hyperedges,
            node_embeddings=node_embeddings,
            hyperedge_embeddings=hyperedge_embeddings,
            overview_hops=overview_hops,
            overview_order_buckets=overview_order_buckets,
        )
    if "vc_hidt" in sample:
        return sample["vc_hidt"]["tokens"]
    if "formal_hidt_tokens" in sample:
        return sample["formal_hidt_tokens"]
    if processed_data is None:
        raise ValueError("processed_data is required for formal HIDT runtime construction.")
    return build_formal_node_hidt_tokens(
        node_id=int(sample["id"]),
        processed_data=processed_data,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
        max_depth=formal_hidt_depth,
    )

def _resolve_overview_payload(
    existing_payload: Dict | None,
    builder,
    require_semantic_override: bool = False,
) -> Dict:
    if can_use_overview_payload(existing_payload):
        if not require_semantic_override:
            return existing_payload
        tokens = existing_payload.get("tokens", [])
        if not tokens or all("semantic_override" in token for token in tokens):
            return existing_payload
    return builder()


def _should_require_semantic_override(node_embeddings) -> bool:
    return node_embeddings is not None


def build_node_hidto_tokens(
    sample: Dict,
    processed_data: Dict | None = None,
    max_incident_hyperedges: int = 8,
    max_members_per_hyperedge: int = 8,
    formal_hidt_depth: int = 3,
    max_child_hyperedges: int = 1,
    node_embeddings=None,
    hyperedge_embeddings=None,
    overview_hops: int = 2,
    overview_order_buckets: int = 4,
) -> List[Dict]:
    if processed_data is None:
        raise ValueError("processed_data is required for HIDT+O runtime construction.")
    detail_tokens = build_node_hidt_tokens(
        sample=sample,
        processed_data=processed_data,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        template="HIDT",
        formal_hidt_depth=formal_hidt_depth,
        max_child_hyperedges=max_child_hyperedges,
    )
    overview_payload = _resolve_overview_payload(
        sample.get("vc_overview"),
        lambda: build_vertex_overview_payload(
            data_view=processed_data,
            node_id=int(sample["id"]),
            node_embeddings=node_embeddings,
            hyperedge_embeddings=hyperedge_embeddings,
            config=OverviewConfig(hops=overview_hops, order_bucket_count=overview_order_buckets),
            source="vc_hidt",
        ),
        require_semantic_override=_should_require_semantic_override(node_embeddings),
    )
    return detail_tokens + overview_payload["tokens"]

def build_hyperedge_hidto_tokens(
    sample: Dict,
    processed_data: Dict | None = None,
    max_incident_hyperedges: int = 8,
    max_members_per_hyperedge: int = 8,
    formal_hidt_depth: int = 3,
    max_child_hyperedges: int = 1,
    node_embeddings=None,
    hyperedge_embeddings=None,
    overview_hops: int = 2,
    overview_order_buckets: int = 4,
) -> List[Dict]:
    if processed_data is None:
        raise ValueError("processed_data is required for HIDT+O runtime construction.")
    hyperedge_id = int(sample["id"])
    detail_tokens = build_formal_hyperedge_hidt_tokens(
        hyperedge_id=hyperedge_id,
        processed_data=processed_data,
        max_incident_hyperedges=max_incident_hyperedges,
        max_members_per_hyperedge=max_members_per_hyperedge,
        max_child_hyperedges=max_child_hyperedges,
        max_depth=formal_hidt_depth,
    )
    overview_payload = _resolve_overview_payload(
        sample.get("ec_overview"),
        lambda: build_hyperedge_overview_payload(
            data_view=processed_data,
            hyperedge_id=hyperedge_id,
            node_embeddings=node_embeddings,
            hyperedge_embeddings=hyperedge_embeddings,
            config=OverviewConfig(hops=overview_hops, order_bucket_count=overview_order_buckets),
            source="ec_hidt",
        ),
        require_semantic_override=_should_require_semantic_override(node_embeddings),
    )
    return detail_tokens + overview_payload["tokens"]
