"""Build natural-language text descriptions of HIDT-sampled subgraphs.

The text block complements the projector's hypergraph tokens with explicit
titles and abstracts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------------------
# Token type constants in HIDT parallel arrays.
# Matches utils/final_hidt.py: vertex tokens have type_id == 0,
# hyperedge tokens have type_id == 1.
# ---------------------------------------------------------------------------
_TYPE_VERTEX = 0
_TYPE_HYPEREDGE = 1


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_value(values: Sequence[Any] | None, index: int) -> Any:
    if values is None:
        return None
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_title(processed_data: Dict[str, Any], node_id: int) -> str:
    return _normalize_text(_safe_value(processed_data.get("title"), node_id))


def _get_abs(processed_data: Dict[str, Any], node_id: int) -> str:
    return _normalize_text(_safe_value(processed_data.get("abs"), node_id))


def _hyperedge_source_id(processed_data: Dict[str, Any], he_id: int) -> int:
    hs = processed_data.get("hyperedge_source")
    if hs is None or he_id < 0:
        return -1
    try:
        return int(hs[he_id])
    except (IndexError, TypeError, ValueError):
        return -1


def _fmt_title(title: str) -> str:
    return title if title else "(no title)"


# ---------------------------------------------------------------------------
# Slot index built from the 6 parallel HIDT arrays.
# ---------------------------------------------------------------------------

def _build_slot_index(arrays: Dict[str, Sequence[int]]) -> Dict[int, Dict[str, int]]:
    """Return ``{slot_id: {type_id, depth, node_id, he_id, parent_slot_id}}``.

    Tokens with ``slot_id < 0`` (overview tokens) are skipped; they are not
    part of the HIDT tree.
    """
    slot_ids = arrays["token_slot_ids"]
    type_ids = arrays["token_type_ids"]
    depths = arrays["token_depths"]
    node_ids = arrays["token_node_ids"]
    he_ids = arrays["token_he_ids"]
    parent_slot_ids = arrays["token_parent_slot_ids"]

    slots: Dict[int, Dict[str, int]] = {}
    for i in range(len(slot_ids)):
        sid = int(slot_ids[i])
        if sid < 0:
            continue
        slots[sid] = {
            "slot_id": sid,
            "type_id": int(type_ids[i]),
            "depth": int(depths[i]),
            "node_id": int(node_ids[i]),
            "he_id": int(he_ids[i]),
            "parent_slot_id": int(parent_slot_ids[i]),
        }
    return slots


def _children(
    slots: Dict[int, Dict[str, int]],
    parent_slot_id: int,
    required_type: int,
) -> List[Dict[str, int]]:
    """Return non-pad child slots whose parent matches and type matches.

    Sorted by ``slot_id`` so output order is deterministic and matches the
    HIDT linearization.
    """
    out: List[Dict[str, int]] = []
    for sid in sorted(slots.keys()):
        s = slots[sid]
        if s["parent_slot_id"] != parent_slot_id:
            continue
        if s["type_id"] != required_type:
            continue
        if required_type == _TYPE_VERTEX and s["node_id"] < 0:
            continue
        if required_type == _TYPE_HYPEREDGE and s["he_id"] < 0:
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# NC: vertex-centered text block
# ---------------------------------------------------------------------------

def build_text_context_nc(
    arrays: Dict[str, Sequence[int]],
    center_node_id: int,
    processed_data: Dict[str, Any],
) -> str:
    slots = _build_slot_index(arrays)
    lines: List[str] = []

    center_title = _get_title(processed_data, center_node_id)
    center_abs = _get_abs(processed_data, center_node_id)
    if center_title:
        lines.append(f'Center node: "{center_title}"')
    else:
        lines.append("Center node: (no title available)")
    if center_abs:
        lines.append(f"  Abstract: {center_abs}")
    lines.append("")

    he_slots = _children(slots, parent_slot_id=0, required_type=_TYPE_HYPEREDGE)
    for he_idx, he in enumerate(he_slots, start=1):
        he_id = he["he_id"]
        src_id = _hyperedge_source_id(processed_data, he_id)
        src_title = _get_title(processed_data, src_id) if src_id >= 0 else ""

        if src_id == center_node_id:
            induced_clause = "induced by the center paper itself"
        elif src_title:
            induced_clause = f'induced by source paper "{src_title}"'
        else:
            induced_clause = "induced by source paper with unknown title"

        lines.append(f"Connected through hyperedge HE-{he_idx} ({induced_clause}):")
        lines.append("  with members:")

        member_slots = _children(slots, parent_slot_id=he["slot_id"], required_type=_TYPE_VERTEX)
        if not member_slots:
            lines.append("    (none)")
        else:
            for m in member_slots:
                m_title = _get_title(processed_data, m["node_id"])
                lines.append(f'    - "{_fmt_title(m_title)}"')
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HECLS: hyperedge-centered text block
# ---------------------------------------------------------------------------

def build_text_context_hecls(
    arrays: Dict[str, Sequence[int]],
    center_he_id: int,
    processed_data: Dict[str, Any],
) -> str:
    slots = _build_slot_index(arrays)
    lines: List[str] = []

    src_id = _hyperedge_source_id(processed_data, center_he_id)
    src_title = _get_title(processed_data, src_id) if src_id >= 0 else ""
    src_abs = _get_abs(processed_data, src_id) if src_id >= 0 else ""

    if src_title:
        lines.append(f'Center hyperedge (induced by source paper "{src_title}"):')
    else:
        lines.append("Center hyperedge (induced by source paper with unknown title):")
    if src_abs:
        lines.append(f"  Abstract: {src_abs}")
    lines.append("")

    member_slots = _children(slots, parent_slot_id=0, required_type=_TYPE_VERTEX)
    lines.append("Includes members:")
    global_he_counter = 0
    if not member_slots:
        lines.append("  (none)")
    else:
        for m in member_slots:
            m_title = _get_title(processed_data, m["node_id"])
            lines.append(f'  Member "{_fmt_title(m_title)}":')
            lines.append("    also connected through other hyperedges:")

            child_hes = _children(slots, parent_slot_id=m["slot_id"], required_type=_TYPE_HYPEREDGE)
            if not child_hes:
                lines.append("      (none)")
            else:
                for ch in child_hes:
                    global_he_counter += 1
                    ch_src_id = _hyperedge_source_id(processed_data, ch["he_id"])
                    ch_src_title = _get_title(processed_data, ch_src_id) if ch_src_id >= 0 else ""
                    if ch_src_title:
                        clause = f'induced by source paper "{ch_src_title}"'
                    else:
                        clause = "induced by source paper with unknown title"
                    lines.append(f"      - Hyperedge HE-{global_he_counter} ({clause})")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def build_text_context_block(
    task: str,
    entity_id: int,
    arrays: Dict[str, Sequence[int]],
    processed_data: Dict[str, Any],
) -> str:
    """Generate a natural-language description of the HIDT subgraph.

    Args:
        task: ``nc`` (vertex-centered) or ``hecls`` (hyperedge-centered).
        entity_id: center node_id (NC) or center hyperedge_id (HECLS).
        arrays: parallel HIDT arrays as stored in prebaked JSONL rows.
        processed_data: the ``processed_data.pt`` dict for this dataset.

    Returns:
        A string ready to be substituted into the ``{details}`` placeholder
        of the prompt template (no trailing newline, no header).
    """
    if task == "nc":
        return build_text_context_nc(arrays, int(entity_id), processed_data)
    if task == "hecls":
        return build_text_context_hecls(arrays, int(entity_id), processed_data)
    raise ValueError(
        f"Unsupported task for text context: {task!r}; supported tasks are 'nc' and 'hecls'."
    )
