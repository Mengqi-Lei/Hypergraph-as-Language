from __future__ import annotations

from typing import Any, Dict, List, Sequence

from utils.constants import DEFAULT_HYPERGRAPH_TOKEN

# ---------------------------------------------------------------------------
# Three-stage prompt convention: Background -> {details} -> Question
#
# Every prompt template (both dataset-specific in processed_data["prompt_templates"]
# and the hard-coded fallbacks below) MUST contain a ``{details}`` placeholder.
# At runtime ``_inject_details`` either:
#   - replaces ``{details}`` with ``Details of the hypergraph:\n<text_context>``
#     when text_context is provided, or
#   - scrubs the placeholder (and surrounding blank lines) when text_context
#     is None, gracefully falling back to a two-stage "Background + Question"
#     prompt with no awkward gap.
# ---------------------------------------------------------------------------

_DETAILS_PLACEHOLDER = "{details}"
_DETAILS_HEADER = "Details of the hypergraph:"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_sequence_value(values: Sequence[Any] | None, index: int) -> Any:
    if values is None:
        return None
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _inject_details(prompt: str, text_context: str | None) -> str:
    """Inject (or scrub) the ``{details}`` placeholder in a three-stage prompt.

    See module docstring for the convention. Behaviour:

    * ``text_context`` is provided
        - ``{details}`` present: replaced by ``Details of the hypergraph:\\n<text>``
        - ``{details}`` absent: block appended at end
    * ``text_context`` is None
        - ``{details}`` present: placeholder and neighboring blank lines removed
        - ``{details}`` absent: no-op
    """
    if text_context:
        block = f"{_DETAILS_HEADER}\n{text_context}"
        if _DETAILS_PLACEHOLDER in prompt:
            return prompt.replace(_DETAILS_PLACEHOLDER, block)
        return f"{prompt}\n\n{block}"

    if _DETAILS_PLACEHOLDER not in prompt:
        return prompt
    cleaned = prompt.replace(f"\n\n{_DETAILS_PLACEHOLDER}\n\n", "\n\n")
    cleaned = cleaned.replace(f"\n{_DETAILS_PLACEHOLDER}\n", "\n")
    cleaned = cleaned.replace(_DETAILS_PLACEHOLDER, "")
    return cleaned


def _resolve_prompt_template(
    task: str,
    label_texts: Sequence[str],
    processed_data: Dict[str, Any],
) -> str | None:
    """If processed_data carries a dataset-specific prompt template for *task*,
    fill in the {labels} placeholder and return it.  Otherwise return None so
    the caller falls back to the built-in default.

    ``{details}`` is intentionally left untouched here; it is processed
    later by ``_inject_details``.
    """
    templates = processed_data.get("prompt_templates")
    if not isinstance(templates, dict):
        return None
    raw = templates.get(task)
    if not raw:
        return None
    labels = ", ".join(str(label) for label in label_texts)
    return str(raw).replace("{labels}", labels)


# ---------------------------------------------------------------------------
# Hard-coded fallback prompts (used by ogbn-arxiv-hg / -u2-hg, which do not
# carry a ``prompt_templates`` field).  All fallbacks follow the three-stage
# convention with a ``{details}`` placeholder.
# ---------------------------------------------------------------------------

def build_hypergraph_nc_prompt(
    label_texts: Sequence[str],
    prompt_style: str | None = None,
    processed_data: Dict[str, Any] | None = None,
) -> str:
    if processed_data is not None:
        custom = _resolve_prompt_template("nc", label_texts, processed_data)
        if custom is not None:
            return custom
    labels = ", ".join(str(label) for label in label_texts)
    if (prompt_style or "").strip() == "2uniform_edge_lift":
        return (
            f"Given a node-centered 2-uniform hypergraph: {DEFAULT_HYPERGRAPH_TOKEN}, "
            "where nodes represent papers and each size-2 hyperedge represents one directed "
            "citation relation in which the first node cites the second node.\n\n"
            "{details}\n\n"
            "Question: Please tell me which class the center node belongs to. "
            f"The 40 classes are: {labels}. Directly output the class name."
        )
    return (
        f"Given a node-centered hypergraph: {DEFAULT_HYPERGRAPH_TOKEN}, "
        "where nodes represent papers and hyperedges represent sets of papers "
        "co-cited by a source paper.\n\n"
        "{details}\n\n"
        "Question: Please tell me which class the center node belongs to. "
        f"The 40 classes are: {labels}. Directly output the class name."
    )


def build_hypergraph_hecls_prompt(
    label_texts: Sequence[str],
    processed_data: Dict[str, Any] | None = None,
) -> str:
    if processed_data is not None:
        custom = _resolve_prompt_template("hecls", label_texts, processed_data)
        if custom is not None:
            return custom
    labels = ", ".join(str(label) for label in label_texts)
    return (
        f"Given a hyperedge-centered hypergraph: {DEFAULT_HYPERGRAPH_TOKEN}, "
        "where the center hyperedge is induced by one source paper and its member "
        "nodes are the cited papers.\n\n"
        "{details}\n\n"
        "Question: Please tell me which class the source paper belongs to. "
        f"The 40 classes are: {labels}. Directly output the class name."
    )


def resolve_node_task_fields(row: Dict[str, Any], processed_data: Dict[str, Any]) -> Dict[str, Any]:
    node_id = int(row["id"])
    labels = processed_data["label_texts"]
    ys = processed_data["y"]
    titles = processed_data.get("title")

    label_id = int(ys[node_id])
    label_text = _normalize_text(_safe_sequence_value(labels, label_id))
    title = _normalize_text(_safe_sequence_value(titles, node_id))
    return {
        "id": node_id,
        "label_id": label_id,
        "label_text": label_text,
        "title": title,
    }


def resolve_hyperedge_task_fields(row: Dict[str, Any], processed_data: Dict[str, Any]) -> Dict[str, Any]:
    hyperedge_id = int(row["id"])
    hyperedge_source = processed_data["hyperedge_source"]
    ys = processed_data["y"]
    labels = processed_data["label_texts"]
    titles = processed_data.get("title")

    source_node_id = int(hyperedge_source[hyperedge_id])
    label_id = int(ys[source_node_id])
    label_text = _normalize_text(_safe_sequence_value(labels, label_id))
    source_title = _normalize_text(_safe_sequence_value(titles, source_node_id))
    return {
        "id": hyperedge_id,
        "source_node_id": source_node_id,
        "label_id": label_id,
        "label_text": label_text,
        "source_title": source_title,
    }


def build_hypergraph_task_conversations(
    task: str,
    row: Dict[str, Any],
    processed_data: Dict[str, Any],
    text_context: str | None = None,
) -> List[Dict[str, str]]:
    """Build [human, gpt] conversation turns for a single task example.

    Args:
        task: one of ``nc`` / ``hecls``.
        row: a row from the prebaked JSONL (must carry ``id``).
        processed_data: dataset-level metadata loaded from ``processed_data.pt``.
        text_context: optional natural-language description of the HIDT subgraph
            generated by ``utils.hypergraph_text_context.build_text_context_block``.
            When provided it is injected into the human prompt's ``{details}``
            placeholder; when None the placeholder is scrubbed cleanly.
    """
    label_texts = processed_data["label_texts"]
    prompt_style = processed_data.get("graph_prompt_style")

    if task == "nc":
        node_fields = resolve_node_task_fields(row, processed_data)
        human_prompt = build_hypergraph_nc_prompt(
            label_texts, prompt_style=prompt_style, processed_data=processed_data,
        )
        return [
            {"from": "human", "value": _inject_details(human_prompt, text_context)},
            {"from": "gpt", "value": node_fields["label_text"]},
        ]
    if task == "hecls":
        hyperedge_fields = resolve_hyperedge_task_fields(row, processed_data)
        human_prompt = build_hypergraph_hecls_prompt(
            label_texts, processed_data=processed_data,
        )
        return [
            {"from": "human", "value": _inject_details(human_prompt, text_context)},
            {"from": "gpt", "value": hyperedge_fields["label_text"]},
        ]
    raise ValueError(
        f"Unsupported hypergraph task: {task!r}; supported tasks are 'nc' and 'hecls'."
    )
