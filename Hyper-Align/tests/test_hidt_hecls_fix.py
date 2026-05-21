"""Smoke tests for vertex- and hyperedge-centered HIDT construction.

The tests use a tiny in-memory hypergraph, so they run without loading any
dataset files.

Run::

    cd <path-to-Hyper-Align>
    python tests/test_hidt_hecls_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.final_hidt import (  # noqa: E402  (path setup above)
    FinalHIDTConfig,
    TorchFinalHIDTAccessor,
    build_final_hidt_instance,
)


def _make_tiny_hg() -> dict:
    return {
        "he_ptr": torch.tensor([0, 3, 6], dtype=torch.long),
        "he_node_index": torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.long),
        "node_ptr": torch.tensor([0, 1, 3, 5, 6], dtype=torch.long),
        "node_hyperedge_index": torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long),
        "hyperedge_size": torch.tensor([3, 3], dtype=torch.long),
        "hyperedge_source": torch.tensor([0, 1], dtype=torch.long),
        "num_nodes": 4,
        "num_hyperedges": 2,
    }


_CFG = FinalHIDTConfig(
    max_depth=3,
    max_incident_hyperedges=4,
    max_members_per_hyperedge=4,
    max_child_hyperedges=2,
)


def _slots_at_depth(payload: dict, depth: int) -> list[dict]:
    return [t for t in payload["tokens"] if int(t["depth"]) == depth]


def _real_vertex_ids_at_depth(payload: dict, depth: int) -> list[int]:
    return [
        int(t["node_id"])
        for t in _slots_at_depth(payload, depth)
        if int(t["type_id"]) == 0  # vertex non-pad
    ]


def _real_he_ids_at_depth(payload: dict, depth: int) -> list[int]:
    return [
        int(t["hyperedge_id"])
        for t in _slots_at_depth(payload, depth)
        if int(t["type_id"]) == 1  # hyperedge non-pad
    ]


def test_vertex_center_unchanged() -> None:
    pd = _make_tiny_hg()
    acc = TorchFinalHIDTAccessor(pd)

    payload = build_final_hidt_instance(
        center_kind="vertex", center_id=1, accessor=acc, config=_CFG,
    )

    d0_v = _real_vertex_ids_at_depth(payload, 0)
    assert d0_v == [1], f"depth-0 vertex must be [1], got {d0_v}"

    d1_he = sorted(_real_he_ids_at_depth(payload, 1))
    assert d1_he == [0, 1], f"depth-1 hyperedges must be [0,1], got {d1_he}"

    d2_v = sorted(set(_real_vertex_ids_at_depth(payload, 2)))
    assert d2_v == [0, 2, 3], f"depth-2 vertices must be [0,2,3], got {d2_v}"

    print("[ok] test_vertex_center_unchanged")


def test_hyperedge_center_natural_expansion() -> None:
    pd = _make_tiny_hg()
    acc = TorchFinalHIDTAccessor(pd)

    payload = build_final_hidt_instance(
        center_kind="hyperedge", center_id=0, accessor=acc, config=_CFG,
    )

    d0_he = _real_he_ids_at_depth(payload, 0)
    assert d0_he == [0], f"depth-0 hyperedge must be [0], got {d0_he}"

    d1_v = sorted(_real_vertex_ids_at_depth(payload, 1))
    assert d1_v == [0, 1, 2], (
        f"depth-1 vertices (members of center he0) must be "
        f"[0, 1, 2], got {d1_v}"
    )

    d2_he = sorted(_real_he_ids_at_depth(payload, 2))
    assert d2_he == [1, 1], f"depth-2 hyperedges must be [1,1], got {d2_he}"

    print("[ok] test_hyperedge_center_natural_expansion")


def test_hyperedge_center_forced_members_still_works() -> None:
    pd = _make_tiny_hg()
    acc = TorchFinalHIDTAccessor(pd)

    payload = build_final_hidt_instance(
        center_kind="hyperedge", center_id=0, accessor=acc, config=_CFG,
        forced_member_node_ids=[2],
    )

    d1_v = sorted(_real_vertex_ids_at_depth(payload, 1))
    assert d1_v == [2], f"forced expansion must give exactly [2], got {d1_v}"
    print("[ok] test_hyperedge_center_forced_members_still_works")


def test_hyperedge_center_explicit_empty_list_is_no_expansion() -> None:
    pd = _make_tiny_hg()
    acc = TorchFinalHIDTAccessor(pd)

    payload = build_final_hidt_instance(
        center_kind="hyperedge", center_id=0, accessor=acc, config=_CFG,
        forced_member_node_ids=[],
    )

    d1_v = _real_vertex_ids_at_depth(payload, 1)
    assert d1_v == [], (
        f"explicit empty list must keep depth-1 empty, got {d1_v}"
    )
    print("[ok] test_hyperedge_center_explicit_empty_list_is_no_expansion")


def main() -> None:
    test_vertex_center_unchanged()
    test_hyperedge_center_natural_expansion()
    test_hyperedge_center_forced_members_still_works()
    test_hyperedge_center_explicit_empty_list_is_no_expansion()
    print("\nAll HIDT tests passed.")


if __name__ == "__main__":
    main()
