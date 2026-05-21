from __future__ import annotations

from pathlib import Path


_DATASET_ROOT_CANDIDATES = {
    "arxiv_hg": (
        "../HyperAlign-Bench/dataset/arxiv_hg",
        "../HyperAlign-Bench/dataset/ogbn-arxiv-hg",
    ),
    "ogbn-arxiv-hg": (
        "../HyperAlign-Bench/dataset/arxiv_hg",
        "../HyperAlign-Bench/dataset/ogbn-arxiv-hg",
    ),
    "cora_cc": (
        "../HyperAlign-Bench/dataset/cora_cc",
        "../HyperAlign-Bench/dataset/cora_co_hg",
    ),
    "cora_co_hg": (
        "../HyperAlign-Bench/dataset/cora_cc",
        "../HyperAlign-Bench/dataset/cora_co_hg",
    ),
    "pubmed": (
        "../HyperAlign-Bench/dataset/pubmed",
        "../HyperAlign-Bench/dataset/pubmed_hg",
    ),
    "pubmed_hg": (
        "../HyperAlign-Bench/dataset/pubmed",
        "../HyperAlign-Bench/dataset/pubmed_hg",
    ),
    "dblp": (
        "../HyperAlign-Bench/dataset/dblp",
        "../HyperAlign-Bench/dataset/dblp_a_hg",
    ),
    "dblp_a_hg": (
        "../HyperAlign-Bench/dataset/dblp",
        "../HyperAlign-Bench/dataset/dblp_a_hg",
    ),
    "imdb": (
        "../HyperAlign-Bench/dataset/imdb",
        "../HyperAlign-Bench/dataset/imdb_hg",
    ),
    "imdb_hg": (
        "../HyperAlign-Bench/dataset/imdb",
        "../HyperAlign-Bench/dataset/imdb_hg",
    ),
}

DEFAULT_HYPERGRAPH_DATA_ROOTS = {
    dataset_name: candidates[0]
    for dataset_name, candidates in _DATASET_ROOT_CANDIDATES.items()
}


def _first_existing_or_preferred(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def is_hypergraph_dataset_name(dataset_name: str | None) -> bool:
    dataset_name = (dataset_name or "").strip()
    return dataset_name in _DATASET_ROOT_CANDIDATES or dataset_name.endswith("_hg")


def resolve_hypergraph_data_root(dataset_name: str | None, configured_root: str | None) -> str | None:
    dataset_name = (dataset_name or "").strip()
    configured_root = configured_root or None
    candidates = _DATASET_ROOT_CANDIDATES.get(dataset_name)

    if configured_root is None:
        return _first_existing_or_preferred(candidates) if candidates else None

    if candidates and configured_root in candidates:
        if Path(configured_root).exists():
            return configured_root
        return _first_existing_or_preferred(candidates)

    arxiv_defaults = set(_DATASET_ROOT_CANDIDATES["arxiv_hg"])
    if candidates and configured_root in arxiv_defaults and dataset_name not in {"arxiv_hg", "ogbn-arxiv-hg"}:
        return _first_existing_or_preferred(candidates)

    return configured_root
