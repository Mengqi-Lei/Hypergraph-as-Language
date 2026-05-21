from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import NodePropPredDataset
from tqdm import tqdm

HYPERALIGN_ROOT = Path(
    os.environ.get("HYPERALIGN_ROOT")
    or os.environ.get("HYPERLM_ROOT")
    or Path(__file__).resolve().parents[2] / "Hyper-Align"
)
if not HYPERALIGN_ROOT.exists():
    raise FileNotFoundError(
        "Cannot find the Hyper-Align runtime. Set HYPERALIGN_ROOT=/path/to/Hyper-Align "
        "or place Hyper-Align next to HyperAlign-Bench. Legacy HYPERLM_ROOT is also accepted."
    )
if str(HYPERALIGN_ROOT) not in sys.path:
    sys.path.append(str(HYPERALIGN_ROOT))

from utils.final_hidt import (  # noqa: E402
    build_final_hidt_contract_summary,
)

from .constants import (
    ARXIV_LABEL_TEXTS,
)


@dataclass
class BuildConfig:
    raw_root: str
    output_dir: str
    titleabs_path: str | None = None
    min_hyperedge_size: int = 2
    max_hyperedge_size: int = 64
    max_incident_hyperedges: int = 8
    max_members_per_hyperedge: int = 8
    max_child_hyperedges: int = 1
    formal_hidt_depth: int = 3
    overview_hops: int = 2
    overview_order_buckets: int = 4
    seed: int = 42


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_file(root: str | Path, filename: str) -> Path:
    root_path = Path(root)
    matches = list(root_path.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {root_path}")
    return matches[0]


def resolve_titleabs_path(raw_root: str | Path, titleabs_path: str | None = None) -> Path | None:
    if titleabs_path:
        candidate = Path(titleabs_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot find titleabs file at {candidate}")
        return candidate
    try:
        return find_file(raw_root, "titleabs.tsv")
    except FileNotFoundError:
        return None


def load_ogbn_arxiv(raw_root: str | Path) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(raw_root))
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    normalized_splits = {
        key: np.asarray(value).reshape(-1).astype(np.int64) for key, value in split_idx.items()
    }
    return graph, labels, normalized_splits


def build_nodes_frame(
    raw_root: str | Path,
    graph: dict[str, Any],
    labels: np.ndarray,
    titleabs_path: str | None = None,
    allow_missing_text: bool = False,
) -> pd.DataFrame:
    node_map_path = find_file(raw_root, "nodeidx2paperid.csv.gz")
    resolved_titleabs_path = resolve_titleabs_path(raw_root, titleabs_path)

    node_map = pd.read_csv(node_map_path, compression="infer")
    if node_map.shape[1] < 2:
        raise ValueError(f"Unexpected node map format: {node_map_path}")
    node_map = node_map.iloc[:, :2].copy()
    node_map.columns = ["node_id", "paper_id"]
    node_map["node_id"] = node_map["node_id"].astype(np.int64)
    node_map["paper_id"] = node_map["paper_id"].astype(np.int64)

    if resolved_titleabs_path is not None:
        titleabs = pd.read_csv(
            resolved_titleabs_path,
            sep="\t",
            header=None,
            names=["paper_id", "title", "abstract"],
            keep_default_na=False,
            na_filter=False,
        )
        # The public OGB misc dump may include a header row; coerce non-numeric
        # values (such as 'paper id' or 'titleabs.tsv') to NaN and drop them.
        titleabs["paper_id"] = pd.to_numeric(titleabs["paper_id"], errors="coerce")
        titleabs = titleabs.dropna(subset=["paper_id"]).copy()
        titleabs["paper_id"] = titleabs["paper_id"].astype(np.int64)
        nodes = node_map.merge(titleabs, on="paper_id", how="left")
    else:
        if not allow_missing_text:
            raise FileNotFoundError(
                "titleabs.tsv was not found under the OGB download directory. "
                "The standard ogbn-arxiv package does not include raw title/abstract text. "
                "Please download the raw text mapping separately and pass --titleabs-path."
            )
        nodes = node_map.copy()
        nodes["title"] = ""
        nodes["abstract"] = ""

    nodes = nodes.sort_values("node_id").drop_duplicates(subset=["node_id"]).reset_index(drop=True)
    nodes["title"] = nodes["title"].fillna("").astype(str)
    nodes["abstract"] = nodes["abstract"].fillna("").astype(str)

    node_year = graph.get("node_year")
    if node_year is None:
        raise RuntimeError("graph['node_year'] is required for temporal splits")
    nodes["year"] = np.asarray(node_year).reshape(-1).astype(np.int64)
    nodes["label"] = labels
    nodes["raw_text"] = (
        nodes["title"].str.strip() + " " + nodes["abstract"].str.strip()
    ).str.strip()
    nodes["has_text"] = ((nodes["title"].str.len() > 0) | (nodes["abstract"].str.len() > 0)).astype(np.int64)

    num_nodes = int(graph["num_nodes"])
    if not nodes["node_id"].is_unique:
        raise AssertionError("node_id must be unique after join")
    if len(nodes) != num_nodes:
        raise AssertionError(f"Expected {num_nodes} nodes, found {len(nodes)} after text join")
    nodes.attrs["text_available"] = bool(resolved_titleabs_path is not None)
    nodes.attrs["titleabs_path"] = str(resolved_titleabs_path) if resolved_titleabs_path is not None else None
    return nodes


def member_sort_key(node_id: int, node_year: np.ndarray) -> tuple[int, int]:
    return (-int(node_year[node_id]), int(node_id))


def build_canonical_hyperedges(
    nodes: pd.DataFrame,
    edge_index: np.ndarray,
    config: BuildConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    source_to_members: dict[int, set[int]] = {}
    src = edge_index[0]
    dst = edge_index[1]
    for source, target in zip(src, dst):
        source = int(source)
        target = int(target)
        source_to_members.setdefault(source, set()).add(target)

    node_year = nodes["year"].to_numpy(dtype=np.int64)
    raw_texts = nodes["raw_text"].tolist()

    hyperedge_rows: list[dict[str, Any]] = []
    he_ptr = [0]
    he_node_chunks: list[int] = []
    stats = {
        "raw_source_hyperedges": len(source_to_members),
        "filtered_too_small": 0,
        "filtered_too_large": 0,
    }

    for hyperedge_id, source_node_id in enumerate(sorted(source_to_members)):
        members = sorted(
            source_to_members[source_node_id],
            key=lambda node_id: member_sort_key(node_id, node_year),
        )
        if len(members) < config.min_hyperedge_size:
            stats["filtered_too_small"] += 1
            continue
        if len(members) > config.max_hyperedge_size:
            stats["filtered_too_large"] += 1
            continue

        current_hyperedge_id = len(hyperedge_rows)
        hyperedge_rows.append(
            {
                "hyperedge_id": current_hyperedge_id,
                "source_node_id": int(source_node_id),
                "hyperedge_year": int(node_year[source_node_id]),
                "hyperedge_size": int(len(members)),
                "hyperedge_text": raw_texts[source_node_id],
            }
        )
        he_node_chunks.extend(members)
        he_ptr.append(len(he_node_chunks))

    hyperedges = pd.DataFrame(hyperedge_rows)
    he_ptr_array = np.asarray(he_ptr, dtype=np.int64)
    he_node_index = np.asarray(he_node_chunks, dtype=np.int64)
    return hyperedges, he_ptr_array, he_node_index, stats


def build_node_reverse_index(
    num_nodes: int,
    hyperedges: pd.DataFrame,
    he_ptr: np.ndarray,
    he_node_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    members_per_node: list[list[int]] = [[] for _ in range(num_nodes)]
    hyperedge_year = hyperedges["hyperedge_year"].to_numpy(dtype=np.int64)
    hyperedge_size = hyperedges["hyperedge_size"].to_numpy(dtype=np.int64)

    for hyperedge_id in range(len(hyperedges)):
        members = he_node_index[he_ptr[hyperedge_id] : he_ptr[hyperedge_id + 1]]
        for member in members:
            members_per_node[int(member)].append(hyperedge_id)

    def incident_sort_key(hyperedge_id: int) -> tuple[int, int, int]:
        return (
            -int(hyperedge_year[hyperedge_id]),
            -int(hyperedge_size[hyperedge_id]),
            int(hyperedge_id),
        )

    node_ptr = [0]
    node_hyperedge_index: list[int] = []
    for hyperedge_ids in members_per_node:
        hyperedge_ids.sort(key=incident_sort_key)
        node_hyperedge_index.extend(hyperedge_ids)
        node_ptr.append(len(node_hyperedge_index))

    return np.asarray(node_ptr, dtype=np.int64), np.asarray(node_hyperedge_index, dtype=np.int64)


def split_hyperedges(hyperedges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = hyperedges["hyperedge_year"].to_numpy(dtype=np.int64)
    return (
        np.where(years <= 2017)[0].astype(np.int64),
        np.where(years == 2018)[0].astype(np.int64),
        np.where(years >= 2019)[0].astype(np.int64),
    )


def build_processed_data(
    graph: dict[str, Any],
    nodes: pd.DataFrame,
    splits: dict[str, np.ndarray],
    hyperedges: pd.DataFrame,
    he_ptr: np.ndarray,
    he_node_index: np.ndarray,
    node_ptr: np.ndarray,
    node_hyperedge_index: np.ndarray,
    he_train_idx: np.ndarray,
    he_valid_idx: np.ndarray,
    he_test_idx: np.ndarray,
) -> dict[str, Any]:
    processed = {
        "dataset_name": "arxiv_hg",
        "graph_origin": "source_excluded_cocitation_hypergraph",
        "supported_tasks": ["nc", "hecls", "nd", "hed"],
        "deferred_tasks": [],
        "num_nodes": int(graph["num_nodes"]),
        "title": nodes["title"].tolist(),
        "abs": nodes["abstract"].tolist(),
        "node_year": torch.tensor(nodes["year"].to_numpy(dtype=np.int64), dtype=torch.long),
        "y": torch.tensor(nodes["label"].to_numpy(dtype=np.int64), dtype=torch.long),
        "label_texts": list(ARXIV_LABEL_TEXTS),
        "train_idx": torch.tensor(splits["train"], dtype=torch.long),
        "valid_idx": torch.tensor(splits["valid"], dtype=torch.long),
        "test_idx": torch.tensor(splits["test"], dtype=torch.long),
        "num_hyperedges": int(len(hyperedges)),
        "hyperedge_source": torch.tensor(
            hyperedges["source_node_id"].to_numpy(dtype=np.int64), dtype=torch.long
        ),
        "hyperedge_size": torch.tensor(
            hyperedges["hyperedge_size"].to_numpy(dtype=np.int64), dtype=torch.long
        ),
        "he_ptr": torch.tensor(he_ptr, dtype=torch.long),
        "he_node_index": torch.tensor(he_node_index, dtype=torch.long),
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "node_hyperedge_index": torch.tensor(node_hyperedge_index, dtype=torch.long),
        "he_train_idx": torch.tensor(he_train_idx, dtype=torch.long),
        "he_valid_idx": torch.tensor(he_valid_idx, dtype=torch.long),
        "he_test_idx": torch.tensor(he_test_idx, dtype=torch.long),
    }
    return processed


def extract_numpy_views(processed_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "y": processed_data["y"].numpy(),
        "label_texts": processed_data["label_texts"],
        "title": processed_data["title"],
        "hyperedge_source": processed_data["hyperedge_source"].numpy(),
        "hyperedge_size": processed_data["hyperedge_size"].numpy(),
        "he_ptr": processed_data["he_ptr"].numpy(),
        "he_node_index": processed_data["he_node_index"].numpy(),
        "node_ptr": processed_data["node_ptr"].numpy(),
        "node_hyperedge_index": processed_data["node_hyperedge_index"].numpy(),
        "train_idx": processed_data["train_idx"].numpy(),
        "valid_idx": processed_data["valid_idx"].numpy(),
        "test_idx": processed_data["test_idx"].numpy(),
        "he_train_idx": processed_data["he_train_idx"].numpy(),
        "he_valid_idx": processed_data["he_valid_idx"].numpy(),
        "he_test_idx": processed_data["he_test_idx"].numpy(),
    }


def build_local_node_sample(center_node_id: int) -> dict[str, Any]:
    return {"id": int(center_node_id)}


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_node_samples(
    output_dir: Path,
    processed_data: dict[str, Any],
    config: BuildConfig,
) -> dict[str, int]:
    views = extract_numpy_views(processed_data)
    split_map = {
        "train": views["train_idx"],
        "valid": views["valid_idx"],
        "test": views["test_idx"],
    }
    sample_counts: dict[str, int] = {}

    for split_name, node_ids in split_map.items():
        samples = [
            build_local_node_sample(int(node_id))
            for node_id in tqdm(node_ids, desc=f"Building node samples ({split_name})")
        ]
        out_path = output_dir / f"node_task_hg_{config.max_incident_hyperedges}_{config.max_members_per_hyperedge}_{split_name}.jsonl"
        write_jsonl(out_path, samples)
        sample_counts[split_name] = len(samples)
    return sample_counts


def build_single_he_task_sample(hyperedge_id: int) -> dict[str, Any]:
    return {"id": int(hyperedge_id)}


def build_he_task_samples(
    output_dir: Path,
    processed_data: dict[str, Any],
    config: BuildConfig,
) -> dict[str, int]:
    views = extract_numpy_views(processed_data)
    split_map = {
        "train": views["he_train_idx"],
        "valid": views["he_valid_idx"],
        "test": views["he_test_idx"],
    }
    sample_counts: dict[str, int] = {}
    for split_name, hyperedge_ids in split_map.items():
        samples = [
            build_single_he_task_sample(int(hyperedge_id))
            for hyperedge_id in tqdm(hyperedge_ids, desc=f"Building hyperedge task samples ({split_name})")
        ]
        out_path = output_dir / f"he_task_hg_{config.max_incident_hyperedges}_{config.max_members_per_hyperedge}_{split_name}.jsonl"
        write_jsonl(out_path, samples)
        sample_counts[split_name] = len(samples)
    return sample_counts


def build_meta(
    config: BuildConfig,
    nodes: pd.DataFrame,
    hyperedges: pd.DataFrame,
    split_idx: dict[str, np.ndarray],
    he_train_idx: np.ndarray,
    he_valid_idx: np.ndarray,
    he_test_idx: np.ndarray,
    hyperedge_build_stats: dict[str, int],
    node_sample_counts: dict[str, int],
    he_task_sample_counts: dict[str, int],
) -> dict[str, Any]:
    hyperedge_sizes = hyperedges["hyperedge_size"].to_numpy(dtype=np.int64)
    return {
        "dataset_name": "ogbn-arxiv-hg",
        "canonical_definition": "source-excluded co-citation hyperedge",
        "primary_tasks": ["nc", "hecls", "nd", "hed"],
        "deferred_tasks": [],
        "hecls_label_space": list(ARXIV_LABEL_TEXTS),
        "config": asdict(config),
        "node_stats": {
            "num_nodes": int(len(nodes)),
            "num_classes": int(nodes["label"].nunique()),
            "missing_text_nodes": int((nodes["has_text"] == 0).sum()),
            "year_min": int(nodes["year"].min()),
            "year_max": int(nodes["year"].max()),
            "train_nodes": int(len(split_idx["train"])),
            "valid_nodes": int(len(split_idx["valid"])),
            "test_nodes": int(len(split_idx["test"])),
        },
        "hyperedge_stats": {
            **hyperedge_build_stats,
            "num_hyperedges": int(len(hyperedges)),
            "num_train_hyperedges": int(len(he_train_idx)),
            "num_valid_hyperedges": int(len(he_valid_idx)),
            "num_test_hyperedges": int(len(he_test_idx)),
            "min_size": int(hyperedge_sizes.min()) if len(hyperedge_sizes) else 0,
            "max_size": int(hyperedge_sizes.max()) if len(hyperedge_sizes) else 0,
            "mean_size": float(hyperedge_sizes.mean()) if len(hyperedge_sizes) else 0.0,
        },
        "sample_files": {
            "node_task_samples": node_sample_counts,
            "he_task_samples": he_task_sample_counts,
            "payload_materialization": "runtime_only",
            "runtime_materialized_payloads": [
                "vc_hidt",
                "vc_overview",
                "ec_hidt",
                "ec_overview",
            ],
            "sample_families": {
                "node_task_hg": {
                    "tasks": ["nc", "nd"],
                    "canonical_center_key": "id",
                    "optional_payloads": [],
                    "payload_materialization": "runtime_only",
                    "splits": node_sample_counts,
                },
                "he_task_hg": {
                    "tasks": ["hecls", "hed"],
                    "canonical_center_key": "id",
                    "optional_payloads": [],
                    "payload_materialization": "runtime_only",
                    "splits": he_task_sample_counts,
                },
            },
            "final_hidt_embedded": False,
            "overview_embedded": False,
            "overview_semantic_embedded": False,
        },
        "task_contract": {
            "nc_target": "y[center_node_id]",
            "hecls_target": "y[source_node_id]",
            "hecls_input": "EC-HIDT+O(e)",
            "nc_input": "VC-HIDT+O(v)",
            "nd_target": "templated description(label_text, title)",
            "hed_target": "templated description(y[source_node_id], title[source_node_id])",
            "nd_input": "VC-HIDT+O(v)",
            "hed_input": "EC-HIDT+O(e)",
        },
        "final_hidt": {
            "runtime_compatible": True,
            "formal_hidt_depth": int(config.formal_hidt_depth),
            "max_child_hyperedges": int(config.max_child_hyperedges),
            "max_incident_hyperedges": int(config.max_incident_hyperedges),
            "max_members_per_hyperedge": int(config.max_members_per_hyperedge),
            "contract": build_final_hidt_contract_summary(),
        },
        "hidt_o": {
            "enabled": True,
            "overview_hops": int(config.overview_hops),
            "overview_order_buckets": int(config.overview_order_buckets),
            "suffix_length": int(config.overview_hops * config.overview_order_buckets),
            "primary_task_views": ["open"],
            "exported_fields": [],
            "runtime_materialized_fields": ["vc_overview", "ec_overview"],
        },
    }


def build_ogbn_arxiv_hg(config: BuildConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    graph, labels, split_idx = load_ogbn_arxiv(config.raw_root)
    nodes = build_nodes_frame(
        config.raw_root,
        graph,
        labels,
        titleabs_path=config.titleabs_path,
        allow_missing_text=False,
    )

    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    hyperedges, he_ptr, he_node_index, hyperedge_build_stats = build_canonical_hyperedges(
        nodes,
        edge_index,
        config,
    )
    node_ptr, node_hyperedge_index = build_node_reverse_index(
        int(graph["num_nodes"]),
        hyperedges,
        he_ptr,
        he_node_index,
    )
    he_train_idx, he_valid_idx, he_test_idx = split_hyperedges(hyperedges)

    processed_data = build_processed_data(
        graph=graph,
        nodes=nodes,
        splits=split_idx,
        hyperedges=hyperedges,
        he_ptr=he_ptr,
        he_node_index=he_node_index,
        node_ptr=node_ptr,
        node_hyperedge_index=node_hyperedge_index,
        he_train_idx=he_train_idx,
        he_valid_idx=he_valid_idx,
        he_test_idx=he_test_idx,
    )
    torch.save(processed_data, output_dir / "processed_data.pt")

    node_sample_counts = build_node_samples(
        output_dir,
        processed_data,
        config,
    )
    he_task_sample_counts = build_he_task_samples(
        output_dir,
        processed_data,
        config,
    )

    meta = build_meta(
        config=config,
        nodes=nodes,
        hyperedges=hyperedges,
        split_idx=split_idx,
        he_train_idx=he_train_idx,
        he_valid_idx=he_valid_idx,
        he_test_idx=he_test_idx,
        hyperedge_build_stats=hyperedge_build_stats,
        node_sample_counts=node_sample_counts,
        he_task_sample_counts=he_task_sample_counts,
    )
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    return meta
