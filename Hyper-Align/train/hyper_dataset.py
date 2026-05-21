from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch.utils.data import Dataset

from utils.final_hidt import (
    build_projector_metadata_from_arrays,
    build_training_targets_from_arrays,
)
from utils.hypergraph_features import build_graph_tensors, load_hypergraph_semantic_embeddings
from utils.hypergraph_templates import normalize_hyper_template


class HyperSupervisedDataset(Dataset):
    """Hypergraph-native supervised dataset using prebaked JSONL + overview PT."""

    def __init__(
        self,
        tokenizer,
        data_args,
        preprocess_fn: Callable,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.preprocess_fn = preprocess_fn
        self.hyper_data_root = Path(data_args.hyper_data_root)
        self.processed_data = torch.load(
            self.hyper_data_root / "processed_data.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.node_embeddings, self.hyperedge_embeddings = load_hypergraph_semantic_embeddings(
            hyper_data_root=str(self.hyper_data_root),
            node_embedding_path=data_args.node_embedding_path,
            hyperedge_embedding_path=data_args.hyperedge_embedding_path,
            pretrained_embedding_type=data_args.pretrained_embedding_type,
        )
        self.max_hypergraph_tokens = data_args.max_hypergraph_tokens
        self.max_incident_hyperedges = data_args.max_incident_hyperedges
        self.max_members_per_hyperedge = data_args.max_members_per_hyperedge
        self.max_child_hyperedges = data_args.max_child_hyperedges
        self.formal_hidt_depth = data_args.formal_hidt_depth
        self.projector_incidence_mode = getattr(
            data_args,
            "projector_incidence_mode",
            "sample_real",
        )
        self.max_incidence_pairs = data_args.max_incidence_pairs
        self.max_co_member_pairs = data_args.max_co_member_pairs
        self.max_unrelated_pairs = data_args.max_unrelated_pairs
        self.hyper_template = normalize_hyper_template(data_args.hyper_template or data_args.template)

        mih = self.max_incident_hyperedges
        mmh = self.max_members_per_hyperedge

        self.task_names = [task.strip() for task in data_args.use_task.split("-") if task.strip()]
        declared_tasks = self.processed_data.get("supported_tasks")
        if declared_tasks is not None:
            declared_task_set = {str(task).strip() for task in declared_tasks}
            unsupported = [task for task in self.task_names if task not in declared_task_set]
            if unsupported:
                raise ValueError(
                    f"Dataset at {self.hyper_data_root} supports only {sorted(declared_task_set)}, "
                    f"but got {unsupported}."
                )

        self.samples: List[Dict] = []
        loaded_overview: Dict[str, Dict[int, torch.Tensor]] = {}

        emb_type = data_args.pretrained_embedding_type

        for task_name in self.task_names:
            if task_name == "nc":
                center_kind = "vertex"
                ov_name = str(Path("overview") / emb_type / f"node_{mih}_{mmh}_train.pt")
            elif task_name == "hecls":
                center_kind = "hyperedge"
                ov_name = str(Path("overview") / emb_type / f"he_{mih}_{mmh}_train.pt")
            else:
                raise ValueError(
                    f"Unsupported hypergraph task: {task_name!r}; "
                    "supported tasks are 'nc' and 'hecls'."
                )

            prefix = "node" if center_kind == "vertex" else "he"
            jsonl_name = f"{prefix}_task_{task_name}_hg_{mih}_{mmh}_train_prebaked.jsonl"
            jsonl_path = self._resolve_sample_path(jsonl_name)

            if ov_name not in loaded_overview:
                ov_path = self.hyper_data_root / ov_name
                if ov_path.exists():
                    loaded_overview[ov_name] = torch.load(ov_path, map_location="cpu")
                else:
                    loaded_overview[ov_name] = {}
            ov_dict = loaded_overview[ov_name]

            for row in self._read_jsonl(jsonl_path):
                entity_id = int(row["id"])
                self.samples.append({
                    "id": entity_id,
                    "conversations": row["conversations"],
                    "token_node_ids": torch.tensor(row["token_node_ids"], dtype=torch.long),
                    "token_he_ids": torch.tensor(row["token_he_ids"], dtype=torch.long),
                    "token_type_ids": torch.tensor(row["token_type_ids"], dtype=torch.long),
                    "token_depths": torch.tensor(row["token_depths"], dtype=torch.long),
                    "token_slot_ids": torch.tensor(row["token_slot_ids"], dtype=torch.long),
                    "token_parent_slot_ids": torch.tensor(row["token_parent_slot_ids"], dtype=torch.long),
                    "overview_support_bucket_ids": (
                        torch.tensor(row["overview_support_bucket_ids"], dtype=torch.long)
                        if row.get("overview_support_bucket_ids")
                        else None
                    ),
                    "_overview_semantics": ov_dict.get(entity_id),
                    "_center_kind": center_kind,
                })

        random.shuffle(self.samples)

    def _read_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def _resolve_sample_path(self, filename: str) -> Path:
        for parent in [self.hyper_data_root / "samples", self.hyper_data_root]:
            path = parent / filename
            if path.exists():
                return path
        raise FileNotFoundError(
            "Cannot find the required prebaked hypergraph sample file:\n"
            f"{self.hyper_data_root / 'samples' / filename}\n"
            "Run scripts/prebake_data.py first to generate prebaked data."
        )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self) -> List[int]:
        return [
            sum(len(conv["value"].split()) for conv in row["conversations"])
            + self.max_hypergraph_tokens
            for row in self.samples
        ]

    @property
    def modality_lengths(self) -> List[int]:
        return [
            sum(len(conv["value"].split()) for conv in row["conversations"])
            for row in self.samples
        ]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.samples[index]

        token_node_ids = row["token_node_ids"]
        token_he_ids = row["token_he_ids"]
        token_type_ids = row["token_type_ids"]
        token_depths = row["token_depths"]
        token_slot_ids = row["token_slot_ids"]
        token_parent_slot_ids = row["token_parent_slot_ids"]

        center_kind = row["_center_kind"]
        ov_sbids = row.get("overview_support_bucket_ids")

        graph, graph_emb = build_graph_tensors(
            token_node_ids=token_node_ids,
            token_he_ids=token_he_ids,
            token_type_ids=token_type_ids,
            token_depths=token_depths,
            token_slot_ids=token_slot_ids,
            overview_semantics=row["_overview_semantics"],
            node_embeddings=self.node_embeddings,
            hyperedge_embeddings=self.hyperedge_embeddings,
            processed_data=self.processed_data,
            max_tokens=self.max_hypergraph_tokens,
            overview_support_bucket_ids=ov_sbids,
            hidt_center_kind=center_kind,
            hidt_max_depth=self.formal_hidt_depth,
            hidt_max_incident_hyperedges=self.max_incident_hyperedges,
            hidt_max_members_per_hyperedge=self.max_members_per_hyperedge,
            hidt_max_child_hyperedges=self.max_child_hyperedges,
        )

        # Auxiliary targets only cover HIDT tokens (slot_id >= 0), not overview
        n_hidt = int((token_slot_ids >= 0).sum())
        graph_aux = build_training_targets_from_arrays(
            token_type_ids=token_type_ids[:n_hidt],
            token_slot_ids=token_slot_ids[:n_hidt],
            token_parent_slot_ids=token_parent_slot_ids[:n_hidt],
            token_node_ids=token_node_ids[:n_hidt],
            token_he_ids=token_he_ids[:n_hidt],
            processed_data=self.processed_data,
            max_tokens=self.max_hypergraph_tokens,
            seed=int(row["id"]),
            max_incidence_pairs=self.max_incidence_pairs,
            max_co_member_pairs=self.max_co_member_pairs,
            max_unrelated_pairs=self.max_unrelated_pairs,
        )
        graph_aux.update(
            build_projector_metadata_from_arrays(
                token_type_ids=token_type_ids,
                token_depths=token_depths,
                token_slot_ids=token_slot_ids,
                token_parent_slot_ids=token_parent_slot_ids,
                token_node_ids=token_node_ids,
                token_he_ids=token_he_ids,
                processed_data=self.processed_data,
                max_tokens=self.max_hypergraph_tokens,
                incidence_mode=self.projector_incidence_mode,
            )
        )
        graph_aux = {k: v.unsqueeze(0) for k, v in graph_aux.items()}

        conversations = row["conversations"]
        sources = copy.deepcopy([conversations])
        data_dict = self.preprocess_fn(
            sources,
            self.tokenizer,
            has_graph=True,
        )
        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "graph": graph,
            "graph_emb": graph_emb,
            "graph_aux": graph_aux,
        }
