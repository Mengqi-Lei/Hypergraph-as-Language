from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from utils.constants import GRAPH_TOKEN_INDEX
from utils.conversation import conv_templates
from utils.final_hidt import build_projector_metadata_from_arrays
from utils.hypergraph_features import build_graph_tensors
from utils.utils import tokenizer_graph_token


class HypergraphEvalDataset(Dataset):
    """Build hypergraph evaluation tensors from prebaked parallel-array rows."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        task: str,
        tokenizer,
        conv_mode: str,
        processed_data: Dict[str, Any],
        node_embeddings: torch.Tensor,
        hyperedge_embeddings: torch.Tensor,
        overview_semantics: Dict[int, torch.Tensor],
        max_hypergraph_tokens: int,
        projector_incidence_mode: str = "sample_real",
        hidt_center_kind: str = "vertex",
        hidt_max_depth: int = 3,
        hidt_max_incident_hyperedges: int = 8,
        hidt_max_members_per_hyperedge: int = 8,
        hidt_max_child_hyperedges: int = 1,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.task = task
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.processed_data = processed_data
        self.node_embeddings = node_embeddings
        self.hyperedge_embeddings = hyperedge_embeddings
        self.overview_semantics = overview_semantics or {}
        self.max_hypergraph_tokens = max_hypergraph_tokens
        self.projector_incidence_mode = projector_incidence_mode
        self.hidt_center_kind = hidt_center_kind
        self.hidt_max_depth = hidt_max_depth
        self.hidt_max_incident_hyperedges = hidt_max_incident_hyperedges
        self.hidt_max_members_per_hyperedge = hidt_max_members_per_hyperedge
        self.hidt_max_child_hyperedges = hidt_max_child_hyperedges

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        conversations = row["conversations"]
        prompt_text = conversations[0]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_graph_token(
            prompt,
            self.tokenizer,
            GRAPH_TOKEN_INDEX,
            return_tensors="pt",
        )

        token_node_ids = torch.tensor(row["token_node_ids"], dtype=torch.long)
        token_he_ids = torch.tensor(row["token_he_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(row["token_type_ids"], dtype=torch.long)
        token_depths = torch.tensor(row["token_depths"], dtype=torch.long)
        token_slot_ids = torch.tensor(row["token_slot_ids"], dtype=torch.long)
        token_parent_slot_ids = torch.tensor(row["token_parent_slot_ids"], dtype=torch.long)

        ov_sem = self.overview_semantics.get(int(row["id"]))
        ov_sbids_raw = row.get("overview_support_bucket_ids")
        ov_sbids = torch.tensor(ov_sbids_raw, dtype=torch.long) if ov_sbids_raw else None

        graph, graph_emb = build_graph_tensors(
            token_node_ids=token_node_ids,
            token_he_ids=token_he_ids,
            token_type_ids=token_type_ids,
            token_depths=token_depths,
            token_slot_ids=token_slot_ids,
            overview_semantics=ov_sem,
            node_embeddings=self.node_embeddings,
            hyperedge_embeddings=self.hyperedge_embeddings,
            processed_data=self.processed_data,
            max_tokens=self.max_hypergraph_tokens,
            overview_support_bucket_ids=ov_sbids,
            hidt_center_kind=self.hidt_center_kind,
            hidt_max_depth=self.hidt_max_depth,
            hidt_max_incident_hyperedges=self.hidt_max_incident_hyperedges,
            hidt_max_members_per_hyperedge=self.hidt_max_members_per_hyperedge,
            hidt_max_child_hyperedges=self.hidt_max_child_hyperedges,
        )
        graph_aux = build_projector_metadata_from_arrays(
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

        return {
            "question_id": int(row["id"]),
            "prompt": prompt_text,
            "gt": conversations[1]["value"],
            "input_ids": input_ids,
            "graph": graph,
            "graph_emb": graph_emb,
            "graph_aux": graph_aux,
        }


class HypergraphEvalCollator:
    """Collate hypergraph eval samples into a fixed-shape batch."""

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not instances:
            raise ValueError("HypergraphEvalCollator received an empty batch.")

        input_lengths = {instance["input_ids"].shape[0] for instance in instances}
        if len(input_lengths) != 1:
            raise ValueError(
                "Batched hypergraph evaluation currently requires identical prompt lengths. "
                "Please keep one task per run or set --eval-batch-size 1."
            )

        return {
            "question_ids": [instance["question_id"] for instance in instances],
            "prompts": [instance["prompt"] for instance in instances],
            "gts": [instance["gt"] for instance in instances],
            "input_ids": torch.stack([instance["input_ids"] for instance in instances], dim=0),
            "attention_mask": torch.ones(
                len(instances),
                next(iter(input_lengths)),
                dtype=torch.long,
            ),
            "graph": torch.cat([instance["graph"] for instance in instances], dim=0),
            "graph_emb": torch.cat([instance["graph_emb"] for instance in instances], dim=0),
            "graph_aux": {
                key: torch.stack([instance["graph_aux"][key] for instance in instances], dim=0)
                for key in instances[0]["graph_aux"]
            },
        }
