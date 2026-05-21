from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from utils.final_hidt import (
    FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE,
    FINAL_HIDT_PROJECTOR_ROLE_OVERVIEW,
    FINAL_HIDT_PROJECTOR_ROLE_PAD,
    FINAL_HIDT_PROJECTOR_ROLE_VERTEX,
)


@dataclass
class HTPOutput:
    projected: torch.Tensor
    fused_states: torch.Tensor
    type_group_ids: torch.Tensor
    role_ids: torch.Tensor
    semantic_core: torch.Tensor
    structure_sidecar: torch.Tensor


class MaskedSetAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5

    def forward(self, states: torch.Tensor, neighbor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = neighbor_mask.to(dtype=torch.bool)
        query = self.query(states)
        key = self.key(states)
        value = self.value(states)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        has_neighbor = mask.any(dim=-1, keepdim=True)
        attn = torch.where(has_neighbor, attn, torch.zeros_like(attn))
        return torch.matmul(attn, value), has_neighbor.squeeze(-1)


class ResidualUpdate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        states: torch.Tensor,
        messages: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(active_mask.any()):
            return states
        updated = self.norm(states + self.mlp(torch.cat([states, messages], dim=-1)))
        return torch.where(active_mask.unsqueeze(-1), updated, states)


class HyperIncidenceBlock(nn.Module):
    """Vertex-hyperedge bidirectional message passing block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.vertex_to_hyperedge = MaskedSetAttention(dim)
        self.hyperedge_to_vertex = MaskedSetAttention(dim)
        self.hyperedge_update = ResidualUpdate(dim)
        self.vertex_update = ResidualUpdate(dim)

    def forward(
        self,
        states: torch.Tensor,
        role_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        incidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        vertex_mask = (role_ids == FINAL_HIDT_PROJECTOR_ROLE_VERTEX) & valid_mask
        hyperedge_mask = (role_ids == FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE) & valid_mask

        v_to_e_mask = incidence_mask & vertex_mask.unsqueeze(1)
        v_to_e_messages, hyperedge_has_neighbor = self.vertex_to_hyperedge(states, v_to_e_mask)
        states = self.hyperedge_update(states, v_to_e_messages, hyperedge_mask & hyperedge_has_neighbor)

        e_to_v_mask = incidence_mask & hyperedge_mask.unsqueeze(1)
        e_to_v_messages, vertex_has_neighbor = self.hyperedge_to_vertex(states, e_to_v_mask)
        states = self.vertex_update(states, e_to_v_messages, vertex_mask & vertex_has_neighbor)

        return states


class HigherOrderTypedProjector(nn.Module):
    """Semantic-core + structural-sidecar projector with incidence message passing."""

    requires_graph_aux = True

    def __init__(
        self,
        semantic_dim: int,
        structure_dim: int,
        semantic_core_dim: int,
        structure_sidecar_dim: int,
        output_dim: int,
        num_layers: int = 1,
        order_num_classes: int = 4,
        relation_num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.structure_dim = int(structure_dim)
        self.semantic_core_dim = int(semantic_core_dim)
        self.structure_sidecar_dim = int(structure_sidecar_dim)
        self.state_dim = self.semantic_core_dim + self.structure_sidecar_dim
        self.output_dim = int(output_dim)
        self.num_layers = max(1, int(num_layers))

        self.semantic_norm = nn.LayerNorm(self.semantic_dim)
        self.semantic_branch = nn.Linear(self.semantic_dim, self.semantic_core_dim)

        self.structure_norm = nn.LayerNorm(self.structure_dim)
        self.structure_stems = nn.ModuleDict(
            {
                "vertex": nn.Linear(self.structure_dim, self.structure_sidecar_dim),
                "hyperedge": nn.Linear(self.structure_dim, self.structure_sidecar_dim),
                "overview": nn.Linear(self.structure_dim, self.structure_sidecar_dim),
                "pad": nn.Linear(self.structure_dim, self.structure_sidecar_dim),
            }
        )

        self.state_norm = nn.LayerNorm(self.state_dim)
        self.blocks = nn.ModuleList(
            [HyperIncidenceBlock(dim=self.state_dim) for _ in range(self.num_layers)]
        )
        self.output_projector = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.output_dim),
        )
        self.order_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, order_num_classes),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, relation_num_classes),
        )

        self.lambda_ord = 0.01
        self.lambda_rel = 0.01
        self.consistency_scale = 0.0

    def set_consistency_weights(self, lambda_ord: float, lambda_rel: float) -> None:
        self.lambda_ord = float(lambda_ord)
        self.lambda_rel = float(lambda_rel)

    def set_consistency_scale(self, scale: float) -> None:
        self.consistency_scale = float(max(0.0, min(1.0, scale)))

    def _extract_projector_metadata(
        self,
        graph_aux: Dict[str, torch.Tensor] | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if graph_aux is None:
            raise ValueError("HTP requires projector metadata in graph_aux.")

        required_keys = (
            "projector_role_ids",
            "projector_valid_mask",
            "projector_incidence_mask",
        )
        missing = [key for key in required_keys if key not in graph_aux]
        if missing:
            raise KeyError(f"HTP missing projector metadata keys: {missing}")

        role_ids = graph_aux["projector_role_ids"].to(device=device, dtype=torch.long)
        valid_mask = graph_aux["projector_valid_mask"].to(device=device, dtype=torch.bool)
        incidence_mask = graph_aux["projector_incidence_mask"].to(device=device, dtype=torch.bool)
        return role_ids, valid_mask, incidence_mask

    def _role_ids_to_type_groups(self, role_ids: torch.Tensor) -> torch.Tensor:
        type_group_ids = torch.full_like(role_ids, 2)
        type_group_ids[role_ids == FINAL_HIDT_PROJECTOR_ROLE_VERTEX] = 0
        type_group_ids[
            (role_ids == FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE)
            | (role_ids == FINAL_HIDT_PROJECTOR_ROLE_OVERVIEW)
        ] = 1
        return type_group_ids

    def forward(self, graph_emb: torch.Tensor, graph_aux: Dict[str, torch.Tensor] | None = None) -> HTPOutput:
        semantic = graph_emb[..., : self.semantic_dim]
        structure = graph_emb[..., self.semantic_dim :]

        role_ids, valid_mask, incidence_mask = self._extract_projector_metadata(
            graph_aux=graph_aux,
            device=graph_emb.device,
        )

        semantic_core = self.semantic_branch(self.semantic_norm(semantic))
        normalized_structure = self.structure_norm(structure)
        structure_sidecar = semantic_core.new_zeros(
            *semantic_core.shape[:-1],
            self.structure_sidecar_dim,
        )
        for role_id, group_name in (
            (FINAL_HIDT_PROJECTOR_ROLE_VERTEX, "vertex"),
            (FINAL_HIDT_PROJECTOR_ROLE_HYPEREDGE, "hyperedge"),
            (FINAL_HIDT_PROJECTOR_ROLE_OVERVIEW, "overview"),
            (FINAL_HIDT_PROJECTOR_ROLE_PAD, "pad"),
        ):
            group_mask = role_ids == role_id
            if not bool(group_mask.any()):
                continue
            structure_sidecar[group_mask] = self.structure_stems[group_name](normalized_structure[group_mask])

        fused_states = self.state_norm(torch.cat([semantic_core, structure_sidecar], dim=-1))

        for block in self.blocks:
            fused_states = block(
                states=fused_states,
                role_ids=role_ids,
                valid_mask=valid_mask,
                incidence_mask=incidence_mask,
            )

        projected = self.output_projector(fused_states)
        return HTPOutput(
            projected=projected,
            fused_states=fused_states,
            type_group_ids=self._role_ids_to_type_groups(role_ids),
            role_ids=role_ids,
            semantic_core=semantic_core,
            structure_sidecar=structure_sidecar,
        )

    def compute_auxiliary_losses(
        self,
        projector_output: HTPOutput,
        graph_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fused_states = projector_output.fused_states
        device = fused_states.device
        order_targets = graph_aux["order_bucket_ids"].to(device)
        relation_pair_indices = graph_aux["relation_pair_indices"].to(device)
        relation_pair_labels = graph_aux["relation_pair_labels"].to(device)

        order_loss = fused_states.new_zeros(())
        relation_loss = fused_states.new_zeros(())

        valid_order_mask = order_targets >= 0
        if valid_order_mask.any():
            order_logits = self.order_head(fused_states[valid_order_mask])
            order_loss = nn.functional.cross_entropy(order_logits, order_targets[valid_order_mask])

        valid_relation_mask = relation_pair_labels >= 0
        if valid_relation_mask.any():
            pair_indices = relation_pair_indices.clone().clamp(min=0)
            left_indices = pair_indices[..., 0].unsqueeze(-1).expand(-1, -1, self.state_dim)
            right_indices = pair_indices[..., 1].unsqueeze(-1).expand(-1, -1, self.state_dim)
            left_states = torch.gather(fused_states, 1, left_indices)
            right_states = torch.gather(fused_states, 1, right_indices)
            relation_input = torch.cat([left_states, right_states], dim=-1)
            relation_logits = self.relation_head(relation_input[valid_relation_mask])
            relation_loss = nn.functional.cross_entropy(
                relation_logits,
                relation_pair_labels[valid_relation_mask],
            )

        total_aux_loss = self.consistency_scale * (
            self.lambda_ord * order_loss + self.lambda_rel * relation_loss
        )
        return {
            "order_loss": order_loss.detach(),
            "relation_loss": relation_loss.detach(),
            "consistency_scale": fused_states.new_tensor(self.consistency_scale),
            "aux_loss": total_aux_loss,
        }
