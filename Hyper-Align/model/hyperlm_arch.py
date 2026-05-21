#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import re

from .hyper_projector import HigherOrderTypedProjector
from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_PAD_ID


def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'htp')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type in ('htp', 'htp_v3'):
        structure_dim = getattr(config, 'hypergraph_structure_dim', None)
        if structure_dim is None:
            raise ValueError("HTP requires config.hypergraph_structure_dim.")
        semantic_dim = getattr(config, 'hypergraph_semantic_dim', int(config.mm_hidden_size) - int(structure_dim))
        return HigherOrderTypedProjector(
            semantic_dim=semantic_dim,
            structure_dim=structure_dim,
            semantic_core_dim=getattr(config, 'htp_semantic_core_dim',
                                     getattr(config, 'htp_v3_semantic_core_dim', 384)),
            structure_sidecar_dim=getattr(config, 'htp_structure_sidecar_dim',
                                         getattr(config, 'htp_v3_structure_sidecar_dim', 64)),
            output_dim=hidden_dim,
            num_layers=getattr(config, 'htp_num_layers',
                               getattr(config, 'htp_v3_num_layers', 1)),
        )
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_dim)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')



class HyperLMMetaModel:

    def __init__(self, config):
        super(HyperLMMetaModel, self).__init__(config)

        if hasattr(config, "mm_hidden_size"):
            self.mm_projector = build_graph_projector(config)


    def initialize_graph_modules(self, model_args, fsdp=None):
        pretrain_mm_mlp_adapter = getattr(model_args, 'pretrain_mm_mlp_adapter', None)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'htp')
        self.config.mm_hidden_size = getattr(model_args, 'mm_hidden_size')
        self.config.hypergraph_structure_dim = getattr(model_args, 'hypergraph_structure_dim', None)
        self.config.hypergraph_semantic_dim = getattr(model_args, 'hypergraph_semantic_dim', None)
        self.config.lambda_ord = getattr(model_args, 'lambda_ord', 0.01)
        self.config.lambda_rel = getattr(model_args, 'lambda_rel', 0.01)
        self.config.htp_semantic_core_dim = getattr(model_args, 'htp_semantic_core_dim', 384)
        self.config.htp_structure_sidecar_dim = getattr(model_args, 'htp_structure_sidecar_dim', 64)
        self.config.htp_num_layers = getattr(model_args, 'htp_num_layers', 1)


        self.mm_projector = build_graph_projector(self.config)
        if hasattr(self.mm_projector, "set_consistency_weights"):
            self.mm_projector.set_consistency_weights(self.config.lambda_ord, self.config.lambda_rel)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class HyperLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def _move_graph_aux_to_device(self, graph_aux, device):
        if graph_aux is None:
            return None
        return {
            key: value.to(device=device) if torch.is_tensor(value) else value
            for key, value in graph_aux.items()
        }

    def _get_projector_dtype_device(self, projector, reference_tensor):
        for param in projector.parameters():
            return param.dtype, param.device
        for buffer in projector.buffers():
            return buffer.dtype, buffer.device
        for module in projector.modules():
            for value in module._parameters.values():
                if value is not None:
                    return value.dtype, value.device
            former_parameters = getattr(module, "_former_parameters", None)
            if former_parameters:
                for value in former_parameters.values():
                    if value is not None:
                        return value.dtype, value.device
            for value in module._buffers.values():
                if value is not None:
                    return value.dtype, value.device
            for value in module.__dict__.values():
                if torch.is_tensor(value):
                    return value.dtype, value.device
        for value in projector.state_dict().values():
            return value.dtype, value.device
        return reference_tensor.dtype, reference_tensor.device

    def encode_graphs(self, graph, graph_emb, graph_aux=None):
        projector = self.get_model().mm_projector
        proj_dtype, proj_device = self._get_projector_dtype_device(projector, graph_emb)
        projector_inputs = graph_emb.to(device=proj_device, dtype=proj_dtype)
        graph_on_projector_device = graph.to(device=proj_device)
        projector_graph_aux = self._move_graph_aux_to_device(graph_aux, proj_device)
        if hasattr(projector, "compute_auxiliary_losses"):
            if getattr(projector, "requires_graph_aux", False):
                graph_features = projector(projector_inputs, graph_aux=projector_graph_aux).projected
            else:
                graph_features = projector(projector_inputs).projected
        else:
            graph_features = projector(projector_inputs)
        graph_features[graph_on_projector_device == DEFAULT_GRAPH_PAD_ID] = 0.
        return graph_features

    def project_graph_features(self, graph, graph_emb, graph_aux=None):
        projector = self.get_model().mm_projector
        proj_dtype, proj_device = self._get_projector_dtype_device(projector, graph_emb)
        projector_inputs = graph_emb.to(device=proj_device, dtype=proj_dtype)
        graph_on_projector_device = graph.to(device=proj_device)
        projector_graph_aux = self._move_graph_aux_to_device(graph_aux, proj_device)
        if hasattr(projector, "compute_auxiliary_losses"):
            if getattr(projector, "requires_graph_aux", False):
                projector_output = projector(projector_inputs, graph_aux=projector_graph_aux)
            else:
                projector_output = projector(projector_inputs)
            graph_features = projector_output.projected
        else:
            projector_output = None
            graph_features = projector(projector_inputs)
        graph_features[graph_on_projector_device == DEFAULT_GRAPH_PAD_ID] = 0.
        return graph_features, projector_output

    def set_consistency_schedule(self, scale):
        projector = self.get_model().mm_projector
        if hasattr(projector, "set_consistency_scale"):
            projector.set_consistency_scale(scale)

    def compute_projector_aux_loss(self, projector_output, graph_aux):
        projector = self.get_model().mm_projector
        if projector_output is None or graph_aux is None or not hasattr(projector, "compute_auxiliary_losses"):
            return None, {}
        aux_results = projector.compute_auxiliary_losses(projector_output, graph_aux)
        aux_loss = aux_results["aux_loss"]
        aux_logs = {
            key: value
            for key, value in aux_results.items()
            if key != "aux_loss"
        }
        return aux_loss, aux_logs

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb, graph_aux=None, return_projector_aux=False
    ):
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            if return_projector_aux:
                return input_ids, attention_mask, past_key_values, None, labels, None
            return input_ids, attention_mask, past_key_values, None, labels

        graph_features, projector_output = self.project_graph_features(graphs, graph_emb, graph_aux=graph_aux)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # Keep graph parameters in the graph for ZeRO-3 when a batch item
                # has no graph token.
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                graph_token_start = graph_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                cur_new_input_embeds.append(cur_graph_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:graph_token_start])
                    cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[graph_token_start+1:]
                cur_graph_idx += 1
                cur_input_ids = cur_input_ids[graph_token_start+1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        if return_projector_aux:
            return None, attention_mask, past_key_values, new_input_embeds, new_labels, projector_output
        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_graph_tokenizer(self, model_args, tokenizer):
        pass
