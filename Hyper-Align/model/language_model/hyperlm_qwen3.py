from typing import List, Optional, Tuple, Union

import os
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..hyperlm_arch import HyperLMMetaModel, HyperLMMetaForCausalLM
from utils.constants import IGNORE_INDEX

# Optional: liger-kernel fused linear cross-entropy
_LIGER_FLCE_CLS = None
_LIGER_FLCE_REASON = ""
if os.environ.get("HYPERLM_DISABLE_LIGER_FLCE", "0") == "1":
    _LIGER_FLCE_REASON = "DISABLED via HYPERLM_DISABLE_LIGER_FLCE=1"
else:
    try:
        from liger_kernel.transformers import (
            LigerFusedLinearCrossEntropyLoss as _LIGER_FLCE_CLS,
        )
        _LIGER_FLCE_REASON = "ENABLED (liger-kernel fused linear CE)"
    except ImportError as _e:
        _LIGER_FLCE_CLS = None
        _LIGER_FLCE_REASON = f"DISABLED (liger-kernel not importable: {_e})"
        warnings.warn(
            "[hyperlm_qwen3] liger-kernel unavailable; falling back to native "
            "CrossEntropyLoss. This will use significantly more peak GPU memory "
            "for long sequences. Install with: pip install liger-kernel"
        )

if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    print(f"[hyperlm_qwen3] fused-linear-CE: {_LIGER_FLCE_REASON}", flush=True)


class HyperLMQwen3Config(Qwen3Config):
    model_type = "hyperlm_qwen3"


class HyperLMQwen3Model(HyperLMMetaModel, Qwen3Model):
    config_class = HyperLMQwen3Config

    def __init__(self, config: Qwen3Config):
        super(HyperLMQwen3Model, self).__init__(config)


class HyperLMQwen3ForCausalLM(Qwen3ForCausalLM, HyperLMMetaForCausalLM):
    config_class = HyperLMQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = HyperLMQwen3Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        graph_aux: Optional[dict] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, projector_output = (
            self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                graph,
                graph_emb,
                graph_aux=graph_aux,
                return_projector_aux=True,
            )
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss = None
        logits = None

        if labels is not None and _LIGER_FLCE_CLS is not None:
            # The fused CE path avoids materializing the full (B, L, V) logits tensor.
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_hidden = shift_hidden.view(-1, hidden_states.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_hidden.device)
            loss_fct = _LIGER_FLCE_CLS(ignore_index=IGNORE_INDEX)
            loss = loss_fct(self.lm_head.weight, shift_hidden, shift_labels)
        elif labels is not None:
            logits = self.lm_head(hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            logits = self.lm_head(hidden_states)

        if loss is not None:
            aux_loss, aux_logs = self.compute_projector_aux_loss(projector_output, graph_aux)
            if aux_loss is not None:
                loss = loss + aux_loss.to(loss.device)
                self.latest_aux_logs = {key: value.detach().float().cpu() for key, value in aux_logs.items()}
            else:
                self.latest_aux_logs = {}

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
                "graph_aux": kwargs.get("graph_aux", None),
            }
        )
        return model_inputs


AutoConfig.register("hyperlm_qwen3", HyperLMQwen3Config)
AutoModelForCausalLM.register(HyperLMQwen3Config, HyperLMQwen3ForCausalLM)
