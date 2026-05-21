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


import json
import logging
import os
import warnings
from contextlib import contextmanager

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from model import *
from model import get_hyperalign_model_class
from utils.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download


def _read_local_model_type(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("model_type")
    except (OSError, json.JSONDecodeError):
        return None


class _ExpectedProjectorInitWarningFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if (
            "Some weights of" in message
            and "were not initialized from the model checkpoint" in message
            and "mm_projector" in message
        ):
            return False
        if "You should probably TRAIN this model on a down-stream task" in message:
            return False
        return True


@contextmanager
def _suppress_expected_projector_init_warnings():
    logger = logging.getLogger("transformers.modeling_utils")
    warning_filter = _ExpectedProjectorInitWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)




_DTYPE_ALIAS = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16,  "float16":  torch.float16,  "half": torch.float16,
    "fp32": torch.float32,  "float32":  torch.float32,  "float": torch.float32,
}


def _resolve_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    if key not in _DTYPE_ALIAS:
        raise ValueError(
            f"Unsupported dtype string: {dtype!r}. Choose from {sorted(_DTYPE_ALIAS)}."
        )
    return _DTYPE_ALIAS[key]


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    cache_dir="../../checkpoint",
    dtype=None,
):
    kwargs = {"device_map": device_map}
    is_local_projector_dir = os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "mm_projector.bin"))
        or os.path.exists(os.path.join(model_path, "non_lora_trainables.bin"))
    )
    model_type = _read_local_model_type(model_path) if os.path.isdir(model_path) else None
    model_name_lower = model_name.lower()
    is_hyperlm_checkpoint = (
        'hyperlm' in model_name_lower
        or 'hyperalign' in model_name_lower
        or model_type in {"hyperlm", "hyperlm_qwen3"}
        or is_local_projector_dir
    )

    resolved_dtype = _resolve_dtype(dtype)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # Hyper-Align checkpoints are trained in bf16; use the caller-provided
        # dtype when available.
        kwargs['torch_dtype'] = resolved_dtype if resolved_dtype is not None else torch.float16

    if is_hyperlm_checkpoint:
        # Load Hyper-Align model. Internal module/class names remain HyperLM for checkpoint compatibility.
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading Hyper-Align from base model...')
            model = HyperLMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional Hyper-Align weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            print('Loading Hyper-Align from base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=True, trust_remote_code=True,
            )
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model_cls = get_hyperalign_model_class(model_base)
            with _suppress_expected_projector_init_warnings():
                model = model_cls.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                    cache_dir=cache_dir, **kwargs,
                )
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                print("Load from local path")
            else:
                from huggingface_hub import hf_hub_download
                model_path_hf = hf_hub_download(repo_id=model_path,  filename='mm_projector.bin')
                mm_projector_weights = torch.load(model_path_hf, map_location='cpu')
                print("Load from huggingface")
            target_dtype = resolved_dtype if resolved_dtype is not None else torch.float16
            mm_projector_weights = {k: v.to(target_dtype) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, trust_remote_code=True,
                )
            model_cls = get_hyperalign_model_class(model_path)
            model = model_cls.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        if model_base is not None:
            from peft import PeftModel
            non_hyperlm_dtype = resolved_dtype if resolved_dtype is not None else torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=non_hyperlm_dtype, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print(f'Convert to {non_hyperlm_dtype}...')
            model.to(non_hyperlm_dtype)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                cache_dir=cache_dir, **kwargs,
            )


    if is_hyperlm_checkpoint:
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        if mm_use_graph_start_end:
            tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
