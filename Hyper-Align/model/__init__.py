import logging

_logger = logging.getLogger(__name__)

from .language_model.hyperlm_llama import HyperLMLlamaForCausalLM, HyperLMConfig

try:
    from .language_model.hyperlm_qwen3 import HyperLMQwen3ForCausalLM, HyperLMQwen3Config
except ImportError:
    HyperLMQwen3ForCausalLM = None
    _logger.debug("Qwen3 wrapper unavailable (transformers too old?); skipping.")

HYPERLM_MODEL_CLASS_MAP = {
    "llama": HyperLMLlamaForCausalLM,
    "mistral": HyperLMLlamaForCausalLM,
    "hyperlm": HyperLMLlamaForCausalLM,
}

if HyperLMQwen3ForCausalLM is not None:
    HYPERLM_MODEL_CLASS_MAP["qwen3"] = HyperLMQwen3ForCausalLM
    HYPERLM_MODEL_CLASS_MAP["hyperlm_qwen3"] = HyperLMQwen3ForCausalLM

HYPERALIGN_MODEL_CLASS_MAP = HYPERLM_MODEL_CLASS_MAP


def get_hyperlm_model_class(model_name_or_path: str):
    """Pick the correct Hyper-Align/HyperLM wrapper class based on config.json model_type."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "llama")
    cls = HYPERLM_MODEL_CLASS_MAP.get(model_type)
    if cls is None:
        raise ValueError(
            f"No Hyper-Align/HyperLM wrapper registered for model_type={model_type!r}. "
            f"Available: {list(HYPERLM_MODEL_CLASS_MAP.keys())}"
        )
    return cls


def get_hyperalign_model_class(model_name_or_path: str):
    """Public-name alias kept compatible with existing HyperLM checkpoints."""
    return get_hyperlm_model_class(model_name_or_path)
