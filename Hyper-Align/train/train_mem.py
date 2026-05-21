# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
import warnings

sys.path.append(".")
sys.path.append("./utils")

try:
    from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(
        f"FlashAttention is unavailable ({exc}). Falling back to standard attention, "
        "which is slower and may use more GPU memory."
    )
else:
    replace_llama_attn_with_flash_attn()

from train import _train

if __name__ == "__main__":
    _train()
