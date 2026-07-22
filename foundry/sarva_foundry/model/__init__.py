"""sarva_foundry.model — the from-scratch transformer architecture
(spec §3.6a). Teaching baseline first (this module); frontier-class
extensions (MoE, long-context scaling, native multimodal) build on it.
"""

from sarva_foundry.model.attention import GroupedQueryAttention, repeat_kv
from sarva_foundry.model.layers import RMSNorm, SwiGLU, apply_rope, precompute_rope
from sarva_foundry.model.transformer import (
    DecoderOnlyTransformer,
    TransformerBlock,
    TransformerConfig,
)

__all__ = [
    "DecoderOnlyTransformer",
    "GroupedQueryAttention",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "apply_rope",
    "precompute_rope",
    "repeat_kv",
]
