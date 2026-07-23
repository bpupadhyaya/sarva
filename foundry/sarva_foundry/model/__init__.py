"""sarva_foundry.model — the from-scratch transformer architecture
(spec §3.6a). Teaching baseline first (dense SwiGLU FFN, unscaled RoPE);
frontier-class extensions build on it — Mixture-of-Experts (`moe.py`,
swapped in via `TransformerConfig.moe`) and long-context RoPE scaling
(`RopeScalingConfig`, via `TransformerConfig.rope_scaling`) are the
first two; native multimodal input remains future work.
"""

from sarva_foundry.model.attention import GroupedQueryAttention, repeat_kv
from sarva_foundry.model.layers import (
    RMSNorm,
    RopeScalingConfig,
    SwiGLU,
    apply_rope,
    precompute_rope,
)
from sarva_foundry.model.moe import MoEConfig, MoEFeedForward
from sarva_foundry.model.transformer import (
    DecoderOnlyTransformer,
    TransformerBlock,
    TransformerConfig,
)

__all__ = [
    "DecoderOnlyTransformer",
    "GroupedQueryAttention",
    "MoEConfig",
    "MoEFeedForward",
    "RMSNorm",
    "RopeScalingConfig",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "apply_rope",
    "precompute_rope",
    "repeat_kv",
]
