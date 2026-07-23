"""sarva_foundry.model — the from-scratch transformer architecture
(spec §3.6a). Teaching baseline first (dense SwiGLU FFN); frontier-class
extensions build on it — Mixture-of-Experts (`moe.py`) is the first one,
swapped in via `TransformerConfig.moe`; long-context scaling and native
multimodal input remain future work.
"""

from sarva_foundry.model.attention import GroupedQueryAttention, repeat_kv
from sarva_foundry.model.layers import RMSNorm, SwiGLU, apply_rope, precompute_rope
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
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "apply_rope",
    "precompute_rope",
    "repeat_kv",
]
