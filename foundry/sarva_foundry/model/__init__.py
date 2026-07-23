"""sarva_foundry.model — the from-scratch transformer architecture
(spec §3.6a). Teaching baseline first (dense SwiGLU FFN, unscaled RoPE);
all three named frontier-class extensions build on it — Mixture-of-
Experts (`moe.py`, via `TransformerConfig.moe`), long-context RoPE
scaling (`RopeScalingConfig`, via `TransformerConfig.rope_scaling`), and
native multimodal input (`vision.py`'s `VisionEncoder` + `Projector`,
via `DecoderOnlyTransformer.forward_multimodal`).
"""

from sarva_foundry.model.attention import GroupedQueryAttention, repeat_kv
from sarva_foundry.model.kv_cache import KVCache
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
from sarva_foundry.model.vision import PatchEmbed, Projector, VisionEncoder, VisionEncoderConfig

__all__ = [
    "DecoderOnlyTransformer",
    "GroupedQueryAttention",
    "KVCache",
    "MoEConfig",
    "MoEFeedForward",
    "PatchEmbed",
    "Projector",
    "RMSNorm",
    "RopeScalingConfig",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "VisionEncoder",
    "VisionEncoderConfig",
    "apply_rope",
    "precompute_rope",
    "repeat_kv",
]
