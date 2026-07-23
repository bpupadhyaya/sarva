"""sarva_foundry.model.vision — native multimodal input (spec §3.6a: "a
vision encoder + projector"). The last named piece of §3.6a's
architecture list; Mixture-of-Experts (`moe.py`) and long-context RoPE
scaling (`RopeScalingConfig` in `layers.py`) came first.

Three real, standard pieces of every LLaVA-class vision-language model,
each reusing this project's existing, already-tested substrate rather
than reimplementing it in parallel:

1. **`PatchEmbed`**: splits an image into non-overlapping patches and
   linearly projects each to `dim` — a single strided `nn.Conv2d`
   (kernel == stride == patch_size), the standard "patchify" trick,
   mathematically identical to flattening each patch and applying one
   shared `nn.Linear` (verified directly, not assumed, in
   `test_patch_embed_matches_manual_flatten_and_linear`).
2. **`VisionEncoder`**: patchify, then N *bidirectional* transformer
   blocks. Reuses `GroupedQueryAttention`/`RMSNorm`/`SwiGLU` from
   `attention.py`/`layers.py` with `causal=False` — the same math the
   text decoder uses, just without the causal mask, since an image
   patch needs to see every other patch, not just ones that happen to
   come "earlier" in some arbitrary flatten order.
3. **`Projector`**: a 2-layer MLP with a GELU nonlinearity mapping the
   vision encoder's output dim to the text decoder's `dim` — the
   "connector" every LLaVA-class VLM uses (LLaVA-1.5's own ablation
   found a 2-layer MLP beats a single linear projection).

**Honestly named simplification, not silently assumed equivalent:**
`VisionEncoder` uses 1D RoPE over the flattened patch sequence — the
same positional mechanism the text decoder already has, reused for
consistency and because it's already tested — rather than a 2D-aware
positional scheme (2D RoPE or learned 2D embeddings) a production vision
encoder would use to actually encode row/column structure. Real,
deferred follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.model.attention import GroupedQueryAttention
from sarva_foundry.model.layers import RMSNorm, SwiGLU, default_swiglu_hidden_dim


@dataclass
class VisionEncoderConfig:
    image_size: int
    patch_size: int
    n_channels: int = 3
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4  # full MHA by default: no KV-cache pressure for a one-shot encoder pass
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def n_patches(self) -> int:
        return self.patches_per_side**2

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.n_channels, config.dim, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """`pixel_values`: (batch, channels, H, W) -> (batch, n_patches, dim),
        patches in row-major (raster) order."""
        x = self.proj(pixel_values)  # (batch, dim, H/patch, W/patch)
        return x.flatten(2).transpose(1, 2)  # (batch, n_patches, dim)


class VisionEncoderBlock(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.n_patches,
            causal=False,
        )
        self.mlp_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = SwiGLU(config.dim, default_swiglu_hidden_dim(config.dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.layers = nn.ModuleList([VisionEncoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        x = self.patch_embed(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)  # (batch, n_patches, dim)


class Projector(nn.Module):
    """Maps vision encoder output dim to the text decoder's dim."""

    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or text_dim
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, text_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
