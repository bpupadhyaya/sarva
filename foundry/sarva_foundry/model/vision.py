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
        # A real bug found by a much later fresh-eyes sweep, on the one
        # field every prior sweep of this exact __post_init__ (five
        # rounds' worth, one per sibling field below) never got to:
        # n_channels feeds straight into PatchEmbed's nn.Conv2d as
        # in_channels, but was never validated for being positive.
        # n_channels=-3 constructs cleanly and only surfaces as a
        # confusing raw RuntimeError ("negative dimension") deep inside
        # nn.Conv2d's weight allocation once PatchEmbed is actually
        # built -- the identical failure shape every sibling fix below
        # already closed for its own field. n_channels=0 is worse:
        # PyTorch's Conv2d silently "succeeds" with a zero-element
        # weight tensor, producing an output with 0 channels instead of
        # `dim` channels, which then crashes several layers deeper
        # inside RMSNorm.forward with an unrelated tensor-size-mismatch
        # RuntimeError that names neither n_channels nor anything a
        # caller could connect back to the real root cause. Confirmed
        # live both ways.
        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {self.n_channels}")
        # A real bug found by a later fresh-eyes sweep, on the very
        # class this __post_init__ was already revisited twice for
        # (n_layers, n_kv_heads, below): image_size/patch_size were only
        # ever checked for divisibility against EACH OTHER, never for
        # being positive. patch_size=0 raises the modulo check's own
        # raw ZeroDivisionError instead of a clean ValueError; a
        # negative patch_size that happens to divide image_size evenly
        # (Python's `%` follows the divisor's sign, so 16 % -4 == 0)
        # passes silently, producing a negative `patches_per_side` that
        # only surfaces as a confusing raw RuntimeError deep inside
        # nn.Conv2d's construction ("negative dimension") once
        # VisionEncoder is actually built. Confirmed live both ways.
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        # A real bug found by a later fresh-eyes sweep, one field over
        # from the identical TransformerConfig.n_heads gap in
        # transformer.py: n_heads is used as a divisor on the very next
        # line, but was never validated for being positive. n_heads=0
        # raised the modulo check's own raw ZeroDivisionError instead of
        # a clean ValueError. Confirmed live.
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        # A real bug found by a much later fresh-eyes sweep, the
        # identical gap just closed one file over in TransformerConfig's
        # own __post_init__: dim is the value n_heads divides *into* on
        # the very next line, but was never checked for being positive
        # itself. dim=0 passes `dim % n_heads == 0` trivially and
        # constructs cleanly with no error anywhere. Confirmed live.
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        # A real bug found by a fresh-eyes sweep: the identical shape
        # round 133 already fixed for TransformerConfig.n_layers, one
        # sibling config class over in this same package -- range() of a
        # non-positive number is silently empty in Python, so
        # VisionEncoder.__init__'s `[VisionEncoderBlock(config) for _ in
        # range(config.n_layers)]` never raised for n_layers <= 0. It
        # silently built an encoder with ZERO transformer blocks instead:
        # forward() still ran, patch-embedding pixels then immediately
        # normalizing them with no attention or MLP pass at all, and
        # returned a plausibly-shaped tensor with no exception anywhere.
        # This class already validates two OTHER fields in this exact
        # __post_init__ (image_size, dim) -- n_layers was simply the one
        # they missed. Confirmed live: VisionEncoderConfig(..., n_layers=0)
        # constructed cleanly and produced a working 0-block encoder.
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        # A real bug found by a later fresh-eyes sweep, the identical gap
        # just closed one field over (n_layers, above) and one file over
        # (TransformerConfig.n_kv_heads): n_kv_heads feeds straight into
        # GroupedQueryAttention the same way it does for the text
        # decoder, and was never validated here either. n_kv_heads=0
        # constructs cleanly, then raises a raw ZeroDivisionError only
        # when VisionEncoder is actually built; a negative n_kv_heads
        # whose magnitude happens to divide n_heads evenly slips past
        # GroupedQueryAttention's own `n_heads % n_kv_heads` check
        # entirely (Python's modulo is defined for negative divisors
        # too), then crashes nn.Linear with a raw "negative dimension"
        # RuntimeError instead.
        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be positive, got {self.n_kv_heads}")

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
