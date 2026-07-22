"""Building blocks for the teaching-baseline dense decoder (spec §3.6a):
RMSNorm, RoPE, and a SwiGLU feedforward — each written from the underlying
math, not imported from `transformers`.

Where this stops being "from scratch": `torch.nn.Linear`/`nn.Embedding`
and `F.scaled_dot_product_attention` (used in `attention.py`) are treated
as commodity substrate, the same tier as `torch.matmul` — Sarva's "no
black boxes" principle (design doc §2.9) draws the line at PyTorch/CUDA
itself, not at every tensor op built on top of it. The actual model
*math* — how RMSNorm normalizes, how RoPE rotates, how GQA groups heads,
how the residual stream is composed — is ours and is what these modules
implement directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich 2019), as used by
    LLaMA/Mistral/Qwen-class models in place of LayerNorm: normalizes by
    RMS only (no mean-centering, no bias), which is cheaper and empirically
    just as effective for transformer residual streams.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Upcast to float32 for the variance computation regardless of the
        # input's dtype — standard practice (matches LLaMA's RMSNorm) since
        # the sum-of-squares can lose precision in bf16/fp16.
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Precompute the cos/sin tables for rotary position embeddings
    (Su et al. 2021), "rotate-half" convention (GPT-NeoX/LLaMA-style):
    each pair of dimensions `(i, i + head_dim/2)` rotates together at a
    frequency that decreases geometrically across the head dimension, so
    nearby positions get high-frequency (fast-changing) rotation and
    distant positions get low-frequency rotation — this is what encodes
    relative position directly into the attention dot product without any
    learned positional parameters.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to `x` (..., seq_len, head_dim).
    `cos`/`sin` must already be sliced to `x`'s sequence length and be
    broadcastable against `x`'s leading dims (see `precompute_rope`)."""
    return x * cos + _rotate_half(x) * sin


def default_swiglu_hidden_dim(dim: int, multiple_of: int = 256) -> int:
    """SwiGLU has three weight matrices where a ReLU MLP has two, so a
    literal `4 * dim` hidden size would cost ~50% more parameters for the
    same FFN. Scaling by 2/3 keeps the parameter count roughly matched to
    a standard 4x-ReLU MLP, then rounding up to `multiple_of` keeps the
    dimension hardware-friendly — the same convention LLaMA uses."""
    hidden = int(2 * (4 * dim) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """Gated feedforward with a SiLU-activated gate (Shazeer 2020): the
    gate branch (`w1`) modulates the value branch (`w3`) elementwise before
    the down-projection (`w2`). Outperforms a plain ReLU MLP in every
    LLaMA-class ablation since its introduction, at the parameter cost
    `default_swiglu_hidden_dim` accounts for."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down-projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # value

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
