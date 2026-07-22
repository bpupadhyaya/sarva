"""Grouped-query causal self-attention with RoPE (spec §3.6a).

GQA (Ainslie et al. 2023) is the middle ground between multi-head
attention (one KV head per query head — expensive KV cache) and
multi-query attention (one shared KV head — cheap but quality-degrading):
query heads are split into groups that each share one KV head, which is
what every current frontier-class open model (LLaMA 3, Qwen, Mistral)
actually ships for exactly this cache/quality tradeoff.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.model.layers import apply_rope, precompute_rope


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand `x`'s KV-head dimension by `n_rep` so grouped query heads can
    attend against a KV tensor shaped as if every query head had its own
    KV head. `x`: (batch, n_kv_heads, seq_len, head_dim)."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        cos, sin = precompute_rope(head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        if seq_len > self.rope_cos.shape[0]:
            # Without this check, slicing rope_cos/rope_sin past their
            # length silently returns a shorter table than seq_len instead
            # of raising, and the real error only surfaces several calls
            # later as a confusing broadcast-shape mismatch inside
            # apply_rope — caught by actually running example 03 with a
            # generation loop that grows past max_seq_len, not by the
            # shape-only unit tests, which all used a fixed sequence length.
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len "
                f"{self.rope_cos.shape[0]} the RoPE tables were precomputed for"
            )

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # The fused attention kernel itself (QK^T, softmax, causal mask,
        # the V weighting) is PyTorch/CUDA substrate, not model logic — see
        # layers.py's module docstring for where the "from scratch" line is
        # drawn. `is_causal=True` is what makes this a decoder: position i
        # can only attend to positions <= i (verified directly in
        # tests/foundry/test_model.py, not just assumed from the flag).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)
