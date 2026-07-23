"""Grouped-query self-attention with RoPE (spec §3.6a).

GQA (Ainslie et al. 2023) is the middle ground between multi-head
attention (one KV head per query head — expensive KV cache) and
multi-query attention (one shared KV head — cheap but quality-degrading):
query heads are split into groups that each share one KV head, which is
what every current frontier-class open model (LLaMA 3, Qwen, Mistral)
actually ships for exactly this cache/quality tradeoff.

`causal` defaults to `True` (the decoder-only text path, unchanged from
before this parameter existed) — `causal=False` is what
`vision.py`'s bidirectional encoder blocks use: an image patch needs to
see every other patch, not just earlier ones in some arbitrary
flatten order, since "earlier" isn't a meaningful notion for spatial
patches the way it is for a left-to-right token sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sarva_foundry.model.kv_cache import KVCache
from sarva_foundry.model.layers import RopeScalingConfig, apply_rope, precompute_rope


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
        rope_scaling: RopeScalingConfig | None = None,
        causal: bool = True,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.causal = causal

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        cos, sin = precompute_rope(head_dim, max_seq_len, rope_theta, scaling=rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer_idx: int = 0) -> Tensor:
        batch, seq_len, _ = x.shape
        # `start_pos` is where in the full sequence this call's tokens
        # actually sit — 0 for a plain (uncached) call or a cache's first
        # (prefill) call, `cache.seq_len` for every subsequent incremental
        # call. RoPE is a *relative*-position encoding (see layers.py), so
        # generating token 500 must use position 500's rotation even
        # though `x` here is a single new token at sequence index 0 — the
        # bug this would otherwise cause (every generated token silently
        # rotated as if it were position 0) would still produce
        # plausible-looking, syntactically valid text, making it exactly
        # the kind of error a shape-only test can't catch.
        start_pos = cache.seq_len if cache is not None else 0
        if start_pos + seq_len > self.rope_cos.shape[0]:
            # Without this check, slicing rope_cos/rope_sin past their
            # length silently returns a shorter table than seq_len instead
            # of raising, and the real error only surfaces several calls
            # later as a confusing broadcast-shape mismatch inside
            # apply_rope — caught by actually running example 03 with a
            # generation loop that grows past max_seq_len, not by the
            # shape-only unit tests, which all used a fixed sequence length.
            raise ValueError(
                f"sequence length {start_pos + seq_len} exceeds max_seq_len "
                f"{self.rope_cos.shape[0]} the RoPE tables were precomputed for"
            )

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[start_pos : start_pos + seq_len]
        sin = self.rope_sin[start_pos : start_pos + seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            # Every position already generated has its key/value sitting
            # in the cache from a previous call -- only this call's NEW
            # position(s) need a fresh key/value projection at all.
            k, v = cache.write(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # The fused attention kernel itself (QK^T, softmax, masking, the V
        # weighting) is PyTorch/CUDA substrate, not model logic — see
        # layers.py's module docstring for where the "from scratch" line
        # is drawn. Without a cache, q_len == k_len always (start_pos=0),
        # so SDPA's own `is_causal=True` is the ordinary, well-known
        # "position i attends to positions <= i" mask (verified directly
        # in tests/foundry/test_model.py). WITH a cache, q_len can be
        # SHORTER than k_len (only the new tokens vs. every cached
        # position) — `is_causal=True` does NOT handle that the way an
        # offset causal mask would (confirmed empirically while building
        # this, not assumed from the docs: it produced visibly wrong
        # logits, caught by `test_kv_cache.py` comparing cached generation
        # against plain full-recompute generation token-for-token). The
        # correct mask for row i (of the L new query positions, absolute
        # position start_pos+i) is "attend to every key at absolute
        # position <= start_pos+i" — built explicitly below via
        # `tril(diagonal=start_pos)` rather than relying on `is_causal`'s
        # own (non-offset) alignment assumption. This also subsumes the
        # no-cache case exactly: start_pos=0, q_len==k_len reduces to the
        # ordinary causal mask.
        if self.causal:
            total_len = k.shape[2]
            mask = x.new_ones((seq_len, total_len), dtype=torch.bool).tril(diagonal=start_pos)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # `self.causal=False` (vision.py's bidirectional encoder) lets
            # every position attend to every other position — unaffected
            # by any of the above, since vision never uses a cache.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)
