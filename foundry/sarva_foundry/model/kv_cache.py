"""KV-cache — real incremental decoding for `DecoderOnlyTransformer`
(spec §3.6f: "inference/serving stack"). Named explicitly as a deferred
gap in `sarva.providers.foundry_provider`'s own module docstring ("no
batching, no KV-cache reuse across calls — one naive forward pass per
generated token"); this closes the KV-cache half of that gap.

The problem this solves: `DecoderOnlyTransformer.forward` recomputes
every position's key/value projections on every call, including
positions already generated in a previous step. Generating N tokens the
naive way costs O(N^2) total attention work — position 500's key/value
gets recomputed 500 times over the course of a 500-token generation.
`KVCache` remembers each layer's key/value projections across calls, so
each new token only computes its own new key/value once and reuses
everything already cached, dropping the *per-step* cost from O(seq_len)
attention-input-size to O(1) new projections (attention itself is still
O(seq_len) per step — that's inherent to full/non-sparse attention, not
something a cache removes).

Batching multiple concurrent requests together (the other named gap) is
real, separate, deferred work — this cache is single-sequence (batch
dimension is preserved in the tensor shapes for API consistency with the
rest of the model, but every caller in this codebase passes batch=1).
"""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    """One pre-allocated `(n_layers, batch, n_kv_heads, max_seq_len,
    head_dim)` buffer per key/value. Pre-allocating to `max_seq_len`
    up front (rather than growing/concatenating on every step) avoids
    the O(seq_len) tensor-reallocation-and-copy that a naive
    `torch.cat`-per-step approach would otherwise reintroduce — the
    exact cost this cache exists to eliminate."""

    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_seq_len = max_seq_len
        self.seq_len = 0  # how many positions are actually filled so far
        shape = (n_layers, batch, n_kv_heads, max_seq_len, head_dim)
        self._k = torch.zeros(shape, device=device, dtype=dtype)
        self._v = torch.zeros(shape, device=device, dtype=dtype)

    def write(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Writes this layer's new `k`/`v` (batch, n_kv_heads, new_len,
        head_dim) at the cache's current position, and returns every
        position filled so far INCLUDING the new one — the full key/value
        set this layer's attention should attend against. Does not itself
        advance `self.seq_len` (every layer in one forward pass writes at
        the same starting position; see `DecoderOnlyTransformer`, which
        advances the cache exactly once per forward call, after every
        layer has written)."""
        new_len = k.shape[2]
        end = self.seq_len + new_len
        if end > self.max_seq_len:
            raise ValueError(
                f"KVCache overflow: writing {new_len} new positions at "
                f"offset {self.seq_len} would exceed max_seq_len {self.max_seq_len}"
            )
        self._k[layer_idx, :, :, self.seq_len : end, :] = k
        self._v[layer_idx, :, :, self.seq_len : end, :] = v
        return self._k[layer_idx, :, :, :end, :], self._v[layer_idx, :, :, :end, :]

    def advance(self, new_len: int) -> None:
        self.seq_len += new_len
