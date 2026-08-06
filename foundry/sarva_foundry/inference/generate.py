"""sarva_foundry.inference — real incremental generation, the first
piece of §3.6f's "inference/serving stack" (batched inference + KV-cache
reuse, named explicitly as a deferred gap in
`sarva.providers.foundry_provider`'s own module docstring). Batching
multiple concurrent requests together is the other half of that gap and
stays real, deferred, separate scope — `generate_with_cache` here is
single-sequence, matching every caller in this codebase today.

`sarva_foundry.train.rl.sample_completion` already does greedy/sampled
autoregressive generation for RL rollouts, but recomputes every position
from scratch on every step — fine for the short completions RL rollouts
need, wasteful for a real inference path. This module is the KV-cached
counterpart: functionally equivalent generation (same greedy/temperature
sampling semantics), but each step only computes the ONE new token's
key/value projection instead of recomputing the whole prefix, using
`sarva_foundry.model.kv_cache.KVCache`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sarva_foundry.model.kv_cache import KVCache


def generate_with_cache(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    stop_token_id: int | None = None,
) -> list[int]:
    """Same contract as `sarva_foundry.train.rl.sample_completion` (only
    the NEW tokens are returned, not the prompt; `temperature<=0` is
    greedy/argmax, otherwise multinomial sampling over the temperature-
    scaled softmax) — deliberately kept a drop-in match so callers can
    swap between the two without touching anything but the import, and so
    `tests/foundry/test_kv_cache.py` can compare them token-for-token as
    its correctness proof. `model.config.max_seq_len` bounds the cache's
    allocation, the same limit `sample_completion`/RoPE are bounded by
    elsewhere in this codebase."""
    # A real bug found by a fresh-eyes sweep: this is a genuinely public,
    # documented entry point (used directly in examples/15_kv_cache_
    # inference.py), and its own docstring makes no exception for an
    # empty prompt -- but nothing here ever validated one. An empty
    # `prompt_ids` (e.g. `tokenizer.encode("")` on an empty message, a
    # completely realistic input with zero upstream validation --
    # `sarva.providers.foundry_provider`'s own generate() already
    # documents this exact repro before its own caller-side guard) makes
    # `torch.tensor([prompt_ids])` an empty tensor with no elements to
    # infer a dtype from, so PyTorch defaults it to float32 instead of
    # int64 -- fed straight into `tok_embeddings`, which requires an
    # integer index tensor, crashing with a raw, implementation-leaking
    # RuntimeError ("Expected tensor for argument #1 'indices' to have
    # one of the following scalar types: Long, Int; but got
    # torch.FloatTensor instead"), confirmed live before this fix. Even
    # fixing just the dtype wouldn't make an empty prompt generatable
    # anyway -- prefilling zero positions leaves no "last" logit for
    # `next_logits = logits[0, -1]` to sample from either, an IndexError
    # one line later. `foundry_provider.py`'s own caller already guards
    # against this at its own layer, but that protects only that one
    # caller -- every OTHER caller of this public function (the bundled
    # example script, a test, a future adapter) hit the identical
    # unguarded crash. A clear, actionable error here protects all of
    # them at once, at the one place the real architectural constraint
    # (a KV-cache prefill needs at least one token) actually lives.
    if not prompt_ids:
        raise ValueError(
            "prompt_ids must not be empty -- there's no prefix to prefill the KV cache from"
        )
    # A real bug found by a much later fresh-eyes sweep, the identical
    # gap found and fixed in this function's own drop-in-compatible
    # sibling, `sarva_foundry.train.rl.sample_completion` (see that
    # function's own comment for the full confirmed-live repro):
    # temperature=nan was never rejected. `nan <= 0` is `False` in
    # Python, so a NaN temperature fell through to the sampling branch
    # below instead of the greedy one, and `F.softmax(next_logits /
    # nan, ...)` produces an all-NaN probability tensor that crashes
    # `torch.multinomial` with a raw, implementation-leaking
    # RuntimeError instead of a clear one. `temperature=+inf` is
    # deliberately still allowed -- confirmed live, it produces a
    # valid (if degenerate, uniform-random) sample with no error.
    if math.isnan(temperature):
        raise ValueError(f"temperature must not be NaN, got {temperature}")
    model.eval()
    device = next(model.parameters()).device
    cache = KVCache(
        n_layers=model.config.n_layers,
        batch=1,
        n_kv_heads=model.config.n_kv_heads,
        max_seq_len=model.config.max_seq_len,
        head_dim=model.config.head_dim,
        device=device,
    )

    generated: list[int] = []
    with torch.no_grad():
        # Prefill: every prompt token in one forward call, populating the
        # cache with the prompt's own key/values -- this is where the real
        # savings start, since generation steps 2..N then only ever
        # compute ONE new position each instead of re-doing the whole
        # (growing) prefix every time.
        logits = model(torch.tensor([prompt_ids], device=device), cache=cache)
        next_logits = logits[0, -1]

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = int(next_logits.argmax())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))
            generated.append(next_id)
            if stop_token_id is not None and next_id == stop_token_id:
                break
            logits = model(torch.tensor([[next_id]], device=device), cache=cache)
            next_logits = logits[0, -1]

    return generated
