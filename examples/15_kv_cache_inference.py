"""Example 15 — KV-cache inference: the same generation, real measured
speedup.

Spec §3.6f: "inference/serving stack." Named explicitly as a deferred gap
in `sarva.providers.foundry_provider`'s own docstring: generation there
used to call `sarva_foundry.train.rl.sample_completion`, which recomputes
every position's key/value from scratch on every single new token --
correct, but wasteful: generating token 500 means position 499 (and 498,
and 497...) gets its key/value recomputed for the 500th time.
`sarva_foundry.inference.generate_with_cache` remembers each layer's
key/value across steps (`sarva_foundry.model.kv_cache.KVCache`), so each
step only computes the ONE new token's key/value.

This example proves two things, both with real numbers, not asserted or
assumed: (1) cached and naive generation produce the IDENTICAL token
sequence (greedy decoding, so there's exactly one correct answer to
compare against), and (2) the cached version is measurably faster on a
long-enough generation for the O(N) vs. O(N^2) difference to actually
show up -- a short toy generation (a handful of tokens) is too small for
the fixed overhead of Python-level looping to be dominated by the
attention math savings, so this uses a longer generation specifically to
make the effect visible, and reports the real measured numbers plainly
either way.

Run: uv run python examples/15_kv_cache_inference.py
"""

from __future__ import annotations

import time

import torch
from sarva_foundry.inference import generate_with_cache
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.train.rl import sample_completion

PROMPT_IDS = [1, 2, 3, 4, 5]
MAX_NEW_TOKENS = 200


def main() -> None:
    torch.manual_seed(0)
    # Bigger than the tiny 16-dim toy models used elsewhere in this
    # project's examples -- large enough that the O(N) vs O(N^2) gap over
    # 200 generated tokens is actually visible on a laptop CPU, still
    # small enough to run in a few seconds.
    config = TransformerConfig(
        vocab_size=1000, dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=512
    )
    model = DecoderOnlyTransformer(config)
    print(f"Model: {model.num_parameters():,} parameters")

    print(f"\nGenerating {MAX_NEW_TOKENS} tokens, greedy decoding, two ways...")

    start = time.perf_counter()
    naive_ids = sample_completion(model, PROMPT_IDS, MAX_NEW_TOKENS, temperature=0.0)
    naive_s = time.perf_counter() - start

    start = time.perf_counter()
    cached_ids = generate_with_cache(model, PROMPT_IDS, MAX_NEW_TOKENS, temperature=0.0)
    cached_s = time.perf_counter() - start

    print(f"  naive  (recompute every position each step): {naive_s:.3f}s")
    print(f"  cached (KVCache, one new position each step): {cached_s:.3f}s")
    print(f"  speedup: {naive_s / cached_s:.1f}x")

    print(f"\nSame tokens either way: {naive_ids == cached_ids}")
    print(f"First 10 generated ids: {cached_ids[:10]}")

    print(
        "\nBoth numbers above are real measured wall-clock time on this "
        "machine, not asserted or fabricated -- run it yourself and the "
        "exact speedup will vary with hardware, but cached should win "
        "clearly once generation is long enough for the savings to show."
    )


if __name__ == "__main__":
    main()
