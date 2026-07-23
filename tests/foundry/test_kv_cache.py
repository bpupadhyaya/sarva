"""Conformance tests for sarva_foundry.model.kv_cache / sarva_foundry.
inference.generate_with_cache -- real KV-cached incremental decoding
(spec §3.6f). The bar this has to clear isn't "runs without crashing" --
a subtly wrong cache offset would still produce plausible-looking text,
so the tests prove numerical/token-level equivalence against the known-
correct naive full-recompute path directly, not just shape checks."""

from __future__ import annotations

import pytest
import torch
from sarva_foundry.inference import generate_with_cache
from sarva_foundry.model import DecoderOnlyTransformer, KVCache, TransformerConfig
from sarva_foundry.train.rl import sample_completion


def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=50, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64
    )


def test_kv_cache_write_returns_every_position_filled_so_far():
    cache = KVCache(n_layers=2, batch=1, n_kv_heads=2, max_seq_len=16, head_dim=4)
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    k_full, v_full = cache.write(layer_idx=0, k=k1, v=v1)
    assert k_full.shape == (1, 2, 3, 4)
    assert torch.equal(k_full, k1)
    cache.advance(3)

    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    k_full2, v_full2 = cache.write(layer_idx=0, k=k2, v=v2)
    assert k_full2.shape == (1, 2, 4, 4)
    assert torch.equal(k_full2[:, :, :3, :], k1)
    assert torch.equal(k_full2[:, :, 3:, :], k2)


def test_kv_cache_raises_on_overflow_instead_of_silently_truncating():
    cache = KVCache(n_layers=1, batch=1, n_kv_heads=1, max_seq_len=4, head_dim=2)
    cache.write(layer_idx=0, k=torch.randn(1, 1, 3, 2), v=torch.randn(1, 1, 3, 2))
    cache.advance(3)
    with pytest.raises(ValueError, match="overflow"):
        cache.write(layer_idx=0, k=torch.randn(1, 1, 2, 2), v=torch.randn(1, 1, 2, 2))


def test_forward_with_no_cache_is_unchanged_from_before_the_parameter_existed():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    ids = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        default_call = model(ids)
        explicit_none = model(ids, cache=None)
    assert torch.equal(default_call, explicit_none)


def test_forward_with_cache_matches_full_recompute_for_a_single_prefill_call():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    ids = torch.randint(0, 50, (1, 10))
    cache = KVCache(
        n_layers=model.config.n_layers,
        batch=1,
        n_kv_heads=model.config.n_kv_heads,
        max_seq_len=model.config.max_seq_len,
        head_dim=model.config.head_dim,
    )
    with torch.no_grad():
        no_cache = model(ids)
        cached = model(ids, cache=cache)
    assert torch.allclose(no_cache, cached, atol=1e-5)
    assert cache.seq_len == 10


def test_forward_with_cache_matches_full_recompute_across_incremental_steps():
    # The real proof: prefill a prompt, then feed new tokens ONE AT A TIME
    # through the cache and confirm each step's logits match what a full,
    # from-scratch forward pass over the whole sequence-so-far would give
    # -- not just that prefill alone works.
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    prompt = torch.randint(0, 50, (1, 6))
    extra_tokens = [7, 23, 41]

    cache = KVCache(
        n_layers=model.config.n_layers,
        batch=1,
        n_kv_heads=model.config.n_kv_heads,
        max_seq_len=model.config.max_seq_len,
        head_dim=model.config.head_dim,
    )
    with torch.no_grad():
        cached_logits = model(prompt, cache=cache)[0, -1]
        full_ids = prompt
        full_logits = model(full_ids)[0, -1]
        assert torch.allclose(cached_logits, full_logits, atol=1e-5)

        for tok in extra_tokens:
            full_ids = torch.cat([full_ids, torch.tensor([[tok]])], dim=1)
            cached_logits = model(torch.tensor([[tok]]), cache=cache)[0, -1]
            full_logits = model(full_ids)[0, -1]
            assert torch.allclose(cached_logits, full_logits, atol=1e-4)


def test_generate_with_cache_matches_naive_greedy_generation_token_for_token():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    prompt_ids = [1, 2, 3, 4]

    cached = generate_with_cache(model, prompt_ids, max_new_tokens=8, temperature=0.0)
    naive = sample_completion(model, prompt_ids, max_new_tokens=8, temperature=0.0)

    assert cached == naive


def test_generate_with_cache_stops_at_the_stop_token():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(_tiny_config())
    prompt_ids = [1, 2, 3]
    first_token = generate_with_cache(model, prompt_ids, max_new_tokens=1, temperature=0.0)[0]

    completion = generate_with_cache(
        model, prompt_ids, max_new_tokens=10, temperature=0.0, stop_token_id=first_token
    )
    assert completion == [first_token]


def test_generate_with_cache_raises_a_clear_error_past_max_seq_len():
    # The RoPE table's own bound (same `config.max_seq_len`) is checked
    # before the cache ever gets a chance to overflow -- both are sized
    # from the same config, so in practice this is the error a caller
    # actually sees, not KVCache's own (defensive, otherwise-unreachable
    # through this code path) overflow guard.
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=8
    )
    model = DecoderOnlyTransformer(config)
    prompt_ids = list(range(6))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        generate_with_cache(model, prompt_ids, max_new_tokens=10, temperature=0.0)
