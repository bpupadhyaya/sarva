"""Conformance tests for sarva_foundry.model — the from-scratch dense
decoder-only transformer. Definition of done goes beyond shapes: RoPE
must actually encode *relative* position (not absolute), causal masking
must actually prevent attending to future tokens (not just claim to via
an unverified `is_causal=True` flag), and the whole stack must be
trainable end to end (gradients flow, loss decreases on a toy task)."""

from __future__ import annotations

import torch
from sarva_foundry.model import (
    RMSNorm,
    TransformerConfig,
    apply_rope,
    precompute_rope,
    repeat_kv,
)
from sarva_foundry.model.transformer import DecoderOnlyTransformer

torch.manual_seed(0)


def _tiny_config(**overrides) -> TransformerConfig:
    defaults = dict(vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_rmsnorm_normalizes_to_unit_rms_when_weight_is_one():
    norm = RMSNorm(dim=16)
    x = torch.randn(4, 16) * 37.0  # arbitrary scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-3)


def test_rmsnorm_handles_zero_input_without_nan():
    norm = RMSNorm(dim=8)
    out = norm(torch.zeros(2, 8))
    assert torch.isfinite(out).all()


def test_rope_encodes_relative_not_absolute_position():
    # The defining mathematical property of RoPE: the dot product of a
    # rotated query at position m with a rotated key at position n depends
    # only on (m - n), not on m and n individually. Verified directly
    # rather than assumed from a correct-looking implementation.
    head_dim = 8
    cos, sin = precompute_rope(head_dim=head_dim, max_seq_len=100)
    q = torch.randn(1, 1, head_dim)
    k = torch.randn(1, 1, head_dim)

    def rotated_dot(m: int, n: int) -> torch.Tensor:
        q_rot = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        k_rot = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (q_rot * k_rot).sum()

    # Same relative offset (5), different absolute positions -> same dot product.
    assert torch.allclose(rotated_dot(10, 5), rotated_dot(50, 45), atol=1e-4)
    assert torch.allclose(rotated_dot(0, 0), rotated_dot(30, 30), atol=1e-4)
    # Different relative offset -> generically a different dot product.
    assert not torch.allclose(rotated_dot(10, 5), rotated_dot(10, 0), atol=1e-4)


def test_rope_at_zero_relative_offset_matches_unrotated_dot_product():
    head_dim = 8
    cos, sin = precompute_rope(head_dim=head_dim, max_seq_len=10)
    q = torch.randn(1, 1, head_dim)
    k = torch.randn(1, 1, head_dim)
    q_rot = apply_rope(q, cos[3:4], sin[3:4])
    k_rot = apply_rope(k, cos[3:4], sin[3:4])
    assert torch.allclose((q_rot * k_rot).sum(), (q * k).sum(), atol=1e-4)


def test_repeat_kv_duplicates_each_head_contiguously():
    x = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4).float()  # (batch, n_kv_heads=2, seq=3, hd=4)
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (1, 6, 3, 4)
    # Head 0 repeated 3x, then head 1 repeated 3x (grouped-query convention).
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.equal(out[:, 1], x[:, 0])
    assert torch.equal(out[:, 2], x[:, 0])
    assert torch.equal(out[:, 3], x[:, 1])


def test_repeat_kv_is_a_noop_for_n_rep_one():
    x = torch.randn(1, 4, 3, 8)
    assert torch.equal(repeat_kv(x, n_rep=1), x)


def test_forward_pass_shape():
    model = DecoderOnlyTransformer(_tiny_config())
    tokens = torch.randint(0, 50, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 50)


def test_forward_pass_is_deterministic_in_eval_mode():
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    tokens = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        out1 = model(tokens)
        out2 = model(tokens)
    assert torch.equal(out1, out2)


def test_weight_tying_shares_the_same_parameter():
    model = DecoderOnlyTransformer(_tiny_config())
    assert model.lm_head.weight is model.tok_embeddings.weight
    # num_parameters() must not double-count the tied matrix.
    total = model.num_parameters()
    body_only = model.num_parameters(include_embedding=False)
    assert total - body_only == model.tok_embeddings.weight.numel()


def test_causal_masking_prevents_attending_to_future_tokens():
    # The real correctness test for `is_causal=True`: an early position's
    # output must be bit-for-bit unaffected by changing a later token,
    # since if causal masking were silently broken (wrong flag, wrong
    # mask), the model would still produce plausible-looking logits of the
    # right shape while leaking future information — a bug shape checks
    # alone can never catch.
    model = DecoderOnlyTransformer(_tiny_config())
    model.eval()
    tokens = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        out_a = model(tokens)

    perturbed = tokens.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 1) % 50  # change only the last token
    with torch.no_grad():
        out_b = model(perturbed)

    # Every position except the last (which legitimately sees the change)
    # must be untouched by perturbing the final token.
    assert torch.equal(out_a[:, :-1, :], out_b[:, :-1, :])
    assert not torch.equal(out_a[:, -1, :], out_b[:, -1, :])


def test_gqa_config_rejects_indivisible_heads():
    import pytest

    with pytest.raises(ValueError):
        _tiny_config(n_heads=5, n_kv_heads=2)


def test_forward_raises_a_clear_error_past_max_seq_len():
    # Found by running examples/03_train_toy_transformer.py's generation
    # loop, which grows the sequence one token at a time past
    # max_seq_len: without this check, RoPE table slicing silently
    # returns a too-short table and the real error only surfaces several
    # calls later as a confusing broadcast-shape mismatch deep inside
    # apply_rope. This test pins the fix: forward() itself must raise,
    # immediately and clearly, when asked to process a sequence longer
    # than what its RoPE tables were precomputed for.
    import pytest

    model = DecoderOnlyTransformer(_tiny_config(max_seq_len=8))
    tokens = torch.randint(0, 50, (1, 9))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(tokens)


def test_model_is_trainable_loss_decreases_on_a_toy_task():
    # The end-to-end proof: gradients actually flow through RMSNorm, RoPE,
    # GQA, SwiGLU, and the tied embedding/head, and optimization actually
    # reduces loss. A shape bug or an accidental `.detach()`/stop-gradient
    # anywhere in the stack would make this fail (or fail to improve)
    # even though every shape-only test above would still pass.
    torch.manual_seed(0)
    config = _tiny_config(
        vocab_size=20, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # A trivial next-token task: predict token (t + 1) % vocab_size.
    seq_len = 8
    x = torch.arange(seq_len).unsqueeze(0) % config.vocab_size
    targets = (x + 1) % config.vocab_size

    losses = []
    for _ in range(50):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), targets.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5
