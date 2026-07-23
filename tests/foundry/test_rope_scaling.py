"""Conformance tests for RopeScalingConfig / long-context RoPE scaling
(spec §3.6a: "position-interpolation/NTK scaling"). Same bar as
test_model.py's RoPE tests: verify the actual mathematical properties
that distinguish "linear" from "ntk" scaling, not just that both run
without crashing — a table full of the wrong numbers would still have
the right shape."""

from __future__ import annotations

import torch
from sarva_foundry.model.layers import RopeScalingConfig, apply_rope, precompute_rope
from sarva_foundry.model.transformer import DecoderOnlyTransformer, TransformerConfig

torch.manual_seed(0)

_HEAD_DIM = 16
_THETA = 10000.0


def test_no_scaling_is_bit_identical_to_before_this_feature_existed():
    # Regression guard: scaling=None (the default, and every pre-existing
    # call site) must produce exactly what precompute_rope always did.
    cos_a, sin_a = precompute_rope(_HEAD_DIM, 32, _THETA)
    cos_b, sin_b = precompute_rope(_HEAD_DIM, 32, _THETA, scaling=None)
    assert torch.equal(cos_a, cos_b)
    assert torch.equal(sin_a, sin_b)


def test_linear_scaling_squeezes_positions_by_exactly_the_factor():
    # Chen et al.'s position interpolation: rotation angle at raw table
    # index i*factor in the scaled table must equal the angle at index i
    # in the unscaled table -- positions are literally divided by
    # `factor`, not approximately or up to some tolerance from a
    # different mechanism.
    factor = 4
    cos_base, sin_base = precompute_rope(_HEAD_DIM, 32, _THETA)
    cos_lin, sin_lin = precompute_rope(
        _HEAD_DIM, 32 * factor, _THETA, scaling=RopeScalingConfig("linear", factor)
    )
    for i in range(32):
        assert torch.allclose(cos_lin[i * factor], cos_base[i], atol=1e-5)
        assert torch.allclose(sin_lin[i * factor], sin_base[i], atol=1e-5)


def test_ntk_scaling_leaves_the_highest_frequency_dimension_unchanged():
    # The defining property that distinguishes NTK from linear scaling:
    # the highest-frequency dimension's rotation rate is theta^0 = 1
    # regardless of theta, so NTK scaling (which only changes theta)
    # must leave it bit-for-bit identical to the unscaled table at every
    # position -- unlike linear scaling, which rescales every dimension,
    # including this one, uniformly.
    cos_base, sin_base = precompute_rope(_HEAD_DIM, 64, _THETA)
    cos_ntk, sin_ntk = precompute_rope(
        _HEAD_DIM, 64, _THETA, scaling=RopeScalingConfig("ntk", factor=4.0)
    )
    assert torch.allclose(cos_ntk[:, 0], cos_base[:, 0], atol=1e-6)
    assert torch.allclose(sin_ntk[:, 0], sin_base[:, 0], atol=1e-6)


def test_linear_scaling_does_change_the_highest_frequency_dimension():
    # The direct contrast with the NTK test above -- proves these are
    # two genuinely different techniques, not the same math under two
    # names. At the SAME raw position index (20), linear scaling's
    # position division changes the angle for every dimension, including
    # dimension 0.
    cos_base, _ = precompute_rope(_HEAD_DIM, 64, _THETA)
    cos_lin, _ = precompute_rope(
        _HEAD_DIM, 64, _THETA, scaling=RopeScalingConfig("linear", factor=4.0)
    )
    assert not torch.allclose(cos_lin[20, 0], cos_base[20, 0], atol=1e-4)


def test_ntk_scaling_does_stretch_the_lowest_frequency_dimension():
    # NTK isn't a no-op either -- it just concentrates the change on the
    # long-range (low-frequency) dimensions instead of applying it
    # uniformly.
    last_dim = _HEAD_DIM // 2 - 1
    cos_base, _ = precompute_rope(_HEAD_DIM, 64, _THETA)
    cos_ntk, _ = precompute_rope(
        _HEAD_DIM, 64, _THETA, scaling=RopeScalingConfig("ntk", factor=4.0)
    )
    assert not torch.allclose(cos_ntk[20, last_dim], cos_base[20, last_dim], atol=1e-6)


def test_factor_must_be_positive():
    import pytest

    with pytest.raises(ValueError, match="factor"):
        RopeScalingConfig("linear", factor=0.0)
    with pytest.raises(ValueError, match="factor"):
        RopeScalingConfig("ntk", factor=-1.0)


def test_relative_position_invariance_still_holds_under_scaling():
    # Mirrors test_model.py's test_rope_encodes_relative_not_absolute_position
    # for a scaled table: a rotated dot product must still depend only on
    # the (raw-index) relative offset between two positions, for a fixed
    # scaling config -- scaling changes the rotation RATE, not the
    # fundamental "only relative position matters" property.
    def rotated_dot(cos, sin, q, k, m: int, n: int) -> torch.Tensor:
        q_rot = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        k_rot = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (q_rot * k_rot).sum()

    for scaling in [RopeScalingConfig("linear", 4.0), RopeScalingConfig("ntk", 4.0)]:
        cos, sin = precompute_rope(_HEAD_DIM, 100, _THETA, scaling=scaling)
        q = torch.randn(1, 1, _HEAD_DIM)
        k = torch.randn(1, 1, _HEAD_DIM)
        assert torch.allclose(
            rotated_dot(cos, sin, q, k, 10, 5), rotated_dot(cos, sin, q, k, 50, 45), atol=1e-4
        )


def test_transformer_config_wires_rope_scaling_into_the_attention_layer():
    unscaled_config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    scaled_config = TransformerConfig(
        vocab_size=30,
        dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        rope_scaling=RopeScalingConfig("ntk", factor=4.0),
    )
    unscaled_model = DecoderOnlyTransformer(unscaled_config)
    scaled_model = DecoderOnlyTransformer(scaled_config)

    assert not torch.equal(
        unscaled_model.layers[0].attn.rope_cos, scaled_model.layers[0].attn.rope_cos
    )

    tokens = torch.randint(0, 30, (1, 6))
    unscaled_out = unscaled_model(tokens)
    scaled_out = scaled_model(tokens)
    assert unscaled_out.shape == scaled_out.shape == (1, 6, 30)


def test_scaled_rope_transformer_is_trainable_loss_decreases_on_a_toy_task():
    import torch.nn.functional as F

    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=20,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        rope_scaling=RopeScalingConfig("linear", factor=2.0),
    )
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    seq_len = 8
    x = torch.arange(seq_len).unsqueeze(0) % config.vocab_size
    targets = (x + 1) % config.vocab_size

    losses = []
    for _ in range(50):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5
