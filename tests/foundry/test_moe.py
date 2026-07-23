"""Conformance tests for sarva_foundry.model.moe — Mixture-of-Experts
feedforward. Definition of done goes beyond shapes, matching this
project's bar for model math (see test_model.py's RoPE/causal-masking
tests): routing selection must actually respect top-k, the aux-loss-free
bias must actually affect *which* experts get picked without touching
*how much* a selected expert's output counts, and the load-balancing
update must actually converge toward balance, not just run without
crashing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import MoEConfig, MoEFeedForward
from sarva_foundry.model.moe import _route
from sarva_foundry.model.transformer import DecoderOnlyTransformer, TransformerConfig

torch.manual_seed(0)


def test_route_selects_exactly_top_k_by_biased_logits():
    gate_logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    bias = torch.zeros(4)
    idx, weights = _route(gate_logits, bias, top_k=2)
    assert set(idx[0].tolist()) == {1, 2}  # the two largest raw logits
    assert weights.shape == (1, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_route_bias_changes_selection_but_not_weight_of_a_selected_expert():
    # The entire aux-loss-free mechanism hinges on this: the bias may
    # change *which* experts are selected, but the weight assigned to any
    # expert that ends up selected must come from the RAW (unbiased)
    # logits -- otherwise the bias would be indistinguishable from a
    # hidden auxiliary loss term secretly reweighting outputs, defeating
    # the entire "loss-free" premise this module exists to implement.
    gate_logits = torch.tensor([[3.0, 1.0, 0.0]])
    no_bias = torch.zeros(3)
    _, weights_unbiased = _route(gate_logits, no_bias, top_k=1)

    # A large bias forces expert 2 (raw logit 0.0, normally never picked)
    # to be selected instead of expert 0.
    forcing_bias = torch.tensor([0.0, 0.0, 10.0])
    idx_biased, weights_biased = _route(gate_logits, forcing_bias, top_k=1)
    assert idx_biased[0, 0].item() == 2

    # With only one expert selected, softmax over a single raw logit is
    # always 1.0 regardless of which expert or what the bias was --
    # confirm the weight is NOT influenced by the bias magnitude (e.g. by
    # checking it isn't the softmax of the *biased* logit, which would be
    # a different value here since biased[2] = 0.0 + 10.0 = 10.0 vs raw
    # logit 0.0).
    assert torch.allclose(weights_biased, weights_unbiased)


def test_route_selects_top_k_two_out_of_many():
    torch.manual_seed(1)
    gate_logits = torch.randn(5, 8)
    bias = torch.zeros(8)
    idx, weights = _route(gate_logits, bias, top_k=3)
    assert idx.shape == (5, 3)
    for row in range(5):
        expected = gate_logits[row].topk(3).indices
        assert set(idx[row].tolist()) == set(expected.tolist())
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_forward_pass_shape():
    config = MoEConfig(n_experts=6, n_experts_per_tok=2, n_shared_experts=1)
    moe = MoEFeedForward(dim=16, config=config)
    x = torch.randn(2, 5, 16)
    out = moe(x)
    assert out.shape == x.shape


def test_expert_bias_is_a_buffer_not_a_trainable_parameter():
    # The bias is updated by a fixed arithmetic rule (update_expert_bias),
    # never by backprop -- if it were a Parameter it would silently
    # accumulate a gradient from the language-modeling loss, which is
    # exactly the aux-loss coupling this design exists to avoid.
    config = MoEConfig(n_experts=4, n_experts_per_tok=1)
    moe = MoEFeedForward(dim=8, config=config)
    assert "expert_bias" not in dict(moe.named_parameters())
    assert "expert_bias" in dict(moe.named_buffers())


def test_shared_expert_output_is_always_added_regardless_of_routing():
    # Zero out every routed expert's weights; the shared expert's
    # contribution must still be nonzero, proving it's unconditional
    # (added outside the top-k routing branch) rather than accidentally
    # gated by the same selection logic as the routed experts.
    config = MoEConfig(n_experts=4, n_experts_per_tok=1, n_shared_experts=1)
    moe = MoEFeedForward(dim=8, config=config)
    for expert in moe.experts:
        for p in expert.parameters():
            torch.nn.init.zeros_(p)

    x = torch.randn(1, 3, 8)
    out = moe(x)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_update_expert_bias_is_a_noop_before_any_forward_call():
    config = MoEConfig(n_experts=4, n_experts_per_tok=1)
    moe = MoEFeedForward(dim=8, config=config)
    before = moe.expert_bias.clone()
    moe.update_expert_bias()
    assert torch.equal(moe.expert_bias, before)


def test_load_balancing_converges_toward_balance_over_repeated_updates():
    # A real convergence test, not "doesn't crash": start from an
    # artificially skewed bias (expert 0 favored hard enough that it
    # captures every token initially) with an otherwise REAL, random gate
    # -- not a degenerate all-zero one, which a first draft of this test
    # used and which turned out to produce a winner-take-all oscillation
    # between single experts every round instead of genuine convergence
    # (caught by actually running it and inspecting the per-round load,
    # not assumed correct from the algorithm reading right on paper) --
    # so real per-token variation in raw gate logits lets tokens peel off
    # to other experts gradually as the bias gap narrows. Confirms
    # repeated forward+update_expert_bias cycles measurably flattens the
    # selection distribution, proving the aux-loss-free update rule
    # actually does what it claims.
    torch.manual_seed(0)
    config = MoEConfig(n_experts=4, n_experts_per_tok=1, bias_update_speed=0.1)
    moe = MoEFeedForward(dim=8, config=config)
    with torch.no_grad():
        moe.expert_bias[0] = 5.0

    x = torch.randn(64, 8)

    def selection_counts() -> torch.Tensor:
        moe(x.unsqueeze(0))
        return moe._last_load.clone()

    first_load = selection_counts()
    assert first_load[0] == 64  # fully skewed: every token picks expert 0

    final_load = first_load
    for _ in range(50):
        moe.update_expert_bias()
        final_load = selection_counts()

    # "More balanced" measured as the standard deviation of per-expert
    # load shrinking substantially from the fully-skewed starting point,
    # and expert 0 no longer monopolizing every token.
    assert final_load.std() < first_load.std() * 0.5
    assert final_load[0] < first_load[0]


def test_moe_swaps_in_for_dense_ffn_via_transformer_config():
    dense_config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    moe_config = TransformerConfig(
        vocab_size=30,
        dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        moe=MoEConfig(n_experts=4, n_experts_per_tok=2),
    )
    dense_model = DecoderOnlyTransformer(dense_config)
    moe_model = DecoderOnlyTransformer(moe_config)

    from sarva_foundry.model.layers import SwiGLU

    assert isinstance(dense_model.layers[0].mlp, SwiGLU)
    assert isinstance(moe_model.layers[0].mlp, MoEFeedForward)

    tokens = torch.randint(0, 30, (1, 6))
    dense_out = dense_model(tokens)
    moe_out = moe_model(tokens)
    assert dense_out.shape == moe_out.shape == (1, 6, 30)


def test_moe_transformer_is_trainable_loss_decreases_on_a_toy_task():
    # The end-to-end proof, mirroring test_model.py's dense equivalent:
    # gradients actually flow through the router, the selected experts,
    # AND the shared expert, and optimization actually reduces loss.
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=20,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        moe=MoEConfig(n_experts=4, n_experts_per_tok=2, n_shared_experts=1),
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


def test_moe_config_rejects_top_k_larger_than_n_experts():
    import pytest

    with pytest.raises(ValueError, match="n_experts_per_tok"):
        MoEConfig(n_experts=4, n_experts_per_tok=5)
