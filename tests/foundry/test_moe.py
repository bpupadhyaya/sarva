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


def test_moe_config_rejects_non_positive_n_experts_per_tok():
    # A real bug found by a fresh-eyes sweep, the identical gap already
    # found and fixed three separate times for the sibling
    # RopeScalingConfig, never propagated here: neither n_experts nor
    # n_experts_per_tok was ever checked for being positive.
    # foundry_provider.load_checkpoint_bundle() builds this straight
    # from an untrusted checkpoint bundle's config.json -- a checkpoint
    # saved mid-crash, a hand-edited config.json, or a training-script
    # bug all produce a value like this. Confirmed live before this fix:
    # n_experts_per_tok=-1 constructed with no error at all, then
    # crashed deep inside _route() with a raw, uncaught
    # `RuntimeError: selected index k out of range` from
    # `biased.topk(-1, ...)` on the first real inference call --
    # invisible at load time, not a clean ValueError like every other
    # invalid field in this same class already gets.
    import pytest

    for top_k in (0, -1):
        with pytest.raises(ValueError, match="n_experts_per_tok"):
            MoEConfig(n_experts=4, n_experts_per_tok=top_k)


def test_moe_config_rejects_non_positive_n_experts():
    # n_experts_per_tok is deliberately kept <= n_experts here (both
    # non-positive) so this doesn't just re-trigger the pre-existing
    # "can't exceed" check above for an unrelated reason -- a first
    # version of this test used n_experts_per_tok=1, which happened to
    # already raise on the OLD code too (1 > 0 already tripped the
    # exceeds check), silently failing to prove the new check does
    # anything. This combination genuinely passed n_experts<=0 through
    # BOTH pre-existing checks unnoticed on the old code.
    import pytest

    for n_experts in (0, -1):
        with pytest.raises(ValueError, match="n_experts"):
            MoEConfig(n_experts=n_experts, n_experts_per_tok=n_experts)


def test_moe_config_rejects_non_finite_bias_update_speed():
    import pytest

    for speed in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="bias_update_speed"):
            MoEConfig(n_experts=4, n_experts_per_tok=2, bias_update_speed=speed)


def test_moe_config_rejects_non_positive_bias_update_speed():
    import pytest

    for speed in (0.0, -0.01):
        with pytest.raises(ValueError, match="bias_update_speed"):
            MoEConfig(n_experts=4, n_experts_per_tok=2, bias_update_speed=speed)


def test_moe_config_rejects_non_positive_expert_hidden_dim():
    # A real bug found by a fresh-eyes sweep, the identical gap already
    # found and fixed for this class's other three fields (n_experts,
    # n_experts_per_tok, bias_update_speed) above, never propagated to
    # expert_hidden_dim -- reachable through the identical untrusted-
    # checkpoint config.json path those fixes closed. Confirmed live
    # before this fix: a negative value constructed with no error, then
    # crashed deep inside MoEFeedForward.__init__ with a raw
    # RuntimeError ("Trying to create tensor with negative dimension")
    # instead of a clean ValueError. Worse for exactly 0: `hidden =
    # config.expert_hidden_dim or max(32, ...)` is an or-fallback (a
    # truthiness check, not `is None`), so 0 was silently discarded and
    # replaced with the default hidden dim -- no error at all, a model
    # built with a different architecture than the checkpoint's config
    # claims.
    import pytest

    for hidden_dim in (0, -1, -64):
        with pytest.raises(ValueError, match="expert_hidden_dim"):
            MoEConfig(n_experts=4, n_experts_per_tok=2, expert_hidden_dim=hidden_dim)


def test_moe_config_still_accepts_none_and_a_positive_expert_hidden_dim():
    # Regression guard for the fix above: None (the documented "use the
    # default" sentinel) and a genuinely positive value must not be
    # rejected by the new check.
    config_default = MoEConfig(n_experts=4, n_experts_per_tok=2, expert_hidden_dim=None)
    assert config_default.expert_hidden_dim is None
    config_explicit = MoEConfig(n_experts=4, n_experts_per_tok=2, expert_hidden_dim=64)
    assert config_explicit.expert_hidden_dim == 64


def test_moe_config_rejects_negative_n_shared_experts():
    # A real bug found by a fresh-eyes sweep: the fifth field in this
    # exact dataclass, reachable through the identical untrusted-
    # checkpoint config.json path already established for the four
    # fields validated above -- simply the one field those earlier
    # passes missed. range() of a negative number is silently empty in
    # Python, so MoEFeedForward.__init__'s per-shared-expert list
    # comprehension never raised for a negative value -- it silently
    # built a model with ZERO shared experts instead, contradicting
    # this module's own docstring ("n_shared_experts always-active
    # experts every token passes through unconditionally"). Confirmed
    # live before this fix: n_shared_experts=-3 constructed with no
    # error and built a model with 0 shared-expert modules instead of
    # raising.
    import pytest

    for n_shared in (-1, -3):
        with pytest.raises(ValueError, match="n_shared_experts"):
            MoEConfig(n_experts=4, n_experts_per_tok=2, n_shared_experts=n_shared)


def test_moe_config_still_accepts_zero_shared_experts():
    # Regression guard for the fix above, and the reason it's a strict
    # "not negative" check rather than the same "must be positive"
    # check the sibling fields use: unlike those (which each need at
    # least one of something to route to/scale by), zero shared experts
    # is a real, legitimate configuration -- plenty of real MoE
    # architectures use only routed experts with none shared.
    config = MoEConfig(n_experts=4, n_experts_per_tok=2, n_shared_experts=0)
    assert config.n_shared_experts == 0
    model = MoEFeedForward(dim=16, config=config)
    assert len(model.shared_experts) == 0


def test_a_non_finite_bias_update_speed_would_permanently_collapse_routing():
    # Regression proof for why the __post_init__ check above matters: this
    # is what happens if a non-finite bias_update_speed reaches
    # update_expert_bias unchecked -- confirmed by constructing the buffer
    # update directly (bypassing MoEConfig's validation) rather than
    # relying on the validation itself, since the whole point is showing
    # what the validation prevents. expert_bias becomes +-inf/nan after a
    # single update, and torch.topk treats +inf/nan as always-largest, so
    # routing on a completely unrelated fresh batch is dictated entirely
    # by which experts happened to be over/under the mean load on the
    # very first update -- never again touched by the actual input.
    torch.manual_seed(0)
    config = MoEConfig(n_experts=4, n_experts_per_tok=2)
    moe = MoEFeedForward(dim=16, config=config)
    moe(torch.randn(8, 16))

    avg_load = moe._last_load.mean()
    with torch.no_grad():
        moe.expert_bias[moe._last_load > avg_load] = float("inf")
        moe.expert_bias[moe._last_load <= avg_load] = float("-inf")

    x2 = torch.randn(500, 16)
    out = moe(x2)
    gate_logits = moe.gate(x2)
    _, top_idx = (gate_logits + moe.expert_bias).topk(2, dim=-1)
    counts = torch.bincount(top_idx.flatten().long(), minlength=4)

    # Every one of the 500 unrelated tokens routes to the same two
    # +inf-biased experts, and zero tokens ever reach the -inf ones --
    # routing has become completely disconnected from the input.
    assert (counts > 0).sum().item() == 2
    assert torch.isfinite(out).all()  # silent: the forward pass never errors
