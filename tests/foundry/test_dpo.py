"""Conformance tests for sarva_foundry.train.dpo — Direct Preference
Optimization (spec §3.6e's second post-training step). The math has a
known, exact numeric fixed point (loss == ln(2) when policy == reference
— not approximately, exactly, straight from the DPO paper's derivation),
which is a much stronger test than "loss is finite" or "loss decreases":
it pins the loss formula itself, not just its general shape."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from sarva_foundry.data import DOCUMENT_SEPARATOR
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train.dpo import DPOExample, build_dpo_batch, dpo_loss, sequence_logprobs
from sarva_foundry.train.trainer import Trainer

torch.manual_seed(0)


@pytest.fixture
def tokenizer() -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(
        ["what color is the sky blue green", "four two plus response"],
        vocab_size=280,
        special_tokens=[DOCUMENT_SEPARATOR],
    )
    return tok


def _tiny_config(vocab_size: int) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32
    )


def test_dpo_loss_at_zero_logratio_difference_is_a_fixed_point():
    # dpo_loss's own contract, tested in isolation from any model: when
    # the policy-vs-reference log-ratio for chosen exactly equals that
    # for rejected, logits == 0 and loss == -log(sigmoid(0)) == ln(2)
    # exactly -- a direct algebraic check of the formula, no training
    # involved.
    zero = torch.zeros(3)
    loss = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_dpo_loss_rewards_a_larger_chosen_over_rejected_margin():
    # Increasing the policy's preference for chosen over rejected
    # (relative to the reference) must strictly decrease the loss --
    # the direction DPO is supposed to optimize in.
    policy_chosen = torch.tensor([0.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    small_margin_rejected = torch.tensor([-1.0])
    large_margin_rejected = torch.tensor([-5.0])

    small_margin_loss = dpo_loss(policy_chosen, small_margin_rejected, ref_chosen, ref_rejected)
    large_margin_loss = dpo_loss(policy_chosen, large_margin_rejected, ref_chosen, ref_rejected)
    assert large_margin_loss.item() < small_margin_loss.item()


def test_dpo_loss_rejects_a_non_finite_or_non_positive_beta():
    # A real bug found by a fresh-eyes sweep, the same severity class as
    # TrainerConfig.grad_clip/WarmupCosineSchedule.peak_lr's own fixes:
    # beta was never validated, and nothing in dpo_loss's own math
    # guards its sign. Confirmed live with a real model and real DPO
    # training steps: beta=-0.1 took a real (chosen - rejected)
    # log-probability margin from +5.07 to -6.30 over 10 real dpo_step
    # calls -- the policy was actively trained to PREFER the rejected
    # response, the exact opposite of what DPO exists to do.
    zero = torch.zeros(3)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(zero, zero, zero, zero, beta=-0.1)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(zero, zero, zero, zero, beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(zero, zero, zero, zero, beta=float("nan"))


def test_sequence_logprobs_only_sums_masked_positions(tokenizer):
    config = _tiny_config(tokenizer.vocab_size)
    model = DecoderOnlyTransformer(config)
    x = torch.randint(0, tokenizer.vocab_size, (1, 6))
    y = torch.randint(0, tokenizer.vocab_size, (1, 6))
    all_masked = torch.ones(1, 6)
    none_masked = torch.zeros(1, 6)

    lp_all = sequence_logprobs(model, x, y, all_masked)
    lp_none = sequence_logprobs(model, x, y, none_masked)
    assert lp_none.item() == 0.0
    assert lp_all.item() != 0.0


def test_build_dpo_batch_reuses_sft_batch_shapes(tokenizer):
    examples = [DPOExample(prompt="what color is the sky? ", chosen="blue", rejected="green")]
    chosen, rejected = build_dpo_batch(examples, tokenizer)
    for x, y, mask in (chosen, rejected):
        assert x.shape == y.shape == mask.shape
        assert x.shape[0] == 1


def test_build_dpo_batch_rejects_an_empty_example_list_with_a_clear_message(tokenizer):
    # build_dpo_batch just calls build_sft_batch twice, so it inherits
    # the same real bug found by calling build_dpo_batch([], ...): a
    # bare "ValueError: max() iterable argument is empty" instead of a
    # message actually naming the problem (no examples at all).
    with pytest.raises(ValueError, match="requires at least one example"):
        build_dpo_batch([], tokenizer)


def test_dpo_step_initial_loss_is_exactly_ln2_when_policy_equals_reference(tokenizer):
    # The strongest possible correctness check on the FULL dpo_step path
    # (real model forward passes, not the isolated-tensor test above):
    # a policy that's a fresh, untrained copy of its own reference model
    # must produce a loss of exactly ln(2) on the very first step, since
    # pi_logratios == ref_logratios identically when the two models are
    # identical.
    config = _tiny_config(tokenizer.vocab_size)
    model = DecoderOnlyTransformer(config)
    ref_model = copy.deepcopy(model)

    examples = [DPOExample(prompt="what color is the sky? ", chosen="blue", rejected="green")]
    chosen, rejected = build_dpo_batch(examples, tokenizer)

    trainer = Trainer(model)
    loss = trainer.dpo_step(ref_model, chosen, rejected, beta=0.1)
    assert loss == pytest.approx(math.log(2), abs=1e-4)


def test_dpo_step_never_puts_a_gradient_on_the_reference_model(tokenizer):
    config = _tiny_config(tokenizer.vocab_size)
    model = DecoderOnlyTransformer(config)
    ref_model = copy.deepcopy(model)

    examples = [DPOExample(prompt="what color is the sky? ", chosen="blue", rejected="green")]
    chosen, rejected = build_dpo_batch(examples, tokenizer)

    trainer = Trainer(model)
    trainer.dpo_step(ref_model, chosen, rejected)

    assert all(p.grad is None for p in ref_model.parameters())
    assert any(p.grad is not None for p in model.parameters())


def test_dpo_step_applies_the_configured_schedule_like_train_step_does(tokenizer):
    # A real bug found by a fresh-eyes sweep: dpo_step never applied
    # self.config.schedule at all, unlike its two structural siblings
    # in the same Trainer class (train_step, grpo_step), both of which
    # set the optimizer's LR from schedule.lr_at(self.step) right at the
    # top of the method. TrainerConfig.schedule is framed as a
    # Trainer-level opt-in with no indication of being scoped to
    # specific training modes -- confirmed live before this fix: with a
    # WarmupCosineSchedule configured, the optimizer's LR stayed frozen
    # at the initial config.lr across every dpo_step call regardless of
    # the advancing step counter, while train_step on an identical
    # config tracked the schedule exactly.
    from sarva_foundry.train import TrainerConfig, WarmupCosineSchedule

    config = _tiny_config(tokenizer.vocab_size)
    model = DecoderOnlyTransformer(config)
    ref_model = copy.deepcopy(model)

    examples = [DPOExample(prompt="what color is the sky? ", chosen="blue", rejected="green")]
    chosen, rejected = build_dpo_batch(examples, tokenizer)

    schedule = WarmupCosineSchedule(peak_lr=1e-2, min_lr=1e-4, warmup_steps=4, total_steps=20)
    trainer = Trainer(model, TrainerConfig(schedule=schedule))

    lrs = []
    for _ in range(6):
        trainer.dpo_step(ref_model, chosen, rejected)
        lrs.append(trainer.optimizer.param_groups[0]["lr"])

    # Steps 0-3 are warmup (ramping up), step 4+ is past warmup_steps=4
    # and should be decaying -- confirms dpo_step is actually pulling a
    # fresh LR from the schedule every call, not leaving it frozen.
    assert lrs[0] < lrs[3]
    assert lrs[4] > lrs[5]


def test_dpo_training_increases_the_policys_preference_margin(tokenizer):
    # The end-to-end trainability proof: after real DPO training, the
    # policy must prefer the chosen response over the rejected one by a
    # LARGER margin than it did at initialization -- the actual thing
    # DPO training is supposed to accomplish, not just "loss went down."
    config = _tiny_config(tokenizer.vocab_size)
    model = DecoderOnlyTransformer(config)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False

    examples = [DPOExample(prompt="what color is the sky? ", chosen="blue", rejected="green")]
    chosen, rejected = build_dpo_batch(examples, tokenizer)

    with torch.no_grad():
        initial_margin = (
            sequence_logprobs(model, *chosen) - sequence_logprobs(model, *rejected)
        ).item()

    trainer = Trainer(model)
    losses = [trainer.dpo_step(ref_model, chosen, rejected, beta=0.1) for _ in range(60)]

    with torch.no_grad():
        final_margin = (
            sequence_logprobs(model, *chosen) - sequence_logprobs(model, *rejected)
        ).item()

    assert losses[-1] < losses[0]
    assert final_margin > initial_margin
