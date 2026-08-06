"""Conformance tests for sarva_foundry.ablation (design doc §3: the
ablation harness). "Trustworthy" is the word this module's own docstring
takes literally, so the tests prove the actual mechanisms behind that
claim directly -- that identical configs really do see identical data in
identical order (not just asserted), that a real capacity gap really is
detected, and that a real near-tie really is reported honestly as not
trustworthy -- not just that the code runs without crashing."""

from __future__ import annotations

import pytest
from sarva_foundry.ablation import AblationArm, AblationResult, ArmResult, run_ablation
from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR, tokenize_corpus
from sarva_foundry.model import TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "she sells seashells by the seashore and the shells are pretty",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
] * 4


def _tokenized_corpus() -> tuple[ByteLevelBPETokenizer, list[int]]:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(_CORPUS, vocab_size=300, special_tokens=[DOCUMENT_SEPARATOR])
    return tokenizer, tokenize_corpus(_CORPUS, tokenizer)


def test_arm_result_stdev_is_zero_for_a_single_seed_not_a_crash():
    result = ArmResult(name="x", final_losses=[1.0], loss_curves=[[2.0, 1.0]], param_count=10)
    assert result.mean_final_loss == 1.0
    assert result.stdev_final_loss == 0.0


def test_arm_result_computes_real_mean_and_stdev_across_seeds():
    result = ArmResult(
        name="x", final_losses=[1.0, 2.0, 3.0], loss_curves=[[], [], []], param_count=10
    )
    assert result.mean_final_loss == 2.0
    assert abs(result.stdev_final_loss - 1.0) < 1e-9  # sample stdev of [1,2,3] is exactly 1.0


def test_arm_result_stdev_is_nan_not_a_crash_when_a_seed_diverged():
    # A real bug found by giving this module's own statistics their own
    # fresh-eyes sweep: an ablation's whole job is comparing an
    # unproven architecture idea against a baseline, and a diverging
    # (NaN) final loss is one of the most ORDINARY real outcomes of
    # that comparison, not a contrived input. statistics.mean
    # propagates NaN silently (confirmed by reading its own
    # implementation), but statistics.stdev does not: confirmed live,
    # it raised a bare AttributeError ('float' object has no attribute
    # 'numerator') from deep inside its own exact-fraction arithmetic
    # instead of returning NaN, crashing any comparison touching a
    # diverged arm with an opaque, internal-implementation-detail
    # exception -- the same shape this module already fixed once for
    # get()'s bare StopIteration.
    import math

    result = ArmResult(
        name="diverged", final_losses=[float("nan"), 1.0, 2.0], loss_curves=[[]], param_count=10
    )
    assert math.isnan(result.stdev_final_loss)


def test_is_difference_trustworthy_returns_false_not_a_crash_when_an_arm_diverged():
    # The real, end-to-end proof the fix above actually closes the gap
    # at the one method this module's docstring frames as its core API,
    # not just that the property alone stops crashing.
    good = ArmResult(name="baseline", final_losses=[1.0, 1.1, 0.9], loss_curves=[[]], param_count=1)
    diverged = ArmResult(
        name="unstable",
        final_losses=[float("nan"), float("nan"), float("nan")],
        loss_curves=[[]],
        param_count=1,
    )
    result = AblationResult(arms=[good, diverged])

    # diff > NaN is False under IEEE-754 (every comparison against NaN
    # is) -- an honest "not established as trustworthy by this specific
    # statistic," not a crash and not a false claim the arms are similar.
    assert result.is_difference_trustworthy("baseline", "unstable") is False


def test_ablation_result_ranked_orders_by_mean_final_loss_ascending():
    good = ArmResult(name="good", final_losses=[0.1], loss_curves=[[]], param_count=1)
    bad = ArmResult(name="bad", final_losses=[9.0], loss_curves=[[]], param_count=1)
    result = AblationResult(arms=[bad, good])
    assert [a.name for a in result.ranked()] == ["good", "bad"]


def test_get_raises_a_clear_keyerror_on_an_unknown_arm_name_not_a_bare_stopiteration():
    # A real bug found by actually calling `result.get("baslein")` (a
    # plausible one-character typo of a real arm name, the exact kind
    # of mistake a researcher calling is_difference_trustworthy from a
    # notebook or example script would make): `next(a for a in
    # self.arms if a.name == name)` on no match raised a bare
    # StopIteration with no message at all -- not even the name that
    # failed to match, let alone what arms actually exist.
    good = ArmResult(name="baseline", final_losses=[0.1], loss_curves=[[]], param_count=1)
    bad = ArmResult(name="moe", final_losses=[9.0], loss_curves=[[]], param_count=1)
    result = AblationResult(arms=[good, bad])

    with pytest.raises(KeyError, match="baslein") as excinfo:
        result.get("baslein")

    # The available arm names must reach the message too, not just the
    # one that failed to match.
    assert "baseline" in str(excinfo.value)
    assert "moe" in str(excinfo.value)


def test_is_difference_trustworthy_raises_the_same_clear_keyerror_on_a_bad_arm_name():
    # is_difference_trustworthy calls get() twice per comparison, so
    # the same gap reached this method's own callers too -- pinned
    # separately since it's the actual public entry point most callers
    # (including this module's own examples) use, not get() directly.
    good = ArmResult(name="baseline", final_losses=[0.1], loss_curves=[[]], param_count=1)
    result = AblationResult(arms=[good])

    with pytest.raises(KeyError, match="baslein"):
        result.is_difference_trustworthy("baslein", "baseline")


def test_identical_configs_produce_bit_identical_losses_given_the_same_seed():
    # The actual mechanism behind "controls for data-order/seed
    # confounds" -- proven directly, not just claimed in a docstring:
    # two arms with the IDENTICAL model config, trained under the
    # IDENTICAL seed, must see the identical data in the identical order
    # and therefore land on the exact same final loss.
    _, token_ids = _tokenized_corpus()
    config = TransformerConfig(
        vocab_size=300, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    arms = [
        AblationArm(name="a", model_config=config),
        AblationArm(name="b", model_config=config),
    ]

    result = run_ablation(arms, token_ids, seq_len=16, batch_size=2, steps=15, seeds=[0])

    assert result.get("a").final_losses == result.get("b").final_losses


def test_run_ablation_detects_a_real_capacity_gap_as_trustworthy():
    _, token_ids = _tokenized_corpus()
    tiny = TransformerConfig(
        vocab_size=300, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    bigger = TransformerConfig(
        vocab_size=300, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    arms = [
        AblationArm(name="tiny", model_config=tiny),
        AblationArm(name="bigger", model_config=bigger),
    ]

    result = run_ablation(arms, token_ids, seq_len=16, batch_size=4, steps=200, seeds=[0, 1, 2])

    assert result.ranked()[0].name == "bigger"  # the bigger model really does win
    assert result.is_difference_trustworthy("tiny", "bigger") is True
    assert result.get("bigger").param_count > result.get("tiny").param_count


def test_run_ablation_reports_a_near_tie_as_not_trustworthy():
    # Two configs differing only in a way that shouldn't matter much at
    # this tiny scale/budget (a purely cosmetic head-count change at the
    # same total dim) -- the harness must not manufacture a "winner" out
    # of run-to-run noise.
    _, token_ids = _tokenized_corpus()
    a_config = TransformerConfig(
        vocab_size=300, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    b_config = TransformerConfig(
        vocab_size=300, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    arms = [
        AblationArm(name="a", model_config=a_config),
        AblationArm(name="b", model_config=b_config),
    ]

    result = run_ablation(arms, token_ids, seq_len=16, batch_size=4, steps=40, seeds=[0, 1, 2])

    assert result.is_difference_trustworthy("a", "b") is False


def test_run_ablation_records_a_real_loss_curve_not_just_a_final_number():
    _, token_ids = _tokenized_corpus()
    config = TransformerConfig(
        vocab_size=300, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    arms = [AblationArm(name="a", model_config=config)]

    result = run_ablation(
        arms, token_ids, seq_len=16, batch_size=2, steps=30, seeds=[0], record_every=5
    )

    curve = result.get("a").loss_curves[0]
    assert len(curve) > 1  # more than just the final loss
    assert curve[-1] == result.get("a").final_losses[0]  # the curve's last point IS the final loss


def test_run_ablation_rejects_a_non_positive_record_every_instead_of_crashing_mid_training():
    # A real bug found by a fresh-eyes sweep: record_every was a plain,
    # unvalidated int -- a caller wanting maximum recording granularity,
    # or one who derives it programmatically (e.g. `steps // 10` for a
    # quick smoke-test ablation with steps < 10), can land on 0 with
    # nothing anywhere to catch it. The training loop's own `step %
    # record_every` used to raise a raw ZeroDivisionError on its very
    # first iteration -- AFTER trainer.train_step() had already run one
    # real (if wasted) gradient-update pass. The same "unvalidated
    # numeric parameter leaking a raw implementation exception" shape
    # AblationResult.get()/ArmResult.stdev_final_loss were already fixed
    # for, one parameter over neither of those earlier fixes reached.
    _, token_ids = _tokenized_corpus()
    config = TransformerConfig(
        vocab_size=300, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    arms = [AblationArm(name="a", model_config=config)]

    with pytest.raises(ValueError, match="record_every"):
        run_ablation(arms, token_ids, seq_len=16, batch_size=2, steps=5, seeds=[0], record_every=0)


def test_run_ablation_rejects_a_non_positive_batch_size_instead_of_crashing_mid_training():
    # A real bug found by a later fresh-eyes sweep, the sibling
    # parameter one over from record_every above, in this exact
    # function: batch_size was likewise a plain, unvalidated int -- a
    # caller computing it programmatically (e.g. `tokens_per_step //
    # seq_len`, which is 0 for a small smoke-test corpus) can land on 0
    # (or a typo'd negative value) with nothing to catch it.
    # _make_batch's own `range(batch_size)` used to be empty, so
    # `torch.stack([])` raised a raw, undocumented RuntimeError -- AFTER
    # Trainer/DecoderOnlyTransformer construction had already happened
    # for that arm/seed.
    _, token_ids = _tokenized_corpus()
    config = TransformerConfig(
        vocab_size=300, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    arms = [AblationArm(name="a", model_config=config)]

    with pytest.raises(ValueError, match="batch_size"):
        run_ablation(arms, token_ids, seq_len=16, batch_size=0, steps=5, seeds=[0])
    with pytest.raises(ValueError, match="batch_size"):
        run_ablation(arms, token_ids, seq_len=16, batch_size=-2, steps=5, seeds=[0])
