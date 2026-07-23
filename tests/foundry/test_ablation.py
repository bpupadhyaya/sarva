"""Conformance tests for sarva_foundry.ablation (design doc §3: the
ablation harness). "Trustworthy" is the word this module's own docstring
takes literally, so the tests prove the actual mechanisms behind that
claim directly -- that identical configs really do see identical data in
identical order (not just asserted), that a real capacity gap really is
detected, and that a real near-tie really is reported honestly as not
trustworthy -- not just that the code runs without crashing."""

from __future__ import annotations

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


def test_ablation_result_ranked_orders_by_mean_final_loss_ascending():
    good = ArmResult(name="good", final_losses=[0.1], loss_curves=[[]], param_count=1)
    bad = ArmResult(name="bad", final_losses=[9.0], loss_curves=[[]], param_count=1)
    result = AblationResult(arms=[bad, good])
    assert [a.name for a in result.ranked()] == ["good", "bad"]


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
