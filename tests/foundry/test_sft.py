"""Conformance tests for sarva_foundry.train.sft — SFT data prep (spec
§3.6e). The property that actually matters is the loss mask's
alignment: shape-correct tensors that mask the wrong positions would
still "work" in the sense of not crashing while silently training the
model on the wrong objective (predicting prompts instead of responses)
— every test here checks the mask against known, hand-traced token
positions, not just tensor shapes."""

from __future__ import annotations

import pytest
from sarva_foundry.data import DOCUMENT_SEPARATOR
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train.sft import SFTExample, build_sft_batch, encode_sft_example


@pytest.fixture
def tokenizer() -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(
        ["hello world", "what is 2 2", "the cat sat on the mat", "four"],
        vocab_size=280,
        special_tokens=[DOCUMENT_SEPARATOR],
    )
    return tok


def test_encode_sft_example_masks_exactly_the_prompt(tokenizer):
    example = SFTExample(prompt="what is 2 2 ", response="four")
    token_ids, mask = encode_sft_example(example, tokenizer)

    prompt_len = len(tokenizer.encode(example.prompt))
    response_len = len(tokenizer.encode(example.response)) + len(
        tokenizer.encode(DOCUMENT_SEPARATOR)
    )
    assert len(token_ids) == prompt_len + response_len
    assert mask == [0] * prompt_len + [1] * response_len


def test_encode_sft_example_response_ends_with_end_of_turn(tokenizer):
    example = SFTExample(prompt="hello ", response="world")
    token_ids, mask = encode_sft_example(example, tokenizer)
    eot_ids = tokenizer.encode(DOCUMENT_SEPARATOR)
    assert token_ids[-len(eot_ids) :] == eot_ids
    assert mask[-len(eot_ids) :] == [1] * len(eot_ids)


def test_encode_sft_example_requires_end_of_turn_in_special_tokens():
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world"], vocab_size=260)  # no special_tokens at all
    with pytest.raises(ValueError, match="special token"):
        encode_sft_example(SFTExample(prompt="hi ", response="there"), tok)


def test_build_sft_batch_shift_is_standard_next_token_framing(tokenizer):
    example = SFTExample(prompt="hello ", response="world")
    token_ids, _ = encode_sft_example(example, tokenizer)
    x, y, _ = build_sft_batch([example], tokenizer)

    assert x.shape[1] == len(token_ids) - 1
    assert x[0].tolist() == token_ids[:-1]
    assert y[0].tolist() == token_ids[1:]


def test_build_sft_batch_mask_excludes_prompt_and_includes_response(tokenizer):
    example = SFTExample(prompt="hello ", response="world")
    _, full_mask = encode_sft_example(example, tokenizer)
    _, _, mask = build_sft_batch([example], tokenizer)

    # The batch's mask is aligned to TARGETS (shifted by one), so it must
    # equal the un-shifted per-token mask sliced from index 1 onward --
    # position i's target is token_ids[i+1], and that target's mask
    # entry is what determines whether predicting it counts.
    assert mask[0].tolist() == [float(m) for m in full_mask[1:]]


def test_build_sft_batch_pads_shorter_examples_and_masks_the_padding(tokenizer):
    short = SFTExample(prompt="hi ", response="x")
    long = SFTExample(prompt="the cat sat on the mat ", response="four")

    x, y, mask = build_sft_batch([short, long], tokenizer)

    assert x.shape == y.shape == mask.shape
    assert x.shape[0] == 2
    short_len = (
        len(tokenizer.encode(short.prompt))
        + len(tokenizer.encode(short.response))
        + len(tokenizer.encode(DOCUMENT_SEPARATOR))
    )
    # Padding positions (past the short example's real length - 1, since
    # shifting drops one position) must be masked out.
    real_target_len = short_len - 1
    assert mask[0, real_target_len:].sum().item() == 0.0


def test_build_sft_batch_rejects_a_batch_too_short_to_form_any_pair(tokenizer):
    # A real, constructible pathological case: an empty prompt and an
    # empty response encode to zero tokens each, leaving only the single
    # end_of_turn token -- one token total, nothing left to predict once
    # shifted for next-token training. Not contrived via mocking; this
    # is exactly what an empty SFTExample produces through the real
    # encode path.
    empty_example = SFTExample(prompt="", response="")
    assert encode_sft_example(empty_example, tokenizer)[0] == tokenizer.encode(DOCUMENT_SEPARATOR)

    with pytest.raises(ValueError, match="at least 2 tokens"):
        build_sft_batch([empty_example], tokenizer)
