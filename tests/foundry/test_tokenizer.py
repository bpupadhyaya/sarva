"""Conformance tests for sarva_foundry.tokenizer.bpe — the from-scratch
byte-level BPE tokenizer. Definition of done: lossless round-trip on any
input (the byte-level guarantee), deterministic training, and a vocab
that actually respects the requested size."""

from __future__ import annotations

import os

import pytest
from sarva_foundry.tokenizer import ByteLevelBPETokenizer

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
]


def _trained(vocab_size: int = 300, special_tokens: tuple[str, ...] = ()) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(_CORPUS, vocab_size=vocab_size, special_tokens=special_tokens)
    return tok


def test_untrained_vocab_is_exactly_the_byte_alphabet():
    tok = ByteLevelBPETokenizer()
    assert tok.vocab_size == 256


def test_roundtrip_ascii():
    tok = _trained()
    text = "the quick brown fox"
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_unseen_unicode_and_emoji():
    # The byte-level guarantee: text never seen during training, including
    # non-Latin scripts and emoji, still round-trips exactly — there is no
    # <unk>, because every byte value has a dedicated symbol.
    tok = _trained()
    text = "héllo wörld —日本語 🎉🚀"
    assert tok.decode(tok.encode(text)) == text


def test_empty_text_roundtrips_to_empty():
    tok = _trained()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_vocab_size_is_respected():
    tok = _trained(vocab_size=280)
    assert tok.vocab_size <= 280
    assert tok.vocab_size == 256 + len(tok.merges)


def test_vocab_size_below_byte_alphabet_raises():
    tok = ByteLevelBPETokenizer()
    with pytest.raises(ValueError):
        tok.train(_CORPUS, vocab_size=100)


def test_merges_compress_a_training_sentence():
    tok = _trained(vocab_size=300)
    text = "the quick brown fox"
    byte_level = ByteLevelBPETokenizer()  # no merges learned
    assert len(tok.encode(text)) < len(byte_level.encode(text))


def test_training_is_deterministic():
    a = _trained(vocab_size=290)
    b = _trained(vocab_size=290)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_special_tokens_are_atomic_and_roundtrip():
    tok = _trained(vocab_size=300, special_tokens=("<|endoftext|>",))
    text = "the quick brown fox<|endoftext|>the lazy dog"
    ids = tok.encode(text)
    special_id = tok.special_tokens["<|endoftext|>"]
    assert special_id in ids
    assert tok.decode(ids) == text
    # The special token must never be split by ordinary byte-level merges.
    assert ids.count(special_id) == 1


def test_decode_replaces_invalid_utf8_instead_of_raising():
    # Real, not hypothetical: encode() always produces valid UTF-8 by
    # construction, but decode() also has to handle arbitrary token id
    # sequences a model might actually generate (RL rollout, an
    # undertrained checkpoint, adversarial input) -- those aren't
    # guaranteed to concatenate into valid UTF-8. A tokenizer used for
    # real inference must decode gracefully, not crash the whole
    # generation. Byte 0xCC alone is an invalid lone UTF-8 continuation
    # byte -- confirmed directly (not assumed) by finding the vocab id
    # that maps to that exact raw byte value.
    tok = ByteLevelBPETokenizer()  # untrained: vocab ids 0..255 are the raw byte alphabet
    from sarva_foundry.tokenizer.bpe import _byte_to_unicode

    invalid_byte_char = _byte_to_unicode()[0xCC]
    invalid_id = tok.vocab[invalid_byte_char]

    result = tok.decode([invalid_id])  # must not raise UnicodeDecodeError

    assert "�" in result  # the standard Unicode replacement character


def test_decode_still_roundtrips_valid_text_around_invalid_bytes():
    # The replacement behavior must be scoped to genuinely invalid bytes
    # only -- real, validly-encoded text around them must still decode
    # exactly, not get mangled as collateral damage.
    tok = _trained()
    valid_prefix = tok.encode("hello ")
    from sarva_foundry.tokenizer.bpe import _byte_to_unicode

    invalid_id = tok.vocab[_byte_to_unicode()[0xCC]]
    valid_suffix = tok.encode(" world")

    result = tok.decode(valid_prefix + [invalid_id] + valid_suffix)

    assert result.startswith("hello ")
    assert result.endswith(" world")
    assert "�" in result


def test_decode_replaces_an_out_of_vocabulary_id_instead_of_raising():
    # A real bug found by actually calling decode() with an id outside
    # [0, vocab_size): id_to_symbol[token_id] raised an uncaught
    # KeyError, propagating all the way up through
    # FoundryProvider.generate() into AgentLoop's own except Exception,
    # silently failing the ENTIRE generation (already-sampled valid text
    # included) with an unhelpful numeric detail. Reachable whenever a
    # bundle's config.json vocab_size (which sizes the model's own
    # lm_head, so sampling can legitimately produce any id in that
    # range) disagrees with what tokenizer.json actually serializes --
    # the same "corrupted/mismatched checkpoint" threat model already
    # hardened against elsewhere. Extends the exact "decode gracefully"
    # discipline the invalid-UTF-8 tests above already establish, just
    # for an id missing from the vocabulary entirely rather than one
    # whose bytes don't form valid UTF-8.
    tok = _trained()
    out_of_vocab_id = tok.vocab_size + 50

    result = tok.decode([out_of_vocab_id])  # must not raise KeyError

    assert "�" in result


def test_decode_still_roundtrips_valid_text_around_an_out_of_vocabulary_id():
    tok = _trained()
    valid_prefix = tok.encode("hello ")
    out_of_vocab_id = tok.vocab_size + 50
    valid_suffix = tok.encode(" world")

    result = tok.decode(valid_prefix + [out_of_vocab_id] + valid_suffix)

    assert result.startswith("hello ")
    assert result.endswith(" world")
    assert "�" in result


def test_load_rejects_a_tokenizer_json_where_a_special_token_id_collides_with_vocab(
    tmp_path,
):
    # A real bug found by actually corrupting special_tokens to collide
    # with a real vocab id: a legitimately-trained tokenizer can never
    # produce this (train() always assigns special-token ids starting
    # after len(vocab), strictly increasing), so this needs the same
    # corrupted/hand-edited tokenizer.json precondition as the
    # out-of-vocabulary decode bug above. decode() checks id_to_special
    # before id_to_symbol, so a colliding id used to silently decode as
    # the special token's text instead of the real character, with no
    # error at all -- confirmed live: a real vocab id whose correct
    # decoding was "2" decoded as the special token instead.
    import json

    tok = _trained(special_tokens=("<|endoftext|>",))
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    data = json.loads(path.read_text())
    real_vocab_id = tok.vocab["t"]  # a real, ordinary byte-alphabet id
    data["special_tokens"]["<|endoftext|>"] = real_vocab_id
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="collide"):
        ByteLevelBPETokenizer.load(path)


def test_save_and_load_roundtrip(tmp_path):
    tok = _trained(vocab_size=300, special_tokens=("<|endoftext|>",))
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    reloaded = ByteLevelBPETokenizer.load(path)
    text = "the quick brown fox<|endoftext|>日本語"
    assert reloaded.encode(text) == tok.encode(text)
    assert reloaded.decode(tok.encode(text)) == text
    assert reloaded.merges == tok.merges
    assert reloaded.special_tokens == tok.special_tokens


def test_save_does_not_destroy_the_previous_file_if_interrupted_mid_write(tmp_path, monkeypatch):
    # A real bug found by actually simulating an interrupted write: this
    # used to path.write_text() directly, which truncates the file to 0
    # bytes the instant it's opened, before a single byte of new content
    # is written. A crash mid-write destroyed a previously-trained, real
    # tokenizer -- hours of BPE merge-learning, not regenerable from the
    # saved file itself. Simulated here by making os.replace() raise
    # partway through a second save.
    tok = _trained(vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    good_bytes = path.read_bytes()

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    other = _trained(vocab_size=280)
    with pytest.raises(OSError):
        other.save(path)

    assert path.read_bytes() == good_bytes
    reloaded = ByteLevelBPETokenizer.load(path)  # must not raise
    assert reloaded.merges == tok.merges
