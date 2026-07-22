"""Conformance tests for sarva_foundry.data.dataset."""

from __future__ import annotations

import pytest
import torch
from sarva_foundry.data import TextChunkDataset, tokenize_corpus
from sarva_foundry.tokenizer import ByteLevelBPETokenizer


def test_chunking_produces_correct_shapes_and_shifted_targets():
    token_ids = list(range(21))  # 0..20, 21 tokens
    ds = TextChunkDataset(token_ids, seq_len=5)
    # (21 - 1) // 5 = 4 whole chunks; the 21st token (leftover) is dropped.
    assert len(ds) == 4
    x0, y0 = ds[0]
    assert x0.tolist() == [0, 1, 2, 3, 4]
    assert y0.tolist() == [1, 2, 3, 4, 5]
    x1, y1 = ds[1]
    assert x1.tolist() == [5, 6, 7, 8, 9]


def test_chunking_dtype_is_long_for_embedding_lookup():
    ds = TextChunkDataset(list(range(11)), seq_len=5)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_too_few_tokens_raises_clear_error():
    with pytest.raises(ValueError, match="at least"):
        TextChunkDataset([1, 2, 3], seq_len=5)


def test_invalid_seq_len_raises():
    with pytest.raises(ValueError):
        TextChunkDataset(list(range(10)), seq_len=0)


def test_tokenize_corpus_requires_the_separator_as_a_special_token():
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world", "goodbye world"], vocab_size=280)  # no special tokens
    with pytest.raises(ValueError, match="special token"):
        tokenize_corpus(["hello"], tok)


def test_tokenize_corpus_inserts_separator_between_documents():
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world", "goodbye world"], vocab_size=280, special_tokens=["<|endoftext|>"])
    sep_id = tok.special_tokens["<|endoftext|>"]

    ids = tokenize_corpus(["hello", "world"], tok)
    hello_ids = tok.encode("hello")
    world_ids = tok.encode("world")
    assert ids == [*hello_ids, sep_id, *world_ids, sep_id]
