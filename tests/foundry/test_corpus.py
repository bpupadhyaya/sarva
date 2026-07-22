"""Conformance tests for sarva_foundry.data.corpus — local corpus
sourcing/cleaning/dedup (spec §3.6c)."""

from __future__ import annotations

import pytest
from sarva_foundry.data import (
    TextChunkDataset,
    dedup_documents,
    filter_by_length,
    load_text_files,
    tokenize_corpus,
)
from sarva_foundry.tokenizer import ByteLevelBPETokenizer


def test_load_text_files_reads_in_sorted_order(tmp_path):
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")
    (tmp_path / "c.txt").write_text("third")

    docs = load_text_files(tmp_path)

    assert docs == ["first", "second", "third"]


def test_load_text_files_respects_the_glob_pattern(tmp_path):
    (tmp_path / "doc.txt").write_text("keep me")
    (tmp_path / "notes.md").write_text("ignore me")

    docs = load_text_files(tmp_path, pattern="*.txt")

    assert docs == ["keep me"]


def test_load_text_files_raises_a_clear_error_on_no_matches(tmp_path):
    with pytest.raises(ValueError, match="no files matching"):
        load_text_files(tmp_path)


def test_load_text_files_raises_rather_than_silently_corrupting_bad_encoding(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe not valid utf-8 on its own \x80\x81")
    with pytest.raises(UnicodeDecodeError):
        load_text_files(tmp_path, encoding="utf-8")


def test_dedup_documents_drops_exact_duplicates_keeping_first_occurrence_order():
    docs = ["alpha", "beta", "alpha", "gamma", "beta"]
    assert dedup_documents(docs) == ["alpha", "beta", "gamma"]


def test_dedup_documents_is_a_noop_with_no_duplicates():
    docs = ["alpha", "beta", "gamma"]
    assert dedup_documents(docs) == docs


def test_filter_by_length_drops_documents_shorter_than_min_chars():
    docs = ["a", "ab", "abc", "abcd"]
    assert filter_by_length(docs, min_chars=3) == ["abc", "abcd"]


def test_filter_by_length_boundary_is_inclusive():
    docs = ["abc"]  # exactly 3 chars
    assert filter_by_length(docs, min_chars=3) == ["abc"]
    assert filter_by_length(docs, min_chars=4) == []


def test_filter_by_length_drops_documents_longer_than_max_chars():
    docs = ["short", "this one is much too long"]
    assert filter_by_length(docs, max_chars=10) == ["short"]


def test_filter_by_length_with_no_max_chars_keeps_everything_above_min():
    docs = ["a" * 1000]
    assert filter_by_length(docs, min_chars=1) == docs


def test_sourcing_pipeline_plugs_into_the_existing_tokenize_and_chunk_stages(tmp_path):
    # The point of this test: load -> dedup -> filter -> tokenize_corpus
    # -> TextChunkDataset is a real pipeline, not three functions that
    # happen to share a module. Two files are exact duplicates of each
    # other and must collapse to one document; a near-empty file must be
    # filtered out before it ever reaches the tokenizer.
    (tmp_path / "doc1.txt").write_text("the quick brown fox jumps over the lazy dog")
    (tmp_path / "doc2.txt").write_text("the quick brown fox jumps over the lazy dog")  # dup of doc1
    (tmp_path / "doc3.txt").write_text("she sells seashells by the seashore")
    (tmp_path / "doc4.txt").write_text("hi")  # too short, gets filtered

    docs = load_text_files(tmp_path)
    docs = dedup_documents(docs)
    docs = filter_by_length(docs, min_chars=10)
    assert len(docs) == 2  # doc1/doc2 collapsed to one, doc4 filtered out

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(docs, vocab_size=300, special_tokens=["<|endoftext|>"])
    token_ids = tokenize_corpus(docs, tokenizer)
    dataset = TextChunkDataset(token_ids, seq_len=8)

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
