"""Conformance tests for sarva_foundry.data.near_dedup — MinHash-based
near-duplicate detection, the deferred scope named in dedup_documents'
own docstring."""

from __future__ import annotations

from sarva_foundry.data import dedup_documents
from sarva_foundry.data.near_dedup import (
    _estimated_jaccard_similarity,
    _minhash_signature,
    _shingles,
    dedup_near_duplicates,
)

_DOC_A = (
    "The quick brown fox jumps over the lazy dog near the old wooden "
    "fence at the edge of the sleepy little village."
)
# doc_b: doc_a with a single word changed -- a realistic near-duplicate
# (e.g. a re-published article with a minor copy edit). Deliberately NOT
# a whole appended sentence: that dilutes shingle-set Jaccard similarity
# far more than intuition suggests (verified empirically at ~0.66 true
# similarity for this doc's length, well below any reasonable
# near-duplicate threshold) -- a small in-place edit is what stays
# genuinely "near-duplicate" by shingle overlap.
_DOC_B = _DOC_A.replace("sleepy", "quiet")
# doc_c: genuinely unrelated text, same rough length as doc_a.
_DOC_C = (
    "She sells seashells by the seashore while the tide slowly rises "
    "over the smooth grey stones scattered along the coast."
)


def test_shingles_produces_overlapping_windows_of_the_requested_size():
    shingles = _shingles("abcdef", size=3)
    assert shingles == {"abc", "bcd", "cde", "def"}


def test_shingles_of_text_shorter_than_size_is_the_whole_text():
    assert _shingles("ab", size=5) == {"ab"}


def test_shingles_of_empty_text_is_empty():
    assert _shingles("", size=3) == set()


def test_minhash_signature_is_deterministic():
    shingles = _shingles(_DOC_A, size=5)
    sig1 = _minhash_signature(shingles, num_hashes=32)
    sig2 = _minhash_signature(shingles, num_hashes=32)
    assert sig1 == sig2


def test_minhash_signature_of_identical_shingle_sets_matches_exactly():
    shingles = _shingles(_DOC_A, size=5)
    sig_a = _minhash_signature(shingles, num_hashes=32)
    sig_b = _minhash_signature(set(shingles), num_hashes=32)  # a fresh, equal set
    assert sig_a == sig_b


def test_estimated_similarity_of_identical_signatures_is_exactly_one():
    sig = _minhash_signature(_shingles(_DOC_A, size=5), num_hashes=64)
    assert _estimated_jaccard_similarity(sig, sig) == 1.0


def test_estimated_similarity_is_high_for_near_duplicates_and_low_for_unrelated_text():
    sig_a = _minhash_signature(_shingles(_DOC_A, size=5), num_hashes=128)
    sig_b = _minhash_signature(_shingles(_DOC_B, size=5), num_hashes=128)
    sig_c = _minhash_signature(_shingles(_DOC_C, size=5), num_hashes=128)

    similarity_near_dup = _estimated_jaccard_similarity(sig_a, sig_b)
    similarity_unrelated = _estimated_jaccard_similarity(sig_a, sig_c)

    assert similarity_near_dup > 0.75  # true similarity ~0.85; comfortable margin below it
    assert similarity_unrelated < 0.15  # true similarity ~0.04; comfortable margin above it
    assert similarity_near_dup > similarity_unrelated


def test_dedup_near_duplicates_drops_a_near_identical_document():
    result = dedup_near_duplicates([_DOC_A, _DOC_B], threshold=0.8)
    assert result == [_DOC_A]


def test_dedup_near_duplicates_keeps_genuinely_different_documents():
    result = dedup_near_duplicates([_DOC_A, _DOC_C], threshold=0.8)
    assert result == [_DOC_A, _DOC_C]


def test_dedup_near_duplicates_keeps_first_occurrence_order():
    result = dedup_near_duplicates([_DOC_C, _DOC_A, _DOC_B], threshold=0.8)
    assert result == [_DOC_C, _DOC_A]  # doc_b dropped as a near-dup of doc_a


def test_dedup_near_duplicates_is_a_noop_below_threshold_similarity():
    # A very high threshold means even a near-duplicate isn't similar
    # enough to drop -- confirms the threshold parameter actually gates
    # the decision rather than always dropping/always keeping.
    result = dedup_near_duplicates([_DOC_A, _DOC_B], threshold=0.999)
    assert result == [_DOC_A, _DOC_B]


def test_dedup_near_duplicates_handles_empty_documents():
    result = dedup_near_duplicates(["", ""], threshold=0.8)
    assert result == [""]  # two empty strings genuinely are duplicates


def test_dedup_near_duplicates_composes_with_exact_dedup():
    docs = [_DOC_A, _DOC_A, _DOC_B, _DOC_C]  # exact dup, then a near-dup, then unrelated
    after_exact = dedup_documents(docs)
    assert after_exact == [_DOC_A, _DOC_B, _DOC_C]  # exact dup of doc_a collapsed

    after_near = dedup_near_duplicates(after_exact, threshold=0.8)
    assert after_near == [_DOC_A, _DOC_C]  # doc_b now also dropped as a near-dup
