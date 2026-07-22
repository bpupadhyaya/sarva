"""Conformance tests for sarva.memory.vector — TF-IDF + cosine-similarity
semantic memory. Definition of done goes beyond "runs without crashing":
relevance ranking must actually reflect real topical similarity, not
just return something."""

from __future__ import annotations

import pytest
from sarva.memory.vector import VectorMemoryStore, _cosine_similarity, _tfidf_vector, _tokenize


@pytest.fixture
def store(tmp_path):
    return VectorMemoryStore(tmp_path / "memory.db")


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert _tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_cosine_similarity_of_identical_vectors_is_one():
    vec = {"a": 1.0, "b": 2.0}
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_of_disjoint_vectors_is_zero():
    assert _cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_similarity_handles_a_zero_vector_without_dividing_by_zero():
    assert _cosine_similarity({}, {"a": 1.0}) == 0.0


def test_tfidf_gives_zero_weight_to_query_terms_never_seen_in_any_document():
    idf = {"seen": 1.0}
    vec = _tfidf_vector(["seen", "never_seen_anywhere"], idf)
    assert vec["seen"] > 0
    assert vec["never_seen_anywhere"] == 0.0


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search("anything") == []


def test_search_ranks_the_topically_relevant_entry_first(store):
    store.add("s1", "The quick brown fox jumps over the lazy dog in the meadow.")
    store.add("s1", "Quarterly revenue increased due to strong enterprise software sales.")
    store.add("s1", "A dog and a fox are both common animals found in rural meadows.")

    results = store.search("fox and dog in the meadow", top_k=3)

    assert len(results) == 3
    top_entry, top_score = results[0]
    assert "fox" in top_entry.text.lower() or "dog" in top_entry.text.lower()
    # The unrelated financial entry must rank last, with a lower score.
    scores_by_text = {entry.text: score for entry, score in results}
    revenue_score = next(s for t, s in scores_by_text.items() if "revenue" in t)
    assert top_score > revenue_score


def test_search_respects_top_k(store):
    for i in range(10):
        store.add("s1", f"document number {i} about various topics")
    results = store.search("document topics", top_k=3)
    assert len(results) == 3


def test_search_is_scoped_to_session_id_when_given(store):
    store.add("session-a", "apples and oranges are fruit")
    store.add("session-b", "apples and oranges are fruit")  # same text, different session

    results = store.search("apples", session_id="session-a")

    assert len(results) == 1
    assert results[0][0].session_id == "session-a"


def test_search_without_session_id_searches_everything(store):
    store.add("session-a", "apples and oranges")
    store.add("session-b", "apples and oranges")

    results = store.search("apples")

    assert len(results) == 2


def test_add_returns_an_incrementing_row_id(store):
    first_id = store.add("s1", "first entry")
    second_id = store.add("s1", "second entry")
    assert second_id > first_id


def test_data_persists_across_separate_store_instances(tmp_path):
    db_path = tmp_path / "memory.db"
    store_a = VectorMemoryStore(db_path)
    store_a.add("s1", "a persisted memory entry")
    store_a.close()

    store_b = VectorMemoryStore(db_path)
    results = store_b.search("persisted memory")
    assert len(results) == 1
    assert results[0][0].text == "a persisted memory entry"


def test_query_with_no_overlapping_vocabulary_still_returns_results_with_zero_score(store):
    store.add("s1", "completely unrelated content about gardening")
    results = store.search("xyzzy nonexistent zzqq")
    assert len(results) == 1
    assert results[0][1] == 0.0
