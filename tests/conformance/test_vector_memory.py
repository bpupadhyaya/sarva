"""Conformance tests for sarva.memory.vector — TF-IDF + cosine-similarity
semantic memory. Definition of done goes beyond "runs without crashing":
relevance ranking must actually reflect real topical similarity, not
just return something."""

from __future__ import annotations

import stat
import sys

import pytest
from sarva.memory.vector import (
    MemoryStoreError,
    VectorMemoryStore,
    _cosine_similarity,
    _tfidf_vector,
    _tokenize,
)

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.chmod's real per-user isolation is POSIX-only -- see sarva.config's docstring",
)


@pytest.fixture
def store(tmp_path):
    return VectorMemoryStore(tmp_path / "memory.db")


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert _tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tokenize_does_not_drop_or_truncate_non_ascii_text():
    # A real bug found by giving this module -- never fully swept in
    # 13+ prior rounds -- its own dedicated fresh-eyes read: the old
    # pattern, `[a-z0-9]+`, was ASCII-only. An accented Latin word like
    # "café" silently truncated to "caf" (dropped at the accented "é"),
    # and a pure-CJK string (no ASCII characters at all) tokenized to
    # an empty list.
    assert _tokenize("café") == ["café"]
    assert _tokenize("東京にラーメンを食べに行った") != []


def test_tokenize_drops_common_english_stopwords():
    assert _tokenize("The dog is in the yard") == ["dog", "yard"]


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


@_posix_only
def test_db_file_and_parent_directory_are_owner_only(store, tmp_path):
    # remember/recall_memory can hold text at least as sensitive as a
    # saved session -- the same class of gap already fixed for
    # sarva.config and sarva.memory.session, checked here too.
    store.add("sess1", "a note")

    db_path = tmp_path / "memory.db"
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


@_posix_only
def test_parent_directory_tightened_before_the_db_file_is_ever_created(tmp_path):
    # The directory is chmod'd BEFORE sqlite3.connect() creates the
    # file, not after -- closing the window a file-level-only fix would
    # still leave open, however brief.
    nested = tmp_path / "sub"
    VectorMemoryStore(nested / "memory.db")
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700


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


def test_search_ignores_shared_stopwords_when_ranking_topical_relevance(store):
    # A real bug found live, not by reading the scoring math in
    # isolation: search("tell me about the user's pet", ...) against
    # five real memories ranked "The user's favorite programming
    # language is Python." ABOVE "The user's dog is named Max and is a
    # golden retriever." -- the wrong entry first, for a query naming
    # the topic ("pet") the second entry is actually about. The
    # smoothed IDF formula never lets a term's weight hit true zero
    # even when it appears in every document (correct on its own), so
    # shared common-function-word overlap ("the", "is", "a") between a
    # query and a topically unrelated memory still contributed real,
    # non-negligible cosine-similarity mass. This is the clean,
    # deterministic version of that repro: a memory built entirely out
    # of stopwords now tokenizes to nothing (a zero-norm vector,
    # scoring 0.0 against everything) instead of accumulating spurious
    # similarity from words like "the"/"a"/"is"/"are" alone.
    store.add("s1", "The cat sat on the mat while the dog ran in the yard.")
    store.add("s1", "The the the a a a is is is are are are for for for.")

    results = store.search("the a dog is running", top_k=2, session_id="s1")

    scores_by_text = {entry.text: score for entry, score in results}
    dog_score = next(s for t, s in scores_by_text.items() if "dog" in t)
    stopword_only_score = next(s for t, s in scores_by_text.items() if "yard" not in t)
    assert dog_score > 0.0
    assert stopword_only_score == 0.0


def test_search_can_find_a_non_ascii_memory_by_its_own_exact_text(store):
    # The end-to-end proof of the tokenizer fix above: before it, a
    # pure-CJK memory tokenized to an empty token list, giving it a
    # zero-norm TF-IDF vector that _cosine_similarity short-circuits to
    # 0.0 for EVERY query -- including its own verbatim text used as
    # the query -- tying it indistinguishably with unrelated English
    # memories and making it structurally unreachable via search().
    store.add("s1", "東京にラーメンを食べに行った")
    store.add("s1", "I bought a new laptop yesterday")
    store.add("s1", "The weather in Tokyo is nice today")

    results = store.search("東京にラーメンを食べに行った", top_k=3, session_id="s1")

    top_entry, top_score = results[0]
    assert top_entry.text == "東京にラーメンを食べに行った"
    assert top_score > 0.0


def test_search_respects_top_k(store):
    for i in range(10):
        store.add("s1", f"document number {i} about various topics")
    results = store.search("document topics", top_k=3)
    assert len(results) == 3


def test_search_rejects_a_negative_top_k_instead_of_silently_dropping_results(store):
    # A real gap found by a fresh-eyes sweep, one layer below where this
    # class's own only real caller (RecallMemoryTool) already validates
    # the identical value: this class's own `scored[:top_k]` had no
    # guard of its own. Confirmed live: `search(..., top_k=-1)` against
    # three real entries returned two, not zero and not an error --
    # Python's own list-slice semantics turn a negative top_k into "drop
    # the last |top_k| results" (the worst-scoring one, specifically),
    # the exact opposite of a "return up to top_k, highest first"
    # contract. RecallMemoryTool can't reach this today (it rejects a
    # negative top_k before ever calling in), but a future direct caller
    # of this public method shouldn't have to rediscover the same fix.
    store.add("s1", "the quick brown fox")
    store.add("s1", "jumps over the lazy dog")
    store.add("s1", "a completely unrelated sentence about weather")

    with pytest.raises(ValueError, match="top_k must be non-negative"):
        store.search("quick brown fox", top_k=-1)

    # top_k=0 is a genuinely valid "give me nothing" request, unaffected.
    assert store.search("quick brown fox", top_k=0) == []


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


def test_construction_fails_cleanly_on_a_file_that_is_not_a_database(tmp_path):
    # A real bug found by actually writing garbage bytes to a real
    # memory.db path and constructing a real VectorMemoryStore:
    # sqlite3.connect() itself never fails (connections are lazy), but
    # the first real query -- the CREATE TABLE IF NOT EXISTS this
    # constructor already runs -- raised a raw, uncaught
    # sqlite3.DatabaseError. The fourth instance of the "corrupted
    # on-disk state" bug class already fixed for config.json, a saved
    # session file, and a foundry checkpoint bundle.
    db_path = tmp_path / "memory.db"
    db_path.write_text("not a real sqlite database, just garbage bytes")

    with pytest.raises(MemoryStoreError, match="not a valid SQLite database"):
        VectorMemoryStore(db_path)


def test_construction_fails_cleanly_on_a_truncated_real_database(tmp_path):
    # A related real bug: a real, previously-valid database truncated
    # mid-file (simulating an interrupted write or disk corruption --
    # the same realistic failure mode already used for the foundry
    # checkpoint fix) raises a *different* sqlite3.DatabaseError message
    # ("database disk image is malformed") than the "not a database at
    # all" case above -- confirmed both are the same exception class
    # (sqlite3.DatabaseError) so one except clause covers both.
    db_path = tmp_path / "memory.db"
    store = VectorMemoryStore(db_path)
    for i in range(50):
        store.add("s1", f"some real content padding out the file {i} " * 10)
    store.close()

    raw = db_path.read_bytes()
    db_path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(MemoryStoreError, match="not a valid SQLite database"):
        VectorMemoryStore(db_path)
