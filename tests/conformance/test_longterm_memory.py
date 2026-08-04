"""Conformance tests for sarva.memory.longterm — the design doc's own
"long-term memory as plain markdown files" tier. Definition of done goes
beyond "runs without crashing": notes must actually be real, appendable,
human-readable markdown on disk, search must actually find real text,
and concurrent writers to the same topic must not lose an update."""

from __future__ import annotations

import stat
import sys
import threading

import pytest
from sarva.memory.longterm import LongTermMemoryError, LongTermMemoryStore, _slugify

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.chmod's real per-user isolation is POSIX-only -- see sarva.config's docstring",
)


@pytest.fixture
def store(tmp_path):
    return LongTermMemoryStore(tmp_path / "memory")


def test_slugify_lowercases_and_collapses_non_alphanumerics():
    assert _slugify("User Preferences!!") == "user-preferences"


def test_slugify_rejects_a_topic_with_no_alphanumeric_content():
    with pytest.raises(LongTermMemoryError, match="invalid topic name"):
        _slugify("!!!")


def test_write_creates_a_real_readable_markdown_file(store):
    path = store.write("project status", "the launch is scheduled for next week")

    assert path.is_file()
    assert path.suffix == ".md"
    text = path.read_text()
    assert "# project status" in text
    assert "the launch is scheduled for next week" in text


def test_read_returns_none_for_a_topic_never_written(store):
    assert store.read("never written") is None


def test_repeated_writes_to_the_same_topic_append_not_overwrite(store):
    store.write("decisions", "first decision")
    store.write("decisions", "second decision")

    text = store.read("decisions")
    assert "first decision" in text
    assert "second decision" in text


def test_different_topics_land_in_different_files(store):
    store.write("topic-a", "content a")
    store.write("topic-b", "content b")

    assert store.list_topics() == ["topic-a", "topic-b"]
    assert "content b" not in store.read("topic-a")
    assert "content a" not in store.read("topic-b")


def test_search_finds_the_right_topic_by_exact_text(store):
    store.write("recipes", "the secret ingredient is nutmeg")
    store.write("travel", "pack an umbrella for Seattle")

    matches = store.search("nutmeg")

    assert len(matches) == 1
    assert matches[0].topic == "recipes"
    assert "nutmeg" in matches[0].snippet


def test_search_is_case_insensitive(store):
    store.write("recipes", "the secret ingredient is Nutmeg")
    assert len(store.search("NUTMEG")) == 1


def test_search_with_no_matches_returns_empty(store):
    store.write("recipes", "the secret ingredient is nutmeg")
    assert store.search("a completely unrelated query") == []


def test_two_different_topic_names_that_slugify_the_same_share_one_file(store):
    # "Project Status" and "project-status" both slugify to the same
    # file -- a real, deliberate consequence of using a human-friendly
    # slug as the identity, not a bug: notes about the same topic
    # written with slightly different capitalization/punctuation still
    # land together, which is the more useful behavior for a
    # human-organized note system.
    store.write("Project Status", "first note")
    store.write("project-status", "second note")

    text = store.read("Project Status")
    assert "first note" in text
    assert "second note" in text


def test_a_differently_phrased_topic_that_shares_a_slug_is_traceable_not_silent(store):
    # A real bug found by a fresh-eyes sweep: the test above correctly
    # documents that merging near-duplicate topic strings onto one file
    # is intentional, not a bug -- but the merge used to be completely
    # SILENT. The second write's own entry carried no record anywhere
    # that a differently-phrased topic string ("q3-planning") was
    # actually used -- only the file's original heading ("Q3 Planning",
    # from the first write) survived, so the second call's real topic
    # string was permanently unrecoverable after the write. Confirmed
    # live before this fix: write("Q3 Planning", ...) then
    # write("q3-planning", ...) landed both entries in one file with no
    # way to tell, from the file alone, that the second note was
    # actually filed under a different literal string. Fixed narrowly:
    # only the silence is closed, not the merge -- an entry whose own
    # topic string differs from the file's original heading now records
    # that literal string alongside its timestamp.
    store.write("Q3 Planning", "Revenue targets for Q3.")
    store.write("q3-planning", "Unrelated: my favorite pizza topping is mushroom.")
    # A same-topic repeat write must NOT get an annotation -- only a
    # genuinely different literal topic string should.
    store.write("Q3 Planning", "More revenue detail.")

    text = store.read("Q3 Planning")
    assert text is not None
    assert '(topic: "q3-planning")' in text
    lines = text.splitlines()
    same_topic_headers = [
        line for line in lines if line.startswith("## ") and "(topic:" not in line
    ]
    assert len(same_topic_headers) == 2  # the two "Q3 Planning" writes, unannotated


@_posix_only
def test_directory_and_files_are_owner_only(store):
    store.write("secrets", "sensitive content")
    dir_mode = stat.S_IMODE(store._directory.stat().st_mode)
    file_mode = stat.S_IMODE((store._directory / "secrets.md").stat().st_mode)
    assert dir_mode == 0o700
    assert file_mode == 0o600
    assert store.list_topics() == ["secrets"]


def test_concurrent_writes_to_the_same_topic_never_lose_an_update(tmp_path):
    # The same "unlocked read-modify-write" lost-update race already
    # found and fixed twice in this codebase (sarva.config, SessionStore)
    # -- proven here with real OS threads racing store.write() against
    # the SAME topic. Without the per-topic exclusive_lock, two threads
    # can both read the same "before" content, each append their own
    # note, and one write clobbers the other -- confirmed live before
    # the lock was added (see BUILD-JOURNAL.md).
    store = LongTermMemoryStore(tmp_path / "memory")
    n_writers = 8

    def write_one(i: int) -> None:
        store.write("shared-topic", f"note number {i}")

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = store.read("shared-topic")
    for i in range(n_writers):
        assert f"note number {i}" in text, f"note {i} was lost to a concurrent-write race"
