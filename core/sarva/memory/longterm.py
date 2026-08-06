"""sarva.memory.longterm — durable, cross-session, human-readable
long-term memory: plain markdown files, one per topic. The design doc's
own literal promise (§3.4): "long-term memory as plain markdown files
(human-readable, greppable)" — a third memory tier, distinct from and
complementary to the other two already built:

- `sarva.memory.session` (JSONL) — one file per CONVERSATION, holding
  the turn-by-turn transcript of a single session.
- `sarva.memory.vector` (SQLite + TF-IDF) — semantic recall, primarily
  used SESSION-scoped by `RememberTool`/`RecallMemoryTool` (a session's
  own notes don't bleed into another session's recall by default).
- This module — DURABLE and CROSS-SESSION by design: a note written
  from one conversation is visible to every future one, organized by
  topic rather than by session, and stored as literal, exact markdown
  text a person can open in any editor and read or hand-edit directly
  — no query tool required, matching the design doc's own "greppable"
  word precisely. Search here is deliberately exact substring matching,
  not semantic — `sarva.memory.vector` already covers semantic recall;
  the whole point of this tier is that it's plain text, so search
  should match that promise rather than duplicate the other tier's.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sarva.atomic_write import atomic_write_text
from sarva.file_lock import exclusive_lock

DEFAULT_LONGTERM_MEMORY_DIR = Path.home() / ".sarva" / "memory"

# A real bug found by a fresh-eyes sweep, the identical "ASCII-only
# normalization pattern" shape already found and fixed once for
# sarva.memory.vector's own _tokenize() -- this sibling function, doing
# the identical normalization job one memory-tier module over, never
# got the same fix. `[^a-z0-9]+` only ever preserves ASCII letters/
# digits, so a topic written entirely in a non-Latin script (Japanese,
# Russian, Arabic, ...) -- an entirely ordinary thing for a
# non-English-speaking user or a model conversing in another language
# to ask this tool to remember something under -- slugified to an empty
# string and was rejected outright with LongTermMemoryError, making the
# `note` tool completely unusable for that topic; an accented Latin
# topic like "café" wasn't rejected but was silently mangled to "caf".
# Confirmed live before this fix: `_slugify("日本語のメモ")` raised
# LongTermMemoryError; `_slugify("café")` returned "caf". `\w` is
# Unicode-aware by default for a `str` pattern in Python 3, matching
# vector.py's own fix -- `_` is explicitly still collapsed to "-" too,
# preserving this function's own prior behavior for underscores (`\w`
# alone would treat `_` as a keep-worthy word character, a silent
# behavior change this fix deliberately avoids).
_SLUG_PATTERN = re.compile(r"[\W_]+")


class LongTermMemoryError(Exception):
    """An invalid topic name -- the same "reject, don't guess" discipline
    `sarva.memory.session.SessionStore`'s own name validation already
    applies, so a caller gets one clean, documented failure instead of
    a raw OSError from an unusable path (e.g. a topic that slugifies to
    an empty string, or one long enough to exceed the filesystem's own
    filename-length limit)."""


# Safely under every mainstream filesystem's max filename-component length
# (typically 255 bytes) even after the ".md"/".md.lock" suffix. A real bug
# found by actually writing a 500-character topic name: with no cap here,
# _path_for() built a filename long enough to make the OS itself reject
# it -- LongTermMemoryStore.write() raised a raw OSError (with a real
# local filesystem path in the message) instead of this module's own
# documented LongTermMemoryError, and NoteTool only catches the latter,
# so the raw OS error (and the local path inside it) surfaced straight
# through to the tool result text a model/user sees.
_MAX_TOPIC_SLUG_LENGTH = 200


def _slugify(topic: str) -> str:
    slug = _SLUG_PATTERN.sub("-", topic.strip().lower()).strip("-")
    if not slug:
        raise LongTermMemoryError(
            f"invalid topic name: {topic!r} -- must contain at least one alphanumeric character"
        )
    # A real bug found alongside the Unicode-slugification fix above:
    # this cap was measured in Python characters, which was exactly
    # equivalent to bytes back when the slug was ASCII-only -- but a
    # non-Latin slug's characters can each take up to 4 bytes in UTF-8
    # (the actual unit the filesystem's own limit this cap exists to
    # protect against is measured in), so a slug well under 200
    # *characters* could still exceed 255 *bytes* and reintroduce the
    # exact raw-OSError bug this cap was built to prevent, just for
    # Unicode topics instead of long ASCII ones.
    slug_bytes = len(slug.encode("utf-8"))
    if slug_bytes > _MAX_TOPIC_SLUG_LENGTH:
        raise LongTermMemoryError(
            f"invalid topic name: {slug_bytes} bytes after slugifying, "
            f"must be at most {_MAX_TOPIC_SLUG_LENGTH}"
        )
    return slug


@dataclass(frozen=True)
class NoteMatch:
    topic: str
    snippet: str


class LongTermMemoryStore:
    """One markdown file per topic under `directory` (default
    `~/.sarva/memory/`), each a real, appendable, human-readable
    document. Owner-only permissions on both the directory and every
    file in it -- the same "notes can be at least as sensitive as a
    saved session" posture `VectorMemoryStore`/`SessionStore` already
    apply. Writes to the SAME topic from concurrent callers (two agent
    runs in a long-running `sarva serve` both noting to the same topic
    near-simultaneously) are serialized via a real, per-topic
    cross-process lock -- an unlocked read-modify-write here would be
    the identical lost-update race already found and fixed twice in
    this codebase (`sarva.config`'s `_exclusive_lock`, `SessionStore.
    locked`); reintroducing it in new code would be a regression, not
    an oversight."""

    def __init__(self, directory: Path = DEFAULT_LONGTERM_MEMORY_DIR):
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self._directory, 0o700)

    def _path_for(self, topic: str) -> Path:
        return self._directory / f"{_slugify(topic)}.md"

    def write(self, topic: str, content: str) -> Path:
        """Appends a timestamped entry to `topic`'s markdown file,
        creating it (with a top-level heading naming the topic) if this
        is the first note under that topic.

        A real bug found by a fresh-eyes sweep: `_slugify()` is a lossy,
        many-to-one normalization (lowercased, punctuation/whitespace
        collapsed to `-`) with no collision check against the file's own
        original topic string -- two different topic strings that
        happen to slugify identically (differing only in case, spacing,
        or punctuation -- an entirely ordinary, non-adversarial thing
        for the SAME model to produce across separate, independent
        conversations, since this tier is explicitly cross-session)
        silently share one file. Confirmed live: `write("Q3 Planning",
        ...)` then `write("q3-planning", ...)` landed both entries in
        the identical file, the second silently filed under the FIRST
        call's original heading with no signal anywhere that a
        differently-phrased topic string was actually used.
        `list_topics()`/`read()` only ever see the one slug -- the
        second topic string is completely unrecoverable after the
        write. Merging near-duplicate topic strings onto one file is
        the intended point of slugifying in the first place (so a model
        doesn't accidentally fragment "Meeting Notes" across three
        near-identical files) -- rejecting the write would defeat that
        purpose and add real friction to the common, harmless case of a
        model rephrasing the same topic slightly. Fixed narrowly: only
        the *silence* is the bug, not the merge -- each entry now
        records the literal topic string it was actually written under
        whenever that differs from the file's own original heading, so
        the file stays a complete, honest record instead of quietly
        misattributing content to whichever topic string happened to
        create the file first."""
        path = self._path_for(topic)
        lock_path = path.with_suffix(".md.lock")
        stripped_topic = topic.strip()
        with exclusive_lock(lock_path):
            file_exists = path.is_file()
            existing = path.read_text(encoding="utf-8") if file_exists else f"# {stripped_topic}\n"
            # A real bug found by a fresh-eyes sweep: only "the file
            # doesn't exist yet" was special-cased -- a file that
            # EXISTS but is empty (0 bytes) fell straight through to
            # `existing.splitlines()[0]`, and `"".splitlines()` is `[]`,
            # not `[""]`, so indexing `[0]` raised a raw, uncaught
            # `IndexError` before `atomic_write_text` was ever reached,
            # silently losing the write. Not a contrived state: this
            # store's own docstring calls these files "human-readable
            # ... a person can open in any editor and read or hand-edit
            # directly," so a user clearing a note file's contents (or
            # any external process leaving it empty) is ordinary use,
            # not adversarial -- confirmed live before this fix.
            # `NoteTool.run()` only catches `LongTermMemoryError`, so
            # this leaked past its own documented "reject, don't guess"
            # discipline every sibling case here already follows
            # correctly, saved only by `AgentLoop.run_one()`'s
            # incidental broad `except Exception` one layer up, which
            # any other/future direct caller wouldn't have. Fixed by
            # treating an existing-but-blank file the same as
            # "doesn't exist yet" for heading purposes: there's no real
            # original topic left to compare against or preserve, so
            # the file is reseeded with a fresh `# Topic` heading
            # exactly like a brand-new file gets, rather than crashing.
            if not existing.strip():
                existing = f"# {stripped_topic}\n"
                file_exists = False
            original_topic = existing.splitlines()[0].removeprefix("#").strip()
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            heading = (
                f"## {timestamp}"
                if not file_exists or stripped_topic == original_topic
                else f'## {timestamp} (topic: "{stripped_topic}")'
            )
            entry = f"{heading}\n\n{content.strip()}\n"
            atomic_write_text(path, existing.rstrip("\n") + "\n\n" + entry + "\n")
            os.chmod(path, 0o600)
        return path

    def read(self, topic: str) -> str | None:
        path = self._path_for(topic)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def list_topics(self) -> list[str]:
        return sorted(p.stem for p in self._directory.glob("*.md"))

    def search(self, query: str) -> list[NoteMatch]:
        """Exact, case-insensitive substring search across every note's
        full text. One match per topic (the first occurrence), not one
        per occurrence -- this is meant to help a caller find WHICH
        topics are relevant, not enumerate every hit within one."""
        query_lower = query.lower()
        matches: list[NoteMatch] = []
        for path in sorted(self._directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            idx = text.lower().find(query_lower)
            if idx == -1:
                continue
            start = max(0, idx - 60)
            end = min(len(text), idx + len(query) + 60)
            snippet = text[start:end].strip().replace("\n", " ")
            matches.append(NoteMatch(topic=path.stem, snippet=snippet))
        return matches
