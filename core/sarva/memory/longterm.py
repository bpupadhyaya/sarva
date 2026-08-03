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

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


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
    if len(slug) > _MAX_TOPIC_SLUG_LENGTH:
        raise LongTermMemoryError(
            f"invalid topic name: {len(slug)} characters after slugifying, "
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
        is the first note under that topic."""
        path = self._path_for(topic)
        lock_path = path.with_suffix(".md.lock")
        with exclusive_lock(lock_path):
            existing = path.read_text() if path.is_file() else f"# {topic.strip()}\n"
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            entry = f"## {timestamp}\n\n{content.strip()}\n"
            atomic_write_text(path, existing.rstrip("\n") + "\n\n" + entry + "\n")
            os.chmod(path, 0o600)
        return path

    def read(self, topic: str) -> str | None:
        path = self._path_for(topic)
        return path.read_text() if path.is_file() else None

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
            text = path.read_text()
            idx = text.lower().find(query_lower)
            if idx == -1:
                continue
            start = max(0, idx - 60)
            end = min(len(text), idx + len(query) + 60)
            snippet = text[start:end].strip().replace("\n", " ")
            matches.append(NoteMatch(topic=path.stem, snippet=snippet))
        return matches
