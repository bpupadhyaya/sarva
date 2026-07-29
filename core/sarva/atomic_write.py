"""sarva.atomic_write — one shared helper for "write real, valuable data
to disk without a crash mid-write destroying whatever was there before,"
used everywhere in `core` this project persists data that matters.

A real, systemic gap found the same way an earlier reward-hacking bug
was: `sarva.config`/`sarva.memory.session` each independently invented
their own temp-file-then-`os.replace()` fix for an interrupted-write
data-loss bug, but that discipline never propagated to the *other* real
call sites in this package writing equally real, pre-existing data
directly (`WriteFileTool.run()`, `distill.save_jsonl()`,
`save_checkpoint_bundle()`'s own `config.json` write) — a fix applied
once, never centralized, so nothing carried it to structurally
identical code elsewhere. Confirmed live for each: `WriteFileTool` is
the tool the agent loop uses on essentially every file-editing turn,
operating on arbitrary real user files, not just this project's own
state — a crash mid-write (OOM-kill, `SIGKILL`, power loss, disk-full)
truncated a real 5000-byte file to 0 bytes the instant `write_text()`
opened it, before a single byte of new content was written.

One shared helper now, instead of a fourth hand-rolled copy: every
future writer of real data in `core` should reach for this rather than
re-deriving the same fix (or forgetting to) a fifth time.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Calls `write_fn(tmp_path)` to produce the new content at a
    sibling temp path in the same directory, then atomically renames it
    into place via `os.replace()` — so `path` always ends up holding
    either the last fully-written version or the new one, never a
    partial one, regardless of what `write_fn` actually does (a plain
    text/bytes write, or something more involved). `os.replace()` is
    atomic on both POSIX and Windows. Generic over *how* the content is
    produced deliberately — `save_checkpoint_bundle`'s siblings in
    `sarva_foundry` need `torch.save`'s own file-writing logic, not a
    raw bytes/text write, and mirror this exact helper for that reason
    (`sarva_foundry.atomic_write` — a real, intentional duplication:
    `core` has no dependency on `sarva_foundry`, and `sarva_foundry` has
    no dependency on `core`, the same disjoint-dependency boundary
    `sarva.distill`'s own docstring already documents)."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o644) -> None:
    """The common case: write literal bytes, created via `os.open(...,
    mode)` directly rather than the platform-default `open()` mode —
    matters for anything holding sensitive data (callers writing
    credentials/session content pass `mode=0o600`)."""

    def _write(tmp_path: Path) -> None:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    atomic_write(path, _write)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8", mode: int = 0o644) -> None:
    atomic_write_bytes(path, content.encode(encoding), mode=mode)
