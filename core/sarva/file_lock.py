"""sarva.file_lock — a small, reusable cross-process exclusive lock
(`flock` on POSIX, `msvcrt.locking` on Windows) on a dedicated sibling
`.lock` file, for serializing concurrent read-modify-write callers
against a shared on-disk resource.

Extracted as its own module because this exact mechanism now has three
independent, hand-mirrored implementations in this codebase
(`sarva.config._exclusive_lock`, `sarva.memory.session.SessionStore.
locked`, and `sarva.memory.longterm.LongTermMemoryStore`'s own use of
this module) — the same "centralize once duplicated three times"
discipline already applied to `sarva.atomic_write`. `config.py`/
`session.py` are deliberately NOT refactored onto this in the same
change that introduces it (a real, named follow-up, not silently
forgotten) to keep the long-term-memory feature's own diff focused on
one thing; a future pass can fold them in without re-deriving the
reasoning below.

Design, carried over unchanged from the two prior implementations
(both already found and fixed the same real bugs; repeating them here
would be a regression, not a coincidence):

- **A dedicated `.lock` file, never the resource file itself.** A
  reader never needs to acquire this lock at all as long as the
  resource's own writes are atomic (`sarva.atomic_write`) — locking is
  only needed to serialize writers against EACH OTHER, not readers
  against writers.
- **The lock file is opened without truncating it, and the marker byte
  written only if it isn't already there.** A version that re-opens
  with truncating `"wb"` mode and rewrites the marker byte on every
  acquisition is harmless on POSIX (`flock()` is purely advisory) but
  breaks on Windows: `msvcrt.locking()` is a *mandatory* byte-range
  lock the OS enforces against ANY I/O to that region from other
  processes, including a plain `write()` that never itself calls
  `locking()`. A second, contending caller's truncating write targets
  exactly the byte range the first caller already holds locked, failing
  with a sharing-violation `OSError` before the second caller ever
  reaches its own `msvcrt.locking()` call — contention crashes instead
  of waiting, the opposite of what this exists to do. Fixed with
  `os.open(..., O_CREAT)` (no `O_TRUNC`): the marker byte is written
  once, only if the file turns out to be empty.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Holds a real, cross-process exclusive lock on `path` (a dedicated
    lock file, created if missing) for the duration of the `with` block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    f = os.fdopen(fd, "r+b")
    try:
        # Content is irrelevant -- this file exists purely as a lock
        # target -- but msvcrt.locking() requires the byte range being
        # locked to actually exist in the file, so one byte is written
        # the first time this lock file is ever used, never again after.
        if f.seek(0, os.SEEK_END) < 1:
            f.seek(0)
            f.write(b"\0")
            f.flush()
        f.seek(0)
        if sys.platform == "win32":
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()
