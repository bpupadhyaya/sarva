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
import threading
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
    `sarva.distill`'s own docstring already documents).

    **The temp filename includes the calling thread's id, not just the
    process id — a real bug found by actually racing two real threads
    against this exact function.** A PID-only temp name
    (`{path.name}.tmp-{os.getpid()}`) is unique across processes but
    identical across every thread of the SAME process, since all
    threads in one process share one PID. Two threads calling
    `atomic_write` on the same `path` concurrently then race the exact
    same temp file: both `write_fn` calls target it, and whichever
    thread's `os.replace()` runs second finds the temp path already
    renamed away by the winner and raises an uncaught `FileNotFoundError`
    — confirmed live, deterministically, 10/10 trials with two real
    `threading.Thread`s. Not yet reachable through any current call site
    (each one either already serializes writes some other way, e.g.
    `sarva.config`'s own separate `flock`, or is never actually invoked
    from two genuinely concurrent threads today) — but this module's own
    docstring invites every future writer of real data to use this
    helper without re-auditing it, so the helper itself needs to be
    correct under concurrency, not just under its current callers.
    Thread id is genuinely unique among the threads alive at any one
    moment (unlike a PID, which is stable but shared by every thread of
    one process), so appending it removes the collision entirely.

    **The temp file is cleaned up if `write_fn` raises, not left behind
    forever -- a real bug found by actually making `write_fn` raise
    partway through a real write and checking the directory afterward.**
    A crash-safe *target* file (the whole point of this module) still
    leaked its own *temp* file on every write failure: nothing ever
    removed the sibling `path.tmp-<pid>-<tid>` this function creates,
    so a disk-full write, an interrupted `torch.save`, or any other
    `write_fn` failure left a stray, permanently orphaned file on disk
    with no code path anywhere that ever cleaned it up. Confirmed live:
    a `write_fn` that wrote 15000 bytes then raised left exactly that
    file sitting on disk after the exception propagated. Worst for
    `Trainer.save_checkpoint` (multi-GB checkpoint writes during long
    training runs) -- a disk-full failure there is a realistic scenario
    that this bug made actively worse, not just unhandled: the leaked
    partial checkpoint consumes disk space, making the *next* save more
    likely to fail too, the same underlying cause compounding itself.
    Fixed with a `try`/`except`/`finally`-shaped cleanup: on any
    exception from `write_fn`, the temp file is removed (best-effort --
    a failure to remove it doesn't mask or replace the original
    exception, which always still propagates) before re-raising.

    **That fix only covered `write_fn` failing, not `os.replace()`
    itself -- a real bug found by a third sweep of this exact module,
    re-reading it line by line specifically because it had already had
    two real bugs found in it.** The rename call sat *after* the
    `try` block, uncovered by the same cleanup. If `os.replace()` itself
    raises -- a real, non-hypothetical case on Windows, where `os.
    replace` maps to `MoveFileEx`/`MOVEFILE_REPLACE_EXISTING` and fails
    with a sharing-violation `PermissionError` when the destination is
    open or locked by another process (antivirus/backup software
    holding a handle, or another reader mid-`open()`) -- the temp file
    (by this point fully, successfully written by `write_fn`) leaked
    exactly like the already-fixed case, just reached through a
    different trigger the first fix didn't cover. Confirmed live:
    monkeypatching `os.replace` to raise *after* `write_fn` had already
    produced valid content at the temp path left that fully-written
    file on disk, permanently, while the original target was correctly
    untouched (a leak, not data loss -- but the identical unbounded-
    accumulation-under-repeated-failure concern the first fix's own
    docstring already named). Fixed by widening the `try` block to cover
    `os.replace()` too, so any failure at either step -- not just
    `write_fn`'s -- triggers the identical cleanup."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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
