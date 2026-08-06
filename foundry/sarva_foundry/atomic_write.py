"""sarva_foundry.atomic_write — the `sarva.atomic_write` helper, mirrored
here rather than imported: `core` has no dependency on `sarva_foundry`,
and `sarva_foundry` has no dependency on `core` (their `pyproject.toml`s
name completely disjoint dependency sets — `torch`/`numpy` vs.
`anthropic`/`openai`/`google-genai`/etc., the same boundary
`sarva.distill`'s own docstring documents). A real, checked gap: this
package's own checkpoint/tokenizer saves never got the atomic-write fix
`core` applied to `config.json`/session storage — confirmed live by
truncating a real, trained 76128-byte `DecoderOnlyTransformer` checkpoint
to simulate an interrupted `torch.save`, and `Trainer.load_checkpoint`
then raised `RuntimeError: PytorchStreamReader failed reading zip
archive: failed finding central directory` — a training run's actual
GPU-hours of progress, not just a config file, destroyed by a crash at
exactly the wrong moment.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Calls `write_fn(tmp_path)` to produce the new content at a
    sibling temp path in the same directory, then atomically renames it
    into place via `os.replace()` -- so `path` always ends up holding
    either the last fully-written version or the new one, never a
    partial one. Generic over `write_fn` on purpose: `Trainer.
    save_checkpoint` needs `torch.save`'s own zip-serialization logic,
    not a raw bytes/text write.

    Temp filename includes the calling thread's id, not just the
    process id -- mirrors the identical fix in `sarva.atomic_write`
    (see that module's own docstring for the live-confirmed race: a
    PID-only temp name collides across every thread of one process,
    since they all share one PID, so two threads writing the same
    `path` concurrently raced the same temp file and one of them raised
    an uncaught `FileNotFoundError`).

    The temp file is cleaned up if `write_fn` OR `os.replace()` itself
    raises, not left behind forever -- mirrors the identical fix in
    `sarva.atomic_write` (see that module's own docstring for both
    live-confirmed leaks this closes: a `write_fn` failure, e.g. a
    disk-full `torch.save` mid-checkpoint, and a real, non-hypothetical
    `os.replace()` failure on Windows, where a locked/open destination
    -- antivirus or backup software holding a handle, another reader
    mid-`open()` -- makes the rename itself raise `PermissionError`
    after `write_fn` already succeeded. Either way, the fully-written
    temp file used to leak permanently -- worse than just unhandled,
    since a leaked partial checkpoint consumes disk space, making the
    *next* save more likely to hit the same failure again)."""
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


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8", mode: int = 0o644) -> None:
    """A real, checked gap found by a fresh-eyes sweep, against this
    module's own docstring claim of mirroring `sarva.atomic_write`:
    that sibling's `atomic_write_text` writes through `atomic_write_bytes`,
    which `os.fsync()`s the temp file's contents before `atomic_write()`
    renames it into place -- this one used a plain `Path.write_text()`
    with no `fsync` at all. The temp-file+rename pattern already fixes
    the truncate-on-open scenario this module's own docstring names, but
    without `fsync`, the write is not durable against an OS-level
    crash/power loss (a laptop battery dying, a kernel panic, an
    unclean container shutdown -- all realistic for this project's own
    laptop-scale training story) between `write_text()` returning and
    the kernel actually flushing dirty pages to disk: the well-known
    "rename() without fsync() before it can still leave a zero-length
    or garbage file after a crash" filesystem gotcha, exactly why the
    `core` sibling this module claims to mirror already calls it.
    Confirmed live: monkeypatching `os.fsync` to count calls showed the
    old version never reached it at all. Fixed by writing through
    `os.open`/`os.fdopen` with an explicit `flush()` + `fsync()`,
    matching `sarva.atomic_write.atomic_write_bytes` exactly (including
    the `mode` parameter, for the same reason that sibling has one --
    anything holding sensitive data can pass a tighter mode)."""

    def _write(tmp_path: Path) -> None:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

    atomic_write(path, _write)
