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
    an uncaught `FileNotFoundError`)."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    def _write(tmp_path: Path) -> None:
        tmp_path.write_text(content, encoding=encoding)

    atomic_write(path, _write)
