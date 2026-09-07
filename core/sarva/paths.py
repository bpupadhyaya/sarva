"""sarva.paths — the one place every `~/.sarva`-relative path agrees on
where that home directory actually is.

A real gap found live, by this project's own testing practice: nothing
anywhere in this codebase ever checked an environment variable for an
override — `sarva.config.DEFAULT_CONFIG_PATH`, `sarva.memory.session.
DEFAULT_SESSIONS_DIR`, `sarva.memory.vector.DEFAULT_MEMORY_DB_PATH`, and
`sarva.memory.longterm.DEFAULT_LONGTERM_MEMORY_DIR` each independently
hardcoded `Path.home() / ".sarva" / <subpath>` with no way to redirect
any of them. Confirmed live and consequential, not hypothetical:
several real end-to-end test rounds in this project's own development
assumed a `SARVA_HOME` environment variable would sandbox `sarva serve`/
`sarva chat` invocations away from the real user's home directory —
it silently did nothing, since no such variable was ever read anywhere,
and every one of those "isolated" test runs actually wrote real session
files (session names, message counts, timestamps) straight into this
machine's genuine `~/.sarva/sessions/`, discovered only when `sarva
sessions list` on the real, unmodified home directory turned up dozens
of test-named sessions that were never meant to persist. The same
exposure applies to anyone else trying to test Sarva locally, run it in
CI, or run more than one isolated profile on one machine — a real,
ordinary need, not an edge case.

Every one of the four call sites above now goes through `sarva_home()`
below instead of recomputing `Path.home() / ".sarva"` independently —
the identical "one shared function instead of four places that could
drift" reasoning `sarva.atomic_write`'s own consolidation already
applies. `SARVA_HOME`, once set, redirects config, sessions, vector
memory, and long-term memory together, as one coherent profile, rather
than needing four separate overrides that could disagree.
"""

from __future__ import annotations

import os
from pathlib import Path


def sarva_home() -> Path:
    """The base directory every `~/.sarva`-relative path is computed
    from. Honors a `SARVA_HOME` environment variable override (checked
    fresh on every call, not cached, so a test suite's `monkeypatch.
    setenv` takes effect immediately without needing module reload
    tricks); falls back to `~/.sarva`, unchanged from every prior
    version of this project."""
    override = os.environ.get("SARVA_HOME")
    if override:
        return Path(override)
    return Path.home() / ".sarva"
