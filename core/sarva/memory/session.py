"""sarva.memory.session — file-based session persistence.

A session is a saved conversation: a list[Message], one JSON file per
session name. Deliberately simple and inspectable — `cat
~/.sarva/sessions/default.json` should just work.

Wired for both tool-free (`sarva chat`) and tool-using (`sarva run`)
conversations: `AgentLoop.run(transcript_out=...)` extends a caller-
supplied list in place with the complete final message history —
including every intermediate tool-call/tool-result round, not just the
final assistant turn — which both CLI commands pass straight to
`SessionStore.save()`. See `test_transcript_out_includes_full_tool_use_round`
in `tests/conformance/test_agent.py` for the real, tool-using proof.

**Written with owner-only (0700 dir / 0600 file) permissions**, the same
real gap found and fixed in `sarva.config`'s credential file: a saved
session can hold real tool-use output (file contents `ReadFileTool`
read, `RunShellTool` command output, anything the user typed) — at
least as sensitive as an API key, and until this fix was left at
whatever the platform default happened to be (`0644`/`0755` on this
machine's real umask, confirmed with a real `stat()` call, not
assumed). Same POSIX-only honesty as that fix: real on macOS/Linux,
not genuine per-user isolation on Windows.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
from pathlib import Path

from pydantic import TypeAdapter

from sarva.atomic_write import atomic_write_bytes
from sarva.multimodal.content import Message

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_MESSAGES_ADAPTER: TypeAdapter[list[Message]] = TypeAdapter(list[Message])

DEFAULT_SESSIONS_DIR = Path.home() / ".sarva" / "sessions"

_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# Comfortably under every common filesystem's ~255-byte filename limit even
# after this module's longest suffix (".lock") is appended -- every
# character in _VALID_NAME's charset is a single ASCII byte, so this is
# also an exact byte-length bound, not just a character-count heuristic.
_MAX_NAME_LENGTH = 200


def _sanitize(name: str) -> str:
    # Reject rather than silently strip — silently dropping characters risks
    # two distinct names (e.g. "my session" and "mysession") colliding onto
    # the same file, which would corrupt one or the other's history.
    if not _VALID_NAME.match(name):
        raise ValueError(f"invalid session name: {name!r} (use only letters, digits, '-', and '_')")
    # A real bug found by a fresh-eyes sweep: the character check above
    # says nothing about length, so a session name past the filesystem's
    # max filename length reached os.open() (in `locked`'s `_acquire`)
    # and raised a raw OSError (ENAMETOOLONG) -- a completely different
    # exception type than the ValueError every real call site (cli.py,
    # server/app.py) catches specifically because that's the only
    # exception type _sanitize() was ever documented to raise. Confirmed
    # live: POST /chat with a 300-character session name crashed with a
    # raw 500 (`OSError: [Errno 63] File name too long`); /ws/chat's
    # equivalent crash was worse -- the ASGI call died with no frame sent
    # at all, not even the clean failure pair this handler already sends
    # for every other validation error. Reachable through the server's
    # real public REST/WS surface: ChatRequest.session has no length
    # constraint, and the WS frame is raw schema-less JSON -- a buggy
    # session-id generator, a proxy that mangles a session parameter, or
    # simply a very long user-typed name all trigger this with no
    # adversarial intent needed. Fixed at the source, in _sanitize()
    # itself, rather than patching each of the many call sites
    # individually (cli.py alone has five) -- every existing `except
    # ValueError` already in place gets this fix for free, and any
    # future caller of _sanitize() automatically inherits it too.
    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(
            f"session name too long ({len(name)} chars, max {_MAX_NAME_LENGTH}): {name[:50]!r}..."
        )
    return name


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = root or DEFAULT_SESSIONS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        # Tightens a directory an older version of this module (or
        # anything else) already created with looser permissions, the
        # same self-healing os.chmod sarva.config's save_config uses.
        os.chmod(self.root, 0o700)

    def _path(self, name: str) -> Path:
        return self.root / f"{_sanitize(name)}.json"

    def load(self, name: str) -> list[Message]:
        path = self._path(name)
        if not path.exists():
            return []
        return _MESSAGES_ADAPTER.validate_json(path.read_text())

    def save(self, name: str, messages: list[Message]) -> None:
        # A real bug found by actually simulating an interrupted write:
        # the previous version opened the real session file directly
        # with O_TRUNC, which truncates it to 0 bytes immediately, before
        # a single byte of the new content is written. A crash (OOM-kill,
        # SIGKILL, power loss) between that open() and the write
        # completing destroyed a previously-good, real conversation
        # history -- confirmed live: a 150-byte valid session file became
        # 0 bytes, and `load()` then raised a pydantic ValidationError on
        # data that used to be perfectly fine. Fixed the standard way,
        # now shared via sarva.atomic_write rather than kept as this
        # method's own independent copy (see that module's docstring for
        # why the duplication itself was a real, separate risk).
        content = _MESSAGES_ADAPTER.dump_json(messages, indent=2)
        atomic_write_bytes(self._path(name), content, mode=0o600)

    def clear(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    @contextlib.asynccontextmanager
    async def locked(self, name: str | None):
        """Holds a real, cross-process exclusive lock (`flock` on POSIX,
        `msvcrt.locking` on Windows) on a dedicated `.lock` file sibling
        to the session file, for the entire duration of the caller's
        load-through-save turn -- the identical pattern `sarva.config`'s
        own `_exclusive_lock` already uses for the same reason, applied
        here for the first time.

        A no-op when `name` is `None`: a session-less turn has nothing
        to protect, and locking on a shared "no session" key would
        needlessly serialize every anonymous request against every
        other one.

        **A real bug found by actually racing two genuine OS processes
        against the same session** (not threads, not asyncio tasks
        within one process -- the already-fixed in-process race a prior
        milestone closed with a plain `asyncio.Lock` in `sarva.server.
        app` doesn't cover this): two separate `python` processes doing
        `load()` -> (a stand-in for a real, multi-second model call) ->
        `save()` against the same session file raced every time --
        10/10 trials -- with the loser's entire turn silently discarded,
        no error to either side. The concrete, realistic shape: `sarva
        chat --session default` running in a terminal at the same
        moment someone (or the same person, in a browser tab) is
        chatting into the `"default"` session on a running `sarva
        serve` instance, or two independent CLI invocations. Every
        session write already goes through `sarva.atomic_write` (so no
        caller ever sees a torn file), but atomicity of each individual
        write does nothing to serialize TWO separate load-modify-save
        cycles against each other -- the same distinction `sarva.
        config`'s own `_exclusive_lock` docstring draws between its
        atomic-write fix and its separate concurrent-write-race fix.

        The blocking `flock`/`msvcrt.locking` acquire call runs via
        `asyncio.to_thread` rather than directly on the event loop --
        acquiring this lock can genuinely block for as long as another
        process's real agent turn takes (a real, possibly many-second
        model call), and calling a blocking syscall directly from async
        code for that long would freeze every other unrelated request
        this same process is serving, not just the one waiting on this
        session. Once acquired, the lock is held by keeping the file
        object open for the whole `async with` block -- the thread is
        only needed for the acquire/release syscalls themselves, not to
        babysit the lock the whole time it's held.

        Deliberately a dedicated sibling `.lock` file, never the session
        file itself, so it never interferes with `save()`'s own atomic-
        rename mechanism -- and deliberately not required for `load()`
        on its own (a bare read, outside this context manager, still
        sees only a complete old-or-new version thanks to that same
        atomic write, never a torn one); locking only matters for
        serializing writers against each other, not readers against
        writers.

        **The lock file is opened without truncating it, and the marker
        byte written only if it isn't already there -- mirrors the
        identical fix in `sarva.config`'s own `_exclusive_lock` (see
        that function's own docstring for the real bug this closes: a
        second, contending caller's truncating `open(path, "wb")`
        rewrite on every acquisition attempt targets the exact byte
        range Windows' *mandatory* `msvcrt.locking()` already holds
        locked for a first caller, so the second caller's own `write()`
        call fails with a sharing-violation `OSError` before it ever
        reaches its own lock attempt -- contention crashing instead of
        waiting, the opposite of what this method exists to do). Not
        live-verified here (no Windows machine in this environment),
        reasoned from documented Win32/CRT mandatory-locking semantics."""
        if name is None:
            yield
            return

        lock_path = self._path(name).with_suffix(".lock")

        def _acquire():
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            f = os.fdopen(fd, "r+b")
            # Content is irrelevant -- this file exists purely as a lock
            # target -- but msvcrt.locking() requires the byte range
            # being locked to actually exist in the file, so one byte is
            # written the first time this lock file is ever used, never
            # again after (see this method's own docstring for why never
            # rewriting it matters, not just why it needs to exist).
            if f.seek(0, os.SEEK_END) < 1:
                f.seek(0)
                f.write(b"\0")
                f.flush()
            f.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            return f

        def _release(f) -> None:
            if sys.platform == "win32":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

        f = await asyncio.to_thread(_acquire)
        try:
            yield
        finally:
            await asyncio.to_thread(_release, f)
