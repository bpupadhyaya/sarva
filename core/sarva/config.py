"""sarva.config — a real, persistent config file for provider API keys.

**Written with owner-only (0600) permissions, not the platform default.**
A real, checked gap found by inspecting an actual saved file's mode
bits, not assumed: `Path.write_text`'s default `open()` mode (0666,
reduced by the process umask) left `~/.sarva/config.json` at 0644 on
this machine's real umask (022) -- world-readable, for a file whose
entire purpose is holding plaintext Anthropic/OpenAI/Gemini API keys.
On any shared machine (a real, common case this project's own "free
for everyone" audience includes -- shared dev servers, lab machines,
CI runners with persistent home directories), any other local user
could read another user's credentials straight off disk. `save_config`
now creates the file via `os.open(..., 0o600)` directly (no
create-then-chmod race window where it's briefly world-readable) and
`os.chmod`s it explicitly afterward too, so a file an older version of
this module already created insecurely gets tightened on the very next
save rather than staying exposed forever. **Honestly platform-scoped:**
this is a real, meaningful boundary on POSIX (macOS/Linux, verified
against actual `stat()` mode bits); on Windows, `os.chmod`'s real
effect is limited to toggling the read-only attribute, not genuine
per-user ACL isolation -- true multi-user protection there would need
the Windows ACL APIs, real, separate, deferred work rather than
silently assumed equivalent to the POSIX fix.

Closes a gap the desktop app's own promised first-run flow depends on:
the design doc's own T4 definition of done and the README's own
quickstart text both promise a guided first run that offers "paste an
API key" as an alternative to a local model — but until now there was
nowhere for a pasted key to actually go. `sarva.runtime`'s availability
checks, and every provider's SDK client, only ever looked at real
process environment variables; a key entered once in a UI had no way to
survive past that single process's lifetime.

`~/.sarva/config.json` — the same `~/.sarva/` home this project already
uses for session storage (`sarva.memory.session`). A flat dict of
provider env-var names to values (e.g. `{"ANTHROPIC_API_KEY": "sk-..."}`),
deliberately the exact same names `sarva.runtime` already checks via
`os.environ`, so nothing downstream needs a second, parallel notion of
"which key is this."

**Precedence, stated explicitly and tested, not left implicit:** a real
environment variable always wins over a saved config value. A user who
explicitly exported a key in their shell almost certainly means for it
to take effect for that session; silently overriding an explicit env
var with a stale saved file would be a confusing, hard-to-debug
surprise, the same category of "don't guess when you don't have to"
principle this project applies elsewhere (e.g. session-name validation
rejecting rather than silently sanitizing).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from sarva.atomic_write import atomic_write_text

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

DEFAULT_CONFIG_PATH = Path.home() / ".sarva" / "config.json"

# The exact env-var names sarva.runtime checks -- kept here as the one
# place both sides agree on the set of names this module manages.
KNOWN_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


class ConfigError(RuntimeError):
    """Raised when `~/.sarva/config.json` exists but isn't valid JSON --
    disk corruption, an interrupted write, or a hand-edit gone wrong. A
    real, checked gap found by actually corrupting a config file and
    running `sarva doctor`/`chat`/`config show`: `get_env()` backs
    nearly every provider-availability check in `sarva.runtime`, so a
    bad file crashed almost every command and server endpoint with a
    raw `json.JSONDecodeError` traceback -- the broadest blast radius of
    any "unhandled exception where a clean error belongs" bug found in
    this project so far. Deliberately its own exception type, not a
    `ValueError` -- callers already have `except ValueError` blocks
    scoped to unrelated failures (an invalid session name); reusing that
    base would risk this being silently swallowed by the wrong handler
    with a misleading message instead of failing clearly."""


def load_config(path: Path | None = None) -> dict[str, str]:
    """Returns `{}` if no config file exists yet — a fresh install with
    nothing saved is the expected common case, not an error. A file that
    exists but isn't valid JSON is a different, real failure and raises
    `ConfigError` rather than propagating a raw `JSONDecodeError` --
    every caller (CLI commands, server endpoints) needs a clean, single
    exception type to catch, not a leaky implementation detail of how
    this file happens to be encoded."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file at {path} is corrupted (invalid JSON): {e}") from e


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """Holds a real, cross-process exclusive lock (`flock` on POSIX,
    `msvcrt.locking` on Windows) on a dedicated sibling `.lock` file --
    never the config file itself, so it never interferes with
    `_write_config`'s own atomic-rename mechanism -- for the entire
    duration of the caller's read-modify-write critical section.

    A real bug found by actually simulating two concurrent callers:
    `save_config`/`unset_config` both did a plain, unlocked
    read-then-write (`existing = load_config(path); ...; _write_config(
    path, existing)`). Two callers (e.g. the CLI and the desktop app,
    or two CLI invocations) racing this sequence can both read the SAME
    "before" state, each add a different key, and each write their own
    merged dict back -- the second write silently overwrites the
    first's key with no error, a genuine lost update, confirmed live:
    a real `OPENAI_API_KEY` saved by one caller vanished entirely after
    a second, concurrent caller's own unrelated save. The already-fixed
    atomic-write-on-save bug (`_write_config`'s own docstring) makes
    each individual write crash-safe; it does nothing to serialize TWO
    separate read-modify-write cycles against each other -- a different
    bug class, closed here instead.

    Deliberately a dedicated `.lock` file, never the config file: a
    reader (`load_config`/`get_env`) never needs to acquire this lock at
    all, since `_write_config`'s atomic `os.replace()` already
    guarantees a reader only ever sees a complete old or new version,
    never a torn one -- locking is only needed to serialize writers
    against EACH OTHER, not readers against writers.

    **The lock file is opened without truncating it, and the marker
    byte written only if it isn't already there -- a real bug found by
    reasoning through what a SECOND, contending caller actually does on
    Windows.** The previous version re-opened the lock file with
    truncating `"wb"` mode and rewrote the marker byte on every single
    acquisition attempt. On POSIX that's harmless (`flock()` is purely
    advisory -- it doesn't interact with ordinary reads/writes at all),
    but on Windows, `msvcrt.locking()` is a *mandatory* byte-range lock
    the OS enforces against **any** I/O to that region from other
    processes, including a plain `write()` call that never itself calls
    `locking()`. A second caller's truncating open + write targets
    exactly the byte range the first caller already holds locked -- that
    write fails with a sharing-violation `OSError` *before* the second
    caller ever reaches its own `msvcrt.locking()` call, defeating the
    intended blocking-retry semantics entirely: contention crashes the
    second caller instead of making it wait, the opposite of what this
    function exists to do. Not live-verified here (no Windows machine in
    this environment) -- reasoned from documented Win32/CRT mandatory-
    locking semantics, the same honesty this project already applies to
    every other Windows-only code path, not assumed correct by analogy
    to the POSIX branch. Fixed with `os.open(..., O_CREAT)` (no
    `O_TRUNC`): the marker byte is written once, only if the file turns
    out to be empty, and every later acquisition just opens and locks
    the already-populated file without writing to it at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # A real bug found by a fresh-eyes sweep, the identical fix this
    # module's own docstring already claims for config.json's own file
    # (0600, not the platform default) but never applied to the
    # directory it lives in: sarva.memory.session.SessionStore,
    # sarva.memory.longterm.LongTermMemoryStore, and sarva.memory.
    # vector.VectorMemoryStore all explicitly os.chmod their own
    # `~/.sarva/...` subdirectory to 0o700 right after mkdir -- this
    # module, whose own docstring most directly promises the "shared
    # dev servers, lab machines, CI runners" hardening, never picked up
    # the same fix for `~/.sarva` itself. Confirmed live on a fresh
    # install: `~/.sarva` was left at 0o755 (world-readable, other
    # local users can list its contents and confirm a key is
    # configured) under the common 022 umask this module's own
    # docstring already tested against for the file -- and at 0o777
    # (genuinely world-WRITABLE, letting another local user unlink/
    # rename config.json out from under the app regardless of the
    # file's own 0600 bits) under a real, if less common, 000 umask.
    # Fixed here, in the one function both save_config and unset_config
    # already route every write through, rather than duplicated at each
    # call site.
    os.chmod(path.parent, 0o700)
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


def _write_config(path: Path, data: dict[str, str]) -> None:
    """Writes `data` to `path` atomically, with owner-only permissions.

    A real bug found by actually simulating an interrupted write: both
    `save_config` and `unset_config` used to open the real config file
    directly with `O_TRUNC`, which truncates it to 0 bytes immediately
    -- before a single byte of the new content is written. A crash
    (OOM-kill, SIGKILL, power loss) between that `open()` and the write
    completing destroyed every previously-saved key, not just failed to
    save the new one -- confirmed live: a valid config file with a real
    saved key became 0 bytes, and the next `load_config()` call raised
    `ConfigError` on data that used to be perfectly fine. Fixed the
    standard way: write to a sibling temp file first, then
    `os.replace()` into place -- atomic on both POSIX and Windows, so
    the real path always holds either the last fully-written version or
    the new one, never a partial one. `os.replace()` also means the
    resulting file's permissions are exactly the temp file's (0600),
    with no separate `os.chmod` needed afterward -- a POSIX rename
    replaces the target inode entirely.

    Delegates the actual temp-file-then-`os.replace()` mechanics to
    `sarva.atomic_write` -- this function used to have its own inline
    copy of that logic, independently reinvented from
    `sarva.memory.session.SessionStore.save`'s identical fix rather than
    sharing it, the exact "same fix, implemented twice, could silently
    diverge" risk a later sweep of this codebase found repeated at
    several OTHER real call sites too (`WriteFileTool`, foundry
    checkpoint saving, `distill.save_jsonl`) -- see
    `sarva.atomic_write`'s own docstring. Centralizing here closes that
    gap for this module's own two writers, not just the new ones."""
    atomic_write_text(path, json.dumps(data, indent=2), mode=0o600)


def save_config(values: dict[str, str], path: Path | None = None) -> None:
    """Merges `values` into whatever's already saved (a caller setting
    only `ANTHROPIC_API_KEY` doesn't wipe out a previously saved
    `OPENAI_API_KEY`), then writes the whole file back with owner-only
    permissions -- see this module's own docstring for why. The whole
    read-modify-write cycle holds `_exclusive_lock` (see its own
    docstring for the real concurrent-write race this closes)."""
    path = path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with _exclusive_lock(lock_path):
        existing = load_config(path)
        existing.update(values)
        _write_config(path, existing)


def unset_config(names: list[str], path: Path | None = None) -> list[str]:
    """Removes `names` from the saved config file, if present -- `set`'s
    missing counterpart, closed the same milestone `sarva config
    set`/`show` shipped in rather than left as a real, separate gap: a
    key saved by mistake, or a user switching back to relying purely on
    an env var, had no way to actually remove it short of hand-editing
    or deleting `~/.sarva/config.json` outright (which would also lose
    every *other* saved key). A name not currently saved is silently a
    no-op, not an error -- "make sure this key isn't saved" is a
    reasonable request regardless of whether it happened to be saved in
    the first place. Returns the names that were actually removed, so a
    caller (the CLI) can report exactly what changed rather than
    assuming every requested name was present."""
    path = path or DEFAULT_CONFIG_PATH
    lock_path = path.with_name(f"{path.name}.lock")
    with _exclusive_lock(lock_path):
        existing = load_config(path)
        removed = [name for name in names if name in existing]
        for name in removed:
            del existing[name]
        if removed:
            _write_config(path, existing)
    return removed


def get_env(name: str, path: Path | None = None) -> str | None:
    """What `sarva.runtime` should treat env-var `name` as being set to:
    a real process environment variable if set, else whatever's saved in
    the config file, else `None`. Every provider-key check in
    `sarva.runtime` goes through this instead of `os.environ.get`
    directly, so config-file support can't accidentally be forgotten at
    a new call site."""
    # A real bug found by a fresh-eyes sweep, the same "truthiness
    # treats a legitimate empty/falsy-but-valid value as absent" shape
    # already found and fixed twice in this project's audio subsystem:
    # `os.environ.get(name)` correctly distinguishes "not set" (`None`)
    # from "set to empty string" (`""`), but `if env_value:` collapsed
    # both into the same branch. An env var a caller has explicitly set
    # to `""` -- the ordinary shell idiom `ANTHROPIC_API_KEY= sarva ...`
    # used to override/clear an inherited or previously-saved value for
    # one invocation, not a contrived input -- was indistinguishable
    # here from the variable never being exported at all, so this
    # silently fell through to whatever's saved in config.json instead,
    # directly contradicting this function's own docstring ("a real
    # process environment variable if set, else whatever's saved"): the
    # env var IS set, just empty. Confirmed live through the real public
    # surface (run_diagnostics/build_providers): a previously-`sarva
    # config set`-saved key silently resurrected itself and got used to
    # construct a real, authenticated provider client, even after the
    # user explicitly cleared the env var for that run in their own
    # shell. `is not None` matches `os.environ.get`'s own real
    # None/"" distinction instead of collapsing it via truthiness.
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value
    return load_config(path).get(name)
