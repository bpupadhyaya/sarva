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

import json
import os
from pathlib import Path

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
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file at {path} is corrupted (invalid JSON): {e}") from e


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
    replaces the target inode entirely."""
    content = json.dumps(data, indent=2)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_config(values: dict[str, str], path: Path | None = None) -> None:
    """Merges `values` into whatever's already saved (a caller setting
    only `ANTHROPIC_API_KEY` doesn't wipe out a previously saved
    `OPENAI_API_KEY`), then writes the whole file back with owner-only
    permissions -- see this module's own docstring for why."""
    path = path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
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
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    return load_config(path).get(name)
