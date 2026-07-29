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

import os
import re
from pathlib import Path

from pydantic import TypeAdapter

from sarva.atomic_write import atomic_write_bytes
from sarva.multimodal.content import Message

_MESSAGES_ADAPTER: TypeAdapter[list[Message]] = TypeAdapter(list[Message])

DEFAULT_SESSIONS_DIR = Path.home() / ".sarva" / "sessions"

_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _sanitize(name: str) -> str:
    # Reject rather than silently strip — silently dropping characters risks
    # two distinct names (e.g. "my session" and "mysession") colliding onto
    # the same file, which would corrupt one or the other's history.
    if not _VALID_NAME.match(name):
        raise ValueError(f"invalid session name: {name!r} (use only letters, digits, '-', and '_')")
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
