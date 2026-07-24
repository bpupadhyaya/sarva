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
        content = _MESSAGES_ADAPTER.dump_json(messages, indent=2)
        path = self._path(name)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        # Covers a file that already existed with looser permissions --
        # os.open's mode only applies when it actually creates the file.
        os.chmod(path, 0o600)

    def clear(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
