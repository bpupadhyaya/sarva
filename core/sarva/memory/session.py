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
"""

from __future__ import annotations

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

    def _path(self, name: str) -> Path:
        return self.root / f"{_sanitize(name)}.json"

    def load(self, name: str) -> list[Message]:
        path = self._path(name)
        if not path.exists():
            return []
        return _MESSAGES_ADAPTER.validate_json(path.read_text())

    def save(self, name: str, messages: list[Message]) -> None:
        self._path(name).write_bytes(_MESSAGES_ADAPTER.dump_json(messages, indent=2))

    def clear(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
