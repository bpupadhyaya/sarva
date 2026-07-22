"""sarva.agent.tools — the tool contract, confirmation policies, and built-ins.

Tools declare `spec.destructive`; the loop — not the tool — decides whether
to gate on confirmation. This keeps the security policy in one place: an
"autonomous mode" is a policy swap (`always_allow`), not code edits per tool.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from sarva.memory.vector import DEFAULT_MEMORY_DB_PATH, VectorMemoryStore
from sarva.multimodal.content import TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ToolSpec

_MAX_FETCH_CHARS = 50_000


class ToolContext:
    """Passed to every tool invocation. `emit` is wired by the AgentLoop for
    transcript logging; tools never talk to the provider layer directly.
    `session_id` is optional and `None` by default — most tools don't need
    it; it exists so session-aware tools (e.g. `RememberTool`/`RecallMemoryTool`)
    can scope themselves to the actual conversation session a run belongs
    to, threaded from `AgentLoop.run(session_id=...)`, instead of falling
    back to a tool-constructor-time default that has no idea which
    conversation is actually running."""

    def __init__(
        self,
        workdir: str,
        run_dir: str,
        emit: Callable[[Any], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ):
        self.workdir = workdir
        self.run_dir = run_dir
        self.emit = emit or (lambda event: asyncio.sleep(0))
        self.session_id = session_id


class Tool(Protocol):
    spec: ToolSpec

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock: ...


ConfirmPolicy = Callable[[ToolCallBlock], Awaitable[bool]]


async def always_allow(call: ToolCallBlock) -> bool:
    """Autonomous-mode policy: never ask."""
    return True


def _within_workdir(workdir: str, path: str) -> Path:
    resolved = (Path(workdir) / path).resolve()
    root = Path(workdir).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes workdir: {path!r}")
    return resolved


class ReadFileTool:
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file relative to the working directory.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        p = _within_workdir(ctx.workdir, args["path"])
        text = p.read_text()
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])


class WriteFileTool:
    spec = ToolSpec(
        name="write_file",
        description="Write a UTF-8 text file relative to the working directory. "
        "Creates parent directories as needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        destructive=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        p = _within_workdir(ctx.workdir, args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=f"wrote {p}")])


class RunShellTool:
    """Destructive by default — the loop's default confirm policy asks first."""

    spec = ToolSpec(
        name="run_shell",
        description="Run a shell command in the working directory and return "
        "combined stdout+stderr.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        destructive=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        proc = await asyncio.create_subprocess_shell(
            args["command"],
            cwd=ctx.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        return ToolResultBlock(
            tool_call_id="",
            content=[TextBlock(text=stdout.decode(errors="replace"))],
            is_error=proc.returncode != 0,
        )


class WebFetchTool:
    """Non-destructive: read-only network access, no state changed."""

    spec = ToolSpec(
        name="web_fetch",
        description="Fetch the text content of an http(s) URL. Content is "
        f"truncated to {_MAX_FETCH_CHARS} characters.",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        url = args["url"]
        scheme = urlparse(url).scheme
        if scheme not in ("http", "https"):
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"unsupported URL scheme: {scheme!r}")],
                is_error=True,
            )
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                text = resp.text[:_MAX_FETCH_CHARS]
                if len(resp.text) > _MAX_FETCH_CHARS:
                    text += "\n\n[truncated]"
                return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])
        except httpx.HTTPError as e:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"fetch failed: {e}")],
                is_error=True,
            )


class RememberTool:
    """Non-destructive: appends to the memory store, never overwrites or
    deletes anything a user or the model already saved.

    Session-scoped via `ctx.session_id` when the loop was run with one
    (threaded from the CLI's `--session` flag / the server's `session`
    request field, through `AgentLoop.run(session_id=...)`) — falls back
    to `self._session_id` (default `"default"`) only when the run itself
    has no session identity (e.g. a one-shot `sarva chat` with no
    `--session`), so unrelated sessions' memories don't bleed together
    by default once a real session is in play.

    The default store is opened lazily, on first `run()`, not in
    `__init__` — `BUILTIN_TOOLS` below is a module-level list, so eager
    construction here would open (and create, via `VectorMemoryStore`'s
    own `mkdir`) a real SQLite file at `~/.sarva/memory.db` as a side
    effect of merely *importing* this module, on every machine that ever
    imports `sarva.agent.tools` — including test/CI runs that never
    otherwise touch the filesystem."""

    spec = ToolSpec(
        name="remember",
        description="Save a short note or fact to long-term memory for later recall.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    def __init__(self, store: VectorMemoryStore | None = None, session_id: str = "default"):
        self._store = store
        self._session_id = session_id

    def _get_store(self) -> VectorMemoryStore:
        if self._store is None:
            self._store = VectorMemoryStore(DEFAULT_MEMORY_DB_PATH)
        return self._store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        self._get_store().add(ctx.session_id or self._session_id, args["text"])
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text="Saved to memory.")])


class RecallMemoryTool:
    """Non-destructive: read-only search over the memory store. See
    `RememberTool`'s docstring for both the session-scoping rule
    (`ctx.session_id` preferred, `self._session_id` as fallback) and why
    the default store is opened lazily rather than at
    `__init__`/module-import time."""

    spec = ToolSpec(
        name="recall_memory",
        description="Search previously remembered notes/facts for ones relevant to a query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    def __init__(self, store: VectorMemoryStore | None = None, session_id: str = "default"):
        self._store = store
        self._session_id = session_id

    def _get_store(self) -> VectorMemoryStore:
        if self._store is None:
            self._store = VectorMemoryStore(DEFAULT_MEMORY_DB_PATH)
        return self._store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        top_k = args.get("top_k", 5)
        session_id = ctx.session_id or self._session_id
        results = self._get_store().search(args["query"], top_k=top_k, session_id=session_id)
        if not results:
            text = "No relevant memories found."
        else:
            text = "\n".join(f"- {entry.text} (relevance {score:.2f})" for entry, score in results)
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])


BUILTIN_TOOLS: list[Tool] = [
    ReadFileTool(),
    WriteFileTool(),
    RunShellTool(),
    WebFetchTool(),
    RememberTool(),
    RecallMemoryTool(),
]
