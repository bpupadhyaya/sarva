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
from urllib.parse import urljoin, urlparse

import httpx

from sarva.agent.events import AgentResult
from sarva.agent.subagents import DelegateTool
from sarva.atomic_write import atomic_write_text
from sarva.memory.longterm import (
    DEFAULT_LONGTERM_MEMORY_DIR,
    LongTermMemoryError,
    LongTermMemoryStore,
)
from sarva.memory.vector import DEFAULT_MEMORY_DB_PATH, VectorMemoryStore
from sarva.multimodal.content import TextBlock, ToolCallBlock, ToolResultBlock
from sarva.multimodal.fetch import FetchError, ensure_public_host, ssrf_safe_transport
from sarva.providers.base import ToolSpec

_MAX_FETCH_CHARS = 50_000
_MAX_REDIRECTS = 5
_SHELL_TIMEOUT_SECONDS = 60


class ToolContext:
    """Passed to every tool invocation. `emit` is wired by the AgentLoop for
    transcript logging; tools never talk to the provider layer directly.
    `session_id` is optional and `None` by default — most tools don't need
    it; it exists so session-aware tools (e.g. `RememberTool`/`RecallMemoryTool`)
    can scope themselves to the actual conversation session a run belongs
    to, threaded from `AgentLoop.run(session_id=...)`, instead of falling
    back to a tool-constructor-time default that has no idea which
    conversation is actually running.

    `spawn_subagent` is the one hook `DelegateTool` (`sarva.agent.
    subagents`) needs and no other built-in tool does: a closure built by
    `AgentLoop.run()` itself (the only place with a router/providers/
    budget/spend to build a subagent from), `None` in any context that
    doesn't support delegation (e.g. a bare ToolContext built directly in
    a test). Kept as a narrow closure rather than exposing the router/
    providers/tools themselves on ToolContext, so every OTHER tool's
    surface area stays exactly what it was before subagent fan-out
    existed. Signature matches spec-03's own frozen `ToolContext.
    spawn_subagent: Callable[..., Awaitable[AgentResult]]` -- `(task,
    task_class, budget)`, both keyword-defaultable so a simple caller
    like `DelegateTool` can pass just the task string."""

    def __init__(
        self,
        workdir: str,
        run_dir: str,
        emit: Callable[[Any], Awaitable[None]] | None = None,
        session_id: str | None = None,
        spawn_subagent: Callable[..., Awaitable[AgentResult]] | None = None,
    ):
        self.workdir = workdir
        self.run_dir = run_dir
        self.emit = emit or (lambda event: asyncio.sleep(0))
        self.session_id = session_id
        self.spawn_subagent = spawn_subagent


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
        # Atomic write, not a direct p.write_text(): this tool runs on
        # essentially every agent file-editing turn, against arbitrary
        # real user files -- not just this project's own state. A crash
        # mid-write (OOM-kill, SIGKILL, power loss) between write_text()
        # truncating the file and the new content landing destroys
        # whatever was there before, confirmed live by writing a real
        # 5000-byte file and simulating that exact crash moment: the file
        # became 0 bytes. See sarva.atomic_write for the shared fix, the
        # same one sarva.config/sarva.memory.session already use.
        p = _within_workdir(ctx.workdir, args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, args["content"])
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=f"wrote {p}")])


class EditFileTool:
    """Targeted edit: replaces an exact occurrence of `old_string` with
    `new_string` rather than rewriting the whole file the way
    `WriteFileTool` always does. Named in the design doc's own §3.5
    built-in-tools line ("file read/write/edit") -- the one of the three
    not built until now. Mirrors the same exact-match-required,
    reject-ambiguous-matches shape already proven throughout this
    project's own development (the identical contract this project's
    own coding assistant's file-editing tool uses): an `old_string` that
    doesn't appear at all, or appears more than once without
    `replace_all=true`, is a clean, actionable error rather than a
    guess at which occurrence was meant.

    File-not-found/permission errors are deliberately NOT caught here --
    same convention `ReadFileTool`/`WriteFileTool` already establish,
    relying on the loop's own generic tool-dispatch exception handling
    for those; only genuinely tool-specific validation (an empty or
    ambiguous `old_string`) gets an explicit is_error result here."""

    spec = ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact occurrence of old_string with new_string in a file, "
            "without rewriting the rest of the file. old_string must match exactly "
            "once unless replace_all is true, or an unrelated occurrence could be "
            "edited by mistake."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        destructive=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        p = _within_workdir(ctx.workdir, args["path"])
        old_string = args["old_string"]
        new_string = args["new_string"]
        replace_all = args.get("replace_all", False)
        if not old_string:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text="old_string must not be empty")],
                is_error=True,
            )
        if old_string == new_string:
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(text="old_string and new_string are identical -- nothing to do")
                ],
                is_error=True,
            )
        # p.read_text() -- NOT used here on purpose: Python's text-mode
        # file reading does universal-newlines translation by default
        # (\r\n and \r both silently become \n), and nothing on the write
        # side translates back. A real bug found by actually editing one
        # line of a real CRLF file: every OTHER line's ending silently
        # flipped to LF too, directly contradicting this tool's own
        # "without rewriting the rest of the file" contract -- confirmed
        # live, `line1\r\nline2\r\nline3\r\n` with only "line2" changed
        # came back as `line1\nLINE2\nline3\n`, every line ending
        # rewritten. Reading raw bytes and decoding directly (bytes.decode()
        # does no newline translation at all) preserves whatever the file
        # actually had, byte for byte, for every line this edit doesn't
        # touch.
        text = p.read_bytes().decode("utf-8")
        count = text.count(old_string)
        if count == 0:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"old_string not found in {p}")],
                is_error=True,
            )
        if count > 1 and not replace_all:
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text=f"old_string matches {count} times in {p} -- add more surrounding "
                        "context to old_string, or pass replace_all=true"
                    )
                ],
                is_error=True,
            )
        n = count if replace_all else 1
        new_text = text.replace(old_string, new_string, n)
        atomic_write_text(p, new_text)
        plural = "s" if n != 1 else ""
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=f"edited {p} ({n} replacement{plural})")]
        )


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
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_SHELL_TIMEOUT_SECONDS)
        except TimeoutError:
            # A real bug found by actually running a long-lived shell
            # command against a shortened timeout: asyncio.wait_for()
            # only cancels the *awaiting* communicate() call -- it never
            # touches the child process itself, confirmed directly with
            # a real `sleep`-then-`echo` command still alive (and its
            # trailing side effect still completing) seconds after the
            # "timeout." That matters specifically because this tool is
            # `destructive=True` -- the whole confirmation gate exists to
            # stop unwanted side effects, and a silent timeout defeated
            # it by leaving the command running unattended regardless of
            # what the user actually approved.
            proc.kill()
            await proc.wait()
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text=f"command timed out after {_SHELL_TIMEOUT_SECONDS}s and was killed"
                    )
                ],
                is_error=True,
            )
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
            # follow_redirects is deliberately off and replaced with a
            # bounded manual loop that re-validates the target host on
            # EVERY hop, not just the caller-supplied URL -- an initial
            # URL can be a legitimate public site whose server issues a
            # redirect straight to an internal address, which a
            # validate-once-up-front check would never catch.
            #
            # A real bug found by actually running ensure_public_host
            # against a fake DNS resolver: the check alone is
            # TOCTOU-vulnerable to DNS rebinding -- it resolves once
            # here, then httpx's own default connection logic
            # independently re-resolves the SAME hostname a moment
            # later, so a resolver answering the first query publicly
            # and every later one with a private address slips straight
            # through, landing the real connection on an internal
            # server anyway. transport=ssrf_safe_transport() is the
            # actual enforcement layer that closes this -- it resolves
            # and validates a hostname exactly once per real TCP
            # connection, so the validated answer and the connected-to
            # address can never diverge. Especially important here: this
            # tool is destructive=False, reachable with zero user
            # confirmation, the exact metadata-exfiltration threat model
            # WebFetchTool's own original SSRF fix targeted.
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=15.0, transport=ssrf_safe_transport()
            ) as client:
                for _ in range(_MAX_REDIRECTS + 1):
                    await ensure_public_host(url)
                    resp = await client.get(url)
                    if resp.is_redirect and resp.has_redirect_location:
                        url = urljoin(str(resp.url), resp.headers["location"])
                        continue
                    resp.raise_for_status()
                    text = resp.text[:_MAX_FETCH_CHARS]
                    if len(resp.text) > _MAX_FETCH_CHARS:
                        text += "\n\n[truncated]"
                    return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"too many redirects fetching {args['url']!r}")],
                is_error=True,
            )
        except httpx.HTTPError as e:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"fetch failed: {e}")],
                is_error=True,
            )
        except FetchError as e:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=str(e))],
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

    def _add(self, session_id: str, text: str) -> None:
        self._get_store().add(session_id, text)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        # A real bug found by giving this tool the same sweep that just
        # found NoteTool blocking the whole event loop on a contended
        # cross-process lock (see VectorMemoryStore.__init__'s own
        # docstring for the confirmed repro): `add()` can block for up
        # to sqlite3's own 5-second default timeout waiting for another
        # writer's lock, and this ran directly on the event loop with no
        # `asyncio.to_thread`. `_get_store()`'s own lazy construction
        # (its `VectorMemoryStore(...)` call does its own `CREATE TABLE
        # IF NOT EXISTS` + commit) is just as capable of blocking on that
        # same contended lock as `add()` itself, so both are dispatched
        # together in `_add` -- routing only `add()` through
        # `asyncio.to_thread` and leaving `_get_store()` called directly
        # here would have left the identical freeze reachable on a
        # tool's very first call.
        await asyncio.to_thread(self._add, ctx.session_id or self._session_id, args["text"])
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

    def _search(self, query: str, top_k: int, session_id: str | None):
        return self._get_store().search(query, top_k=top_k, session_id=session_id)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        top_k = args.get("top_k", 5)
        session_id = ctx.session_id or self._session_id
        # See RememberTool.run's own comment -- the identical
        # event-loop-blocking gap (including the same lazy-construction
        # angle), same fix.
        results = await asyncio.to_thread(self._search, args["query"], top_k, session_id)
        if not results:
            text = "No relevant memories found."
        else:
            text = "\n".join(f"- {entry.text} (relevance {score:.2f})" for entry, score in results)
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])


class NoteTool:
    """Non-destructive: appends to a durable, CROSS-SESSION markdown note
    under a topic, never overwrites or deletes anything already written.

    Deliberately NOT session-scoped, unlike `RememberTool` -- this is the
    design doc's own "long-term memory as plain markdown files" tier
    (`sarva.memory.longterm`), meant to persist and be visible across
    every future conversation, organized by topic rather than by
    session. Use `remember`/`recall_memory` for session-scoped semantic
    notes; use this for durable, human-readable knowledge a person could
    also open directly in a text editor.

    The default store is opened lazily, on first `run()`, not in
    `__init__` -- the same reason `RememberTool`'s docstring gives:
    `BUILTIN_TOOLS` below is a module-level list, so eager construction
    here would create real files on disk as a side effect of merely
    *importing* this module."""

    spec = ToolSpec(
        name="note",
        description=(
            "Save a durable note under a topic to long-term memory, visible across "
            "every future conversation (not just this session). Stored as plain, "
            "human-readable markdown -- use for knowledge worth keeping long-term, "
            "not transient details of the current task."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "A short topic name, e.g. 'user-preferences'.",
                },
                "content": {"type": "string"},
            },
            "required": ["topic", "content"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    def __init__(self, store: LongTermMemoryStore | None = None):
        self._store = store

    def _get_store(self) -> LongTermMemoryStore:
        if self._store is None:
            self._store = LongTermMemoryStore(DEFAULT_LONGTERM_MEMORY_DIR)
        return self._store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        # A real bug found by actually racing a genuine second OS process
        # holding the per-topic flock (see LongTermMemoryStore.write's own
        # docstring) against this tool's own call into it: `write()` is a
        # fully synchronous method that blocks on that flock for as long
        # as another writer holds it -- calling it directly from this
        # `async def` runs it straight on the event loop with no
        # `asyncio.to_thread`, exactly the mistake `SessionStore.locked`'s
        # own docstring already documents fixing once for the identical
        # flock-blocking shape. Confirmed live: a real second process
        # holding the topic's `.md.lock` for 3s froze this process's
        # ENTIRE event loop for the whole window -- a heartbeat coroutine
        # that should have ticked roughly every 0.05s recorded ZERO ticks
        # across the full 2.7s call, meaning every other in-flight
        # `/chat`/`/ws/chat` turn in a real `sarva serve` process would
        # have frozen too, not just this one tool call. `SearchNotesTool`
        # below only ever reads (never contends on this lock), so it's
        # unaffected and left as-is.
        try:
            path = await asyncio.to_thread(self._get_store().write, args["topic"], args["content"])
        except LongTermMemoryError as e:
            return ToolResultBlock(tool_call_id="", content=[TextBlock(text=str(e))], is_error=True)
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=f"Noted under {path.stem!r}.")]
        )


class SearchNotesTool:
    """Non-destructive: read-only, exact substring search over every
    long-term note. Deliberately not semantic (that's `recall_memory`'s
    job) -- the whole point of this tier is that it's plain, greppable
    text, so search matches that promise directly rather than
    duplicating the other tier's own similarity ranking."""

    spec = ToolSpec(
        name="search_notes",
        description="Search long-term notes (across every past conversation) for exact text.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    def __init__(self, store: LongTermMemoryStore | None = None):
        self._store = store

    def _get_store(self) -> LongTermMemoryStore:
        if self._store is None:
            self._store = LongTermMemoryStore(DEFAULT_LONGTERM_MEMORY_DIR)
        return self._store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        matches = self._get_store().search(args["query"])
        if not matches:
            text = "No notes matched."
        else:
            text = "\n".join(f"- {m.topic}: ...{m.snippet}..." for m in matches)
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])


BUILTIN_TOOLS: list[Tool] = [
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    RunShellTool(),
    WebFetchTool(),
    RememberTool(),
    RecallMemoryTool(),
    NoteTool(),
    SearchNotesTool(),
    DelegateTool(),
]
