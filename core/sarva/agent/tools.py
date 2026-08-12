"""sarva.agent.tools — the tool contract, confirmation policies, and built-ins.

Tools declare `spec.destructive`; the loop — not the tool — decides whether
to gate on confirmation. This keeps the security policy in one place: an
"autonomous mode" is a policy swap (`always_allow`), not code edits per tool.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
import signal
import subprocess
import threading
import uuid
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from sarva.agent.events import AgentResult
from sarva.agent.subagents import DelegateTool
from sarva.atomic_write import atomic_write_bytes, atomic_write_text
from sarva.config import get_env
from sarva.memory.longterm import (
    DEFAULT_LONGTERM_MEMORY_DIR,
    LongTermMemoryError,
    LongTermMemoryStore,
    NoteMatch,
)
from sarva.memory.vector import DEFAULT_MEMORY_DB_PATH, VectorMemoryStore
from sarva.multimodal.content import ImageBlock, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.multimodal.fetch import FetchError, ensure_public_host, ssrf_safe_transport
from sarva.providers.base import ToolSpec

_MAX_FETCH_CHARS = 50_000
_MAX_REDIRECTS = 5
_SHELL_TIMEOUT_SECONDS = 60
# A real bug found by giving this tool its own fresh-eyes sweep, one
# layer beyond the already-fixed timeout-doesn't-kill-the-process gap:
# `proc.communicate()` buffers the ENTIRE stdout+stderr stream in
# memory with no size limit at all, unlike WebFetchTool right below
# (which caps its own output via _MAX_FETCH_CHARS). Confirmed live: a
# completely ordinary command (`yes A | head -c 300MB` -- the same
# shape as `git log -p`, a verbose build, or `cat`-ing a large data
# file, not a contrived attack) drove real process peak RSS from ~45MB
# to ~1GB, and the full, untruncated 300MB string was then appended
# into `messages`, resent to the provider on every later turn of the
# same run. Bounded here by actually stopping the READ (and killing
# the still-running process, the same "don't leave an unwanted side
# effect running unattended" reasoning the timeout fix above already
# established) once the cap is hit, not by capturing everything and
# truncating the string afterward -- capturing-then-truncating would
# still incur the full memory spike before ever throwing the excess
# away.
_MAX_SHELL_OUTPUT_BYTES = 1_000_000

_MAX_SEARCH_RESULTS = 5
_SEARCH_TIMEOUT_SECONDS = 15.0

# Official, minimal, freely-available images -- no private registry, no
# paid tier. `bash:5` (not `debian:*-slim`) because it's the one
# official image that ships a real `/bin/bash` with nothing else to
# choose between; `python:3.12-slim` matches this project's own minimum
# supported Python version (core/pyproject.toml's `requires-python`).
_CODE_EXEC_IMAGES = {"python": "python:3.12-slim", "bash": "bash:5"}
_CODE_EXEC_TIMEOUT_SECONDS = 30
_MAX_CODE_OUTPUT_BYTES = 200_000
_CONTAINER_RUNTIME_PROBE_TIMEOUT_SECONDS = 5.0

# black-forest-labs/FLUX.1-schnell: confirmed Apache-2.0 (genuinely free
# for personal AND commercial use, unlike several other popular
# open-weight image models -- checked directly against the model's own
# license file, not assumed from name recognition; e.g. Stability AI's
# own SD-Turbo/SDXL-Turbo require a paid membership for commercial use
# above a threshold, so they're the paid-tier shape this tool's local
# path deliberately avoids as its DEFAULT). 4-step distilled generation
# keeps local inference plausible without a GPU, though still slow on
# CPU alone -- honestly not fast, but genuinely free with no account.
_IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"
_IMAGE_GEN_STEPS = 4
IMAGE_GEN_TIMEOUT_SECONDS = 600
_OPENAI_IMAGE_MODEL = "dall-e-3"


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
        # A real bug found by a fresh-eyes sweep: `read_text()` with no
        # `encoding=` uses `locale.getpreferredencoding(False)`, not
        # UTF-8 -- on this project's own minimum Python (3.12; PEP 686's
        # default-UTF-8 mode doesn't land until 3.15) that's genuinely
        # locale-dependent, not automatically UTF-8, on musl-libc
        # containers (Alpine), Windows without UTF-8 mode enabled, or
        # PYTHONCOERCECLOCALE=0 -- all realistic `sarva serve`/`sarva
        # run` deployment targets, not adversarial ones. Confirmed live:
        # a file this tool's own WriteFileTool sibling had just written
        # (always UTF-8 via atomic_write_text) crashed with
        # UnicodeDecodeError reading it straight back, despite this
        # tool's own description promising "Read a UTF-8 text file."
        # WriteFileTool/EditFileTool already force UTF-8 explicitly;
        # this, the plainest and first of the three, never got the same
        # fix.
        # A real bug found by a fresh-eyes sweep: the identical
        # `asyncio.to_thread` fix already applied four separate times in
        # this same file (RememberTool._add, RecallMemoryTool._search,
        # NoteTool._write, SearchNotesTool._search -- each with its own
        # confirmed live repro of the event loop freezing solid for the
        # duration of a slow/contended-disk or network-mounted-filesystem
        # I/O call) never propagated to this tool, or to WriteFileTool/
        # EditFileTool below -- the three tools an agent actually calls
        # on essentially every file-editing turn, against arbitrary real
        # user files, not just this project's own state. Confirmed live
        # with a real `Path.read_text` slowed to simulate a contended
        # disk: a heartbeat coroutine that should tick every 0.05s made
        # ZERO ticks during the whole blocking call, meaning every OTHER
        # concurrent user's in-flight `/chat`/`/ws/chat` turn in a real
        # `sarva serve` process would freeze too, for as long as one file
        # read takes.
        text = await asyncio.to_thread(p.read_text, encoding="utf-8")
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

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, content)

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
        #
        # A real bug found by a fresh-eyes sweep, the same sibling-
        # propagation gap fixed for ReadFileTool above: both the
        # directory creation and the atomic write are real, synchronous
        # filesystem I/O, called directly on the event loop with no
        # `asyncio.to_thread` -- confirmed live with the identical
        # zero-heartbeat-ticks repro used for ReadFileTool. `_write`
        # bundles both calls together (mirroring NoteTool's own
        # `_write`, which bundles its lazy store construction with its
        # actual write for the same reason), dispatched as one unit
        # through `asyncio.to_thread`.
        p = _within_workdir(ctx.workdir, args["path"])
        await asyncio.to_thread(self._write, p, args["content"])
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
        # A real bug found by a fresh-eyes sweep, the same shape already
        # found and fixed for RecallMemoryTool's top_k (see its own
        # comment): `spec.input_schema` above declares `replace_all` as
        # `{"type": "boolean"}`, but nothing enforces that against a
        # real model's actual tool-call arguments -- input_schema is
        # purely descriptive, never validated server-side. `not
        # replace_all` below treats the value as a plain Python truth
        # value, and any non-empty string is truthy -- confirmed live: a
        # model emitting the JSON *string* `"false"` for `replace_all`
        # (a real, plausible model mistake, not a contrived attack) was
        # treated as `replace_all=True`, the exact opposite of what was
        # asked. On a real file with three occurrences of `old_string`,
        # this silently replaced all three instead of erroring on the
        # ambiguity this tool's own docstring says it exists to prevent
        # -- a destructive tool (`spec.destructive=True`) doing the
        # wrong, unrequested thing with no error and no signal, not a
        # crash. Fixed by rejecting anything that isn't a real Python
        # bool, the same "reject, don't guess" treatment RecallMemoryTool
        # already got for the identical gap.
        if not isinstance(replace_all, bool):
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"replace_all must be a boolean, got {replace_all!r}")],
                is_error=True,
            )
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
        #
        # A real bug found by a fresh-eyes sweep, the same sibling-
        # propagation gap fixed for ReadFileTool/WriteFileTool above:
        # this read is real, synchronous filesystem I/O, called directly
        # on the event loop with no `asyncio.to_thread` -- confirmed live
        # with the identical zero-heartbeat-ticks repro. Worse than
        # ReadFileTool alone, since this tool does BOTH a blocking read
        # here and a blocking write below in the same call.
        raw = await asyncio.to_thread(p.read_bytes)
        text = raw.decode("utf-8")
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
        # The write half of the same fix as the read above.
        await asyncio.to_thread(atomic_write_text, p, new_text)
        plural = "s" if n != 1 else ""
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=f"edited {p} ({n} replacement{plural})")]
        )


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kills the WHOLE process group the shell command runs in, not just
    the shell interpreter `proc` itself -- a real deadlock found while
    verifying the size-cap fix below, before it ever shipped: a shell
    pipeline (e.g. `yes A | head -c N`) forks multiple real child
    processes connected by their own internal pipe, and `proc.kill()`
    only sends SIGKILL to the shell that spawned them, not to those
    children. Confirmed live: killing just the shell after stopping an
    early read left the pipeline's actual commands alive and orphaned,
    still blocked trying to write into a pipe nothing was draining
    anymore -- `await proc.wait()` then hung indefinitely waiting for a
    shell that itself died instantly but whose still-running children
    kept the underlying pipe's write end open. `start_new_session=True`
    (set at the call site) puts the shell in its own process group at
    spawn time specifically so this function can reach every process in
    it via `os.killpg`, not just the one PID this project happens to be
    tracking."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already exited on its own between the timeout/cap firing and this call
    await proc.wait()


async def _read_stream_bounded(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Reads from `stream` until EOF or until strictly more than `limit`
    bytes have been accumulated, whichever comes first -- returns
    `(data[:limit], truncated)`. Deliberately stops the READ itself
    rather than reading everything and slicing the result afterward:
    the latter would still incur the full memory cost of whatever the
    stream produced before throwing the excess away, exactly the gap
    this function exists to close for `RunShellTool` below."""
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks), False
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)[:limit], True


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
        # start_new_session=True (POSIX only) puts the shell in its own
        # process group so `_kill_process_group` below can actually
        # terminate the WHOLE pipeline, not just the shell interpreter
        # itself -- see that function's own docstring for the real
        # deadlock this closes, found while verifying the size-cap fix
        # immediately below, before it ever shipped.
        proc = await asyncio.create_subprocess_shell(
            args["command"],
            cwd=ctx.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # `_SHELL_TIMEOUT_SECONDS` is meant to bound the WHOLE command,
        # not just the output read below -- deadline-based so the later
        # `proc.wait()` gets whatever's left of the same budget, not a
        # fresh one.
        deadline = asyncio.get_running_loop().time() + _SHELL_TIMEOUT_SECONDS
        try:
            stdout, truncated = await asyncio.wait_for(
                _read_stream_bounded(proc.stdout, _MAX_SHELL_OUTPUT_BYTES),
                timeout=_SHELL_TIMEOUT_SECONDS,
            )
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
            await _kill_process_group(proc)
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text=f"command timed out after {_SHELL_TIMEOUT_SECONDS}s and was killed"
                    )
                ],
                is_error=True,
            )
        if truncated:
            # The same "don't leave an unwanted side effect running
            # unattended" reasoning as the timeout branch above: once
            # we've decided to stop reading, leaving the process alive
            # to keep producing output nobody will see is exactly the
            # gap that branch already exists to close, just reached via
            # a size cap instead of a time cap.
            await _kill_process_group(proc)
        else:
            # A real bug found by a fresh-eyes sweep: `_read_stream_
            # bounded` returns as soon as the stdout PIPE hits EOF --
            # which happens the instant every process holding the
            # write end closes its stdout/stderr, an ordinary shell
            # idiom (`exec 1>&- 2>&-`, or any command that redirects
            # away its own fds and keeps running -- backgrounding/
            # daemonizing a job is the common real case, not a
            # contrived one). That left `_SHELL_TIMEOUT_SECONDS`
            # bounding only the READ, not the command itself: `await
            # proc.wait()` here had no timeout at all, so a command
            # that closed its fds early but kept running was left
            # completely unbounded -- confirmed live, a command
            # configured against a 2s timeout that closed its fds then
            # slept for 8s ran the full 8s and returned `is_error=False`
            # with no indication the timeout never engaged. The same
            # "a destructive tool's own confirmation gate exists to
            # stop unwanted side effects, and a defeated timeout leaves
            # it running unattended regardless of what was approved"
            # reasoning as the read-timeout branch above -- `proc.wait()`
            # now shares the exact same overall deadline the read
            # already started counting down from, not a fresh budget.
            try:
                await asyncio.wait_for(
                    proc.wait(), timeout=max(0.0, deadline - asyncio.get_running_loop().time())
                )
            except TimeoutError:
                await _kill_process_group(proc)
                return ToolResultBlock(
                    tool_call_id="",
                    content=[
                        TextBlock(
                            text=f"command timed out after {_SHELL_TIMEOUT_SECONDS}s and was killed"
                        )
                    ],
                    is_error=True,
                )
        text = stdout.decode(errors="replace")
        if truncated:
            text += f"\n\n[truncated to {_MAX_SHELL_OUTPUT_BYTES} bytes and killed]"
        return ToolResultBlock(
            tool_call_id="",
            content=[TextBlock(text=text)],
            is_error=proc.returncode != 0,
        )


async def _find_container_runtime() -> str | None:
    """Returns the absolute path to a real, working container runtime
    binary (`docker` preferred, `podman` as a free, open-source
    fallback), or `None` if neither is genuinely usable right now.

    Checked with a real, short-timeout `<binary> info` call, not just
    `shutil.which` -- a binary can be installed with its daemon not
    running (Docker Desktop quit, an entirely ordinary state on a
    laptop, not a contrived one), which `which` alone can't distinguish
    from "usable this instant." `run_code` below must never silently
    fall back to running code unsandboxed if the daemon merely looks
    installed but isn't actually reachable -- the same "reject, don't
    guess" discipline this project already applies to malformed
    on-disk state elsewhere."""
    for binary in ("docker", "podman"):
        path = shutil.which(binary)
        if path is None:
            continue
        try:
            proc = await asyncio.create_subprocess_exec(
                path, "info", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            returncode = await asyncio.wait_for(
                proc.wait(), timeout=_CONTAINER_RUNTIME_PROBE_TIMEOUT_SECONDS
            )
        except (TimeoutError, OSError):
            continue
        if returncode == 0:
            return path
    return None


async def _kill_container(runtime: str, name: str, proc: asyncio.subprocess.Process) -> None:
    """Stops the actual container the daemon is running, not just the
    local `docker run`/`podman run` CLI client process `proc` refers to.

    `docker run` (no `-d`) is a thin client that blocks in the
    foreground attached to a container the DAEMON actually owns and
    runs -- killing the client process, the same `_kill_process_group`
    pattern `RunShellTool` above uses for its own direct child process,
    does not stop the container itself (this is Docker's own documented
    client/daemon split, not a Sarva-specific quirk). Left unaddressed,
    a timed-out `run_code` call would leave the real container running
    orphaned on the daemon, `--network none`-isolated but still
    consuming the CPU/memory/pids quota it was given, for as long as
    whatever code it's running keeps going -- the exact "don't leave an
    unwanted side effect running unattended" gap `RunShellTool`'s own
    timeout fix (see its own docstring above) already exists to close,
    one layer deeper here because the process doing the actual work
    isn't the one this code can `os.killpg` directly.
    `<runtime> kill <name>` is best-effort (the container may have
    already exited on its own between the timeout firing and this
    call); the client process is killed either way, matching
    `_kill_process_group`'s own "kill unconditionally, then wait"
    shape."""
    killer = await asyncio.create_subprocess_exec(
        runtime, "kill", name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        await asyncio.wait_for(killer.wait(), timeout=_CONTAINER_RUNTIME_PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


class RunCodeTool:
    """Destructive by default, the same conservative default
    `RunShellTool` uses -- despite real isolation (`--network none`,
    read-only root filesystem, dropped Linux capabilities, resource
    limits, no host filesystem mount), a container escape is a real,
    if rare, residual risk class, and this project's own RL training
    harness (`sarva_foundry.train.rl_environment`) has already found
    repeated, real bypasses of a DIFFERENT, weaker execution boundary --
    "isolated" is treated here as a real mitigation, never assumed to be
    an absolute guarantee. The loop's confirm-gate (the same one that
    already protects `RunShellTool`) is the actual safety net, not this
    tool's isolation alone.

    No unsandboxed fallback exists anywhere in this class: if no
    container runtime is genuinely reachable, `run` returns a clear,
    honest error rather than ever running the given code directly on
    the host -- the entire point of this tool over `RunShellTool` is
    the isolation guarantee, so silently downgrading to "just run it"
    on a missing dependency would be worse than refusing outright.
    """

    spec = ToolSpec(
        name="run_code",
        description=(
            "Execute a code snippet in an isolated, network-disabled sandbox "
            "container. Requires Docker or Podman (both free, open source) to be "
            "installed and running -- returns a clear error if neither is "
            "reachable, never falls back to running code unsandboxed. Supports "
            "'python' and 'bash'. The container has no access to the host "
            "filesystem, this run's own working directory, or the network. "
            f"Output is truncated at {_MAX_CODE_OUTPUT_BYTES:,} bytes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": sorted(_CODE_EXEC_IMAGES)},
                "code": {"type": "string"},
            },
            "required": ["language", "code"],
            "additionalProperties": False,
        },
        destructive=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        language = args["language"]
        code = args["code"]
        if language not in _CODE_EXEC_IMAGES:
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text=f"unsupported language {language!r} -- supported: "
                        f"{', '.join(sorted(_CODE_EXEC_IMAGES))}"
                    )
                ],
                is_error=True,
            )
        runtime = await _find_container_runtime()
        if runtime is None:
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text="run_code requires a real, running container runtime for "
                        "genuine sandbox isolation -- neither Docker (docker.com) nor "
                        "Podman (podman.io) was found reachable. Install and start one "
                        "(both free and open source) to enable this tool; it never runs "
                        "code unsandboxed as a fallback."
                    )
                ],
                is_error=True,
            )
        image = _CODE_EXEC_IMAGES[language]
        interpreter = (
            ["python3", "-u", "-c", code] if language == "python" else ["bash", "-c", code]
        )
        # A random, unique name (not the auto-generated one `docker run`
        # would otherwise pick) so `_kill_container` above can target
        # exactly this container on timeout, never a sibling from a
        # concurrent run_code call.
        container_name = f"sarva-run-code-{uuid.uuid4().hex}"
        cmd = [
            runtime,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            image,
            *interpreter,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Deadline-based, matching RunShellTool's own already-fixed
        # "the read timeout and the wait timeout must share one overall
        # budget, not each get their own fresh one" pattern above.
        deadline = asyncio.get_running_loop().time() + _CODE_EXEC_TIMEOUT_SECONDS
        try:
            output, truncated = await asyncio.wait_for(
                _read_stream_bounded(proc.stdout, _MAX_CODE_OUTPUT_BYTES),
                timeout=_CODE_EXEC_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await _kill_container(runtime, container_name, proc)
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text=f"code timed out after {_CODE_EXEC_TIMEOUT_SECONDS}s and was killed"
                    )
                ],
                is_error=True,
            )
        if truncated:
            await _kill_container(runtime, container_name, proc)
        else:
            try:
                await asyncio.wait_for(
                    proc.wait(), timeout=max(0.0, deadline - asyncio.get_running_loop().time())
                )
            except TimeoutError:
                await _kill_container(runtime, container_name, proc)
                return ToolResultBlock(
                    tool_call_id="",
                    content=[
                        TextBlock(
                            text=f"code timed out after {_CODE_EXEC_TIMEOUT_SECONDS}s "
                            "and was killed"
                        )
                    ],
                    is_error=True,
                )
        text = output.decode(errors="replace")
        if truncated:
            text += f"\n\n[truncated to {_MAX_CODE_OUTPUT_BYTES:,} bytes and killed]"
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=text)], is_error=proc.returncode != 0
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
                    # client.stream(), not client.get(): the same
                    # "buffer everything then truncate the string" gap
                    # already fixed for RunShellTool's stdout capture --
                    # `client.get()` fully downloads the whole response
                    # body into memory before `.text` is ever sliced, so
                    # a response with a large-but-plausible body (no
                    # adversarial intent needed -- a big JSON API
                    # response, an uncompressed log file, a large HTML
                    # page) incurs the full download's memory cost
                    # before any of it is thrown away. Streaming and
                    # stopping the READ itself once `_MAX_FETCH_CHARS`
                    # is reached bounds that cost directly, the same
                    # "stop reading, don't read-then-discard" fix shape.
                    #
                    # A real bug found by actually fetching an ordinary,
                    # non-adversarial page (en.wikipedia.org, not a
                    # crafted target): with no `User-Agent` header set
                    # here, httpx sends its own default
                    # (`python-httpx/<version>`), and Wikipedia -- like
                    # many real, legitimate sites, not just adversarial
                    # anti-bot ones -- returns a raw 403 to that exact
                    # default, while a plain browser-like UA succeeds
                    # (confirmed directly with both, live). This is the
                    # identical fix `_duckduckgo_search` already applies
                    # for the identical reason (see its own comment) --
                    # it just never propagated to this sibling tool in
                    # the same file, so `web_fetch`'s reachability was
                    # silently narrower than `web_search`'s despite
                    # nothing about the target being unusual.
                    async with client.stream(
                        "GET",
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; sarva-agent/1.0)"},
                    ) as resp:
                        if resp.is_redirect and resp.has_redirect_location:
                            url = urljoin(str(resp.url), resp.headers["location"])
                            continue
                        resp.raise_for_status()
                        parts: list[str] = []
                        total = 0
                        truncated = False
                        async for chunk in resp.aiter_text():
                            parts.append(chunk)
                            total += len(chunk)
                            if total > _MAX_FETCH_CHARS:
                                truncated = True
                                break
                        text = "".join(parts)[:_MAX_FETCH_CHARS]
                        if truncated:
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


def _decode_ddg_redirect(href: str) -> str:
    """DuckDuckGo's own HTML results page never links to a result directly
    -- every `href` points at a same-site redirect endpoint
    (`//duckduckgo.com/l/?uddg=<url-encoded-target>&rut=...`) that wraps
    the real target URL as a query parameter. Decoded here so a caller
    (and the model reading this tool's output) sees the actual site a
    result points to, not an internal DDG redirect link it would have to
    resolve itself.

    A real bug found by a fresh-eyes sweep: `parse_qs` already fully
    percent-decodes each query parameter's value as part of parsing it
    (that's how `real` below recovers the real target URL's own `:`/`/`
    at all) -- so the extra `unquote(real)` this used to end with was a
    SECOND decode pass over an already-decoded string. Harmless for a
    target URL with no percent-encoding of its own, but confirmed live
    to corrupt an entirely ordinary one that does: a search result
    pointing at `https://example.com/search?q=hello%20world` (a genuine
    encoded space in the TARGET's own query string, not contrived --
    any multi-word query embedded in a URL looks like this) decoded to
    `https://example.com/search?q=hello world` instead -- the literal
    space breaks the string's own validity as a URL, and a caller that
    round-trips it back through `urlparse`/an HTTP client would send a
    different request than the one DDG actually pointed at. `parse_qs`'s
    own single decode already produces the correct, real target URL;
    nothing further needs to happen to it."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        real = parse_qs(parsed.query).get("uddg", [None])[0]
        if real:
            return real
    return href


class _DuckDuckGoResultParser(HTMLParser):
    """Extracts (title, url, snippet) triples from `html.duckduckgo.com`'s
    own no-JS HTML results page -- the same page duckduckgo.com/html/
    serves to any browser with JavaScript disabled, not a private or
    undocumented endpoint. Parsed with the stdlib `html.parser`, not a
    regex against HTML (regex can't correctly handle nested tags/
    attribute-quoting edge cases) and not a new third-party HTML-parsing
    dependency -- this project's own from-scratch-first bias, applied to
    a case narrow enough that hand-writing it is genuinely simpler than
    auditing a general-purpose library for something this small.

    Each real result is a `<h2 class="result__title"><a
    class="result__a" href="...">Title</a></h2>` followed later by a
    sibling `<a class="result__snippet">Snippet text</a>` -- title and
    snippet aren't nested in one tag, so a result is only finalized once
    its snippet's closing tag arrives, not its title's."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._title = ""
        self._url = ""
        self._snippet = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        classes = (dict(attrs).get("class") or "").split()
        if "result__a" in classes:
            self._in_title = True
            self._title = ""
            self._url = dict(attrs).get("href") or ""
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        elif self._in_snippet:
            self._snippet += data

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._in_title:
            self._in_title = False
        elif self._in_snippet:
            self._in_snippet = False
            if self._title.strip() and self._url:
                self.results.append(
                    {
                        "title": self._title.strip(),
                        "url": _decode_ddg_redirect(self._url),
                        "snippet": self._snippet.strip(),
                    }
                )
            self._title = self._url = self._snippet = ""


def _search_transport() -> httpx.AsyncBaseTransport | None:
    """The transport `WebSearchTool` sends requests through -- `None`
    (httpx's own real default) in production. Tests monkeypatch this to
    an `httpx.MockTransport` instead of hitting a real network endpoint,
    the same substitution pattern `WebFetchTool`'s own `ssrf_safe_transport`
    already uses. Not SSRF-guarded like `WebFetchTool`: both search
    endpoints below are fixed, hardcoded hosts this tool itself chooses,
    never a caller/model-supplied URL -- the threat model SSRF guards
    against (an attacker steering a fetch at an internal address) doesn't
    apply to a destination this code picks itself."""
    return None


async def _duckduckgo_search(query: str) -> list[dict[str, str]]:
    """The free, keyless default -- no API key, no signup, matching this
    project's own "free tier must truly be free" design principle.
    `html.duckduckgo.com/html/` is DuckDuckGo's own no-JS results page;
    a plain browser User-Agent is sent because DuckDuckGo (like most
    search frontends) serves a different, script-only page to requests
    that look like an empty/unknown client rather than to a genuine
    scripted absence of a UA."""
    async with httpx.AsyncClient(
        timeout=_SEARCH_TIMEOUT_SECONDS, transport=_search_transport()
    ) as client:
        resp = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; sarva-agent/1.0)"},
        )
        resp.raise_for_status()
    parser = _DuckDuckGoResultParser()
    parser.feed(resp.text)
    return parser.results[:_MAX_SEARCH_RESULTS]


async def _brave_search(query: str, api_key: str) -> list[dict[str, str]]:
    """The optional, explicitly opt-in paid path -- only ever used when a
    user has already configured their own `BRAVE_API_KEY` (env or
    `sarva config set`), never a requirement. A real, higher-quality
    index for a user who wants one; the free DuckDuckGo path above stays
    fully functional with zero configuration for everyone else."""
    async with httpx.AsyncClient(
        timeout=_SEARCH_TIMEOUT_SECONDS, transport=_search_transport()
    ) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": _MAX_SEARCH_RESULTS},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        resp.raise_for_status()
    web_results = resp.json().get("web", {}).get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in web_results[:_MAX_SEARCH_RESULTS]
    ]


class WebSearchTool:
    """Non-destructive: read-only network access, no state changed.

    Free by default: DuckDuckGo's own keyless HTML search endpoint, no
    API key or signup required (design doc §2's "free tier must truly be
    free" principle, applied to tools the same way it's already applied
    to the provider layer). If a Brave Search API key is configured, that
    higher-quality paid index is used instead -- always an explicit
    upgrade a user opts into by saving their own key, never a
    requirement to search the web at all.
    """

    spec = ToolSpec(
        name="web_search",
        description="Search the web for a query and return the top "
        f"{_MAX_SEARCH_RESULTS} results (title, URL, snippet). Free by "
        "default (DuckDuckGo, no API key needed); uses a configured "
        "BRAVE_API_KEY instead if one is set.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        query = args["query"]
        if not query.strip():
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text="query must not be empty")],
                is_error=True,
            )
        brave_key = get_env("BRAVE_API_KEY")
        try:
            results = (
                await _brave_search(query, brave_key)
                if brave_key
                else await _duckduckgo_search(query)
            )
        except httpx.HTTPError as e:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"search failed: {e}")],
                is_error=True,
            )
        if not results:
            return ToolResultBlock(
                tool_call_id="", content=[TextBlock(text=f"no results found for {query!r}")]
            )
        lines = [
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
        ]
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text="\n".join(lines))])


class ImageGenerationTool:
    """Destructive=True, matching `WriteFileTool` -- this tool writes a
    real file to `path`, which can overwrite an existing one, and (via
    its optional paid fallback) can incur a real monetary cost, both
    real reasons the loop's confirm-gate should ask first.

    Free by default: a local, open-weight diffusion model
    (`black-forest-labs/FLUX.1-schnell`, genuinely Apache-2.0 -- see
    `_IMAGE_GEN_MODEL`'s own comment for why that license check
    specifically mattered here), matching design doc §2's "free tier
    must truly be free" principle the same way `WebSearchTool` already
    applies it. Requires the optional `sarva[image]` extra (torch +
    diffusers + transformers, all Apache/BSD-licensed commodity
    substrate, not vendored model code) -- NOT part of the default
    install, since the model itself is large and the free tier stays
    genuinely zero-cost only if this one heavy capability needs an
    explicit opt-in the way `sarva[foundry]`/`sarva[audio]` already do.

    If `sarva[image]` isn't installed but `OPENAI_API_KEY` is already
    configured (the same key already used for chat), falls back to the
    paid OpenAI Images API instead -- a real, explicit "additional
    option, only if you already have it" path per the author's own
    direction. Never preferred over the free local model when that IS
    installed; only reached when the free path genuinely isn't
    available."""

    spec = ToolSpec(
        name="generate_image",
        description=(
            "Generate an image from a text prompt and save it as a PNG file "
            "relative to the working directory. Free by default (a local, "
            "open-weight model -- requires the optional sarva[image] install, "
            "and can be slow without a GPU); falls back to the paid OpenAI "
            "Images API if sarva[image] isn't installed but OPENAI_API_KEY is "
            "configured."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["prompt", "path"],
            "additionalProperties": False,
        },
        destructive=True,
    )

    def __init__(self) -> None:
        self._pipeline: Any = None
        # Guards BOTH the lazy pipeline load AND every actual generation
        # call -- deliberately coarser than RememberTool's own
        # double-checked-locking pattern above (which only locks the
        # narrow first-use window): a diffusion pipeline is not
        # documented as safe for concurrent `__call__`s from multiple
        # threads on one instance, and typical hardware this runs on
        # (one GPU, or CPU alone) can't usefully run two generations in
        # parallel anyway, so serializing every local call is the
        # correct behavior here, not just an easy simplification.
        self._pipeline_lock = threading.Lock()

    def _generate_locally(self, prompt: str) -> bytes:
        with self._pipeline_lock:
            if self._pipeline is None:
                import torch
                from diffusers import FluxPipeline

                # cuda > mps (Apple Silicon) > cpu -- confirmed live in
                # this dev environment that FluxPipeline loads and runs
                # correctly on `cpu` (verified against a tiny public
                # test fixture, katuni4ka/tiny-random-flux, since no
                # multi-GB download of the real model was practical
                # here -- see docs/agent-loop.md for the full honest
                # scope of what is and isn't live-verified).
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
                dtype = torch.bfloat16 if device != "cpu" else torch.float32
                pipeline = FluxPipeline.from_pretrained(_IMAGE_GEN_MODEL, torch_dtype=dtype)
                self._pipeline = pipeline.to(device)
            result = self._pipeline(
                prompt=prompt,
                num_inference_steps=_IMAGE_GEN_STEPS,
                guidance_scale=0.0,
            )
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            return buf.getvalue()

    async def _generate_via_openai(self, prompt: str, api_key: str) -> bytes:
        import openai

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.images.generate(
            prompt=prompt,
            model=_OPENAI_IMAGE_MODEL,
            size="1024x1024",
            response_format="b64_json",
            n=1,
        )
        return base64.b64decode(response.data[0].b64_json)

    def _save(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, data)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        prompt = args["prompt"]
        if not prompt.strip():
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text="prompt must not be empty")],
                is_error=True,
            )
        path = _within_workdir(ctx.workdir, args["path"])

        try:
            import diffusers  # noqa: F401

            local_available = True
        except ImportError:
            local_available = False

        if local_available:
            try:
                # A real, documented limitation, not silently assumed
                # away: unlike RunShellTool's subprocess (which a real
                # OS signal can actually terminate), `asyncio.wait_for`
                # timing out here only stops AWAITING the thread --
                # Python has no supported API to forcibly kill a running
                # thread, so a genuinely hung/slow generation keeps
                # running to completion in the background rather than
                # truly stopping. Harmless to the event loop itself
                # (the work is already off it, on a thread-pool worker),
                # but the timeout below bounds how long THIS call waits,
                # not how long the underlying inference actually runs.
                data = await asyncio.wait_for(
                    asyncio.to_thread(self._generate_locally, prompt),
                    timeout=IMAGE_GEN_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                return ToolResultBlock(
                    tool_call_id="",
                    content=[
                        TextBlock(
                            text=f"image generation timed out after "
                            f"{IMAGE_GEN_TIMEOUT_SECONDS}s (local generation can be "
                            "slow without a GPU)"
                        )
                    ],
                    is_error=True,
                )
            except Exception as e:
                # Broad by necessity, the same reasoning
                # FoundryProvider.__init__'s own broad except around
                # load_checkpoint_bundle() already documents: model
                # loading/inference can fail in many library-specific
                # ways (OOM, a corrupted download cache, an
                # incompatible torch build) that don't share one common
                # exception type -- reported as a clean tool error
                # rather than propagating a raw library traceback.
                return ToolResultBlock(
                    tool_call_id="",
                    content=[TextBlock(text=f"local image generation failed: {e}")],
                    is_error=True,
                )
        else:
            openai_key = get_env("OPENAI_API_KEY")
            if not openai_key:
                return ToolResultBlock(
                    tool_call_id="",
                    content=[
                        TextBlock(
                            text="generate_image needs either the optional sarva[image] "
                            "extra installed (free, local, no API key -- see docs) or a "
                            "configured OPENAI_API_KEY (paid, via `sarva config set "
                            "--openai-api-key` or the OPENAI_API_KEY env var) -- neither "
                            "was found."
                        )
                    ],
                    is_error=True,
                )
            try:
                import openai as openai_module

                data = await self._generate_via_openai(prompt, openai_key)
            except openai_module.APIError as e:
                return ToolResultBlock(
                    tool_call_id="",
                    content=[TextBlock(text=f"OpenAI image generation failed: {e}")],
                    is_error=True,
                )

        await asyncio.to_thread(self._save, path, data)
        # A real, confirmed-safe gap found by a fresh-eyes sweep:
        # `ToolResultBlock.content`'s own docstring already says "usually
        # TextBlock/ImageBlock", but until now no tool in this codebase
        # actually returned an ImageBlock -- this tool generated a real
        # image and told the model only a file path, with no way for the
        # model to ever actually SEE what it made (verify it, describe
        # it, or decide to regenerate). `data=data` (the raw bytes
        # already in memory, not re-read from `path`) is embedded
        # directly so the result is self-contained even if the file were
        # later moved or deleted. Safe by construction, not by luck:
        # `AgentLoop.run()`'s own already-existing mid-turn modality
        # check (`required_modalities`/`degrade_message`, see loop.py's
        # own "a tool result produced mid-turn" comment) already handles
        # exactly this case generically -- a vision-capable model sees
        # the real image, a text-only model gets it degraded to metadata
        # text via the same `ImageToTextDegrader` this project's own
        # multimodal pipeline already uses everywhere else, and a run
        # with no degraders configured fails the turn cleanly with a
        # specific reason. No new loop-level code needed for this to
        # work correctly.
        return ToolResultBlock(
            tool_call_id="",
            content=[
                TextBlock(text=f"image saved to {path}"),
                ImageBlock(media_type="image/png", data=data),
            ],
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
        self._store_lock = threading.Lock()

    def _get_store(self) -> VectorMemoryStore:
        # A real bug found by a fresh-eyes sweep: this check-then-act
        # lazy construction runs on a real OS worker thread (via
        # `asyncio.to_thread` in `run()` below, dispatched specifically
        # so it can't block the event loop -- see that comment), not a
        # cooperative asyncio task, so the classic unsynchronized
        # singleton race is genuinely live here, not just theoretical:
        # two concurrent `/ws/chat` connections (two users, or two
        # tabs/windows of the same user) both calling `remember` before
        # this tool's very first call -- most realistically right after
        # server startup -- can both pass `self._store is None`
        # simultaneously on `BUILTIN_TOOLS`' shared, module-level
        # singleton instance and each construct their own independent
        # `VectorMemoryStore` (its own sqlite3 connection AND its own
        # `threading.Lock`). Confirmed live: two concurrent calls
        # produced two distinct `VectorMemoryStore` objects for one
        # lazily-cached field, one connection silently discarded
        # unclosed the moment the other's assignment won the race --
        # momentarily breaking the exact "one lock serializes every
        # real access" invariant `VectorMemoryStore`'s own docstring
        # claims. Fixed with the standard double-checked-locking
        # pattern: the lock is only ever contended during this narrow
        # first-use window, never on the hot path once `self._store` is
        # set.
        if self._store is None:
            with self._store_lock:
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
        self._store_lock = threading.Lock()

    def _get_store(self) -> VectorMemoryStore:
        # A real bug found by a fresh-eyes sweep: the same unsynchronized
        # check-then-act lazy-singleton race found (and fixed) in
        # RememberTool's own `_get_store` -- see that fix's own comment
        # for the confirmed live repro. Fixed identically.
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self._store = VectorMemoryStore(DEFAULT_MEMORY_DB_PATH)
        return self._store

    def _search(self, query: str, top_k: int, session_id: str | None):
        return self._get_store().search(query, top_k=top_k, session_id=session_id)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        top_k = args.get("top_k", 5)
        # A real bug found by a fresh-eyes sweep: `input_schema` above
        # declares `top_k` as `{"type": "integer"}`, but nothing between
        # a real model's tool call and this line ever enforces that --
        # `spec.input_schema` is purely descriptive, sent to the
        # provider so the model knows the expected shape, never
        # validated server-side against the arguments a model actually
        # sends back. Confirmed live two ways: a model emitting
        # `top_k` as the JSON string `"3"` (a real, plausible model
        # mistake, not a contrived attack) reached `VectorMemoryStore.
        # search()`'s unguarded `scored[:top_k]` and raised a raw,
        # non-actionable `TypeError: slice indices must be integers or
        # None or have an __index__ method` -- caught by AgentLoop's
        # outer `except Exception`, so it doesn't crash the run, but the
        # model sees a bare Python exception string instead of a clean,
        # actionable error. Worse, silently: a *negative* `top_k` (the
        # schema has no `minimum` either) doesn't error at all --
        # Python's own slice semantics turn `top_k=-1` into "drop the
        # last result," so `store.search(..., top_k=-1)` against a
        # store holding exactly one genuinely relevant memory returned
        # `[]`, and this tool reported "No relevant memories found"
        # while a relevant memory actually existed -- a silently wrong
        # answer, not just an ugly one. Fixed by validating here, the
        # one place both failure modes are still cheap to reject with a
        # clear, actionable message instead of a raw exception or a
        # silently wrong empty result.
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text=f"top_k must be a non-negative integer, got {top_k!r}")],
                is_error=True,
            )
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
        self._store_lock = threading.Lock()

    def _get_store(self) -> LongTermMemoryStore:
        # A real bug found by a fresh-eyes sweep: the same unsynchronized
        # check-then-act lazy-singleton race found (and fixed) in
        # RememberTool's own `_get_store` -- see that fix's own comment
        # for the confirmed live repro. Fixed identically.
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self._store = LongTermMemoryStore(DEFAULT_LONGTERM_MEMORY_DIR)
        return self._store

    def _write(self, topic: str, content: str) -> Path:
        return self._get_store().write(topic, content)

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
        # below never contends on this lock, but a later sweep found it
        # still blocked the event loop the same way via its own
        # synchronous file I/O -- see its own comment; "no lock
        # contention" turned out not to mean "no blocking I/O."
        #
        # A second, independent instance of the identical mistake, found
        # by a much later fresh-eyes sweep comparing this tool against
        # `RememberTool`'s own `_add` helper: `self._get_store()` used to
        # be called directly as an argument expression here, evaluated
        # eagerly on the event loop *before* `asyncio.to_thread` ever got
        # control -- only the subsequent `.write` call was actually
        # dispatched to a thread. On a process's first `note` call,
        # `_get_store()`'s own lazy construction does real, blocking
        # filesystem I/O (`Path.mkdir` + `os.chmod`), which can be slow on
        # a contended or network-mounted filesystem (this project's own
        # `sarva.config` docstring names "shared dev servers, lab
        # machines, CI runners with persistent home directories" as a
        # real, not hypothetical, scenario). Confirmed live: a
        # deliberately slowed `LongTermMemoryStore.__init__` (simulating
        # exactly that) froze the event loop for the whole construction,
        # near-zero heartbeat ticks during a 1s call. `RememberTool`'s own
        # `_add`/`RecallMemoryTool`'s own `_search` helpers already wrap
        # BOTH the lazy `_get_store()` construction and the store call
        # together for exactly this reason -- that fix never propagated
        # to this sibling tool. Fixed identically: `_write` wraps both,
        # dispatched together through `asyncio.to_thread`.
        try:
            path = await asyncio.to_thread(self._write, args["topic"], args["content"])
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
        self._store_lock = threading.Lock()

    def _get_store(self) -> LongTermMemoryStore:
        # A real bug found by a fresh-eyes sweep: the same unsynchronized
        # check-then-act lazy-singleton race found (and fixed) in
        # RememberTool's own `_get_store` -- see that fix's own comment
        # for the confirmed live repro. Fixed identically.
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self._store = LongTermMemoryStore(DEFAULT_LONGTERM_MEMORY_DIR)
        return self._store

    def _search(self, query: str) -> list[NoteMatch]:
        return self._get_store().search(query)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        # A real bug found by a fresh-eyes sweep: NoteTool.run's own
        # comment above reasoned that this tool "only ever reads (never
        # contends on this lock), so it's unaffected and left as-is" --
        # true for lock contention specifically, but that reasoning
        # conflated "no lock contention" with "no blocking I/O." `search()`
        # is fully synchronous -- it globs every `*.md` file under the
        # notes directory and calls `path.read_text()` on each one in a
        # loop -- and calling it directly here runs all of that straight
        # on the event loop, with no `asyncio.to_thread`, exactly the
        # blocking-call-inside-async-def shape already fixed for
        # `NoteTool.write`, `RememberTool`, and `RecallMemoryTool` in this
        # same file. Confirmed live: 20,000 short notes (~1.8MB total, a
        # plausible amount after long-term real use of the `note` tool
        # across many conversations) froze this process's ENTIRE event
        # loop for the whole call -- a heartbeat coroutine that should
        # have ticked roughly every 0.05s recorded ZERO ticks across the
        # full ~360ms search, meaning every other in-flight `/chat`/
        # `/ws/chat` turn in a real `sarva serve` process would have
        # frozen too, not just this one tool call.
        #
        # A second, independent instance of the identical mistake, found
        # by a much later fresh-eyes sweep, the same one that just found
        # it in `NoteTool` a few lines above: `self._get_store()` was
        # still called directly as an argument expression here, evaluated
        # eagerly on the event loop *before* `asyncio.to_thread` ever got
        # control -- only the subsequent `.search` call was actually
        # dispatched to a thread. On a process's first `search_notes`
        # call, `_get_store()`'s own lazy construction does real, blocking
        # filesystem I/O (`Path.mkdir` + `os.chmod`), the identical gap
        # `NoteTool._write` closes for `note`. Confirmed live: a
        # deliberately slowed `LongTermMemoryStore.__init__` froze the
        # event loop for the whole ~1s construction, near-zero heartbeat
        # ticks during the call. Fixed identically: `_search` wraps both
        # the lazy construction and the store call together, dispatched
        # as one unit through `asyncio.to_thread`.
        matches = await asyncio.to_thread(self._search, args["query"])
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
    RunCodeTool(),
    WebFetchTool(),
    WebSearchTool(),
    ImageGenerationTool(),
    RememberTool(),
    RecallMemoryTool(),
    NoteTool(),
    SearchNotesTool(),
    DelegateTool(),
]
