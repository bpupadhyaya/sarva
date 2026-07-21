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

from sarva.multimodal.content import TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ToolSpec


class ToolContext:
    """Passed to every tool invocation. `emit` is wired by the AgentLoop for
    transcript logging; tools never talk to the provider layer directly."""

    def __init__(
        self,
        workdir: str,
        run_dir: str,
        emit: Callable[[Any], Awaitable[None]] | None = None,
    ):
        self.workdir = workdir
        self.run_dir = run_dir
        self.emit = emit or (lambda event: asyncio.sleep(0))


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


BUILTIN_TOOLS: list[Tool] = [ReadFileTool(), WriteFileTool(), RunShellTool()]
