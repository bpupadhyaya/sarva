"""sarva.mcp_client — MCP client: wraps a remote MCP server's tools as
first-class Sarva `Tool` implementations, so the ecosystem's tools plug in
without any Sarva-specific glue (spec §3.5's "MCP client support").

Uses the official `mcp` SDK's `ClientSession` — the same "official SDK,
not a hand-rolled protocol" pattern the provider adapters already follow
for anthropic/openai/google-genai, not a from-scratch JSON-RPC client.
Only the stdio transport is wired up: the vast majority of real MCP
servers today (npx/uvx-launched local processes) speak stdio, and it's
the one transport genuinely verifiable offline (spawn a real local
subprocess, no network). HTTP/SSE transports are real, deferred scope,
not silently assumed covered.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sarva.agent.tools import ToolContext
from sarva.multimodal.content import ImageBlock, TextBlock, ToolResultBlock
from sarva.providers.base import ToolSpec


class McpToolAdapter:
    """A single remote MCP tool, wrapped as a Sarva `Tool`. Constructed by
    `list_mcp_tools`, never directly — a bare `mcp_types.Tool` has no
    `ClientSession` to actually call through."""

    def __init__(self, session: ClientSession, tool: mcp_types.Tool):
        self.spec = ToolSpec(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
        )
        self._session = session

    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock:
        result = await self._session.call_tool(self.spec.name, args)
        content = [_convert_content(block) for block in result.content]
        return ToolResultBlock(tool_call_id="", content=content, is_error=result.isError)


def _convert_content(block: mcp_types.ContentBlock) -> TextBlock | ImageBlock:
    if isinstance(block, mcp_types.TextContent):
        return TextBlock(text=block.text)
    if isinstance(block, mcp_types.ImageContent):
        return ImageBlock(media_type=block.mimeType, data=base64.b64decode(block.data))
    # Audio/resource-link/embedded-resource content: report what's
    # verifiably known (the block's own declared type) rather than
    # silently dropping it or raising -- the same honesty principle the
    # multimodal degraders use for content a layer can't fully consume.
    return TextBlock(text=f"[MCP tool returned unsupported content type: {block.type}]")


async def list_mcp_tools(session: ClientSession) -> list[McpToolAdapter]:
    """List every tool the connected server exposes, each already wrapped
    as a ready-to-use Sarva `Tool` (append to a tool list passed to
    `AgentLoop`/`BUILTIN_TOOLS`)."""
    result = await session.list_tools()
    return [McpToolAdapter(session, tool) for tool in result.tools]


@asynccontextmanager
async def connect_stdio_mcp_server(
    command: str, args: list[str] | None = None, env: dict[str, str] | None = None
) -> AsyncIterator[ClientSession]:
    """Launch `command` as a subprocess speaking MCP over stdio, and yield a
    live, initialized `ClientSession`. The subprocess is started and torn
    down within this context manager's lifetime."""
    params = StdioServerParameters(command=command, args=args or [], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
