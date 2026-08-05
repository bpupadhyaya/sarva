"""sarva.mcp_client — MCP client: wraps a remote MCP server's tools as
first-class Sarva `Tool` implementations, so the ecosystem's tools plug in
without any Sarva-specific glue (spec §3.5's "MCP client support").

Uses the official `mcp` SDK's `ClientSession` — the same "official SDK,
not a hand-rolled protocol" pattern the provider adapters already follow
for anthropic/openai/google-genai, not a from-scratch JSON-RPC client.

Two transports are wired up. `connect_stdio_mcp_server` was first: the
vast majority of real MCP servers today (npx/uvx-launched local
processes) speak stdio, and it's the one transport genuinely verifiable
offline (spawn a real local subprocess, no network). `connect_http_mcp_
server` closes the gap this module's own docstring used to name as real,
deferred scope — it speaks Streamable HTTP (the protocol's current
standard HTTP transport per spec revision 2025-03-26, superseding the
older separate SSE transport, which the `mcp` SDK still ships as
`mcp.client.sse` for servers that haven't moved off it but which this
client doesn't wire up, matching "current, not every historical
variant").
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

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
            # A real bug found by giving this module its own fresh-eyes
            # sweep: `ToolSpec.destructive` defaults to `False`, and this
            # constructor never set it, so every MCP-provided tool --
            # arbitrary, remote, unaudited code -- silently bypassed the
            # agent loop's own confirm-before-destructive-action gate,
            # regardless of what the tool actually does. `tool.annotations.
            # destructiveHint` exists on the wire, but the MCP spec's own
            # docstring for it says plainly: "Clients should never make
            # tool use decisions based on ToolAnnotations received from
            # untrusted servers" -- a malicious or buggy server could just
            # set `destructiveHint=False` on a genuinely destructive tool
            # to defeat exactly this gate, so using it here would be
            # trusting the attacker's own self-report. Every MCP tool is
            # now treated as destructive unconditionally, matching this
            # project's own "reject, don't guess" posture: the local
            # confirmation gate can always be bypassed with `--auto` by a
            # user who's decided to trust the connected server, but it
            # must never silently not apply at all.
            destructive=True,
        )
        self._session = session

    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock:
        result = await self._session.call_tool(self.spec.name, args)
        content = [_convert_content(block) for block in result.content]
        # A real bug found by a fresh-eyes sweep: `CallToolResult` has TWO
        # separate result fields on the wire -- `content` (unstructured)
        # and `structuredContent` (a JSON object validated against the
        # tool's own declared `outputSchema`, when it has one) -- and only
        # `content` was ever read here. The MCP spec (2025-06-18) only
        # requires a server to include `structuredContent`; duplicating it
        # into `content` as backward-compat text is a SHOULD, not a MUST
        # -- the reference `mcp` SDK auto-enforces that SHOULD only for
        # the simple "tool function returns a plain dict" path, but its
        # own lower-level `Server.call_tool` decorator (a legitimate,
        # documented way to build an MCP server) also supports returning
        # `(unstructured_content, structured_content)` directly, where a
        # server author can legitimately send `content=[]` alongside a
        # real `structuredContent`. Confirmed live: a fake session
        # returning exactly that shape (`content=[], structuredContent=
        # {"sum": 42}, isError=False`) produced a ToolResultBlock the
        # model sees as a clean, empty success -- the actual computed
        # answer nowhere in it, no error, no signal anything was lost.
        # Directly contradicts this same function's own `_convert_content`
        # fallback two lines below, whose whole point is never silently
        # dropping content it can't fully translate. Appended (not
        # replacing `content`) since a spec-compliant server may
        # legitimately have already duplicated it into `content` as text
        # -- there's no reliable way to detect that duplication from the
        # client side, and a harmless repeat is a strictly better outcome
        # than the alternative of sometimes silently losing the only copy.
        if result.structuredContent is not None:
            content.append(
                TextBlock(
                    text=f"[MCP tool structured result: {json.dumps(result.structuredContent)}]"
                )
            )
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


@asynccontextmanager
async def connect_http_mcp_server(
    url: str, headers: dict[str, str] | None = None
) -> AsyncIterator[ClientSession]:
    """Connect to a remote MCP server over Streamable HTTP and yield a
    live, initialized `ClientSession` — the network-reachable counterpart
    to `connect_stdio_mcp_server`; nothing downstream (`list_mcp_tools`,
    `McpToolAdapter`) knows or cares which transport a given session came
    from, since both ultimately just implement `ClientSession`.
    `headers` is the one thing an HTTP server usually needs that a local
    stdio subprocess doesn't — most real deployments put an auth token
    there (e.g. `{"Authorization": "Bearer ..."}`), left entirely to the
    caller since Sarva has no opinion on any particular server's auth
    scheme.

    Builds its own `httpx.AsyncClient` via the `mcp` SDK's own
    `create_mcp_http_client` helper (30s timeout, redirects followed —
    the SDK's own documented defaults) rather than the older
    `streamablehttp_client(url, headers=...)` convenience wrapper, which
    is deprecated in the installed SDK version in favor of this
    explicit-client form."""
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
