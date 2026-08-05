"""A real, minimal MCP server used by test_mcp_client.py — launched as an
actual subprocess speaking MCP over stdio, not a mock. Exposes several
tools so the client test can prove the happy path, error propagation,
and structured-output handling through the real protocol round trip."""

from __future__ import annotations

import os

import mcp.types as mcp_types
from mcp.server.fastmcp import FastMCP

server = FastMCP("sarva-test-echo-server")


@server.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


@server.tool()
def structured_only(a: int, b: int) -> mcp_types.CallToolResult:
    """Returns ONLY structuredContent, with an empty `content` list -- a
    real, spec-legal shape a server built on the lower-level `Server.
    call_tool` decorator can send (this test returns a real
    `CallToolResult` directly, FastMCP's own documented escape hatch for
    exactly this case, rather than relying on FastMCP's own auto content/
    structuredContent duplication). Proves the client actually reads
    `structuredContent`, not just `content`."""
    return mcp_types.CallToolResult(content=[], structuredContent={"sum": a + b}, isError=False)


@server.tool()
def fail(reason: str) -> str:
    """Always raises, to exercise MCP error propagation."""
    raise ValueError(reason)


@server.tool()
def env_var(name: str) -> str:
    """Return the named environment variable's value, or MISSING if unset
    -- proves connect_stdio_mcp_server's `env` parameter actually reaches
    this real subprocess's environment, not just that it's accepted."""
    return os.environ.get(name, "MISSING")


if __name__ == "__main__":
    server.run(transport="stdio")
