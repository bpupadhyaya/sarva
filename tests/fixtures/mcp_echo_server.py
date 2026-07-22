"""A real, minimal MCP server used by test_mcp_client.py — launched as an
actual subprocess speaking MCP over stdio, not a mock. Exposes two tools
so the client test can prove both the happy path and error propagation
through the real protocol round trip."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("sarva-test-echo-server")


@server.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


@server.tool()
def fail(reason: str) -> str:
    """Always raises, to exercise MCP error propagation."""
    raise ValueError(reason)


if __name__ == "__main__":
    server.run(transport="stdio")
