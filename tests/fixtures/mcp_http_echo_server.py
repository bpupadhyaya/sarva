"""The HTTP-transport counterpart to mcp_echo_server.py — a real MCP
server, launched as an actual subprocess speaking real MCP-over-
Streamable-HTTP (the protocol's current standard HTTP transport, per
spec revision 2025-03-26), not a mock. Same two tools, same behavior, so
test_mcp_client_http.py can prove the HTTP transport round-trips
identically to the stdio one, not just that a request/response happens
somehow.

Port is read from the SARVA_TEST_MCP_PORT env var (set by the test,
which picks a real free port first) rather than hardcoded, so parallel
test runs never collide on a fixed port.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

port = int(os.environ["SARVA_TEST_MCP_PORT"])
server = FastMCP("sarva-test-echo-server-http", host="127.0.0.1", port=port)


@server.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


@server.tool()
def fail(reason: str) -> str:
    """Always raises, to exercise MCP error propagation."""
    raise ValueError(reason)


if __name__ == "__main__":
    server.run(transport="streamable-http")
