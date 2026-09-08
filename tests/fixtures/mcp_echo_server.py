"""A real, minimal MCP server used by test_mcp_client.py — launched as an
actual subprocess speaking MCP over stdio, not a mock. Exposes several
tools so the client test can prove the happy path, error propagation,
and structured-output handling through the real protocol round trip."""

from __future__ import annotations

import base64
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
def rich_content() -> mcp_types.CallToolResult:
    """Returns one of each of the mcp SDK's non-text/image ContentBlock
    variants in a single real CallToolResult (the same FastMCP escape
    hatch structured_only above uses) -- AudioContent, an
    EmbeddedResource wrapping BlobResourceContents (real inline base64
    bytes), and an EmbeddedResource wrapping TextResourceContents. Lets
    test_mcp_client.py prove _convert_content's handling of each against
    a real MCP round trip, not a hand-constructed mcp_types object with
    no server on the other end."""
    return mcp_types.CallToolResult(
        content=[
            mcp_types.AudioContent(
                type="audio",
                data=base64.b64encode(b"RIFF....WAVEfmt ").decode(),
                mimeType="audio/wav",
            ),
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.BlobResourceContents(
                    uri="resource://sarva-test/report.csv",
                    mimeType="text/csv",
                    blob=base64.b64encode(b"a,b\n1,2\n").decode(),
                ),
            ),
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.TextResourceContents(
                    uri="resource://sarva-test/note.txt",
                    mimeType="text/plain",
                    text="a real embedded text resource",
                ),
            ),
        ],
        isError=False,
    )


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
