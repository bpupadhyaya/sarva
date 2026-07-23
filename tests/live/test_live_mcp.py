"""Live conformance test for MCP interop against a REAL, independently
published third-party MCP server -- not this repo's own hand-written
test fixture (`tests/fixtures/mcp_http_echo_server.py`), which every
other MCP test in this project talks to. A fixture server written by
the same project as the client it's tested against can share the same
misunderstanding of the protocol; this closes that gap by talking to
`@modelcontextprotocol/server-filesystem`, the official Anthropic
reference filesystem server, launched for real via `npx`.

Skipped by default (pyproject `addopts = "-m 'not live'"`, matching
every other live test) and additionally skipped if `npx` isn't on PATH.
Run explicitly with: `uv run pytest tests/live -m live -k mcp`.
"""

from __future__ import annotations

import shutil

import pytest
from sarva.agent.tools import ToolContext
from sarva.mcp_client import connect_stdio_mcp_server, list_mcp_tools

pytestmark = [
    pytest.mark.live,
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("npx") is None, reason="npx not available"),
]


async def test_real_filesystem_mcp_server_read_and_write_round_trip(tmp_path):
    greeting = tmp_path / "greeting.txt"
    greeting.write_text("hello from a real mcp server\n")
    ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path))

    async with connect_stdio_mcp_server(
        "npx", args=["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)]
    ) as session:
        tools = await list_mcp_tools(session)
        tool_names = {t.spec.name for t in tools}
        # The real server's actual tool set, not assumed from documentation.
        assert {"read_text_file", "write_file", "list_directory"} <= tool_names

        read_tool = next(t for t in tools if t.spec.name == "read_text_file")
        read_result = await read_tool.run({"path": str(greeting)}, ctx)
        assert read_result.is_error is False
        assert read_result.content[0].text == "hello from a real mcp server\n"

        write_tool = next(t for t in tools if t.spec.name == "write_file")
        out_path = tmp_path / "written-by-sarva.txt"
        write_result = await write_tool.run(
            {"path": str(out_path), "content": "sarva wrote this via a real MCP server"}, ctx
        )
        assert write_result.is_error is False

    # The real proof for the write direction: read the raw file back
    # from disk directly, independent of anything the MCP client itself
    # reported -- the server's own tool_result claiming success isn't
    # trusted as the final word.
    assert out_path.read_text() == "sarva wrote this via a real MCP server"
