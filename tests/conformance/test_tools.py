"""Conformance tests for the built-in tools."""

from __future__ import annotations

import pytest
from sarva.agent.tools import ReadFileTool, ToolContext, WebFetchTool, WriteFileTool


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))


@pytest.mark.asyncio
async def test_write_then_read_round_trip(ctx):
    write = WriteFileTool()
    read = ReadFileTool()
    result = await write.run({"path": "note.txt", "content": "hello sarva"}, ctx)
    assert not result.is_error

    result = await read.run({"path": "note.txt"}, ctx)
    assert not result.is_error
    assert result.content[0].text == "hello sarva"


@pytest.mark.asyncio
async def test_write_creates_parent_directories(ctx):
    write = WriteFileTool()
    result = await write.run({"path": "nested/dir/file.txt", "content": "x"}, ctx)
    assert not result.is_error


@pytest.mark.asyncio
async def test_path_escape_is_rejected(ctx):
    read = ReadFileTool()
    with pytest.raises(ValueError, match="escapes workdir"):
        await read.run({"path": "../../etc/passwd"}, ctx)


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_schemes(ctx):
    tool = WebFetchTool()
    result = await tool.run({"url": "file:///etc/passwd"}, ctx)
    assert result.is_error
    assert "unsupported URL scheme" in result.content[0].text


@pytest.mark.live
@pytest.mark.asyncio
async def test_web_fetch_live(ctx):
    """Requires network access — skipped by default (see pyproject `-m 'not live'`)."""
    tool = WebFetchTool()
    result = await tool.run({"url": "https://example.com"}, ctx)
    assert not result.is_error
    assert "Example Domain" in result.content[0].text
