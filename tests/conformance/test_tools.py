"""Conformance tests for the built-in tools."""

from __future__ import annotations

import pytest
from sarva.agent.tools import (
    ReadFileTool,
    RecallMemoryTool,
    RememberTool,
    ToolContext,
    WebFetchTool,
    WriteFileTool,
)
from sarva.memory.vector import VectorMemoryStore


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


@pytest.mark.asyncio
async def test_remember_then_recall_round_trip(ctx, tmp_path):
    store = VectorMemoryStore(tmp_path / "memory.db")
    remember = RememberTool(store=store)
    recall = RecallMemoryTool(store=store)

    result = await remember.run({"text": "the launch code is in the blue folder"}, ctx)
    assert not result.is_error

    result = await recall.run({"query": "where is the launch code"}, ctx)
    assert not result.is_error
    assert "blue folder" in result.content[0].text


@pytest.mark.asyncio
async def test_recall_with_no_memories_says_so(ctx, tmp_path):
    recall = RecallMemoryTool(store=VectorMemoryStore(tmp_path / "memory.db"))
    result = await recall.run({"query": "anything"}, ctx)
    assert "No relevant memories found" in result.content[0].text


@pytest.mark.asyncio
async def test_remember_uses_ctx_session_id_when_present(tmp_path):
    # ctx.session_id (threaded from AgentLoop.run(session_id=...), which
    # in turn comes from the CLI's --session / the server's session
    # field) must win over the tool's own constructor-time default --
    # that default only exists for runs with no session identity at all.
    store = VectorMemoryStore(tmp_path / "memory.db")
    remember = RememberTool(store=store, session_id="fallback")
    ctx = ToolContext(
        workdir=str(tmp_path), run_dir=str(tmp_path / "run"), session_id="real-session"
    )

    await remember.run({"text": "a session-scoped note"}, ctx)

    results = store.search("session-scoped note", session_id="real-session")
    assert len(results) == 1
    assert store.search("session-scoped note", session_id="fallback") == []


@pytest.mark.asyncio
async def test_remember_falls_back_to_constructor_session_id_when_ctx_has_none(tmp_path):
    store = VectorMemoryStore(tmp_path / "memory.db")
    remember = RememberTool(store=store, session_id="fallback")
    ctx = ToolContext(
        workdir=str(tmp_path), run_dir=str(tmp_path / "run")
    )  # session_id defaults to None

    await remember.run({"text": "an unscoped note"}, ctx)

    results = store.search("unscoped note", session_id="fallback")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_recall_uses_ctx_session_id_when_present(tmp_path):
    store = VectorMemoryStore(tmp_path / "memory.db")
    store.add("session-a", "the launch code is blue")
    store.add("session-b", "the launch code is blue")
    recall = RecallMemoryTool(store=store, session_id="fallback")
    ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"), session_id="session-a")

    result = await recall.run({"query": "launch code"}, ctx)

    assert result.content[0].text.count("launch code") == 1  # only session-a's entry, not both


def test_default_memory_tools_do_not_open_the_store_until_first_run():
    # BUILTIN_TOOLS constructs these with no store argument at module
    # import time -- eagerly opening the default store in __init__ would
    # make merely *importing* sarva.agent.tools open (and, via
    # VectorMemoryStore's own mkdir, create) a real file at
    # ~/.sarva/memory.db on every machine that imports it, including
    # test/CI runs that never otherwise touch the filesystem. Checked
    # directly against the internal _store attribute rather than the
    # real filesystem: DEFAULT_MEMORY_DB_PATH is a module-level constant
    # bound to the real Path.home() at import time, so patching Path.home
    # afterwards wouldn't affect it anyway -- this is the precise,
    # hermetic way to verify the laziness property.
    remember = RememberTool()
    recall = RecallMemoryTool()
    assert remember._store is None
    assert recall._store is None
