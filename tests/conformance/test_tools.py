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


@pytest.mark.asyncio
async def test_web_fetch_blocks_loopback_addresses(ctx):
    # A real SSRF risk, not hypothetical: confirmed directly against a
    # real local Ollama server (http://127.0.0.1:11434/api/tags)
    # returning its response with zero confirmation before this fix,
    # since WebFetchTool is marked non-destructive. No mocking needed --
    # 127.0.0.1 needs no DNS lookup or listening server to test the
    # block itself.
    tool = WebFetchTool()
    result = await tool.run({"url": "http://127.0.0.1:11434/api/tags"}, ctx)
    assert result.is_error
    assert "non-public address" in result.content[0].text


@pytest.mark.asyncio
async def test_web_fetch_blocks_the_cloud_metadata_address(ctx):
    # 169.254.169.254 is the well-known cloud-metadata endpoint
    # (AWS/GCP/Azure) -- a classic SSRF target for exfiltrating
    # credentials when a service runs in a cloud VM.
    tool = WebFetchTool()
    result = await tool.run({"url": "http://169.254.169.254/latest/meta-data/"}, ctx)
    assert result.is_error
    assert "non-public address" in result.content[0].text


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_rfc1918_addresses(ctx):
    tool = WebFetchTool()
    result = await tool.run({"url": "http://192.168.1.1/admin"}, ctx)
    assert result.is_error
    assert "non-public address" in result.content[0].text


@pytest.mark.asyncio
async def test_ensure_public_host_rejects_a_redirect_to_an_internal_address(ctx, monkeypatch):
    # The real reason follow_redirects was replaced with a manual,
    # per-hop-validated loop: a caller-supplied URL can be a legitimate
    # public site whose server issues a redirect straight to an
    # internal address. A validate-the-caller's-URL-once check would
    # never catch that. Simulated here (no real attacker-controlled
    # public redirector available to test against) by monkeypatching
    # httpx.AsyncClient.get to return a real Response object carrying a
    # redirect Location header pointing at localhost.
    import httpx

    async def fake_get(self, url, *a, **kw):
        return httpx.Response(
            302,
            headers={"location": "http://127.0.0.1:11434/api/tags"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    tool = WebFetchTool()
    result = await tool.run({"url": "https://example.com/redirector"}, ctx)
    assert result.is_error
    assert "non-public address" in result.content[0].text


@pytest.mark.live
@pytest.mark.asyncio
async def test_web_fetch_live(ctx):
    """Requires network access — skipped by default (see pyproject `-m 'not live'`)."""
    tool = WebFetchTool()
    result = await tool.run({"url": "https://example.com"}, ctx)
    assert not result.is_error
    assert "Example Domain" in result.content[0].text


@pytest.mark.live
@pytest.mark.asyncio
async def test_web_fetch_live_follows_a_real_redirect_to_a_public_site(ctx):
    """Requires network access — skipped by default. http://github.com
    redirects to https://github.com/; proves the manual redirect loop
    genuinely follows a real redirect to another real public site, not
    just that it blocks internal ones."""
    tool = WebFetchTool()
    result = await tool.run({"url": "http://github.com"}, ctx)
    assert not result.is_error
    assert len(result.content[0].text) > 0


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
