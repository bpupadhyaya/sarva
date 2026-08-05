"""Conformance tests for the built-in tools."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest
import sarva.agent.tools as tools_module
from sarva.agent.tools import (
    EditFileTool,
    NoteTool,
    ReadFileTool,
    RecallMemoryTool,
    RememberTool,
    RunShellTool,
    SearchNotesTool,
    ToolContext,
    WebFetchTool,
    WriteFileTool,
)
from sarva.memory.longterm import LongTermMemoryStore
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
async def test_write_does_not_destroy_the_previous_file_if_interrupted_mid_write(ctx, monkeypatch):
    # A real bug found by actually simulating an interrupted write: this
    # tool used to open the target file directly with write_text(),
    # which truncates it to 0 bytes immediately -- before a single byte
    # of new content is written. A crash (OOM-kill, SIGKILL, power loss)
    # mid-write destroyed a previously-good, real user file this tool
    # was writing to -- confirmed live: a real 5000-byte file became 0
    # bytes. Simulated here by making os.replace() (the atomic commit
    # step) raise partway through a second write -- the real file must
    # still hold the first, complete write's content afterward.
    write = WriteFileTool()
    await write.run({"path": "note.txt", "content": "first, good content"}, ctx)

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError):
        await write.run({"path": "note.txt", "content": "second write, interrupted"}, ctx)

    read = ReadFileTool()
    result = await read.run({"path": "note.txt"}, ctx)
    assert result.content[0].text == "first, good content"


@pytest.mark.asyncio
async def test_read_file_passes_an_explicit_utf8_encoding_not_locale_default(ctx, monkeypatch):
    # A real bug found by a fresh-eyes sweep of read_text() call sites
    # across the codebase: this tool used to call `p.read_text()` with
    # no `encoding=`, which uses `locale.getpreferredencoding(False)`,
    # not UTF-8 -- genuinely locale-dependent on this project's own
    # minimum Python (3.12; PEP 686's default-UTF-8 mode doesn't land
    # until 3.15), e.g. on musl-libc containers (Alpine), Windows
    # without UTF-8 mode, or PYTHONCOERCECLOCALE=0 -- all realistic
    # `sarva serve`/`sarva run` deployment targets, not adversarial
    # ones. Confirmed live before this fix: a file WriteFileTool had
    # just written (always UTF-8 via atomic_write_text) crashed with
    # UnicodeDecodeError reading it straight back, despite this tool's
    # own description promising "Read a UTF-8 text file." This spy
    # directly pins the fix (explicit encoding="utf-8"), rather than
    # depending on the OS's actual locale to reproduce it.
    write = WriteFileTool()
    await write.run({"path": "note.txt", "content": "café Ω 日本語 -- bonjour!"}, ctx)

    real_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        assert kwargs.get("encoding") == "utf-8", (
            "read_text() must pass encoding='utf-8' explicitly, not rely on "
            "locale.getpreferredencoding()"
        )
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    read = ReadFileTool()
    result = await read.run({"path": "note.txt"}, ctx)
    assert not result.is_error
    assert result.content[0].text == "café Ω 日本語 -- bonjour!"


@pytest.mark.asyncio
async def test_edit_replaces_the_one_exact_occurrence(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "the quick brown fox"}, ctx)

    result = await edit.run({"path": "file.txt", "old_string": "brown", "new_string": "red"}, ctx)

    assert not result.is_error
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "the quick red fox"


@pytest.mark.asyncio
async def test_edit_leaves_the_rest_of_a_large_file_untouched(ctx):
    # The whole reason this tool exists distinct from WriteFileTool: a
    # targeted change to one part of a real file must not require (or
    # risk) resending/rewriting content that never changed.
    write = WriteFileTool()
    edit = EditFileTool()
    original = "\n".join(f"line {i}" for i in range(1000))
    await write.run({"path": "big.txt", "content": original}, ctx)

    await edit.run({"path": "big.txt", "old_string": "line 500", "new_string": "EDITED"}, ctx)

    read = ReadFileTool()
    text = (await read.run({"path": "big.txt"}, ctx)).content[0].text
    lines = text.splitlines()
    assert lines[500] == "EDITED"
    assert lines[499] == "line 499"
    assert lines[501] == "line 501"
    assert len(lines) == 1000


@pytest.mark.asyncio
async def test_edit_fails_cleanly_when_old_string_is_not_found(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "the quick brown fox"}, ctx)

    result = await edit.run(
        {"path": "file.txt", "old_string": "not present anywhere", "new_string": "x"}, ctx
    )

    assert result.is_error is True
    assert "not found" in result.content[0].text
    # The file itself must be genuinely untouched, not partially edited.
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "the quick brown fox"


@pytest.mark.asyncio
async def test_edit_rejects_an_ambiguous_old_string_without_replace_all(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "cat dog cat bird cat"}, ctx)

    result = await edit.run({"path": "file.txt", "old_string": "cat", "new_string": "fish"}, ctx)

    assert result.is_error is True
    assert "3 times" in result.content[0].text
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "cat dog cat bird cat"  # untouched


@pytest.mark.asyncio
async def test_edit_replace_all_replaces_every_occurrence(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "cat dog cat bird cat"}, ctx)

    result = await edit.run(
        {"path": "file.txt", "old_string": "cat", "new_string": "fish", "replace_all": True}, ctx
    )

    assert not result.is_error
    assert "3 replacements" in result.content[0].text
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "fish dog fish bird fish"


@pytest.mark.asyncio
async def test_edit_rejects_a_non_boolean_replace_all_instead_of_silently_replacing_all(ctx):
    # A real bug found by a fresh-eyes sweep, the same shape already
    # fixed for RecallMemoryTool's top_k: input_schema declares
    # replace_all as {"type": "boolean"}, but nothing enforces that
    # against a real model's actual tool-call arguments. `not
    # replace_all` treated the value as a plain Python truth value, and
    # any non-empty string is truthy -- a model emitting the JSON
    # *string* "false" was treated as replace_all=True, the exact
    # opposite of what was asked, silently rewriting every occurrence
    # in the file instead of erroring on the ambiguity. Confirmed live
    # before this fix: this exact setup replaced all three occurrences.
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "cat dog cat bird cat"}, ctx)

    result = await edit.run(
        {"path": "file.txt", "old_string": "cat", "new_string": "fish", "replace_all": "false"},
        ctx,
    )

    assert result.is_error is True
    assert "replace_all must be a boolean" in result.content[0].text
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "cat dog cat bird cat"  # untouched


@pytest.mark.asyncio
async def test_edit_rejects_an_empty_old_string(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "some content"}, ctx)

    result = await edit.run({"path": "file.txt", "old_string": "", "new_string": "x"}, ctx)

    assert result.is_error is True
    assert "must not be empty" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_rejects_identical_old_and_new_string(ctx):
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "some content"}, ctx)

    result = await edit.run(
        {"path": "file.txt", "old_string": "content", "new_string": "content"}, ctx
    )

    assert result.is_error is True
    assert "identical" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_does_not_destroy_the_file_if_interrupted_mid_write(ctx, monkeypatch):
    # The same atomic-write guarantee WriteFileTool already has, proven
    # the same way: simulate os.replace() raising partway through the
    # edit's own atomic commit -- the file must still hold its last
    # good, complete content afterward, not a truncated 0-byte file.
    write = WriteFileTool()
    edit = EditFileTool()
    await write.run({"path": "file.txt", "content": "the quick brown fox"}, ctx)

    real_replace = os.replace

    def flaky_replace(src, dst):
        raise OSError("simulated crash during os.replace")

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError):
        await edit.run({"path": "file.txt", "old_string": "brown", "new_string": "red"}, ctx)

    monkeypatch.setattr(os, "replace", real_replace)
    read = ReadFileTool()
    text = (await read.run({"path": "file.txt"}, ctx)).content[0].text
    assert text == "the quick brown fox"


@pytest.mark.asyncio
async def test_edit_preserves_crlf_line_endings_on_every_untouched_line(ctx, tmp_path):
    # A real bug found by actually editing one line of a real CRLF file:
    # Path.read_text() does universal-newlines translation on read (\r\n
    # silently becomes \n) with nothing on the write side translating
    # back, so every OTHER line's ending silently flipped to LF too --
    # directly contradicting this tool's own "without rewriting the rest
    # of the file" contract. Written directly as raw bytes (not via
    # WriteFileTool, which doesn't round-trip through read_text() at
    # all and so wouldn't reproduce the bug) to construct a genuine CRLF
    # file the way a real Windows-authored or .gitattributes-enforced
    # file would look on disk.
    path = tmp_path / "file.txt"
    path.write_bytes(b"line1\r\nline2\r\nline3\r\n")
    edit = EditFileTool()

    result = await edit.run({"path": "file.txt", "old_string": "line2", "new_string": "LINE2"}, ctx)

    assert not result.is_error
    assert path.read_bytes() == b"line1\r\nLINE2\r\nline3\r\n"


@pytest.mark.asyncio
async def test_edit_path_escape_is_rejected(ctx):
    edit = EditFileTool()
    with pytest.raises(ValueError, match="escapes workdir"):
        await edit.run({"path": "../../etc/passwd", "old_string": "x", "new_string": "y"}, ctx)


@pytest.mark.asyncio
async def test_path_escape_is_rejected(ctx):
    read = ReadFileTool()
    with pytest.raises(ValueError, match="escapes workdir"):
        await read.run({"path": "../../etc/passwd"}, ctx)


@pytest.mark.asyncio
async def test_run_shell_returns_combined_stdout_stderr_and_exit_status(ctx):
    shell = RunShellTool()
    result = await shell.run({"command": "echo out; echo err >&2; exit 1"}, ctx)
    assert result.is_error
    assert "out" in result.content[0].text
    assert "err" in result.content[0].text


@pytest.mark.asyncio
async def test_run_shell_timeout_kills_the_child_process_and_its_side_effects(ctx, monkeypatch):
    # A real bug found by actually running a long-lived shell command
    # against a shortened timeout: asyncio.wait_for() only cancels the
    # *awaiting* communicate() call on expiry -- it never touches the
    # child process itself, confirmed directly with a real `sleep`
    # process still alive (and its trailing side effect still
    # completing) seconds after the "timeout." This matters
    # specifically because this tool is `destructive=True` -- the whole
    # confirmation gate exists to stop unwanted side effects, and a
    # silent timeout defeated it regardless of what was approved.
    monkeypatch.setattr(tools_module, "_SHELL_TIMEOUT_SECONDS", 0.2)
    shell = RunShellTool()
    marker = ctx.workdir + "/side-effect.txt"

    result = await shell.run({"command": f"sleep 2 && echo done > {marker}"}, ctx)

    assert result.is_error
    assert "timed out" in result.content[0].text
    # Not blank the way a bare TimeoutError's own str() would be -- the
    # actual reason must be visible to the model/user, not just an
    # is_error flag with nothing behind it.
    assert result.content[0].text.strip() != ""

    # The real proof this isn't just a nicer message: the process (and
    # therefore its side effect) must actually be dead, not merely
    # reported as timed out while still running unattended.
    await asyncio.sleep(3)
    assert not os.path.exists(marker)


@pytest.mark.asyncio
async def test_run_shell_output_is_bounded_not_buffered_without_limit(ctx, monkeypatch):
    # A real bug found by giving this tool its own fresh-eyes sweep, one
    # layer beyond the already-fixed timeout-doesn't-kill-the-process
    # gap above: proc.communicate() buffered the ENTIRE stdout+stderr
    # stream with no size limit at all, unlike WebFetchTool right below
    # (which caps its own output via _MAX_FETCH_CHARS). Confirmed live:
    # a completely ordinary command (`yes A | head -c 300MB` -- the same
    # shape as `git log -p`, a verbose build, or `cat`-ing a large data
    # file, not a contrived attack) drove real process peak RSS from
    # ~45MB to ~1GB, and the full, untruncated 300MB string was then
    # appended into `messages`, resent to the provider on every later
    # turn of the same run. A small monkeypatched cap here proves the
    # read genuinely stops early rather than buffering everything then
    # slicing the result -- the latter would still incur the full
    # memory cost this fix exists to avoid.
    monkeypatch.setattr(tools_module, "_MAX_SHELL_OUTPUT_BYTES", 100)
    shell = RunShellTool()

    result = await asyncio.wait_for(
        shell.run({"command": "yes A | head -c 10000000"}, ctx), timeout=10
    )

    assert result.is_error is True  # killed mid-stream -> nonzero/negative returncode
    assert len(result.content[0].text) < 1000  # nowhere near the full 10MB
    assert "truncated" in result.content[0].text


@pytest.mark.asyncio
async def test_run_shell_kills_the_whole_pipeline_not_just_the_shell(ctx, monkeypatch):
    # A real bug found while VERIFYING the size-cap fix above, before it
    # ever shipped: a naive `proc.kill()` only sends SIGKILL to the
    # shell interpreter itself, not to the real child processes a
    # pipeline forks (`yes`/`head` here). Confirmed live: killing just
    # the shell after stopping an early read left the pipeline's actual
    # commands alive and orphaned, still blocked trying to write into a
    # pipe nothing was draining anymore -- `await proc.wait()` then hung
    # indefinitely, since the shell itself died instantly but its
    # still-running children kept the underlying pipe's write end open.
    # This test would hang (and fail via the outer asyncio.wait_for)
    # without `start_new_session=True` + `os.killpg` in the real fix.
    monkeypatch.setattr(tools_module, "_MAX_SHELL_OUTPUT_BYTES", 100)
    shell = RunShellTool()

    result = await asyncio.wait_for(
        shell.run({"command": "yes A | head -c 10000000"}, ctx), timeout=10
    )

    assert result.is_error is True


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
    # httpx.AsyncClient.send -- the common lower-level primitive both
    # `.get()` and the streaming `.stream()` this tool now uses (see
    # test_web_fetch_bounds_the_read_itself_not_just_the_final_string
    # above) ultimately call -- to return a real Response object
    # carrying a redirect Location header pointing at localhost.
    import httpx

    async def fake_send(self, request, **kw):
        return httpx.Response(
            302,
            headers={"location": "http://127.0.0.1:11434/api/tags"},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    tool = WebFetchTool()
    result = await tool.run({"url": "https://example.com/redirector"}, ctx)
    assert result.is_error
    assert "non-public address" in result.content[0].text


@pytest.mark.asyncio
async def test_web_fetch_is_not_bypassable_via_dns_rebinding(ctx, monkeypatch):
    # A real bug found by actually running ensure_public_host against a
    # fake DNS resolver, not a hypothetical: the guard alone is
    # TOCTUO-vulnerable to DNS rebinding. It resolves a hostname once to
    # check it's public, then discards that answer -- the real
    # connection made a moment later re-resolves the SAME hostname
    # independently. Simulated here with a real local HTTP server (a
    # stand-in for something an SSRF guard must never let a request
    # reach) and a fake resolver that answers the FIRST getaddrinfo call
    # (ensure_public_host's own check) with a real public IP and every
    # SUBSEQUENT call (httpx's own internal connection-time resolution)
    # with 127.0.0.1 -- confirmed live before this fix: the check passed
    # and the real request still landed on the local server, leaking its
    # response with zero user confirmation (WebFetchTool is
    # destructive=False). No redirect involved at all.
    import http.server
    import socket
    import threading

    class _InternalHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"internal secret")

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _InternalHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        call_count = {"n": 0}

        async def rebinding_getaddrinfo(host, port_arg, *a, **kw):
            call_count["n"] += 1
            # First call (ensure_public_host's own check) resolves
            # publicly; every later call (the real connection) resolves
            # to the internal server instead.
            target_ip = "93.184.216.34" if call_count["n"] == 1 else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (target_ip, port_arg or 0))]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", rebinding_getaddrinfo)

        tool = WebFetchTool()
        result = await tool.run({"url": f"http://rebind.test:{port}/"}, ctx)

        assert b"internal secret" not in result.content[0].text.encode()
        # Either a clean SSRF rejection or a connection failure to the
        # (now-unreachable, since the pinned IP resolution itself
        # blocks) target is acceptable -- what must never happen is the
        # internal server's real response reaching the model.
    finally:
        server.shutdown()


async def test_web_fetch_bounds_the_read_itself_not_just_the_final_string(ctx, monkeypatch):
    # A real bug found by giving this tool the same lens that had just
    # found RunShellTool buffering its entire stdout with no limit:
    # `client.get(url)` fully downloads the WHOLE response body into
    # memory before `.text` is ever sliced to `_MAX_FETCH_CHARS` --
    # harmless for the ordinary small page this tool usually fetches,
    # but a large-but-plausible response (a big JSON API response, an
    # uncompressed log file, a large HTML page -- no adversarial intent
    # needed) incurs the full download's memory cost before any of it
    # is thrown away. Isolated here from the (separately, thoroughly
    # tested) SSRF guard via a no-op ensure_public_host and a real
    # httpx.MockTransport streaming a huge body -- this test's job is
    # proving the download-bounding fix, not re-verifying SSRF
    # protection. Confirmed live before the fix: the mock server was
    # asked to stream 2000 chunks (125MiB) and all 2000 were consumed
    # before the final string got truncated.
    monkeypatch.setattr(tools_module, "ensure_public_host", lambda url: _noop())

    chunk = b"x" * 65536
    num_chunks = 2000  # ~125MiB if fully consumed
    chunks_yielded = {"n": 0}

    class _HugeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(num_chunks):
                chunks_yielded["n"] += 1
                yield chunk

        async def aclose(self):
            return None

    def handler(request):
        return httpx.Response(200, stream=_HugeStream())

    monkeypatch.setattr(tools_module, "ssrf_safe_transport", lambda: httpx.MockTransport(handler))

    tool = WebFetchTool()
    result = await asyncio.wait_for(tool.run({"url": "http://example.test/big"}, ctx), timeout=10)

    assert not result.is_error
    assert "[truncated]" in result.content[0].text
    assert len(result.content[0].text) < 100_000  # nowhere near the full 125MiB
    # The real, decisive assertion: the read must have stopped early,
    # not consumed the whole stream and thrown the excess away
    # afterward -- that would still incur the full memory cost this fix
    # exists to avoid.
    assert chunks_yielded["n"] < num_chunks, (
        f"consumed {chunks_yielded['n']} of {num_chunks} chunks -- not actually bounded"
    )


async def _noop() -> None:
    return None


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
async def test_remember_does_not_freeze_the_event_loop_while_waiting_on_a_contended_lock(
    tmp_path, monkeypatch
):
    # A real bug found by the same sweep that just found NoteTool's
    # identical shape: VectorMemoryStore's SQLite connection can block
    # for up to sqlite3's own 5-second default timeout waiting for
    # another writer's lock, and RememberTool.run() called `add()`
    # directly with no `asyncio.to_thread`. Confirmed live before the
    # fix: a real second OS process holding a genuine SQLite EXCLUSIVE
    # lock on the same database for 3s froze this process's entire
    # event loop for the whole window -- a heartbeat coroutine that
    # should tick roughly every 0.05s recorded ZERO ticks across the
    # full ~3s call. Also caught along the way: the lazy `_get_store()`
    # construction path (its own CREATE TABLE + commit) is just as
    # capable of blocking as `add()` itself -- exercised directly here
    # by letting RememberTool build its own store lazily on first
    # run(), the same path BUILTIN_TOOLS's real, no-store-argument
    # construction takes, rather than pre-building one synchronously
    # (which would itself block the test on the very same contended
    # lock, before RememberTool.run() is ever even called).
    import subprocess
    import sys

    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(tools_module, "DEFAULT_MEMORY_DB_PATH", db_path)
    holder = tmp_path / "hold_lock.py"
    holder.write_text(
        "import sqlite3, sys, time\n"
        "path, hold = sys.argv[1], float(sys.argv[2])\n"
        "conn = sqlite3.connect(path, timeout=hold + 10)\n"
        "conn.execute('CREATE TABLE IF NOT EXISTS entries "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, text TEXT NOT NULL)')\n"
        "conn.commit()\n"
        "conn.execute('BEGIN EXCLUSIVE')\n"
        "conn.execute(\"INSERT INTO entries (session_id, text) VALUES ('holder', 'x')\")\n"
        "open(path + '.acquired', 'w').write('1')\n"
        "time.sleep(hold)\n"
        "conn.commit()\n"
    )
    marker = Path(str(db_path) + ".acquired")
    proc = subprocess.Popen([sys.executable, str(holder), str(db_path), "1.0"])
    while not marker.exists():
        await asyncio.sleep(0.02)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task actually start ticking first
    remember = RememberTool()  # no store given -- lazily builds one on first run()
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))

    result = await remember.run({"text": "hello"}, ctx_obj)

    hb_task.cancel()
    proc.wait(timeout=10)

    assert not result.is_error
    # The real, decisive assertion: while run() was blocked waiting on
    # the contended lock (both its own lazy store construction and the
    # add() call), the event loop must have kept running other
    # coroutines -- a near-zero tick count is the literal old bug
    # reproducing itself (the loop frozen solid for the whole ~1s wait).
    assert ticks >= 10, f"event loop only ticked {ticks} times -- looks frozen"


@pytest.mark.asyncio
async def test_remember_lazy_store_construction_is_not_racy_under_real_concurrency(
    tmp_path, monkeypatch
):
    # A real bug found by a fresh-eyes sweep: `_get_store()`'s lazy
    # construction runs on a real OS worker thread (via `asyncio.
    # to_thread` above, dispatched specifically so it can't block the
    # event loop), not a cooperative asyncio task -- so the classic
    # unsynchronized check-then-act singleton race is genuinely live
    # here. `BUILTIN_TOOLS` is a module-level singleton list shared by
    # every AgentLoop the server builds, so two concurrent /ws/chat
    # connections both calling `remember` before this tool's very first
    # call -- most realistically right after server startup -- can both
    # pass `self._store is None` simultaneously and each construct
    # their own independent VectorMemoryStore, one connection silently
    # discarded unclosed the moment the other's assignment wins.
    # Confirmed live before this fix (10/10 trials with a small
    # artificial delay inside VectorMemoryStore.__init__ to make the
    # race deterministic rather than "usually"): two distinct store
    # objects constructed for the one lazily-cached `self._store` field.
    monkeypatch.setattr(tools_module, "DEFAULT_MEMORY_DB_PATH", tmp_path / "memory.db")

    created: list[VectorMemoryStore] = []
    real_init = VectorMemoryStore.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.05)  # widen the race window deterministically
        real_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(VectorMemoryStore, "__init__", slow_init)

    remember = RememberTool()  # the shared BUILTIN_TOOLS singleton shape
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))

    await asyncio.gather(
        remember.run({"text": "call A"}, ctx_obj),
        remember.run({"text": "call B"}, ctx_obj),
    )

    assert len(created) == 1, (
        f"{len(created)} distinct VectorMemoryStore objects constructed for one "
        "lazily-cached field -- the unsynchronized race reproduced itself"
    )


@pytest.mark.asyncio
async def test_recall_with_no_memories_says_so(ctx, tmp_path):
    recall = RecallMemoryTool(store=VectorMemoryStore(tmp_path / "memory.db"))
    result = await recall.run({"query": "anything"}, ctx)
    assert "No relevant memories found" in result.content[0].text


@pytest.mark.asyncio
async def test_recall_rejects_a_non_integer_top_k_with_a_clean_error(ctx, tmp_path):
    # A real bug found by a fresh-eyes sweep: the tool's own
    # input_schema declares top_k as {"type": "integer"}, but nothing
    # enforces that against a real model's actual tool-call arguments --
    # input_schema is purely descriptive, never validated server-side.
    # A model emitting top_k as the JSON string "3" (a real, plausible
    # model mistake) reached VectorMemoryStore.search()'s unguarded
    # scored[:top_k] and raised a raw, non-actionable TypeError,
    # confirmed live before this fix.
    store = VectorMemoryStore(tmp_path / "memory.db")
    store.add("default", "some relevant memory")
    recall = RecallMemoryTool(store=store)

    result = await recall.run({"query": "relevant", "top_k": "3"}, ctx)

    assert result.is_error
    assert "top_k must be a non-negative integer" in result.content[0].text


@pytest.mark.asyncio
async def test_recall_rejects_a_negative_top_k_instead_of_silently_dropping_a_result(ctx, tmp_path):
    # A second, silent failure mode from the same missing validation:
    # the schema has no `minimum` either, so a negative top_k didn't
    # error at all -- Python's own slice semantics turn `top_k=-1` into
    # "drop the last result," so a store holding exactly one genuinely
    # relevant memory returned [] and this tool reported "No relevant
    # memories found" while a relevant memory actually existed.
    # Confirmed live before this fix: this exact setup returned that
    # false "no memories found" message instead of the real memory.
    store = VectorMemoryStore(tmp_path / "memory.db")
    store.add("default", "the launch code is blue")
    recall = RecallMemoryTool(store=store)

    result = await recall.run({"query": "launch code", "top_k": -1}, ctx)

    assert result.is_error
    assert "top_k must be a non-negative integer" in result.content[0].text


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


@pytest.mark.asyncio
async def test_note_then_search_notes_round_trip(ctx, tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory")
    note = NoteTool(store=store)
    search = SearchNotesTool(store=store)

    result = await note.run({"topic": "project status", "content": "launch is next week"}, ctx)
    assert not result.is_error

    result = await search.run({"query": "launch"}, ctx)
    assert not result.is_error
    assert "project-status" in result.content[0].text
    assert "launch" in result.content[0].text


@pytest.mark.asyncio
async def test_search_notes_with_no_matches_says_so(ctx, tmp_path):
    search = SearchNotesTool(store=LongTermMemoryStore(tmp_path / "memory"))
    result = await search.run({"query": "anything"}, ctx)
    assert "No notes matched" in result.content[0].text


@pytest.mark.asyncio
async def test_note_rejects_an_unusable_topic_name_cleanly(ctx, tmp_path):
    note = NoteTool(store=LongTermMemoryStore(tmp_path / "memory"))
    result = await note.run({"topic": "!!!", "content": "some content"}, ctx)
    assert result.is_error is True
    assert "invalid topic name" in result.content[0].text


@pytest.mark.asyncio
async def test_note_rejects_an_overlong_topic_name_cleanly_not_a_raw_oserror(ctx, tmp_path):
    # A real bug found by actually writing a 500-character topic name:
    # with no length cap on the slugified topic, _path_for() built a
    # filename long enough that the OS itself rejected it --
    # LongTermMemoryStore.write() raised a raw OSError (carrying a real
    # local filesystem path in its message) instead of this module's
    # own documented LongTermMemoryError, and NoteTool only ever caught
    # the latter, so the raw OS error surfaced straight through to the
    # tool result text a model/user sees.
    note = NoteTool(store=LongTermMemoryStore(tmp_path / "memory"))
    result = await note.run({"topic": "a" * 500, "content": "some content"}, ctx)
    assert result.is_error is True
    assert "invalid topic name" in result.content[0].text
    assert "Errno" not in result.content[0].text  # not a leaked raw OSError


@pytest.mark.asyncio
async def test_note_is_visible_across_different_sessions(tmp_path):
    # The whole point of this tier, unlike RememberTool: a note written
    # from one session must be readable from a completely different
    # session's own ToolContext -- no session_id scoping at all.
    store = LongTermMemoryStore(tmp_path / "memory")
    note = NoteTool(store=store)
    search = SearchNotesTool(store=store)
    ctx_a = ToolContext(
        workdir=str(tmp_path), run_dir=str(tmp_path / "run"), session_id="session-a"
    )
    ctx_b = ToolContext(
        workdir=str(tmp_path), run_dir=str(tmp_path / "run"), session_id="session-b"
    )

    await note.run({"topic": "shared", "content": "written from session a"}, ctx_a)
    result = await search.run({"query": "written from session a"}, ctx_b)

    assert "shared" in result.content[0].text


@pytest.mark.asyncio
async def test_note_does_not_freeze_the_event_loop_while_waiting_on_a_contended_lock(tmp_path):
    # A real bug found by giving NoteTool its own dedicated fresh-eyes
    # sweep (a genuinely new area, after five straight rounds on the
    # agent loop itself): LongTermMemoryStore.write() is a fully
    # synchronous method that blocks on a real cross-process flock for
    # as long as another writer holds it (see its own docstring), and
    # NoteTool.run() called it directly from its `async def` body with
    # no `asyncio.to_thread` -- exactly the mistake `SessionStore.
    # locked`'s own docstring already documents fixing once for the
    # identical flock-blocking shape, never propagated to this newer
    # memory tier. Confirmed live before the fix: a real second OS
    # process holding the topic's `.md.lock` for 3s froze this
    # process's ENTIRE event loop for the whole window -- a heartbeat
    # coroutine that should tick roughly every 0.05s recorded ZERO
    # ticks across the full ~2.7s call, meaning every other in-flight
    # `/chat`/`/ws/chat` turn in a real `sarva serve` process would
    # have frozen too, not just this one tool call.
    import subprocess
    import sys

    memory_dir = tmp_path / "memory"
    store = LongTermMemoryStore(memory_dir)
    lock_path = store._path_for("shared-topic").with_suffix(".md.lock")

    holder = tmp_path / "hold_lock.py"
    holder.write_text(
        "import fcntl, sys, time\n"
        "f = open(sys.argv[1], 'a+b')\n"
        "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "time.sleep(float(sys.argv[2]))\n"
        "fcntl.flock(f.fileno(), fcntl.LOCK_UN)\n"
    )
    proc = subprocess.Popen([sys.executable, str(holder), str(lock_path), "1.0"])
    await asyncio.sleep(0.2)  # let the other process actually acquire the lock first

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb_task = asyncio.create_task(heartbeat())
    note = NoteTool(store=store)
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
    result = await note.run({"topic": "shared-topic", "content": "hello"}, ctx_obj)
    hb_task.cancel()
    proc.wait(timeout=10)

    assert not result.is_error
    # The real, decisive assertion: while NoteTool.run() was blocked
    # waiting on the contended lock, the event loop must have kept
    # running other coroutines -- a near-zero tick count is the literal
    # old bug reproducing itself (the loop frozen solid for the whole
    # ~1s wait).
    assert ticks >= 10, f"event loop only ticked {ticks} times -- looks frozen"


async def test_note_does_not_freeze_the_event_loop_while_lazily_constructing_its_store(
    tmp_path, monkeypatch
):
    # A second, independent instance of the identical mistake the test
    # above already fixed, found by a much later fresh-eyes sweep
    # comparing this tool against RememberTool's own `_add` helper: the
    # test above passes a PRE-BUILT `store=store`, which bypasses
    # `_get_store()`'s own lazy construction entirely and only exercises
    # `write()`'s lock contention -- exactly why this gap slipped past
    # it undetected. `self._get_store()` used to be called directly as
    # an argument expression in `run()`, evaluated eagerly on the event
    # loop *before* `asyncio.to_thread` ever got control -- only the
    # subsequent `.write` call was actually dispatched to a thread. On a
    # process's first `note` call, `_get_store()`'s own lazy
    # construction does real, blocking filesystem I/O (`Path.mkdir` +
    # `os.chmod`), which can be slow on a contended or network-mounted
    # filesystem. Confirmed live before this fix: a deliberately slowed
    # `LongTermMemoryStore.__init__` froze the event loop for the whole
    # ~1s construction, near-zero heartbeat ticks recorded.
    import sarva.memory.longterm as longterm_module

    real_init = longterm_module.LongTermMemoryStore.__init__

    def slow_init(self, directory=longterm_module.DEFAULT_LONGTERM_MEMORY_DIR):
        time.sleep(1.0)
        real_init(self, directory)

    monkeypatch.setattr(longterm_module.LongTermMemoryStore, "__init__", slow_init)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task actually start ticking first
    note = NoteTool()  # no store given -- lazily builds one on first run()
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))

    result = await note.run({"topic": "some-topic", "content": "hello"}, ctx_obj)

    hb_task.cancel()

    assert not result.is_error
    # The real, decisive assertion: while run() was blocked on the slow
    # store construction, the event loop must have kept running other
    # coroutines -- a near-zero tick count is the literal old bug
    # reproducing itself (the loop frozen solid for the whole ~1s wait).
    assert ticks >= 10, f"event loop only ticked {ticks} times -- looks frozen"


@pytest.mark.asyncio
async def test_note_lazy_store_construction_is_not_racy_under_real_concurrency(tmp_path):
    # The LongTermMemoryStore-backed sibling of RememberTool's own
    # identical race, fixed the same way and for the same reason: once
    # `_get_store()`'s lazy construction runs on a real OS worker thread
    # (see the freeze fix above), the unsynchronized check-then-act
    # singleton pattern is genuinely racy under real concurrency -- two
    # concurrent `/ws/chat` connections both calling `note` before
    # `BUILTIN_TOOLS`' shared NoteTool instance has ever built a store
    # can both pass `self._store is None` and each construct their own.
    import sarva.memory.longterm as longterm_module

    created: list[longterm_module.LongTermMemoryStore] = []
    real_init = longterm_module.LongTermMemoryStore.__init__

    def slow_init(self, directory=longterm_module.DEFAULT_LONGTERM_MEMORY_DIR):
        time.sleep(0.05)  # widen the race window deterministically
        real_init(self, directory)
        created.append(self)

    import unittest.mock

    with unittest.mock.patch.object(longterm_module.LongTermMemoryStore, "__init__", slow_init):
        note = NoteTool()  # the shared BUILTIN_TOOLS singleton shape
        ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))

        await asyncio.gather(
            note.run({"topic": "t", "content": "call A"}, ctx_obj),
            note.run({"topic": "t", "content": "call B"}, ctx_obj),
        )

    assert len(created) == 1, (
        f"{len(created)} distinct LongTermMemoryStore objects constructed for one "
        "lazily-cached field -- the unsynchronized race reproduced itself"
    )


async def test_search_notes_does_not_freeze_the_event_loop_on_a_large_notes_directory(tmp_path):
    # A real bug found by a fresh-eyes sweep of this exact same file: the
    # test above's own comment reasoned that SearchNotesTool "only ever
    # reads (never contends on this lock), so it's unaffected and left
    # as-is" -- true for lock contention specifically, but that
    # conflated "no lock contention" with "no blocking I/O."
    # LongTermMemoryStore.search() is fully synchronous -- it globs every
    # *.md file under the notes directory and calls path.read_text() on
    # each one in a loop -- and SearchNotesTool.run() called it directly
    # with no asyncio.to_thread, exactly the same blocking-call-inside-
    # async-def shape NoteTool.write already had fixed above it in this
    # same file. Confirmed live before the fix: 20,000 short notes
    # (~1.8MB total, a plausible amount after long-term real use of the
    # `note` tool across many conversations) froze this process's ENTIRE
    # event loop for the whole ~360ms search -- a heartbeat coroutine
    # that should tick roughly every 0.05s recorded ZERO ticks across
    # that window.
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    for i in range(20000):
        (memory_dir / f"topic-{i}.md").write_text(f"# topic {i}\n\nsome note content here {i}\n")

    store = LongTermMemoryStore(memory_dir)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task actually start ticking first
    search = SearchNotesTool(store=store)
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
    result = await search.run({"query": "nonexistent query xyz"}, ctx_obj)
    hb_task.cancel()

    assert not result.is_error
    # The real, decisive assertion: while search() globbed and read
    # 20,000 files, the event loop must have kept running other
    # coroutines -- a near-zero tick count is the literal old bug
    # reproducing itself (the loop frozen solid for the whole call).
    assert ticks >= 3, f"event loop only ticked {ticks} times -- looks frozen"


async def test_search_notes_does_not_freeze_the_event_loop_while_lazily_constructing_its_store(
    tmp_path, monkeypatch
):
    # A second, independent instance of the identical mistake found and
    # fixed for `NoteTool` a few tests above, found by the same later
    # fresh-eyes sweep: the large-notes-directory test above passes a
    # PRE-BUILT `store=store`, which bypasses `_get_store()`'s own lazy
    # construction entirely and only exercises `search()`'s own file-I/O
    # volume -- exactly why this gap slipped past it undetected.
    # `self._get_store()` was still called directly as an argument
    # expression in `run()`, evaluated eagerly on the event loop *before*
    # `asyncio.to_thread` ever got control -- only the subsequent
    # `.search` call was actually dispatched to a thread. On a process's
    # first `search_notes` call, `_get_store()`'s own lazy construction
    # does real, blocking filesystem I/O (`Path.mkdir` + `os.chmod`),
    # which can be slow on a contended or network-mounted filesystem.
    # Confirmed live before this fix: a deliberately slowed
    # `LongTermMemoryStore.__init__` froze the event loop for the whole
    # ~1s construction, near-zero heartbeat ticks recorded.
    import sarva.memory.longterm as longterm_module

    real_init = longterm_module.LongTermMemoryStore.__init__

    def slow_init(self, directory=longterm_module.DEFAULT_LONGTERM_MEMORY_DIR):
        time.sleep(1.0)
        real_init(self, directory)

    monkeypatch.setattr(longterm_module.LongTermMemoryStore, "__init__", slow_init)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    hb_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task actually start ticking first
    search = SearchNotesTool()  # no store given -- lazily builds one on first run()
    ctx_obj = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))

    result = await search.run({"query": "anything"}, ctx_obj)

    hb_task.cancel()

    assert not result.is_error
    # The real, decisive assertion: while run() was blocked on the slow
    # store construction, the event loop must have kept running other
    # coroutines -- a near-zero tick count is the literal old bug
    # reproducing itself (the loop frozen solid for the whole ~1s wait).
    assert ticks >= 10, f"event loop only ticked {ticks} times -- looks frozen"


def test_default_longterm_memory_tools_do_not_open_the_store_until_first_run():
    # Same laziness property as the memory tools above, same reason:
    # BUILTIN_TOOLS constructs these at module import time with no store
    # argument -- eagerly opening the default store in __init__ would
    # create real files/directories under ~/.sarva/memory/ as a side
    # effect of merely importing sarva.agent.tools.
    note = NoteTool()
    search = SearchNotesTool()
    assert note._store is None
    assert search._store is None
