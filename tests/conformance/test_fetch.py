"""Conformance tests for sarva.multimodal.fetch. Uses httpx.MockTransport
throughout — no real network calls, fully deterministic.

`ensure_public_host`'s own real DNS lookup is deliberately bypassed for
the tests below that aren't *about* it (they use `https://example.com/
...` purely as a stand-in hostname for exercising response handling) --
a real DNS lookup for "example.com" would make this file's own "no real
network calls" promise false, the same test-isolation discipline this
project applies elsewhere. The SSRF guard itself gets its own dedicated
tests further down, using real IP literals that need no DNS at all.
"""

from __future__ import annotations

import httpx
import pytest
import sarva.multimodal.fetch as fetch_module
from sarva.multimodal.content import ImageBlock
from sarva.multimodal.fetch import FetchError, ensure_public_host, fetch_bytes, resolve_media_bytes


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _skip_ssrf_check(monkeypatch) -> None:
    """These tests are about response handling (body, errors, streaming
    limits), not the SSRF guard -- real DNS resolution for their
    stand-in "example.com" URLs would silently make this file dependent
    on real network access."""

    async def _noop(url: str) -> None:
        return None

    monkeypatch.setattr(fetch_module, "ensure_public_host", _noop)


async def test_fetch_bytes_returns_the_response_body(monkeypatch):
    _skip_ssrf_check(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG\r\n fake image bytes")

    result = await fetch_bytes("https://example.com/cat.png", client=_mock_client(handler))
    assert result == b"\x89PNG\r\n fake image bytes"


async def test_fetch_bytes_rejects_non_http_schemes():
    with pytest.raises(FetchError, match="unsupported URL scheme"):
        await fetch_bytes("file:///etc/passwd")
    with pytest.raises(FetchError, match="unsupported URL scheme"):
        await fetch_bytes("ftp://example.com/x")


async def test_fetch_bytes_raises_fetch_error_on_http_error_status(monkeypatch):
    _skip_ssrf_check(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    with pytest.raises(FetchError, match="failed to fetch"):
        await fetch_bytes("https://example.com/missing.png", client=_mock_client(handler))


async def test_fetch_bytes_enforces_max_bytes_while_streaming(monkeypatch):
    _skip_ssrf_check(monkeypatch)

    # No Content-Length trust here: the handler doesn't even set the header,
    # matching a real misbehaving/lying server -- the cap must still trigger
    # from counting streamed bytes directly.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1000)

    with pytest.raises(FetchError, match="exceeded max_bytes"):
        await fetch_bytes(
            "https://example.com/huge.bin", max_bytes=100, client=_mock_client(handler)
        )


async def test_resolve_media_bytes_dispatches_data_source_without_network():
    block = ImageBlock(media_type="image/png", data=b"\x89PNG raw bytes")
    # No client given, and no mock transport -- if this tried to hit the
    # network it would hang/fail; passing means the data path never touches
    # fetch_bytes at all, exactly as intended.
    result = await resolve_media_bytes(block)
    assert result == b"\x89PNG raw bytes"


async def test_resolve_media_bytes_dispatches_path_source(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG from disk")
    block = ImageBlock(media_type="image/png", path=str(path))
    result = await resolve_media_bytes(block)
    assert result == b"\x89PNG from disk"


async def test_resolve_media_bytes_dispatches_url_source(monkeypatch):
    _skip_ssrf_check(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.com/remote.png"
        return httpx.Response(200, content=b"\x89PNG from the network")

    block = ImageBlock(media_type="image/png", url="https://example.com/remote.png")
    result = await resolve_media_bytes(block, client=_mock_client(handler))
    assert result == b"\x89PNG from the network"


# ---------- ensure_public_host / SSRF guard ----------
# The real reason this fetch path needed the identical guard
# WebFetchTool got: nothing in this codebase currently wires a
# url-sourced content block up to external/model-controlled input
# (checked directly before adding this), but the type exists
# specifically to support it, and leaving this path unguarded while
# WebFetchTool's got fixed would be real, avoidable inconsistency the
# moment anything does. Real IP literals below -- no DNS lookup needed,
# genuinely hermetic.


async def test_ensure_public_host_blocks_loopback_addresses():
    with pytest.raises(FetchError, match="non-public address"):
        await ensure_public_host("http://127.0.0.1:11434/api/tags")


async def test_ensure_public_host_blocks_the_cloud_metadata_address():
    with pytest.raises(FetchError, match="non-public address"):
        await ensure_public_host("http://169.254.169.254/latest/meta-data/")


async def test_ensure_public_host_blocks_private_rfc1918_addresses():
    with pytest.raises(FetchError, match="non-public address"):
        await ensure_public_host("http://192.168.1.1/admin")


async def test_fetch_bytes_actually_calls_the_ssrf_guard():
    # Proves the integration point, not just that the guard function
    # works in isolation -- fetch_bytes must reject an internal address
    # even with a client already supplied (so a caller can't bypass the
    # guard just by passing their own client).
    with pytest.raises(FetchError, match="non-public address"):
        await fetch_bytes("http://127.0.0.1:11434/api/tags", client=_mock_client(lambda r: None))


async def test_fetch_bytes_rejects_a_redirect_to_an_internal_address():
    # The real reason follow_redirects was replaced with a manual,
    # per-hop-validated loop: a caller-supplied URL can be a legitimate
    # public site whose server issues a redirect straight to an
    # internal address. Simulated here (no real attacker-controlled
    # public redirector available to test against) via a MockTransport
    # that always answers with a redirect to localhost. Starting URL is
    # a real public IP literal (Cloudflare's public resolver), not a
    # hostname -- keeps this test needing no real DNS lookup either.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/tags"})

    with pytest.raises(FetchError, match="non-public address"):
        await fetch_bytes("http://1.1.1.1/redirector", client=_mock_client(handler))
