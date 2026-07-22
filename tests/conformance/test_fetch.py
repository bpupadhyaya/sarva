"""Conformance tests for sarva.multimodal.fetch. Uses httpx.MockTransport
throughout — no real network calls, fully deterministic."""

from __future__ import annotations

import httpx
import pytest
from sarva.multimodal.content import ImageBlock
from sarva.multimodal.fetch import FetchError, fetch_bytes, resolve_media_bytes


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_bytes_returns_the_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG\r\n fake image bytes")

    result = await fetch_bytes("https://example.com/cat.png", client=_mock_client(handler))
    assert result == b"\x89PNG\r\n fake image bytes"


async def test_fetch_bytes_rejects_non_http_schemes():
    with pytest.raises(FetchError, match="unsupported URL scheme"):
        await fetch_bytes("file:///etc/passwd")
    with pytest.raises(FetchError, match="unsupported URL scheme"):
        await fetch_bytes("ftp://example.com/x")


async def test_fetch_bytes_raises_fetch_error_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    with pytest.raises(FetchError, match="failed to fetch"):
        await fetch_bytes("https://example.com/missing.png", client=_mock_client(handler))


async def test_fetch_bytes_enforces_max_bytes_while_streaming():
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


async def test_resolve_media_bytes_dispatches_url_source():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.com/remote.png"
        return httpx.Response(200, content=b"\x89PNG from the network")

    block = ImageBlock(media_type="image/png", url="https://example.com/remote.png")
    result = await resolve_media_bytes(block, client=_mock_client(handler))
    assert result == b"\x89PNG from the network"
