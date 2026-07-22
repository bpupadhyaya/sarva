"""sarva.multimodal.fetch — resolves url-sourced media blocks to bytes.

`_MediaBlock.resolve_bytes()` (content.py) explicitly punts on url sources:
loading `data`/`path` is synchronous and local, but a url source needs
network I/O, which has no business happening synchronously inside the
agent loop's hot path — and content.py stays dependency-light (no httpx)
by design, since it's the type vocabulary every layer imports. This module
is what that method's own docstring names as the place to look.
"""

from __future__ import annotations

import httpx

from sarva.multimodal.content import _MediaBlock

_ALLOWED_SCHEMES = {"http", "https"}


class FetchError(Exception):
    """A url-sourced media block's bytes could not be retrieved."""


async def fetch_bytes(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int = 20_000_000,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Download `url`'s body. Streams rather than trusting Content-Length
    (which can be absent or dishonest) so a misbehaving server can't
    exhaust memory by lying about — or omitting — its response size.

    `client` lets a caller supply a shared/pre-configured `AsyncClient`
    (reused across calls in production, or backed by an `httpx.MockTransport`
    in tests — see test_fetch.py, which never touches the real network).
    A caller-supplied client is used as-is and never closed here; when none
    is given, one is created and closed for just this call."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme {scheme!r} (only http/https allowed): {url}")

    async def _do_fetch(http_client: httpx.AsyncClient) -> bytes:
        try:
            async with http_client.stream("GET", url) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"{url} exceeded max_bytes={max_bytes} while streaming")
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPError as e:
            raise FetchError(f"failed to fetch {url}: {e}") from e

    if client is not None:
        return await _do_fetch(client)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as owned_client:
        return await _do_fetch(owned_client)


async def resolve_media_bytes(
    block: _MediaBlock, *, client: httpx.AsyncClient | None = None
) -> bytes:
    """The async counterpart to `block.resolve_bytes()` that also handles
    `url` sources. Safe to call on any `_MediaBlock` regardless of which
    source it carries — `data`/`path` resolve exactly as the sync method
    already does, `url` is the only path that actually awaits anything."""
    if block.url is not None:
        return await fetch_bytes(block.url, client=client)
    return block.resolve_bytes()
