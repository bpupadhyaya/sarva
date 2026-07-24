"""sarva.multimodal.fetch — resolves url-sourced media blocks to bytes.

`_MediaBlock.resolve_bytes()` (content.py) explicitly punts on url sources:
loading `data`/`path` is synchronous and local, but a url source needs
network I/O, which has no business happening synchronously inside the
agent loop's hot path — and content.py stays dependency-light (no httpx)
by design, since it's the type vocabulary every layer imports. This module
is what that method's own docstring names as the place to look.

`ensure_public_host` (the same SSRF guard `WebFetchTool`
(`core/sarva/agent/tools.py`) uses -- checked into one shared module so
neither url-fetching path in this codebase can drift out of sync on
what "safe to fetch" means) blocks requests to private/loopback/
link-local/reserved addresses before every fetch, including every
redirect hop. Not reachable through any current attacker-controlled
input path in this codebase (no server endpoint or MCP tool result
constructs a `url`-sourced content block from external input today —
checked directly, not assumed), but the type exists specifically to
support url-sourced media, and leaving the *other* real url-fetching
path in this codebase unguarded while `WebFetchTool` got the fix would
be real, avoidable inconsistency the moment anything does wire a
url-sourced block up to external input.
"""

from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urljoin, urlparse

import httpx

from sarva.multimodal.content import _MediaBlock

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 5


class FetchError(Exception):
    """A url-sourced media block's bytes could not be retrieved."""


async def ensure_public_host(url: str) -> None:
    """Raises `FetchError` if `url`'s hostname resolves to anything but a
    globally-routable public IP address. See this module's own
    docstring and `core/sarva/agent/tools.py`'s `WebFetchTool` for the
    full story -- both call this exact function so the SSRF guard can
    never drift out of sync between the two real url-fetching paths in
    this codebase."""
    host = urlparse(url).hostname
    if host is None:
        raise FetchError(f"URL has no hostname: {url!r}")
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except OSError as e:
        raise FetchError(f"could not resolve host {host!r}: {e}") from e
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise FetchError(
                f"refusing to fetch {url!r}: host {host!r} resolves to "
                f"a non-public address ({ip}) -- possible SSRF"
            )


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

    Every redirect hop is re-validated against `ensure_public_host`, not
    just the caller-supplied `url` — a legitimate public server could
    otherwise redirect straight to an internal address, the same bypass
    `WebFetchTool`'s own fix closes.

    `client` lets a caller supply a shared/pre-configured `AsyncClient`
    (reused across calls in production, or backed by an `httpx.MockTransport`
    in tests — see test_fetch.py, which never touches the real network).
    A caller-supplied client is used as-is and never closed here; when none
    is given, one is created and closed for just this call."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme {scheme!r} (only http/https allowed): {url}")

    async def _do_fetch(http_client: httpx.AsyncClient) -> bytes:
        current_url = url
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                await ensure_public_host(current_url)
                async with http_client.stream("GET", current_url) as response:
                    if response.is_redirect and response.has_redirect_location:
                        current_url = urljoin(str(response.url), response.headers["location"])
                        continue
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise FetchError(
                                f"{url} exceeded max_bytes={max_bytes} while streaming"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            raise FetchError(f"too many redirects fetching {url}")
        except httpx.HTTPError as e:
            raise FetchError(f"failed to fetch {url}: {e}") from e

    if client is not None:
        return await _do_fetch(client)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as owned_client:
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
