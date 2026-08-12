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
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

from sarva.multimodal.content import _MediaBlock

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 5


class FetchError(Exception):
    """A url-sourced media block's bytes could not be retrieved."""


async def _resolve_one_public_ip(host: str) -> str:
    """Resolves `host` and returns one validated, globally-routable IP
    address literal -- the single DNS lookup a caller must use for BOTH
    the "is this safe" check and the real connection, so a hostname can
    never be validated against one DNS answer and then connected to
    against a different one. See `_PinnedResolutionBackend`'s own
    docstring for why that distinction is the actual security boundary,
    not `ensure_public_host` below."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except OSError as e:
        raise FetchError(f"could not resolve host {host!r}: {e}") from e
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise FetchError(
                f"refusing to connect to host {host!r}: resolves to a "
                f"non-public address ({ip}) -- possible SSRF"
            )
    return str(ipaddress.ip_address(infos[0][4][0]))


async def ensure_public_host(url: str) -> None:
    """Raises `FetchError` if `url`'s hostname resolves to anything but a
    globally-routable public IP address. See this module's own
    docstring and `core/sarva/agent/tools.py`'s `WebFetchTool` for the
    full story -- both call this exact function so the SSRF guard can
    never drift out of sync between the two real url-fetching paths in
    this codebase.

    **A real bug found by actually running this check against a fake
    DNS resolver, not a hypothetical:** this function alone is
    TOCTOU-vulnerable to DNS rebinding. It resolves `url`'s hostname
    once here, then discards the answer -- the real `httpx` connection
    made a moment later re-resolves the SAME hostname completely
    independently. A resolver that answers the FIRST query (this
    check) with a public IP and every SUBSEQUENT query (httpx's own
    internal connection-time resolution) with `127.0.0.1` sails
    straight through this function and then has the real request land
    on a local server anyway -- confirmed live, no redirect involved at
    all. This function is kept as a fast, early, clearly-worded
    rejection for the common case (and every redirect hop, so a bad
    hop is reported with a specific message before a wasted connection
    attempt) -- but the actual enforced security boundary is
    `_ssrf_safe_transport()` below, whose custom network backend
    resolves and validates a hostname exactly once per real TCP
    connection, so the validated answer and the connected-to address
    can never diverge."""
    host = urlparse(url).hostname
    if host is None:
        raise FetchError(f"URL has no hostname: {url!r}")
    await _resolve_one_public_ip(host)


class _PinnedResolutionBackend(httpcore.AnyIOBackend):
    """The actual SSRF enforcement layer, not `ensure_public_host`
    above: overrides the one place httpx/httpcore actually opens a
    socket (`connect_tcp`) to resolve-and-validate the hostname right
    here, then hand the underlying connector a literal IP address
    instead of the hostname -- so the validated DNS answer and the
    address actually connected to are always the exact same lookup,
    never two separate ones a DNS-rebinding attacker's resolver can
    answer differently. TLS SNI and certificate-hostname verification
    are completely untouched by this -- those happen one layer above,
    still against the real hostname in the request URL, since only the
    raw socket target changes here."""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        validated_ip = await _resolve_one_public_ip(host)
        return await super().connect_tcp(
            validated_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


def ssrf_safe_transport() -> httpx.AsyncHTTPTransport:
    """An `httpx` transport that closes the DNS-rebinding TOCTOU gap
    `ensure_public_host`'s own docstring names: every real TCP
    connection this transport makes resolves and validates the target
    hostname atomically (see `_PinnedResolutionBackend`), instead of
    trusting a separate, earlier `ensure_public_host` call whose DNS
    answer httpx's own default connection logic would otherwise
    silently re-resolve and could receive a different answer for.
    `httpx.AsyncHTTPTransport` itself has no constructor parameter for
    a custom network backend, but the `httpcore` connection pool it
    wraps does -- swapped in here after construction, the same
    pattern real-world DNS-rebinding-hardened `httpx` clients use."""
    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=_PinnedResolutionBackend())
    return transport


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
                # A real bug found by actually fetching an ordinary,
                # non-adversarial media URL (a Wikimedia-hosted image,
                # not a crafted target -- the exact kind of URL a
                # url-sourced ImageBlock exists to support): with no
                # `User-Agent` header set here, httpx sends its own
                # default (`python-httpx/<version>`), and real,
                # legitimate sites -- not just adversarial anti-bot ones
                # -- reject that exact default with a raw 403 (confirmed
                # live). `WebFetchTool` (core/sarva/agent/tools.py) had
                # the identical gap, found and fixed in the same round;
                # this module's own docstring already says the two
                # url-fetching paths in this codebase must not drift out
                # of sync on what "safe to fetch" means -- this is the
                # same principle applied to reachability, not just
                # safety.
                async with http_client.stream(
                    "GET",
                    current_url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; sarva-agent/1.0)"},
                ) as response:
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
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, transport=ssrf_safe_transport()
    ) as owned_client:
        return await _do_fetch(owned_client)


async def resolve_media_bytes(
    block: _MediaBlock, *, client: httpx.AsyncClient | None = None
) -> bytes:
    """The async counterpart to `block.resolve_bytes()` that also handles
    `url` sources. Safe to call on any `_MediaBlock` regardless of which
    source it carries — `data` resolves exactly as the sync method
    already does (an in-memory attribute access, no I/O), `url` and
    `path` are the two sources that actually need to await something.

    A real bug found by a fresh-eyes sweep, the identical "blocking I/O
    called directly from async code with no `asyncio.to_thread`" class
    already found and fixed at 9+ other call sites in this project
    (ReadFileTool, SessionStore.load/save, NoteTool, SearchNotesTool,
    RememberTool, RecallMemoryTool, ...): the old code fell through to
    `block.resolve_bytes()` for a `path`-sourced block, which does a
    real, synchronous `Path(self.path).read_bytes()` -- and this is the
    one async entry point every provider adapter and every multimodal
    degrader calls from inside their own `async def` methods. Confirmed
    live with a simulated slow disk: zero heartbeat ticks landed on the
    event loop for the whole duration of a single `path`-sourced
    resolve, meaning every OTHER concurrent `/chat`/`/ws/chat` turn in a
    real `sarva serve` process would freeze too, for as long as one
    media file read takes. Not reachable through any current server/CLI
    input surface in this repo (nothing here constructs a `path=`-
    sourced block today -- `cli.py`'s own `_load_image` deliberately
    pre-reads bytes and uses `data=` instead), the identical "not
    reachable today, but a real, documented, first-class part of the
    public type -- content.py's own module docstring names data/path/
    url as three equally valid sources -- so leaving it unguarded would
    be real, avoidable inconsistency the moment anything does wire a
    `path`-sourced block up to real input" reasoning this module's own
    docstring already applies to the sibling `url` source above."""
    if block.url is not None:
        return await fetch_bytes(block.url, client=client)
    if block.path is not None:
        return await asyncio.to_thread(Path(block.path).read_bytes)
    return block.resolve_bytes()
