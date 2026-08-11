# MCP client: plugging in the ecosystem's tools

`sarva.mcp_client` closes the last named gap in §3.5's tool runtime list:
"MCP client support so the ecosystem's tools plug in without
Sarva-specific glue." Any server that speaks the
[Model Context Protocol](https://modelcontextprotocol.io) — filesystem
access, GitHub, databases, whatever a third party ships — becomes a set
of ordinary Sarva `Tool`s with no code written per server.

## Why the official SDK, not a hand-rolled client

Sarva's provider adapters already use the official `anthropic`/`openai`/
`google-genai` SDKs rather than hand-rolled HTTP against each API. The
MCP client follows the same principle: `mcp.ClientSession` from the
official `mcp` Python SDK, not a from-scratch JSON-RPC implementation.
"From scratch" in this project is reserved for the foundry's model math
(tokenizer, transformer, training loop) — commodity protocol clients are
exactly the kind of substrate the provider layer already treats as
commodity.

## What's wired up

Two transports. **Stdio** came first: most real MCP servers today are
local processes launched with `npx`/`uvx`/a plain command — stdio covers
that majority, and it's the one transport genuinely verifiable offline:
spawn a real local subprocess, speak real MCP over its stdin/stdout, no
network call involved.

```python
from sarva.mcp_client import connect_stdio_mcp_server, list_mcp_tools

async with connect_stdio_mcp_server("npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as session:
    tools = await list_mcp_tools(session)
    # tools is a list of ready-to-use Sarva `Tool`s — pass straight into AgentLoop
```

**Streamable HTTP** closes the gap this chapter used to name as real,
deferred scope — MCP's current standard HTTP transport (spec revision
2025-03-26, superseding the older, separate SSE transport). Nothing
downstream cares which transport a session came from; both
`connect_stdio_mcp_server` and `connect_http_mcp_server` just hand back
an initialized `ClientSession`.

```python
from sarva.mcp_client import connect_http_mcp_server, list_mcp_tools

async with connect_http_mcp_server(
    "https://example.com/mcp", headers={"Authorization": "Bearer ..."}
) as session:
    tools = await list_mcp_tools(session)
```

`headers` is the one thing an HTTP server usually needs that a local
stdio subprocess doesn't — most real deployments put an auth token
there. Sarva has no opinion on any particular server's auth scheme, so
it's left entirely to the caller. The older `mcp.client.sse` transport
(plain SSE, pre-2025-03-26) is real and still shipped by the underlying
`mcp` SDK for servers that haven't moved off it, but isn't wired up here
— "current standard transport," not "every historical variant."

Each `McpToolAdapter` implements the same `Tool` protocol as every
built-in tool (`spec` + `async def run(args, ctx)`), so nothing downstream
— the agent loop, the confirmation policy, transcript logging — needs to
know or care that a given tool call is actually a round trip to a
subprocess speaking MCP instead of local Python code.

### A real security gap found by a fresh-eyes sweep: every MCP tool silently bypassed the destructive-confirmation gate — and a colliding name could take over a builtin's

A dedicated sweep of this module, after several rounds focused
elsewhere, found `McpToolAdapter` never set `ToolSpec.destructive` —
which defaults to `False` — so every MCP-provided tool, arbitrary and
unaudited remote code, silently bypassed the agent loop's
confirm-before-destructive-action gate regardless of what it actually
does. MCP's own protocol *does* carry a `destructiveHint` annotation on
the wire, but the spec's own documentation for it is explicit: "clients
should never make tool use decisions based on ToolAnnotations received
from untrusted servers" — a malicious or buggy server could simply set
`destructiveHint=False` on a genuinely destructive tool to defeat
exactly this gate. Every MCP tool is now marked `destructive=True`
unconditionally, deliberately ignoring that hint rather than trusting a
remote server's own self-report.

A second, compounding gap made this worse for name collisions
specifically: `AgentLoop.__init__` builds its tool dispatch table as
`{t.spec.name: t for t in tools}` — a later tool with the same name
silently replaces an earlier one. `sarva run`'s own `--mcp-server` help
text names `@modelcontextprotocol/server-filesystem` as its example
command, and that real, official, unmodified server exports tools
literally named `read_file`/`write_file`/`edit_file` — identical to
Sarva's own builtins of the same name, two of which (`write_file`,
`edit_file`) are marked `destructive=True` specifically so the confirm
gate asks before running them. Connecting it, following the CLI's own
documented example, silently took over those names: confirmed live
with a `confirm` callback that raises if it's ever invoked — it wasn't,
and the (fake, in the repro) MCP server's own implementation ran
unconfirmed while the real local file was left untouched, a completely
different outcome than what the user would reasonably expect.

Fixed with two changes: `McpToolAdapter`'s `ToolSpec` now sets
`destructive=True` always (above), and `sarva run`'s `_run()` now
checks every newly-connected MCP server's tool names against every
already-registered name (builtins and any earlier `--mcp-server`) and
refuses to continue if any collide, printing which name(s) collided
and why, rather than silently letting one replace the other — which of
the two a user actually wanted is genuinely ambiguous, and guessing
wrong in either direction is the wrong failure mode for a
security-relevant gate. Verified live both fixes close the gap: the
confirm callback now correctly fires for an MCP tool call, and a
colliding server name is now rejected with a clean, actionable message
before the run ever starts. 3 new tests.

## CLI usage

```bash
sarva run "list the files in /tmp" \
    --mcp-server "npx -y @modelcontextprotocol/server-filesystem /tmp" \
    --auto

# An http:// or https:// value connects over Streamable HTTP instead:
sarva run "..." --mcp-server https://example.com/mcp --auto

# Most real HTTP MCP deployments need auth -- connect_http_mcp_server()
# has always accepted a headers dict, but nothing threaded one through
# from the command line until --mcp-header closed that gap:
sarva run "..." --mcp-server https://example.com/mcp \
    --mcp-header "Authorization: Bearer sk-..." --auto

# A real npx/uvx-run stdio server often reads its own auth token from
# its process environment rather than an HTTP header --
# connect_stdio_mcp_server() has always accepted an env dict, but
# nothing threaded one through from the command line until --mcp-env
# closed that gap too:
sarva run "..." \
    --mcp-server "npx -y some-server-that-needs-a-token" \
    --mcp-env "GITHUB_TOKEN=ghp_..." --auto
```

`--mcp-server` is repeatable, and each value is dispatched by shape —
`http://`/`https://` connects over Streamable HTTP, anything else is
shell-split and run as a stdio subprocess command — so both transports
can be mixed freely in one run. Each server's tools are listed once at
startup and merged into the same tool registry as the built-ins
(`read_file`, `write_file`, `remember`, `recall_memory`, ...); the model
sees one flat set of tools, with no way to tell which ones came from
where or which transport carried them.

**The startup tool-listing line escapes what it prints, not just what
it processes.** A real gap found by auditing this file's own CLI
command for the same "unescaped externally-sourced text" bug class
already fixed for `sarva doctor`/`sarva transcribe`: the connected
server's own reported tool names were being interpolated straight into
a Rich-markup string with no `escape()` at all — for an `http(s)://`
server, that name comes from a remote, untrusted source, so a
malicious or buggy server naming a tool with embedded Rich markup
could spoof this project's own terminal output (fake status lines,
hidden text). Fixed with `escape()` on both the tool names and the
echoed `--mcp-server` value itself; pinned with a test that fakes a
tool named `"[red]FAKE ERROR[/red] normal_tool"` and asserts it prints
*verbatim*, not interpreted, and confirmed the test genuinely catches
a regression by reverting the fix and watching it fail before
re-applying. Verified against a real stdio MCP server too (the
project's own `echo`/`fail` fixture) to confirm ordinary tool names
still render exactly as before.

**A real bug found by actually running `sarva run --mcp-server
"definitely-not-a-real-command"`, and the equivalent for an
unreachable `https://` URL and a malformed shell string:** the
`--mcp-server` connection loop had zero error handling anywhere on the
path — `connect_stdio_mcp_server`'s subprocess spawn
(`FileNotFoundError`), `connect_http_mcp_server`'s network layer
(`httpx.ConnectError`, wrapped in an anyio `TaskGroup`'s own
`ExceptionGroup`), and even the loop's own `shlex.split(server_cmd)`
(a plain `ValueError` on something like an unterminated quote) all
crashed the whole `sarva run` command with a raw traceback instead of
the same clean, actionable failure `--model`/`--session`/`--image`
already get on bad user input. Caught broadly around the whole
per-server connection attempt — any exception there means "this MCP
server couldn't be reached" — matching the "reject, don't guess"
discipline `--mcp-header`/`--mcp-env` parsing already applies: a
`--mcp-server` the user explicitly asked for silently failing and the
run continuing without its tools would be a materially different,
unexplained result, not something safe to paper over the way a
corrupted on-disk cache entry is. The HTTP case gets one extra
refinement: an `ExceptionGroup` with exactly one sub-exception (the
common single-connection-failure case) is unwrapped so the real reason
(e.g. `[Errno 8] nodename nor servname provided, or not known`) reaches
the user instead of anyio's own unhelpful `"unhandled errors in a
TaskGroup (1 sub-exception)"` summary.

**A much later fresh-eyes sweep found the collision check's own clean
failure path had the identical problem, one layer deeper.** Confirmed
live by actually connecting the real, official
`@modelcontextprotocol/server-filesystem` package (the exact server
this file's own CLI example names) and triggering the tool-name
collision above: both `raise typer.Exit(1)` calls in `_run()` fired
while the real MCP `ClientSession` was still live on the
`AsyncExitStack` -- exiting that stack via an exception forces
`__aexit__` to throw it INTO the still-open session's own `__aexit__`,
and `mcp`'s real `ClientSession`/`stdio_client` use an internal anyio
TaskGroup, which wraps ANY exception threaded through its own
`__aexit__` into a raw `BaseExceptionGroup` -- the identical wrapping
mechanism the connection-failure fix above already names for
`connect_http_mcp_server`'s own network layer, just reached through the
opposite direction (an exception thrown INTO a live session, not
raised out of connecting one). The real command dumped a full,
unhandled `ExceptionGroup` traceback straight past this file's own
already-printed clean red message. The existing collision test's own
fake `connect_stdio_mcp_server` is a plain `@asynccontextmanager` with
no TaskGroup at all, so it structurally couldn't reproduce this -- the
identical "the test double can't represent the real failure mode" gap
already found once this session for `MockProvider`'s own empty-text
bug.

Fixed by never raising `typer.Exit` while a session is still live: a
collision or connection failure now only sets a flag and breaks out of
the server-connection loop, so the `AsyncExitStack` block always exits
NORMALLY (every live session torn down via an ordinary, exception-free
`__aexit__`) -- `typer.Exit(1)` is raised only after that block has
already fully, cleanly closed. Verified with a more faithful fake that
keeps a real anyio `TaskGroup` with a live child task open around its
own `yield` (genuinely reproducing the wrapping mechanism without
needing a real npx-launched subprocess), and against the real
`server-filesystem` package directly. A real gap found writing that
test: `CliRunner`'s own harness never prints a raw traceback to
`stdout` regardless of the underlying exception type, so a
stdout-only assertion would have passed identically whether the fix
was present or not -- the decisive check is `result.exception`
itself, a clean `SystemExit` after the fix versus a raw
`ExceptionGroup` before it.

`--mcp-header` (also repeatable, `"Name: Value"`) applies to every
`http(s)://` `--mcp-server` in the same invocation alike — a real,
named limit, not silently glossed over: the (rare) case of two HTTP
servers in one run needing different auth isn't supported. Malformed
entries (no `:`) fail immediately with a clear error rather than being
silently dropped, the same "reject, don't guess" discipline session-name
validation already applies elsewhere in this file.

**`--mcp-env` (repeatable, `"NAME=VALUE"`) is `--mcp-header`'s stdio
counterpart, and closes a real gap two separate Explore-agent sweeps
found but the first one didn't pick up.**
`connect_stdio_mcp_server`'s `env` parameter has accepted a dict since
MCP support shipped, but nothing threaded one through from the command
line at all — a real, common case (an `npx`/`uvx`-run server reading
its own auth token from its process environment, since a local
subprocess has no HTTP headers to carry one in) genuinely couldn't
receive it. It's merged on top of the underlying MCP SDK's own fixed
safe-to-inherit environment (`PATH`, `HOME`, ...) rather than replacing
it outright — confirmed directly by reading `stdio_client()`'s own
source (`{**get_default_environment(), **server.env}` when `env` is
given), not assumed. Applies to every stdio server in the same
invocation alike, the same named per-run limit `--mcp-header` has.
**Verified against a real spawned subprocess, not just parsed and
passed along:** `tests/fixtures/mcp_echo_server.py` gained a third
tool, `env_var(name)`, returning the named environment variable's real
value (or `"MISSING"`) — a real MCP round trip through a real
subprocess proves the value set via `env=` genuinely lands in that
child process's own environment, and a matching test with `env=None`
confirms nothing leaks in in the other direction either.

## Content conversion, honestly scoped

An MCP tool result can carry text, images, audio, resource links, or
embedded resources. Text and images convert directly to Sarva's
`TextBlock`/`ImageBlock`. Everything else reports its own declared MCP
content type rather than being silently dropped or raising — the same
"report only what's verifiably known" principle the multimodal degraders
use for content a layer can't fully consume.

### A much later fresh-eyes sweep found the client only ever read one of `CallToolResult`'s two result fields

`mcp.types.CallToolResult` carries `content` (unstructured) *and*
`structuredContent` (a JSON object, validated against the tool's own
declared `outputSchema` when it has one) as two genuinely separate
fields — `McpToolAdapter.run()` only ever read `content`. The MCP spec
(2025-06-18) only *requires* a server to send `structuredContent`;
duplicating it into `content` as backward-compat text is a SHOULD, not
a MUST. The reference `mcp` SDK auto-enforces that SHOULD only for the
simple "tool function returns a plain dict" path — its own lower-level
`Server.call_tool` decorator (a legitimate, documented way to build an
MCP server) also supports returning `(unstructured_content,
structured_content)` directly, where a server author can legitimately
send `content=[]` alongside a real `structuredContent`.

Confirmed live against a **real** MCP server, not a mock — matching
this module's own established discipline for every claim in this
file: `tests/fixtures/mcp_echo_server.py` gained a `structured_only`
tool that returns a real `mcp_types.CallToolResult(content=[],
structuredContent={"sum": a + b}, isError=False)` directly (FastMCP's
own documented escape hatch for a tool function that wants to control
its own result shape exactly). Before this fix, the round trip
produced `is_error=False` with an entirely empty `content` list — a
clean, successful-looking tool result the model would see with the
actual computed answer nowhere in it, no error, no signal anything was
lost. Directly contradicts this same section's own principle two
paragraphs up: "report only what's verifiably known... rather than
being silently dropped."

Fixed by appending `structuredContent`, when present, as its own
`TextBlock` (`[MCP tool structured result: <json>]`) — appended rather
than replacing `content`, since a spec-compliant server may have
already duplicated it into `content` as text, and there's no reliable
way to detect that duplication from the client side; a harmless repeat
is a strictly better outcome than sometimes silently losing the only
copy. Verified live the real round trip above now surfaces `"sum": 42`
in the returned `ToolResultBlock`. Verified by reverting and watching
the new test fail with the literal old bug's own shape: `result.
content` completely empty, no trace of the answer anywhere. 1 new
test, 768 → 769 Python tests.

### A much later fresh-eyes sweep found neither connection helper ever bounded a hung server — no timeout at all, not even a generous one

`ClientSession`'s own `read_timeout_seconds` constructor parameter
defaults to `None` — unbounded — and neither `connect_stdio_mcp_server`
nor `connect_http_mcp_server` ever set it. Confirmed live: constructing
a real `ClientSession` over a stream that never delivers a response and
calling `session.initialize()` hung indefinitely, still blocked after a
2-second `asyncio.wait_for` deadline placed around it purely to observe
the hang rather than reproduce it forever. This isn't just the initial
handshake either — `read_timeout_seconds` bounds *every* request the
session ever makes (`initialize()`, `list_tools()`, `call_tool()`
alike), so a connected-but-then-unresponsive server — one that
spawns/connects successfully, so no `FileNotFoundError`/`httpx.
ConnectError` for the existing broad `except Exception` at the CLI's
own call site to catch, but hangs mid-handshake, or stops responding
partway through a session (a slow-starting `npx`-launched process still
downloading its own package on first run, or a server that crashes into
a zombie state without actually exiting — ordinary real-world failure
modes, not adversarial ones) — would hang the entire `sarva run`/`/ws/
chat` turn forever, with no recovery short of killing the process. The
same "blocking call with no timeout" bug class already fixed for the
human-confirmation read (`_CONFIRM_TIMEOUT_SECONDS` in `server/app.py`)
and the generic per-tool-call backstop (`_TOOL_TIMEOUT_SECONDS` in
`agent/loop.py`) elsewhere in this project, just never applied to the
one place a hang can happen *before* either of those two mechanisms is
even wired up — MCP connection and tool-discovery both happen during
`sarva run`'s own setup, ahead of `AgentLoop` starting.

Fixed with a module-level `_MCP_READ_TIMEOUT = timedelta(seconds=30)`
passed to every `ClientSession(...)` construction in both connection
helpers — generous enough for a genuinely slow first-run `npx` install,
bounded enough that a hung server no longer holds the whole run
hostage. The SDK's own timeout raises a clean, descriptive `McpError`
("Timed out while waiting for response to ClientRequest. Waited 30.0
seconds.") rather than a bare, message-less exception, and the CLI's
own existing single-exception `ExceptionGroup` unwrapping (added for
`connect_http_mcp_server`'s real network failures) already surfaces it
to the user cleanly, with no further changes needed there. Verified by
reverting and confirming the new test can't even exercise the fixed
code path without it — an `AttributeError` for the now-missing
`_MCP_READ_TIMEOUT` constant — the safest possible revert check for a
fix whose whole point is "this used to hang forever." 1 new test,
882 → 883 Python tests.

## Verification

`tests/conformance/test_mcp_client.py` runs against a real MCP server
(`tests/fixtures/mcp_echo_server.py`, built with the official SDK's
`FastMCP`), launched as a genuine subprocess over genuine stdio — not a
mock of the protocol. It covers tool listing, a successful call, a
failing call (proving MCP error propagation reaches Sarva's
`ToolResultBlock.is_error`), and — the one that actually proves the
integration, not just the wrapper in isolation — a real `AgentLoop.run()`
driven by a `MockProvider` script that calls the MCP-backed tool and
gets back the exact text the real subprocess produced.

`test_mcp_client_http.py` mirrors every one of those cases against
`tests/fixtures/mcp_http_echo_server.py` — the same two tools, but
launched as a real subprocess serving real MCP-over-Streamable-HTTP on a
real (OS-assigned free) local port, so the two transports are proven
equivalent from a caller's point of view, not just independently
plausible. Both `sarva run`'s `--mcp-server` command-line path and a
custom `Authorization` header were exercised directly against a live
running server before calling this done, not just through pytest.

**Every test above talks to a fixture server this same project wrote**
— a real subprocess, but one that could in principle share this
client's own misunderstanding of the protocol. `tests/live/
test_live_mcp.py` closes that specific gap: a genuinely independent,
third-party MCP server, `@modelcontextprotocol/server-filesystem`
(Anthropic's own official reference filesystem server, launched for
real via `npx`), listed 14 real tools it actually implements, and a
real read + write round trip — the write direction verified by reading
the file back from disk directly afterward, not by trusting the tool
result alone. Also confirmed through the actual CLI (`sarva run
--mcp-server "npx -y @modelcontextprotocol/server-filesystem ..."`),
which connected and printed the real tool list. Live-gated like every
other real-external-service test in this project (`pytest.mark.live`,
skipped by default, additionally skipped if `npx` isn't on `PATH`) — a
CI run depending on npm registry availability on every push isn't a
tradeoff this project makes for any live external verification.
