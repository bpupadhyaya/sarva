# Chapter 3 — The Agent Loop

Chapter 2 covered how Sarva talks to a model. This chapter is about
what drives that conversation forward: `sarva.agent.loop.AgentLoop`,
the one loop every skin (CLI, server, desktop) runs underneath —
`sarva chat`, `sarva run`, and the WebSocket server all construct the
same `AgentLoop` with different tool lists and a different confirm
policy, never their own copy of the control flow.

## An explicit state machine, not implicit control flow

`AgentLoop.run()` is a state machine with its states and legal
transitions written down as data (`sarva.agent.events.LEGAL`), not
scattered across `if`/`elif` branches:

```
INIT -> CALLING_MODEL
CALLING_MODEL -> RUNNING_TOOLS | DONE | FAILED | INTERRUPTED | BUDGET_EXCEEDED
RUNNING_TOOLS -> AWAITING_CONFIRMATION | CALLING_MODEL | FAILED | INTERRUPTED | BUDGET_EXCEEDED
AWAITING_CONFIRMATION -> RUNNING_TOOLS | INTERRUPTED
```

Every transition goes through one function, `transition(to)`, which
asserts the move is actually legal before taking it. Every skin
consumes only the resulting `AgentEvent` stream (`StateChangedEvent`,
`ModelStreamEvent`, `ToolStartedEvent`/`ToolFinishedEvent`,
`NeedsConfirmationEvent`, `RunDoneEvent`) — none of them reach into
loop internals, and the transcript is the same append-only JSONL file
(`run_dir/transcript.jsonl`) regardless of which skin is driving.

The loop is the plan → act → verify cycle made literal: `CALLING_MODEL`
is the model deciding what to do next (plan), `RUNNING_TOOLS` is acting
on that decision, and looping back to `CALLING_MODEL` afterward — with
the tool results appended to the conversation — is how the model
verifies what just happened and decides whether it's actually done.

### Every run's transcript directory used to live forever — a real resource leak in any long-running `sarva serve`

A round-43 sweep, looking specifically for "resources that accumulate
across many calls in a long-running process" rather than blow up on a
single call, found that `run_dir/transcript.jsonl` above was written
into a fresh `run_root/<run_id>/` directory on **every** `AgentLoop.run()`
call, and nothing anywhere ever deleted an old one. Confirmed live:
running 25 ordinary turns against a real router the same way
`sarva.server.app`'s `/chat`/`/ws/chat` handlers do (a fresh `AgentLoop`
per request, discarded after) left exactly 25 directories on disk —
unbounded growth over the lifetime of a long-running server process,
with each directory retaining the complete event stream for that turn
(full message text, tool-call arguments, tool-result content)
indefinitely, with no user-visible retention policy at all. Reachable
through completely ordinary use, no adversarial input needed — every
`sarva run`/`sarva chat` invocation and every request against a busy
`sarva serve` deployment adds one more directory that lives forever.

Fixed with `_prune_old_runs`, called once per `run()` right after
creating that run's own directory: keeps the most recent
`_MAX_RETAINED_RUNS` (200) directories by mtime and deletes the rest —
some retention window is the intended behavior (this file's own opening
paragraph calls a run "inspectable"), just not an unbounded one.
Verified by reverting and watching the new test fail with
`AttributeError: <module 'sarva.agent.loop'> has no attribute
'_MAX_RETAINED_RUNS'` — the test monkeypatches that constant down to 3
so it doesn't need to actually run 200+ turns to prove pruning happens.
All 30 pre-existing agent-loop tests pass unchanged.

### Much later, that same pruning logic deleted a still-running subagent's directory out from under it

Several rounds later, once subagent fan-out existed (see the chapter
below), giving `_prune_old_runs` its own fresh-eyes sweep — the same
lens that had just found a budget-starvation regression in the
concurrent `delegate_task` fix immediately above this one in the
project's own history — surfaced a genuinely separate bug with the
same root shape: concurrent subagents spawned from one parent share a
single `run_root/subagents/` directory (`spawn_subagent`'s own
`sub_run_root`, below), and each subagent's own `run()` independently
called `_prune_old_runs` on that *shared* directory right after
creating its own run_dir — purely by mtime, with zero concept of
"still running." Confirmed live with `_MAX_RETAINED_RUNS` lowered
(matching this file's own retention-cap test above, so this doesn't
need 200+ real subagents to trigger): a slower sibling's still-in-flight
run_dir got deleted by a faster sibling's own prune call moments later
— worse than a clean error, the slower subagent then crashed with a
raw, uncaught `FileNotFoundError` trying to append to a transcript file
whose parent directory no longer existed.

Fixed with a `.active` marker file: `AgentLoop.run()` is now a thin
wrapper around the actual generator body (renamed `_run_impl`) that
creates the marker in the run's own directory before `_run_impl` ever
starts and removes it in a `finally` block however the run ends —
success, an exception, or the caller closing the generator early.
`_prune_old_runs` now excludes any directory carrying that marker from
its removal candidates, regardless of level (a top-level run or a
nested subagent) or how many concurrent siblings share the directory —
while `keep` still bounds the *total* directory count the same way it
always did, so the retention window's steady-state size is unchanged.
**A real off-by-one was caught by the project's own pre-existing
retention-cap test immediately after the first version of this fix**:
comparing `keep` against only the *non-active* count made the
currently-being-created directory (always active at that exact instant,
since the wrapper marks it before calling in) invisible to its own
retention math, silently growing steady-state disk usage to `keep + 1`
instead of `keep` — caught by that pre-existing test going from 3 to 4,
fixed by keeping `keep` compared against the total count while still
only ever *removing* from the non-active pool. Verified live the fix
turns the crash into two subagents completing cleanly under a slow/fast
timing split designed to trigger exactly this interleaving. Verified by
reverting and watching the new test fail with the literal old bug's own
`FileNotFoundError` text reproducing itself in the assertion failure.
2 new tests, 696 total, all pre-existing tests (including the
retention-cap test this fix's own off-by-one nearly broke) pass
unchanged.

## Tool use: concurrent, typed, gated by one policy

A `Tool` is a small, explicit contract:

```python
class Tool(Protocol):
    spec: ToolSpec  # name, description, JSON Schema, and a `destructive: bool` flag
    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock: ...
```

`BUILTIN_TOOLS` ships ten: `ReadFileTool`, `WriteFileTool`,
`EditFileTool` (a targeted find-and-replace edit, distinct from
`WriteFileTool`'s always-rewrite-the-whole-file contract — see the
design doc's own §3.5 "file read/write/edit" line, the last of that
trio to get built), `RunShellTool`, `WebFetchTool`, `RememberTool`,
`RecallMemoryTool` (session-scoped semantic recall), `NoteTool`,
`SearchNotesTool` (durable, cross-session markdown notes — see the
memory chapter for all four), and `DelegateTool` (subagent fan-out —
see below). MCP-backed tools (see the MCP chapter) implement the exact
same `Tool` protocol, which is why the loop never needs to know or
care whether a given tool call is local Python or a round trip to a
subprocess speaking MCP.

`EditFileTool` mirrors a well-proven, simple contract — the same one
this project's own coding assistant tool uses to edit its own source
throughout this codebase's development: `old_string` must match
*exactly once* in the file (or the tool refuses, asking for more
surrounding context) unless `replace_all=true` is passed. Confirmed
live that this is genuinely a targeted edit, not a full rewrite in
disguise: replacing one line in the middle of a 1000-line file leaves
every other line provably untouched. Same atomic-write guarantee
`WriteFileTool` already has (a crash mid-commit leaves the file at its
last good, complete content, never truncated) — verified the identical
way, by making `os.replace()` raise mid-edit.

### A real bug found by giving this newly-shipped tool its own fresh-eyes sweep: CRLF files got their WHOLE line-ending style silently rewritten

A later round applied this project's own "give freshly-shipped code a
dedicated bug-hunting pass" pattern to `EditFileTool`, and found a real
one: `Path.read_text()` does universal-newlines translation on read
(`\r\n` and `\r` both silently become `\n`), and nothing on the write
side translates back. Confirmed live: editing one line of a real,
genuine CRLF file (`line1\r\nline2\r\nline3\r\n`) changed every OTHER
line's ending to LF too — `line1\nLINE2\nline3\n` — directly
contradicting this tool's own "without rewriting the rest of the file"
contract from the very paragraph above. A completely ordinary, non-
adversarial file (any Windows-authored file, anything a real
`.gitattributes` `text eol=crlf` rule enforces) triggers this on the
very first edit, producing a whole-file git diff for what should have
been a one-line change, and potentially breaking `.bat`/`.cmd`/CI
tooling that requires CRLF.

Fixed by reading raw bytes and decoding directly
(`p.read_bytes().decode("utf-8")`) instead of `p.read_text()` —
`bytes.decode()` does no newline translation at all, so every
untouched line's original ending survives byte-for-byte; only the
write side's own `content.encode()` (already used by `atomic_write_text`)
re-encodes whatever the resulting string actually contains. Verified
live that the identical repro now produces
`line1\r\nLINE2\r\nline3\r\n` — only the edited line's content
changed, every CRLF preserved exactly.

### `replace_all` was never actually validated as a boolean -- a stringified `"false"` did the exact opposite of what was asked

A much later fresh-eyes sweep, applying the same lens round 86 had
just used on `RecallMemoryTool`'s `top_k`: `spec.input_schema` above
declares `replace_all` as `{"type": "boolean"}`, but nothing enforces
that against a real model's actual tool-call arguments --
`input_schema` is purely descriptive, sent to the provider so the
model knows the expected shape, never validated server-side anywhere
in the dispatch path. `not replace_all` treated the raw argument as a
plain Python truth value, and any non-empty string is truthy in
Python -- confirmed live: a model emitting the JSON *string* `"false"`
for `replace_all` (a real, plausible model mistake, not a contrived
attack) was treated as `replace_all=True`, the exact opposite of what
was asked. On a real file with three occurrences of `old_string`, this
silently rewrote all three instead of erroring on the ambiguity this
tool's own docstring says it exists to prevent -- a destructive tool
(`spec.destructive=True`) doing the wrong, unrequested thing with no
error and no signal, not a crash. Fixed by rejecting anything that
isn't a genuine Python `bool` with a clear, actionable error, the same
"reject, don't guess" treatment `RecallMemoryTool`'s `top_k` already
got for the identical gap. Verified live across three cases together:
the string `"false"` case (now rejected, file unchanged), a real
`replace_all=True` (still works exactly as before), and the omitted-
argument default (the pre-existing ambiguous-match error, unaffected).
Verified by reverting and watching the new test fail with the literal
old bug reproducing itself -- all three occurrences silently replaced.
1 new test, 743 -> 744 total.

When a model turn ends in `TOOL_USE`, every requested call runs
concurrently via `asyncio.gather` — not sequentially, and not with the
model waiting on one before deciding about the next. **Whether a tool
runs at all is a policy decision, never the tool's own to make**: a
tool declares `spec.destructive`; the loop — not the tool — decides
whether that triggers `AWAITING_CONFIRMATION` and a call out to the
caller's `ConfirmPolicy`. This is what makes `sarva run --auto`
(`always_allow`, never asks) and `sarva run`'s default (a real
`typer.confirm()` prompt per destructive call) the same loop with a
one-argument policy swap, not two different code paths to keep in sync.

A tool that raises never crashes the loop — the exception becomes an
`is_error=True` `ToolResultBlock` the model sees on its next turn, the
same as any other tool failure it needs to react to. An unrecognized
tool name gets the identical treatment rather than a hard stop.

**`WriteFileTool` writes atomically, not via a direct `Path.write_text`.**
A real bug found by actually simulating an interrupted write: this tool
runs on essentially every agent file-editing turn, against arbitrary
real user files — not just this project's own state. `write_text`'s
default mode truncates the target to 0 bytes the instant it's opened,
before a single byte of new content lands; a crash between that moment
and the write completing (an OOM-kill, a `SIGKILL`, a real power loss)
destroys whatever was there before, confirmed live by writing a real
5000-byte file and simulating that exact crash moment. Fixed via the
same shared `sarva.atomic_write` helper `sarva.config`/`sarva.memory.
session` already use — see the memory chapter for the fuller history of
this bug class and where else it was found and closed.

**A real bug found in `RunShellTool`'s own timeout, not just an
uncaught exception:** `asyncio.wait_for(proc.communicate(),
timeout=...)` only cancels the *awaiting* coroutine on expiry — it
never touches the child process itself. Confirmed directly: a shell
command with a `sleep`-then-side-effect tail, run against a shortened
timeout, left both the shell and the sleeping process alive (and the
trailing side effect completed) seconds after the reported "timeout."
This is specifically dangerous for `RunShellTool` because it's the one
built-in tool marked `destructive=True` — the whole confirmation gate
exists to stop unwanted side effects, and a timeout was silently
defeating that guarantee by leaving the command running unattended
regardless of what was approved. Fixed by catching the `TimeoutError`,
calling `proc.kill()` then `await proc.wait()` before returning a real
`is_error=True` result naming the timeout — never a blank message the
way a bare `TimeoutError`'s own `str()` would produce. Verified live: a
real timed-out command's side-effect file genuinely never appears,
confirmed with the process both killed and reaped, not just reported
as failed.

### `RunShellTool` buffered its ENTIRE output with no size limit at all — and the fix for that itself introduced a real deadlock, caught before shipping

A much later fresh-eyes sweep of this same tool, one layer beyond the
timeout fix above, found `proc.communicate()` had no output-size limit
whatsoever, unlike `WebFetchTool`'s own `_MAX_FETCH_CHARS` cap sitting
right next to it in the same module. Confirmed live: a completely
ordinary command (`yes A | head -c 300MB` — the same shape as `git log
-p`, a verbose build, or `cat`-ing a large data file, not a contrived
attack) drove real process peak RSS from ~45MB to ~1GB, and the full,
untruncated 300MB string was then appended into `messages`, resent to
the provider on every later turn of the same run — a real memory- and
context/cost-blowup risk in a long-running `sarva serve` process, the
same "materialize everything before use" DoS shape already fixed for
the video-frame degrader and local STT, never applied here.

Fixed by reading `proc.stdout` in bounded chunks and stopping the READ
itself once a cap is hit (`_MAX_SHELL_OUTPUT_BYTES`), rather than
capturing everything and truncating the resulting string afterward —
the latter would still incur the full memory spike before throwing the
excess away, the actual harm this fix exists to prevent.

**Verifying that fix, before it ever shipped, surfaced a second, real
bug in the fix itself:** killing the process once the size cap fires
(the same "don't leave an unwanted side effect running unattended"
reasoning as the timeout fix above) hung indefinitely on a real shell
*pipeline*. `proc.kill()` only sends `SIGKILL` to the shell
interpreter `asyncio` tracks — not to the actual commands a pipeline
forks (`yes`/`head` here, connected by their own internal pipe).
Confirmed live: after stopping an early read and killing just the
shell, the pipeline's real commands stayed alive and orphaned, still
blocked writing into a pipe nothing was draining anymore — `await
proc.wait()` then hung forever, since the shell itself died instantly
but its still-running children kept the underlying pipe's write end
open. Fixed by spawning the shell with `start_new_session=True` (its
own process group) and killing via `os.killpg` instead of a plain
`proc.kill()` — reaching every process in the pipeline, not just the
one PID this project happens to track — applied to both the size-cap
kill and the pre-existing timeout kill above, since both shared the
identical latent gap once a pipeline is involved. Verified live both
fixes hold together: a 300MB pipeline command now completes in ~0.01s
with peak RSS barely above baseline, correctly truncated and
`is_error=True`; ordinary commands (including real pipelines that
finish on their own) are unaffected; the timeout path still kills
cleanly. Verified by reverting and watching both new tests fail
cleanly (the size-cap constant simply doesn't exist pre-fix — this is
a genuinely new mechanism, not a modified existing one, the same
"revert-and-verify still applies, just to 'does this exist at all'
instead of 'does the old bug reproduce'" shape prior genuinely-new
mechanisms in this project have used). 2 new tests, 720 → 722 Python
tests.

### Two destructive calls sharing the same `tool_call_id` could let a declined one run anyway — the confirm-gate's own key, not a tool bug

**Worse than any crash in this file: a declined destructive call
silently executing anyway, defeating the exact guarantee the
confirmation gate exists for.** Nothing validates that every
`ToolCallBlock` in one model turn has a unique `.id` — a real model is
expected to keep them unique, but a malformed or adversarial response
isn't structurally prevented from repeating one. The confirmation loop
used to track each call's decision in `approvals: dict[str, bool]`,
keyed by `call.id` — the model-supplied string. Two distinct
`ToolCallBlock`s sharing that string meant the second confirmation
answer silently overwrote the first entry, so *both* calls read back
whichever decision was made last.

**Confirmed live, not hypothetical:** two destructive tools, both given
`id="dup"`, with a confirm policy that explicitly denies the first and
approves the second — the first tool ran anyway. The user's own
"no" was silently discarded because a *different* call happened to
share its id and get approved afterward.

Fixed by keying `approvals` on `id(call)` — Python's own object
identity — instead of the model-supplied `.id` string. Every
`ToolCallBlock` in `calls`/`destructive_calls` is the exact same Python
object referenced both when its confirmation is recorded and when
`run_one` later looks it up (list comprehensions preserve identity),
so `id(call)` is a real per-call key that can never collide the way an
arbitrary string can — no validation, rejection, or deduplication of
malformed ids was needed to close this, just tracking each call by
what it actually *is* rather than by a field it merely carries.
`ToolResultBlock.tool_call_id` (the value sent back to the model on the
wire) is completely unaffected — it still faithfully echoes `call.id`
exactly as before; only the loop's own internal confirm-bookkeeping
changed.

**Verified the new test is real:** reverted the fix and watched it fail
— the reverted run had no `tool_finished` result containing "declined"
at all, because the supposedly-denied call actually executed, the
exact bug this closes — before re-applying. All 27 pre-existing
agent-loop tests pass unchanged.

### A hung tool call used to block the whole turn forever — including every other, already-completed tool's result

The concurrent-dispatch design above (`asyncio.gather` over every call
in one turn) has a real consequence that wasn't guarded against until
now: `gather` withholds **every** result until **every** coroutine
finishes. Confirmed live: a turn with one fast tool (returns instantly)
and one that never returns at all left the fast tool's own
`ToolFinishedEvent` undelivered indefinitely, even though it completed
in microseconds — an outer 5-second guard around the whole run never
unblocked. `RunShellTool` self-protects with its own internal 60-second
timeout (see above), but that's the *only* built-in tool that does;
`McpToolAdapter.run()` (a remote or stdio MCP server's `call_tool`,
`sarva.mcp_client`) has none at all, so any hung or unresponsive MCP
server had zero recovery path — not a clean failure, not an error
event, just a run that never finishes.

Fixed by wrapping each tool call's `tool.run(...)` in `asyncio.wait_for`
with a 90-second backstop timeout — deliberately longer than
`RunShellTool`'s own 60 seconds, so a tool with its own internal
timeout gets to fire first and report its own specific reason; this is
purely a backstop for tools that don't self-protect, not a replacement
for `RunShellTool`'s fix. A timeout is scored exactly the way a raised
exception already is: a real, visible `is_error=True` `ToolResultBlock`
the model sees and can react to, not a special case. **Honestly scoped,
not a guaranteed cure for every underlying resource:** cancelling the
`await` stops *this loop* from waiting on it, the same way it already
did for `RunShellTool`'s own pre-fix bug — whether the real resource
behind a specific tool (an MCP server's network connection, say)
actually tears down promptly depends on that tool's own cancellation
handling, not on this timeout alone. What this closes, unconditionally,
is the symptom that actually mattered: the turn itself always reaches a
terminal state instead of hanging forever, and no other concurrent
tool's already-completed result is held hostage by one that never
finishes.

Verified the new test is real: reverted the fix and watched it fail
with `AttributeError: <module 'sarva.agent.loop'> has no attribute
'_TOOL_TIMEOUT_SECONDS'` before re-applying — the test monkeypatches
that constant down to make the check fast, so its absence is itself
proof the fix wasn't there yet. All 28 pre-existing agent-loop tests
pass unchanged.

### `WebFetchTool` and a real SSRF gap it had, found and closed

`WebFetchTool` is marked `destructive=False` — deliberately, since
fetching a URL changes no state — which means it runs with **zero
confirmation**, even in the CLI's default (non-`--auto`) mode. That
made a real, not hypothetical, SSRF (server-side request forgery) gap:
confirmed directly against a real local Ollama server running in this
environment, `web_fetch` on `http://127.0.0.1:11434/api/tags`
succeeded and returned the response straight into the model's own
context — the same shape of request would reach a cloud metadata
endpoint (`http://169.254.169.254/...`, a classic SSRF target for
exfiltrating cloud credentials) or any other internal service with
identical ease.

Closed with the standard mitigation: before every fetch, the target
hostname is resolved and every returned IP is checked against
`ipaddress`'s `is_global` (covers RFC 1918 private ranges, loopback,
link-local — which includes the metadata address — and other reserved
ranges, for both IPv4 and IPv6, in one check). `follow_redirects=True`
was replaced with a bounded manual redirect loop that re-validates the
target host on **every** hop, not just the caller-supplied URL — a
legitimate public site's own server issuing a redirect straight to an
internal address is exactly the bypass a validate-once-up-front check
would miss. Verified against real addresses (a real running local
Ollama server, the real cloud-metadata IP, a real private-range IP,
and a simulated redirect to an internal address) and against real
public traffic (a real `https://example.com` fetch, and a real
`http://github.com` → `https://github.com/` redirect chain) — both
still work exactly as before.

**The guard itself now lives in `sarva.multimodal.fetch`
(`ensure_public_host`), not duplicated here** — `resolve_media_bytes()`
(the multimodal chapter) is the *other* real url-fetching path in this
codebase, and it shares the identical function so the SSRF guard can
never drift out of sync between the two.

### The SSRF guard above was itself TOCTOU-vulnerable to DNS rebinding — a real bypass, no redirect needed

**`ensure_public_host` alone was never actually the security boundary
it looked like.** It resolves a URL's hostname once, checks every
returned address is public, then discards that answer — but the real
`httpx` connection made a moment later re-resolves the *same* hostname
completely independently, by default. A DNS server an attacker
controls can answer the check's query with a genuine public address
and every later query with `127.0.0.1` (or any internal address): the
check passes, and the real request lands somewhere else entirely — the
classic DNS-rebinding bypass, and it needs no cooperating redirect at
all, unlike the gap the per-hop revalidation above already closes.

**Confirmed live, not reasoned about in the abstract:** a real local
HTTP server standing in for an internal target, and a fake resolver
answering the first `getaddrinfo` call publicly and every later one
with `127.0.0.1` — `web_fetch`'s own SSRF check passed, and the real
request still reached the local server and returned its response, with
zero user confirmation (`WebFetchTool` is `destructive=False`). The
identical gap exists in `resolve_media_bytes`'s own `fetch_bytes` path.

**Fixed by moving the actual enforcement down to where httpx opens a
real socket, not where a caller happens to think to check first:** a
custom `httpcore` network backend (`_PinnedResolutionBackend`,
`sarva.multimodal.fetch`) resolves and validates a hostname exactly
once *per real TCP connection*, then hands the underlying connector the
literal validated IP address instead of the hostname — so the DNS
answer that was checked and the address that gets connected to are
always, structurally, the same lookup, never two separate ones an
attacker's resolver can answer differently. `ensure_public_host` stays
in place too, as a fast, clearly-worded early rejection for the
ordinary (non-adversarial) case and every redirect hop — but it's no
longer the thing actually standing between a request and an internal
service; `ssrf_safe_transport()`'s pinned-resolution backend is. TLS
SNI and certificate-hostname verification are completely untouched by
this change, since only the raw socket target changes — the request
URL itself, and therefore what the TLS layer verifies against, is
unmodified.

Verified the new test is real: reverted the fix and watched it fail
with the internal server's own response body (`b"internal secret"`)
showing up in the tool's result — exactly the leak this closes — before
re-applying. `httpcore` (already an `httpx` transitive dependency)
became an explicit `core` dependency, since production code now imports
it directly rather than relying on it resolving implicitly through
`httpx`'s own requirement.

### `WebFetchTool` also had the same unbounded-buffering gap `RunShellTool` did — deliberately left for a later round, then closed with a cleaner test isolation

`RunShellTool`'s stdout-buffering fix (see the tool-use chapter's own
entry) noted `WebFetchTool` had the identical shallower gap:
`client.get(url)` fully downloads the entire response body into memory
before `.text` is ever sliced to `_MAX_FETCH_CHARS` — harmless for the
ordinary small page this tool usually fetches, but a large-but-
plausible response (a big JSON API response, an uncompressed log file,
a large HTML page, no adversarial intent needed) incurs the full
download's memory cost before any of it is thrown away. Deliberately
deferred at the time because testing it cleanly seemed to require
defeating the SSRF guard (`ensure_public_host`/`ssrf_safe_transport`)
in a way that would itself be testing an attack the guard exists to
block, not the download-bounding logic.

Closed one round later with a cleaner isolation: a no-op stand-in for
`ensure_public_host` plus a real `httpx.MockTransport` streaming a
large body tests the bounding fix directly, without needing to defeat
(or even exercise) the SSRF guard at all — that guard already has its
own thorough, separate live coverage elsewhere in this file. Fixed by
switching from `client.get(url)` to `client.stream("GET", url)` and
reading `resp.aiter_text()` in a loop, stopping the READ itself once
`_MAX_FETCH_CHARS` is exceeded — the same "stop reading, don't
read-then-discard" shape as the `RunShellTool` fix, not a
coincidentally similar one. Verified live the read genuinely stops
early, not just that the final string ends up truncated: a mock server
offering 2000 chunks (~125MiB) only ever had 1 consumed before the cap
fired. Verified by reverting and watching the new test fail with the
literal old bug's own number — all 2000 of 2000 chunks consumed —
reproducing itself. Fixing this also required updating one pre-existing
test (`test_ensure_public_host_rejects_a_redirect_to_an_internal_
address`) that had monkeypatched `httpx.AsyncClient.get` directly — no
longer called at all now that this tool uses `.stream()` — to patch the
shared lower-level `AsyncClient.send()` both entry points route
through instead, a small but real ripple from changing which httpx API
this tool actually calls. 1 new test, 722 → 723 Python tests.

## Budgets: exceeding one is a clean stop, not an exception

```python
class Budget(BaseModel):
    max_model_calls: int = 50
    max_total_tokens: int = 2_000_000
    max_wall_seconds: float = 3600.0
    max_cost_usd: float = 10.0
```

`Spend.exceeded(budget)` is checked after every model turn; the first
budget that's actually crossed lands the loop in `BUDGET_EXCEEDED` —
a normal terminal state with a full `Spend` summary attached to
`RunDoneEvent`, not a raised exception a caller has to catch. A
runaway agent stops itself cleanly, with a receipt of exactly how much
it spent before stopping.

## Multimodal-aware routing, and degradation as an opt-in fallback

Before the first model call, the loop scans every message for the
modalities actually present (an image attached alongside text, say)
and asks the router for a model that supports all of them —
`needs=_required_modalities(messages)`. If no available model
qualifies, that's normally a hard `FAILED` — unless the loop was
constructed with `degraders` (see the multimodal degraders described
in the memory/eval chapters' sibling material): with degraders
supplied, the failure becomes *recoverable* — fall back to the best
available text-capable model, and degrade the unsupported content
(video → sampled frames → text, say) into something that model can
actually see, rather than refusing outright. Deliberately opt-in, not
a silent default: nobody gets a lower-fidelity response than they
explicitly asked for without having asked for exactly that tradeoff.

**A real bug found in the fallback's own exception handling, not just
an uncaught error somewhere unrelated:** the degradation attempt was
wrapped in `except (LookupError, UnsupportedModalityError)` — the only
two failure modes anyone had anticipated — but a concrete `Degrader`
can raise its own decode error when it genuinely can't make sense of
the content it was handed. `ImageToTextDegrader.degrade()` does exactly
that (`ImageDecodeError`, its own documented exception type), and
neither exception type above covers it, so it propagated straight out
of `AgentLoop.run()`'s async generator uncaught — a genuinely undecodable
or truncated image attached with no vision-capable model configured
crashed the whole run instead of landing in the clean `FAILED` state
this fallback exists to produce. Fixed by widening that `except` clause
to catch any `Exception` — deliberately broad, since this block's own
purpose is "attempt a best-effort recovery, and if it doesn't work for
*any* reason, fail cleanly," not to enumerate every exception type a
current or future `Degrader` implementation might raise. The
degradation failure's own message (e.g. "could not decode image for
degradation: ...") now reaches the caller's `state_changed.detail` too
— a more specific, actionable reason than the original "no model
supports this modality" `LookupError` it used to fall back to silently
dropping. A second, sibling bug closed in the same pass: a genuinely
*truncated* (not fully unrecognizable) image made Pillow raise a plain
`OSError` reading `.size` rather than `UnidentifiedImageError` —
`ImageToTextDegrader`'s own `except` clause only caught the latter, so
a truncated image reached this fallback as a raw, uncontextualized PIL
error instead of `ImageDecodeError`; fixed by widening that degrader's
own `except` to `(UnidentifiedImageError, OSError)`.

**A third, lower-severity sibling found the same way in a later pass:**
`PIL.Image.DecompressionBombError` (a tiny, hand-crafted PNG declaring
a huge width/height in its header — no real pixel data needed, since
Pillow's check fires from the declared dimensions alone) is a plain
`Exception` subclass, not `OSError`/`UnidentifiedImageError`, so it
reached `ImageToTextDegrader`'s caller unwrapped too. Already caught
one layer up by this loop's own broadened `except Exception` above, so
this was never a crash reaching a real skin — but a direct caller of
the degrader outside that wrapper (a test, a future library user)
still got a raw PIL exception instead of the documented
`ImageDecodeError`. `except` widened once more, to
`(UnidentifiedImageError, OSError, Image.DecompressionBombError)`.

**A fourth bug, upstream of all three above and far more severe: the
fallback never triggered at all for the single most common real case.**
`Router.pick()` (`sarva.providers.registry`) returns the first routing-
chain candidate that both supports the needed modalities and is
`available` — and the `mock` model's own registry entry
(`models.yaml`) had declared `image` (and `document`) support since
this project's very first scaffold commit. `MockProvider` is always
`available` (`build_router()` seeds `available = {"mock"}`
unconditionally) and sits last in every routing chain by design — so
for the ordinary, explicitly-marketed "local Ollama only, no cloud
key" setup, `pick(TaskClass.MAIN, needs={TEXT, IMAGE})` resolved
straight to `mock` instead of ever raising the `LookupError` this
entire degradation mechanism exists to catch. `MockProvider.generate()`
doesn't actually look at images at all — it just echoes text — so the
real image was silently discarded with no signal anywhere: `sarva chat
"describe this" --image photo.png` completed with a clean-looking
`state=done` and text like `"[mock] received: describe this"`, as if
the request had simply been answered. Confirmed live before fixing,
using the real shipped `models.yaml`/`routing.yaml`, not a synthetic
test double. Fixed by removing `image`/`document` from `mock`'s
`modalities_in` — `document` sits in the same reachable bucket even
though no CLI flag constructs one yet (a `DocumentToTextDegrader`
already exists), so it's closed proactively rather than waiting for
that flag to ship and reproduce the same bug. `audio`/`video` stay:
routing.yaml's `audio` chain is `[mock]` alone with no real model ahead
of it to preempt, so mock resolving audio isn't defeating a better
available path — it's the last-resort, zero-config guarantee the
routing file's own header comment promises, working as intended. Two
existing tests had encoded the buggy behavior as the expectation
(`mock's own capabilities include image` was their literal premise)
and needed rewriting to use a genuinely vision-capable test model
instead of relying on mock's now-corrected claim.

## Failure handling, named explicitly rather than left implicit

- A provider crash (any exception escaping `provider.generate()`, not
  just a well-formed `StreamErrorEvent`) is caught at the loop level
  and turned into `FAILED` — it never propagates up into a skin.
- A `StreamErrorEvent` marked `retryable` gets a fixed 1-second backoff
  and retries, up to `_MAX_STREAM_RETRIES` (5) consecutive attempts —
  a non-retryable one is an immediate `FAILED`, and exceeding the retry
  cap is too, with a `detail` naming the count and the underlying error.

**A real bug found by actually running a retryable stream error, not
just reading the retry code:** the retry path above loops back to the
top of the loop's own `while True:` without the state machine ever
leaving `CALLING_MODEL` — and then immediately re-asserts the exact
same transition, `CALLING_MODEL -> CALLING_MODEL`, which `LEGAL`
(`sarva.agent.events`) didn't allow. The very first retryable error on
a real run raised an uncaught `AssertionError: illegal transition
calling_model -> calling_model`, confirmed live with a scripted
`MockProvider` — not a hypothetical: every real adapter sets
`retryable=True` for exactly the cases this mechanism exists to handle
(`RateLimitError`, any 5xx `APIStatusError`), so a single transient rate
limit or server error crashed the entire run instead of being retried,
completely defeating the retry mechanism for every real provider.
Nothing above `AgentLoop.run()` catches a bare `AssertionError` either
— it reached a CLI user as a raw Python traceback, and a `/ws/chat`
client as the same bare `ClosedResourceError` this file's three prior
raw-JSON fixes exist to prevent, just via a trigger none of them cover.
Fixed by adding `CALLING_MODEL` to its own legal-transition set: a
retry genuinely is a legitimate self-transition here, unlike every
other state in that table, which never legitimately re-enters itself.
**Verified the new test is real:** reverted the fix and watched it fail
with the exact same `AssertionError` before re-applying. All 26
pre-existing agent-loop tests pass unchanged.
- `MAX_TOKENS` and `REFUSAL` stop reasons are both `FAILED`, with the
  specific reason recorded in the terminal event's `detail` field —
  distinguishable after the fact, not collapsed into one generic
  failure state.

**A permanently-retryable stream error used to spin forever, ignoring
the caller's own `Budget` entirely — a real, previously-deferred gap
closed here.** The retry path (`if pevent.retryable: ...; break`) falls
through to `if done is None: continue`, jumping back to the top of the
main loop *before* the `spend.exceeded(self._budget)` check a few lines
below ever runs — every other path through the loop is bounded by that
check; this was the one exception. Confirmed live: a `MockProvider`
scripted to always yield a retryable error, driven through `AgentLoop`
with a real `Budget(max_model_calls=50, max_wall_seconds=3600.0)`,
never reached a terminal state after 6 real seconds and 12 real
provider-call attempts — one real API call every second, indefinitely,
regardless of how tight the configured budget was. Previously named in
this chapter as a genuine limitation rather than fixed outright,
because it read like it might be entangled with a bigger "how should
retry policy work" design question; on reinspection it wasn't — the
backoff policy itself (flat 1s) was already settled, all that was
missing was a cap. Fixed with a small, independent `stream_retries`
counter (deliberately not reusing `spend.model_calls`, since a retry
isn't a new model call — the surrounding code already establishes
that) capped at `_MAX_STREAM_RETRIES = 5`, reset to 0 the moment a real
`done` response arrives. Exceeding the cap transitions to `FAILED` the
same way a non-retryable error already does. **Verified the new test is
real, and by an unusually strong margin:** reverting the fix and
re-running the new test didn't just fail an assertion — the reverted
code has no true suspension point in its retry path once `asyncio.sleep`
is mocked to a no-op for the test, so it starves the event loop's own
cooperative cancellation and the test process had to be killed
externally rather than timing out cleanly, about as unambiguous a
confirmation as this project's revert-and-verify discipline has
produced. All 29 pre-existing agent-loop tests pass unchanged.

## Subagent fan-out: `delegate_task`, one level deep

The design doc's own architecture section names "subagent fan-out" and
"verifier subagent" patterns alongside the loop this chapter describes.
This chapter used to say **neither exists in code** — true for a long
time, closed now for the first of the two: `delegate_task`
(`sarva.agent.subagents.DelegateTool`) lets the model spawn a fresh,
independent `AgentLoop` for a self-contained subtask and get back its
final answer as a plain-text tool result.

The spawning logic itself lives in `AgentLoop.run()`, not in the tool —
that's the one place with a router, providers, budget, and live spend
to build a subagent from. `ToolContext` gained one new, optional field
for this: `spawn_subagent`, a closure built fresh by every `run()` call
and left `None` in any context that doesn't support delegation (a bare
`ToolContext` built directly, e.g. in a test). `DelegateTool.run()`
just calls it — no other tool's surface changed at all.

Three deliberate scoping decisions, matching this project's "narrow
first real slice, not the full design" pattern elsewhere:

- **One level of fan-out only.** The spawn closure filters `DelegateTool`
  itself out of the subagent's own tool list, so a delegated task cannot
  itself delegate further. Verified directly: script a subagent to try
  calling `delegate_task` anyway, and its own loop dispatches it as an
  ordinary `unknown tool: delegate_task` error (the same path any
  unrecognized tool name takes) and keeps going — not a crash, not
  actual recursion.
- **Subagent spend counts against the parent's own remaining budget, not
  a fresh independent one.** The subagent is built with
  `Budget(max_model_calls=<parent's remaining>, ...)`, and its own final
  `Spend` is added back into the parent's live `Spend` once it
  completes — so the very next budget check in the parent loop reflects
  the subagent's real cost. Without this, a model could spawn subagents
  to bypass its own budget entirely. Verified with a genuinely tight
  shared budget: the subagent burns through what's left and never
  reaches a real answer, `delegate_task` reports a clean tool error (not
  a crash), and the PARENT's own run correctly ends in
  `BUDGET_EXCEEDED` too — proving the merge is real, not cosmetic.
- **Routed via `TaskClass.SUBTASK`** — cheap delegated work, distinct
  from the parent's own `TaskClass.MAIN`. This is the first real use of
  that enum entry anywhere in the codebase; `routing.yaml` had a
  `subtask:` chain configured since the very first routing policy, but
  nothing had ever actually requested it before this.

**Confirmed end to end with `MockProvider`, not just unit-tested in
isolation:** a full three-call exchange (parent delegates → a real
subagent turn runs → parent uses the subagent's answer to finish) with
the subagent's own model-call cost showing up in the parent's final
`Spend.model_calls` count. Verified by reverting and watching the
whole test file fail to even import (`ModuleNotFoundError: No module
named 'sarva.agent.subagents'`) — the strongest possible confirmation
that this is genuinely new code, not a config flip. 5 new tests, all
pre-existing agent-loop tests pass unchanged.

The design doc's second named pattern, "verifier subagent," is closed
further down this chapter, once the interface correction below is
covered first.

### A real bug found much later: two CONCURRENT `delegate_task` calls in one round could let real spend double the declared `Budget`

A much later round applied the exact lens that had just found `verify=
True` silently exceeding `Budget` (see further down this chapter) to
`delegate_task`'s own budget-merge path, one level deeper: what happens
when a single model turn issues *multiple* `delegate_task` calls at
once? Every tool call in a round already runs concurrently via
`asyncio.gather` (see "Tool use" above) — including `delegate_task`
calls. `spawn_subagent`'s budget clamp reads the parent's live `spend`
synchronously, before its first `await`; the real mutation back into
`spend` only ever happened once a subagent's ENTIRE run had already
finished. Two concurrently-dispatched `delegate_task` calls therefore
both read the exact same, not-yet-decremented `spend` and each got
granted a full, independent slice of the same remaining budget — the
"subagent's own budget slice cannot exceed the parent's remainder"
invariant, verified above for the *sequential* case, was silently
defeated the moment two delegations happened in the same round.
Confirmed live: `Budget(max_model_calls=3)` with two concurrent
`delegate_task` calls made **6** real provider calls — double the
declared cap — through nothing more adversarial than "delegate these
two independent things in parallel," completely ordinary usage.

Fixed with an `asyncio.Lock` scoped to one `run()` call, guarding just
the admission decision (compute the clamp, decide the grant, and
immediately reserve the FULL granted slice against `spend`) rather than
the subagent's entire execution — the subagents themselves still run
fully concurrently once admitted, only the grant itself is serialized.
Reserving the complete granted amount up front is a correct bound, not
a guess: a subagent's own budget check already stops it from ever
exceeding what it was granted, so the reservation can never undercount
real usage. Once a subagent finishes, its reservation is reconciled
down to what it actually used, so the parent's final reported `Spend`
still reflects true cost rather than the conservative up-front grant.
Verified live that the identical repro now makes exactly 4 real calls
(3 declared plus the one standard "checked after the call" overshoot
every other path through this loop already tolerates) instead of 6 —
matching the sequential single-delegation case exactly. Reverting
reproduced the literal doubled call count in the new test's own
assertion failure (`6 <= 4` false). 1 new test, all pre-existing tests
pass unchanged.

### That very fix then starved concurrent siblings to zero — reserving the WHOLE remainder for the first admitted caller

One round later, giving the fix above its own dedicated fresh-eyes
sweep (the project's single most productive lens by this point — see
the four-in-a-row streak this chapter and the verifier chapter below
both note) surfaced a real regression in the fix itself, not a new
independent bug. `DelegateTool`'s own `input_schema` has no `budget`
field at all, so every real call to `spawn_subagent` — through
`delegate_task` or `verify=True` — always passes `budget=None`, "no
explicit request." Before the reservation fix, that just meant
"whatever's left, unreserved until the subagent finishes." After it,
the lock-guarded admission decision reserves the FULL remainder
against `spend` immediately — correct for stopping the overspend race,
but it meant the first of several concurrently-admitted delegations
claimed the *entire* remaining budget outright, unconditionally
starving every sibling dispatched in the same round to exactly zero.
Confirmed live under a realistic, generous default `Budget()`
(`max_model_calls=50`, no artificial tightness): two ordinary
concurrent `delegate_task` calls — "delegate these two independent
things in parallel," not an adversarial trick — left one succeeding
and the other failing `budget_exceeded` after only 1 of 50 calls had
actually been used, with 49 calls of real headroom sitting unused.

A tempting fix — divide by however many concurrent claims are
in-flight — doesn't actually work: the first claimant doesn't know a
second one is coming, so it greedily reserves everything before the
second claimant's request is even visible; by the time the second
checks in, the first has already exhausted the remainder. Fixed
instead by capping an *unspecified* request (`budget is None`
specifically — an explicit `Budget(...)` request is still honored
exactly, unhalved, matching the already-shipped
"tight budget honored precisely" test two sections up) to **half** of
whatever is left at grant time, not the whole thing. A lone delegation
still gets a generous share (half the remainder); each additional
concurrent one gets half of what's left after its predecessors'
reservations — a geometric split (½, ¼, ⅛, …) that approaches but
never reaches zero, so no number of concurrent delegations in one
round can starve one to nothing outright. Verified live: the identical
two-delegation repro under `Budget()` now has both succeed. Reverting
reproduced the exact old failure — `d2` failing with the literal
`budget_exceeded` text — in the new test's own assertion failure. 1
new test, all pre-existing tests (including both budget tests two
sections up) pass unchanged.

A related, narrower gap was found in the same sweep but deliberately
left unfixed: when an explicit `Budget(...)` object sets only *some*
fields, the unset fields' pydantic defaults (e.g.
`max_total_tokens=2_000_000`) are indistinguishable from a genuinely
requested value and get clamped/reserved as if truly requested. This
is currently unreachable in production — neither `DelegateTool` nor
`verify=True` ever construct an explicit `Budget` object, both always
call with `budget=None` — so it's recorded here honestly rather than
solved against a case nothing can reach yet.

### The upfront rejection gate itself only ever checked 2 of `Budget`'s 4 dimensions

A much later fresh-eyes sweep, deliberately continuing the checklist
this exact function's own fix history already established: the gate
right after `sub_budget` is computed -- `if sub_budget.max_model_calls
<= 0 or sub_budget.max_wall_seconds <= 0: return BUDGET_EXCEEDED` --
only ever checked two of `Budget`'s four dimensions.
`max_total_tokens`/`max_cost_usd` are clamped exactly the same way
just above it (and can round down to `0` via `int(remaining *
_UNSPECIFIED_SHARE)` the identical way `max_model_calls` already
does), but were never included in this upfront rejection. Confirmed
live: a parent with generous remaining `max_model_calls`/
`max_wall_seconds` headroom but a nearly-exhausted *token* budget -- an
entirely ordinary, foreseeable case (a long-running conversation
nearing its `Budget.max_total_tokens`), not a contrived one -- let a
subagent with a granted `max_total_tokens == 0` slip straight past
this gate. A full subagent `AgentLoop` was then actually constructed
and run, making one real, wasted `provider.generate()` call before its
own post-call `spend.exceeded()` check caught it -- exactly the "wasted
real call" failure mode the `max_model_calls`/`max_wall_seconds` half
of this same gate already exists to prevent, just never extended to
the other two `Budget` dimensions. A counting provider wrapper
observed 3 real calls (the parent's `delegate_task` call, the
subagent's own wasted call, the parent's retry) where a correct
upfront rejection produces only 2. Fixed by checking all four
dimensions, matching `Spend.exceeded()`'s own complete four-dimension
check exactly. Verified live: the identical repro now stops at 2 real
calls, the subagent never running at all. Verified by reverting and
watching the new test fail with the literal old bug's own number -- 3
calls instead of 2 -- reproducing itself. 1 new test, 744 â 745 total.

### A cancelled subagent left its full budget reservation stuck on the parent's spend forever

A dedicated sweep of this exact reservation/reconcile mechanism —
following this project's own standing practice of treating a recent
fix as the next fix's first suspect — found the reconciliation above
only ever ran on a *normal* `run_done`. `run_one`'s own
`_TOOL_TIMEOUT_SECONDS` (see "Tool use" above) wraps every tool call,
including whatever `delegate_task`-shaped tool called into
`spawn_subagent`, in `asyncio.wait_for`. If a subagent's own provider
call is still in flight when that timeout fires, `CancelledError` is
thrown into the `async for sub_event in sub_loop.run(...)` loop and
`run_done` never arrives — so the reconciliation code, living entirely
inside the `if sub_event.type == "run_done":` branch, simply never
runs, leaving the *full* up-front reservation stuck on the parent's
`spend` permanently. Confirmed live: a subagent cancelled mid-flight
left the parent's reported `spend.model_calls` at **12** against only
**3** real provider calls actually made — an 8-call phantom
reservation that understates headroom for every later sibling
delegation in the same run, the exact starvation shape the previous
two sections' fixes address on the *admission* side, now open on the
*release* side instead.

Fixed with a `try`/`finally` around the `async for` loop: a
`reconciled` flag tracks whether the normal `run_done` reconciliation
ran, and if not — cancellation, or any other exception propagating out
of the subagent's own run — the `finally` block releases the *entire*
reservation instead of leaving any of it stuck. This can only ever
**under**-count that one subagent's own now-unrecoverable real cost
(its own `Spend` lived inside its own abandoned `run()` call and was
never surfaced before cancellation cut it off, so there's no true
partial figure to reconcile down to) — never leave a phantom amount
stuck forever. A one-time, honest undercount on a rare cancellation
path is strictly better than a permanent, silent overcount that starves
every later delegation. Verified live: the identical repro now reports
`spend.model_calls` at **2**, tracking the real **3** provider calls
closely instead of the phantom **12**. Verified by reverting and
watching the new test fail with the literal old bug's own number
reproducing itself (`12 <= 3` false). 1 new test, 696 -> 697 Python
tests, all pre-existing tests pass unchanged.

This closes the third and, as of this sweep, last identified gap in
`spawn_subagent`'s concurrency surface — a companion sweep of the same
area confirmed the `.active`-marker creation sequence above is provably
race-free (no `await` between `mkdir` and the marker write, and asyncio's
single-threaded cooperative scheduling means nothing else can run in
that window) and that the marker's own cleanup, while not deterministic
on every abandonment path (a `/ws/chat` client disconnect mid-stream
depends on Python's cyclic garbage collector eventually finalizing the
orphaned generator, not a guaranteed `finally`), is self-healing rather
than a permanent leak — documented here rather than treated as a
finding, since fixing it would mean changing `/ws/chat`'s own
generator-consumption shape for a gap that already recovers on its own.

### `spawn_subagent`'s real shape was already frozen in spec-03 — a same-session correction, not a second milestone

The paragraphs above describe the feature as first built, against the
design doc's brief one-line mention. **spec-03 (FROZEN) turned out to
already prescribe `spawn_subagent`'s exact interface**, via
`ToolContext.spawn_subagent: Callable[..., Awaitable[AgentResult]]`
(documented there as `(task, task_class, budget)`) and design decision
#7 ("subagents are just recursive loops with their own budget slice
... and their transcript nested under the parent's run dir") —
scaffolded years earlier via the `AgentResult` type in
`sarva.agent.events`, which had sat completely unused until this
milestone. The first version of this feature didn't match that: it
took only a task string, returned a bare `Message | None`, and put a
subagent's run directory as a *sibling* under the shared `run_root`
rather than nested under the parent's own run dir.

Caught by actually reading spec-03 in full before moving on to the next
feature, not just working from the design doc's one-line summary —
reconciled in the same session rather than left to drift:

- `spawn_subagent` now takes `(task, task_class=TaskClass.SUBTASK,
  budget=None)` and returns a real `AgentResult` (`state`,
  `final_message`, `spend`, `run_dir`). `DelegateTool` itself still only
  ever calls it with just a task string — the fuller signature exists
  for other, possibly-future callers (like a verifier subagent) that
  need to pick a different `TaskClass` or request a specific budget
  slice, matching spec-03's own framing of `spawn_subagent` as a
  general primitive on `ToolContext`, not something private to one
  tool.
- A `budget` argument is a **request, not a grant**: every field is
  clamped to what's actually left of the parent's own budget, matching
  spec-03's conformance invariant #8 verbatim ("its budget slice cannot
  exceed the parent's remainder"). Verified two ways: a tight request
  well within the parent's remainder is honored exactly (not silently
  widened to whatever the parent could afford), and a request that
  would exceed the remainder gets clamped down.
- The subagent's transcript is now genuinely nested at
  `<parent_run_dir>/subagents/<sub_run_id>/`, not a sibling under
  `run_root` — `AgentLoop.run()` gained an optional `run_id` parameter
  so the spawn closure can pre-compute the exact path a subagent's
  transcript will land at *before* awaiting its run, letting
  `AgentResult.run_dir` report a real, already-correct path rather than
  a placeholder. Verified directly against the real filesystem layout,
  not just the reported string.

Verified by reverting just this correction (keeping the original
feature) and watching both new tests fail with the exact right reason:
`TypeError: spawn_subagent() got an unexpected keyword argument
'task_class'` — the old closure genuinely couldn't accept what the
frozen spec requires. 2 more new tests, all pre-existing tests
(including the original 5 subagent tests) pass unchanged.

## Verifier subagent: `verify=True`, an advisory check, not a hard gate

The design doc's second named agent-orchestration pattern, closed in
the same milestone as the correction above. Unlike `delegate_task`,
verification isn't something a model chooses to invoke — it's a
loop-level opt-in (`AgentLoop(..., verify=True)`, the same posture as
`degraders`), reachable from `sarva chat --verify`, `sarva run
--verify`, and both `/chat`/`/ws/chat`'s new `verify` request field.

When the main loop reaches a candidate `END_TURN` with `verify=True`,
it calls the SAME `spawn_subagent` primitive `delegate_task` uses
(`task_class=TaskClass.SUBTASK`, no special tool restriction — the
verifier gets the same tools any subagent does, since the existing
confirm-gating discipline already covers destructive calls it might
make) with a prompt asking it to judge the candidate answer against the
original task, prefixing its response with `VERIFIED` or `REJECTED`.
This decision has to happen BEFORE `transition(AgentState.DONE)`, not
after: `DONE` has no legal outgoing transition in the frozen `LEGAL`
table (it's terminal), so rejecting after already transitioning there
would trip `transition()`'s own legality assertion — the check runs
while `state` is still `CALLING_MODEL`, which legally transitions to
either `DONE` or `FAILED`.

**Deliberately advisory, not a hard gate, for this first slice:** only
an unambiguous `REJECTED` verdict turns the run into `FAILED` (with the
verifier's own reason in `StateChangedEvent.detail`, the same "give the
real reason" pattern every other clean-failure path in this loop
already uses). Every other outcome — the verifier subagent itself
fails/refuses/runs out of budget, or its response doesn't
unambiguously start with `REJECTED` — is treated as a pass-through: the
original candidate answer stands, completely unchanged. A flaky or
unavailable verifier can never take down an otherwise-working run. A
stricter mode (ambiguous = reject) is a real, legitimate alternative
design, not implemented here — named as a possible future refinement,
not silently assumed unnecessary.

**Verified end to end with `MockProvider`, covering both outcomes and
the fail-open cases:** an approving verifier leaves the original answer
and the reported spend correctly includes the verifier's own real
model call; an unambiguous rejection produces `FAILED` with the
verifier's reason in `detail` and `final_message=None` (matching the
existing convention every other `FAILED` path already uses); a refused
or ambiguous verifier never blocks completion. Reverting produced
`TypeError: AgentLoop.__init__() got an unexpected keyword argument
'verify'` on every test that passed it. 5 new agent-loop tests plus 3
new server tests (REST + WebSocket wiring, proving the request field
genuinely reaches `AgentLoop`, not just that it's accepted as valid
JSON), every pre-existing test unaffected.

### A real bug found later: the verifier's own real cost could push a run over `Budget` while it still reported `DONE`

A later round applied this project's own "give freshly-shipped code a
dedicated bug-hunting pass" pattern to the interaction between
`verify=True` and `Budget`, and found a real gap: the one
`spend.exceeded(self._budget)` check in this loop's `END_TURN` handling
runs *before* the verify block above ever executes, but `spawn_subagent()`
unconditionally merges the verifier's own real `Spend` into this run's
live `spend` afterward — so a verifier expensive enough to push the
merged total over budget was never caught. Confirmed live: an identical
`Budget(max_total_tokens=60)` correctly reported `DONE` with
`verify=False` (spend stayed at 30 tokens); with `verify=True` against
the exact same budget, the merged spend reached 126 tokens — genuinely
over budget — yet `RunDoneEvent.state` still reported `"done"`. Traced
through to real consequences, not just a theoretical gap: both
`cli.py` and `server/app.py` gate "was this run a failure" (exit code,
`_print_run_failure`) *and* "should the session be saved" on
`state == DONE` alone, so an over-budget `verify=True` run was reported
as an ordinary success everywhere — silently defeating the one purpose
`Budget` exists to serve, specifically whenever verification itself was
what tipped spend over the line.

Fixed by re-checking `spend.exceeded(self._budget)` a second time,
after the verify block (the only other thing in this branch that can
change `spend`), and giving `BUDGET_EXCEEDED` priority over both a
`REJECTED` verdict and an ordinary `DONE` — matching the budget check's
own priority everywhere else in this loop. Verified live: the identical
repro above now correctly reports `BUDGET_EXCEEDED` (with `detail`
naming which dimension — `"tokens"`) for the `verify=True` case, while
the `verify=False` control run is unaffected. Reverting reproduced the
exact old bug in the new test's own assertion failure: `DONE` where
`BUDGET_EXCEEDED` was expected, the real over-budget spend already
attached to the event. 1 new test, every pre-existing test unaffected.

## Build it yourself

- Read `tests/conformance/test_agent.py` — `MockProvider` scripts let
  you drive the loop through every state without a real model, which is
  exactly how this project tests budget exhaustion, tool errors,
  confirmation gating, and the degradation fallback without ever
  touching the network.
- Write a tool that raises on purpose and watch the loop keep going —
  the model sees the error and gets to react, the run doesn't crash.
- Construct a `Budget(max_model_calls=1)` and watch a multi-tool-call
  task land in `BUDGET_EXCEEDED` with a real `Spend` summary instead of
  running forever.
- Try `sarva run --auto "some destructive task"` vs. plain `sarva run`
  and watch the exact same loop take two different paths through
  `AWAITING_CONFIRMATION` based on nothing but which `ConfirmPolicy`
  was passed in.
