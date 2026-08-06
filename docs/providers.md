# Chapter 2 — The Provider Abstraction, Model Registry, and Routing

Chapter 1 named the deal Sarva makes: lean on frontier models today,
behind an abstraction that lets tomorrow's models drop in as a
one-entry registry change instead of a rewrite. This chapter is where
that promise actually lives in code — `sarva.providers`.

## The contract every backend implements

Everything in Sarva that needs a model — the agent loop, the eval
harness, distillation — talks to exactly one interface:

```python
class Provider(Protocol):
    name: str
    def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]: ...
    async def close(self) -> None: ...
```

`GenerateRequest` is provider-agnostic: a model id, an optional system
prompt, a list of typed `Message`s (each holding a list of typed
`ContentBlock`s — text, images, tool calls, tool results), a list of
`ToolSpec`s, and a small `GenerateConfig` (max tokens, an `effort`
knob, `thinking`, stop sequences). `generate()` streams back a sequence
of `ProviderEvent`s — `TextDeltaEvent`, `ThinkingDeltaEvent`,
`ToolCallEvent`, `ContentEvent`, ending in exactly one `DoneEvent`
(carrying the assembled assistant `Message`, a `StopReason`, and
`Usage`) or a `StreamErrorEvent`. `sarva.providers.base.complete()` is
the non-streaming convenience every caller that doesn't need
token-by-token output reaches for: drain the stream, return the
`DoneEvent`, raise `ProviderError` on a `StreamErrorEvent`.

Five real implementations exist: `AnthropicProvider`, `OpenAIProvider`,
`GoogleProvider`, `OllamaProvider` (the free/local/private tier — talks
to a local Ollama server's HTTP API, no API key, no network egress
beyond localhost), and `MockProvider` (fully offline and deterministic,
what makes `sarva chat "hello"` work with zero configuration and what
drives this project's own test suite without needing credentials in
CI). Each adapter is deliberately **thin**: translate `GenerateRequest`
into that backend's wire format, translate wire events back into
`ProviderEvent`s. No retries beyond the SDK's own, no routing, no
degradation logic — those live one layer up, in the router and the
agent loop.

## Every backend disagrees about something, and the adapters are where that friction lives

Writing four real adapters against four real APIs surfaced genuine
differences in how providers represent the same concept — worth
knowing if you're adding a fifth:

- **Tool-call streaming shape.** Anthropic's SDK hands back an
  already-assembled final message via `get_final_message()` — no
  manual accumulation needed. Ollama's chat API sends each tool call
  complete in a single chunk. OpenAI streams a tool call's `arguments`
  as string fragments scattered across many chunks, keyed by an
  `index` — the adapter has to accumulate them itself, and a bug here
  could silently cross-contaminate two concurrent tool calls'
  arguments. (`openai_provider.py`'s own module docstring names this
  directly; `test_openai_provider_streaming.py` has a dedicated test
  that deliberately interleaves two tool calls' fragments to prove the
  accumulation is correct.)
- **No universal "I made a tool call" signal.** Anthropic's
  `stop_reason` says `tool_use`; OpenAI's `finish_reason` says
  `tool_calls`. Gemini has no equivalent — it reports `STOP` even when
  the response includes `function_call` parts. Trusting `finish_reason`
  the way the other two adapters correctly do would silently misreport
  every Gemini tool-use turn as a normal end-of-turn. `google_provider.py`
  infers `TOOL_USE` from the presence of a tool-call block first,
  falling back to the raw finish-reason mapping only when there isn't
  one — a real bug caught by a hermetic test before it ever reached a
  live run, not discovered afterward.
- **Tool-result message shape.** Anthropic lets several tool results
  live inside one `role="user"` content array. OpenAI requires a
  *separate* `role="tool"` message per `tool_call_id`. Gemini bundles
  tool responses as `role="user"` parts carrying a `function_response`,
  correlated back to the original call by an `id` your adapter has to
  track yourself since `ToolResultBlock` (Sarva's own type) doesn't
  carry the function's name — only its call id.
- **Only Anthropic requires a signed round trip for reasoning content.**
  When extended thinking makes a tool call, Anthropic expects the exact
  same `thinking` block — including its original `signature`, an
  anti-tampering check — back in history when the tool result is sent.
  `ThinkingBlock.provider_data` carries that signature (set the moment
  `AnthropicProvider.generate()` produces one); `anthropic_provider.py`
  reconstructs the wire-format block from it on the way back in, and
  drops it (as before) only when no signature is present. Neither
  OpenAI's nor Gemini's reasoning content has an equivalent requirement
  to translate. **A sibling wire type, `redacted_thinking` — returned
  whenever the model's own safety classifier flags part of its
  reasoning, a normal occurrence with thinking on, not a rare one — was
  silently dropped entirely until a later fresh-eyes sweep found it;
  see below.**
- **Ollama has no dedicated tool-result role at all.** Anthropic/OpenAI/
  Gemini each have a real wire shape for "here's what the tool
  returned" (a `tool_result` content block, a `role="tool"` message, a
  `function_response` part, respectively). Ollama's chat API doesn't —
  `_to_ollama_message` renders a `ToolResultBlock`'s text as the plain
  `content` of an ordinary message instead, the honest translation for
  a wire format with no dedicated concept for it. **Losing which result
  came from which tool is an accepted limitation of that translation —
  actively fusing two DIFFERENT results' values together was not, and
  was a real, separate bug; see below.**

None of this is exposed to callers. `AgentLoop`, `run_benchmark`, and
`distill()` all just call `provider.generate(request)` and get back the
same event stream shape regardless of which of the five backends is
underneath.

### Ollama is the one adapter verified fully live in this environment

Anthropic, OpenAI, and Google's adapters are all written to their
documented SDK/API shapes but have never been exercised against a real
credential here — this environment simply has none. Ollama is
different: it needs no API key, only a locally running server, which
this environment *can* actually provide. `brew install ollama`,
`ollama serve`, `ollama pull qwen2.5:0.5b` (a small model — `models.
yaml`'s real registered default, `qwen3:8b`, is ~5GB), then a real
`tests/live/test_live_providers.py::test_ollama_terminal_event_law` run
against it (`OLLAMA_TEST_MODEL` overrides the pulled model), plus
direct streaming and tool-call checks against the same live server. All
passed — the first of the five adapters to move from "written to spec"
to "confirmed working against a real backend" in this environment.

**A real, latent bug this surfaced, not caused:** several
CLI/server conformance tests asserted "zero-config routes to Mock"
without ever mocking away `ollama_reachable()` — true by coincidence in
CI and in this environment right up until a real Ollama server actually
started running, at which point the real router legitimately preferred
`ollama/qwen3:8b` (reachable, but not the small model actually pulled)
over falling back to Mock, and those tests broke. Any contributor
running this suite on a machine with their own local Ollama already
running — a very plausible setup for exactly the kind of person this
project's free/local/private tier is built for — would have hit the
same failures. Fixed by having every affected test explicitly mock
`ollama_reachable` to `False` (CLI tests) or force a mock-only router
(server tests), rather than depending on incidental machine state.

**A second, deeper real bug this same live setup surfaced later:** the
test-isolation fix above papered over the SYMPTOM, not the underlying
issue — `build_router()` itself marked every registered `ollama/*`
model "available" the instant the server merely answered, with no
regard for which model tag was actually pulled. Confirmed directly:
`sarva run ... --auto` in this exact environment (Ollama reachable,
only a small model pulled, not the registered `qwen3:8b`) genuinely
ended with `run ended: failed` — a real request routed to a model that
was never there, the zero-config Mock fallback never getting a chance.
Fixed by making availability per-model, not per-server:
`ollama_pulled_models()` queries the same `/api/tags` endpoint
`ollama_reachable()` already hits, and `build_router()` now only marks
`ollama/<tag>` available when that exact tag is in the real pulled set.
Re-ran the identical failing command afterward and it correctly fell
back to Mock instead. `sarva doctor`'s Ollama check gained the same
real data in its detail message (`pulled: qwen2.5:0.5b`, or an
explicit "no models pulled yet" when the server's up but empty).

**A third bug this same adapter had, unrelated to availability:**
`_to_ollama_message`'s translation loop had no `else` branch at all —
an `ImageBlock` (or any other block type this adapter has no wire
mapping for) was silently skipped, the model answering as if it had
never received it. Fixed to raise the same loud `ValueError` the
Anthropic/OpenAI/Google/Foundry adapters already do for exactly this
case.

**A fourth real cost this same pair of functions had, found much later by a round-45 sweep applying the "expensive work repeated on every ordinary call in a long-running process" lens** (the same one that fixed `AgentLoop`'s leaking run directories and `FoundryProvider`'s repeated checkpoint reloads): `ollama_reachable()` and `ollama_pulled_models()` each independently hit the real `/api/tags` endpoint, and `sarva.server.app`'s `/chat`/`/ws/chat` handlers call `build_router()` immediately followed by `build_providers()` on *every* request — so one request made 2-3 redundant, identical network calls to the same endpoint, even requests that never end up routed to Ollama at all (routing hasn't been decided yet when these functions run). Confirmed live: pointing `OLLAMA_HOST` at a black-holed address (a realistic firewalled or slow-starting host, not just "nothing on localhost") measured 0.61s of pure blocking network wait added to a single request. Fixed with a short-TTL (2s) cache shared by both functions, keyed by host — long enough to collapse the redundant calls within one request, short enough that a server actually starting up mid-session is still noticed within a couple of requests. The cache preserves both functions' exact pre-existing independent semantics (`ollama_reachable()` true on any successful connection regardless of HTTP status; `ollama_pulled_models()` only populated on an actual 2xx), verified directly against a faked 500 response. Verified by reverting and watching the new test fail with `AttributeError: module 'sarva.runtime' has no attribute '_ollama_probe_cache'`. 2 new tests, all pre-existing runtime tests pass unchanged.

**A fifth bug in the same probe, found by a much later fresh-eyes
sweep: the round-45 fix above cut the redundant calls down to one, but
never addressed that the one remaining call still fully freezes the
process's event loop.** `_probe_ollama` calls the plain synchronous
`httpx.get()` — a real blocking network call — directly inside
functions (`ollama_reachable`/`ollama_pulled_models`, and in turn
`build_router`/`build_providers`/`run_diagnostics`) that
`sarva.server.app`'s `/models`, `/doctor`, `POST /config`, `/chat`, and
`/ws/chat` handlers all call inline from their own `async def` bodies,
with no `asyncio.to_thread` anywhere in the chain. A synchronous
network call inside an `async def`, called directly rather than
offloaded to a thread, doesn't just add latency to its own request —
it blocks the *entire* event loop for the full call, exactly the same
bug class already found and fixed for `VectorMemoryStore`'s sqlite
calls, `LongTermMemoryStore`'s flock, and both multimodal degraders'
subprocess decodes. Confirmed live with a real `httpx.AsyncClient`
over the app's own ASGI transport (genuine concurrency, not the
synchronous `TestClient`): racing a slow `/models` call (Ollama probe
patched to block for 0.2s) against a concurrent, completely unrelated
`/health` call — which touches none of the same code and should
return in microseconds — the unrelated call took the *full* 0.205s
too, starved of any chance to run until the slow call's blocking chunk
released the thread. Fixed by wrapping each of the 5 real call sites in
`sarva.server.app` with `asyncio.to_thread`, moving the blocking
network I/O off the event loop — `runtime.py`'s own functions stay
synchronous, since `sarva doctor`/`sarva models` call them from plain,
non-async CLI commands with no event loop to block in the first place;
only the server's concurrent, multi-request context has the severity
that makes this worth fixing. Verified live after the fix: the same
race now measures the unrelated request at ~3ms, independent of the
slow one's duration. Verified by reverting and watching the new test
fail with the literal old number reproducing itself: `/health took
0.205s`. 1 new test, 726 -> 727 Python tests.

**A sixth bug in the same probe, found by yet another fresh-eyes
sweep: `except httpx.HTTPError` only covers connection/status failures,
not a reachable host answering with a body that isn't Ollama's own
`/api/tags` shape.** A completely ordinary real-world condition —
`OLLAMA_HOST` pointing at a corporate captive portal, a stale or
misconfigured reverse proxy, or simple port reuse by an unrelated
service — answers with a real HTTP 200, but a body that isn't the JSON
`{"models": [{"name": ...}, ...]}` shape this probe expects.
`response.json()` raises `json.JSONDecodeError` on a non-JSON body (a
`ValueError` subclass, not an `httpx.HTTPError` subclass); a
differently-shaped-but-*valid* JSON body (a top-level list instead of
a dict, or a model entry missing `"name"`) raises `AttributeError` or
`KeyError` instead — neither of those is an `httpx.HTTPError` subclass
either, and neither narrower type alone would have covered the other.
Confirmed live: crashed `GET /models` and `GET /doctor` with a raw,
plain-text 500 (`sarva.server.app` registers no generic exception
handler besides `ConfigError`'s own, and neither endpoint has its own
try/except around these calls) — `POST /chat` happened to degrade
cleanly instead, purely because `json.JSONDecodeError` is a
`ValueError` subclass that handler's own broad `except` already
covers for an unrelated reason, not because anyone had reasoned about
this specific failure mode there either. Fixed with a second `except
Exception:` clause after the existing `httpx.HTTPError` one, rather
than enumerating `ValueError`/`AttributeError`/`KeyError` individually
— this probe's entire contract, per its own docstring, is
"best-effort," and every malformed-response shape should degrade the
identical way `httpx.HTTPError` already does (treat as
unreachable-or-nothing-pulled), not get re-litigated one exception
type at a time as each new shape is found — the same "third exception
type from the same call, missed by a command already partially fixed"
pattern a sibling round found in `sarva transcribe`'s own `--model-size`
handling, generalized correctly here instead of repeating it a fourth
time. 2 new tests (one for the invalid-JSON case, one for the
valid-but-wrong-shape case), 731 -> 733 Python tests.

**A seventh bug in the same probe, found by yet another fresh-eyes
sweep, one layer beyond the round-45 fix that first introduced the
cache: the cache's own read-check and write-back are two separate,
unsynchronized steps with a real, blocking `httpx.get()` in
between.** `build_router()`/`build_providers()` are called from every
`/chat`, `/ws/chat`, `/models`, and `/doctor` handler via `asyncio.
to_thread` — genuine concurrent requests (the desktop app's own two
independent `useEffect` hooks fire `/doctor` and `/models`
simultaneously on page load, not a contrived scenario) land in
separate real OS threads, and nothing serialized their access to the
shared cache dict. Confirmed live: 6 concurrent `build_router()` calls
against a cold cache made 6 real network calls, not the 1 the cache
exists to guarantee — silently reintroducing the exact redundant-
network-call regression (up to 0.61s of blocking wait per request
against a black-holed `OLLAMA_HOST`) the round-45 fix was built to
eliminate, the identical unguarded check-then-act race class already
found and fixed for `AgentLoop`'s own subagent budget grants (`spend_
lock`), just never applied here. Fixed with a `threading.Lock` around
the whole check-probe-write span, turning the cache into a genuine
single-flight: a thread that loses the race to acquire the lock reads
back the FRESH entry the winner just wrote, instead of racing its own
redundant probe — `httpx.get`'s own `timeout=0.3` bounds how long any
thread can be blocked waiting. Verified live: the same 6-concurrent-
callers repro now measures exactly 1 real network call. Verified by
reverting and watching the new test fail with the literal old number
reproducing itself: 6 calls instead of 1. 1 new test, 815 → 816
Python tests.

### Ollama vision — the named follow-up, closed and verified against a real local vision model

The gap named directly above ("real vision-capable Ollama models do
accept images... this adapter doesn't build yet") stayed open for
exactly one more milestone. `_to_ollama_message` is now `async` (like
the Anthropic/Google translators) so it can call the same
`resolve_media_bytes` they use — an `ImageBlock` becomes a raw base64
string (no `data:` URI prefix, no `media_type` field — Ollama's own
`/api/chat` shape) appended to a per-message `images` array.

**Verified against a real pulled vision model, not just the documented
wire shape:** `ollama pull moondream` (~1.7GB, real, small,
vision-capable — confirmed via its own `/api/tags` entry reporting
`"capabilities":["completion","vision"]` before writing any code), then
a real solid-red PNG built with Pillow sent through the actual
`OllamaProvider.generate()` path end to end (not a raw curl
shortcut) — the model's real reply: `"!!!RED!!!"`, genuinely identifying
the color from real pixel data, not an echo or a hallucinated guess (a
wrong color, or a refusal, would have been just as informative a
result — this was checked, not assumed). `moondream:latest` is now a
second registered `ollama/*` entry in `models.yaml`
(`modalities_in: [text, image]`, `tool_use: false` — matching what its
own capabilities list actually declares, not assumed from its size
class), so the router can genuinely pick it for an image-carrying
request once it's actually pulled, the same `ollama_pulled_models()`
exact-tag-match gate the text model already goes through. 6 new/changed
tests (the file's existing tests all needed `await` once the
translator went async), 458 → 460 Python tests.

### A malformed streaming line crashed `generate()` — closed after zero conformance coverage of the streaming loop itself

`generate()`'s NDJSON parsing (`json.loads(line)` inside `async for
line in response.aiter_lines()`) had no error handling at all — only
`httpx.ConnectError`/`httpx.TimeoutException` and a `status_code >= 400`
response were ever anticipated as failure modes. A genuinely truncated
line (the real-world trigger: a network glitch or proxy cutting the
stream mid-line) raised a raw `json.JSONDecodeError` straight out of
the async generator instead of a clean `StreamErrorEvent` — reproduced
with `httpx.MockTransport` returning deliberately truncated NDJSON
bytes, no real server needed. Since `sarva.providers.base.complete()`
(the non-streaming convenience wrapper `sarva.eval`'s benchmark harness
and `sarva.distill` both use) turns any `StreamErrorEvent` into a
`ProviderError` it already catches per-case, fixing this one adapter
method closes the failure at its source rather than needing separate
fixes in every consumer: a single malformed line from a local Ollama
model no longer aborts an entire benchmark run or distillation batch,
losing every other case's results. Fixed by wrapping the `json.loads`
call in `except json.JSONDecodeError`, `retryable=True` for the same
reason the two `httpx` exception handlers right below it already are —
a transient mid-stream hiccup is worth retrying, not a hard failure.
**A real, first-of-its-kind gap closed alongside the fix:** `generate()`
itself had zero conformance test coverage before this — only ever
exercised live against a real running Ollama server, unlike every other
adapter's own dedicated unit tests. New `httpx.MockTransport`-based
tests (the same hermetic-httpx discipline `test_fetch.py` already
established) cover both the streaming happy path and this regression.
2 new tests, 543 → 545 Python tests.

### A malformed Anthropic SDK response had the identical gap — found by reasoning through the SDK's own exception hierarchy, no API key needed

`AnthropicProvider.generate()`'s three `except` clauses
(`RateLimitError`/`APIConnectionError`/`APIStatusError`) look
comprehensive but miss one real sibling: `anthropic.
APIResponseValidationError`, raised when the SDK receives a response it
can't parse — the exact same "server sent something we can't make sense
of" shape as the Ollama bug just above, just one layer further down
(inside the SDK's own response handling rather than this project's own
`json.loads`). Confirmed by reading the SDK's actual class hierarchy,
not assumed: `RateLimitError` and every other HTTP-status-based error
(`AuthenticationError`, `OverloadedError`, ...) already inherit from
`APIStatusError`, and `APITimeoutError` inherits from
`APIConnectionError` — genuinely already covered — but
`APIResponseValidationError` is a direct sibling of all three under the
SDK's own `APIError` base, not a subclass of any of them. Reproduced
without a live API key: a duck-typed fake client (the same
`test_openai_provider_streaming.py`-style discipline this project
already uses for SDK-based adapters) raising `APIResponseValidationError`
from `messages.stream()`'s `__aenter__` propagated straight out of
`generate()`'s async generator uncaught.

Fixed with a final `except anthropic.APIError as e:` catch-all placed
*after* the three specific handlers — it only ever fires for whatever
they don't already cover, confirmed with a regression test that a
`RateLimitError` still gets its own, more specific `code="rate_limit"`
treatment rather than falling through to the generic one. Deliberately
broad (the SDK's own common base, not just the one named subtype found)
so any other still-unnamed `APIError` subtype the SDK adds in the
future is covered too, rather than needing a fourth (fifth, sixth...)
named `except` clause added reactively each time. 2 new tests, 545 →
547 Python tests.

### The same gap, found and fixed in both remaining adapters in one pass — OpenAI and Gemini

A fresh sweep, pointed specifically at repeating the Ollama/Anthropic
pattern against the two adapters not yet audited: same result, twice.

**`OpenAIProvider.generate()`** had the identical shape:
`openai.APIResponseValidationError` is a direct sibling of
`RateLimitError`/`APIConnectionError`/`APIStatusError` under the SDK's
own `APIError` base — confirmed by introspecting the real installed
`openai` package's class hierarchy, the same way as the Anthropic fix.
Reproduced with a duck-typed fake client raising it from
`chat.completions.create()`; fixed with the identical final
`except openai.APIError as e:` catch-all pattern, verified with the
same "a plain `RateLimitError` still gets its own specific treatment"
regression test.

**`GoogleProvider.generate()`** had a related but structurally
different gap: `google-genai`'s own `_api_client.
_load_json_from_response()` wraps a failed `json.loads()` on a
streaming response chunk in `errors.UnknownApiResponseError` — but
unlike the Anthropic/OpenAI cases, this one is a **`ValueError`
subclass, not an `errors.APIError` subclass**, so it sits in a
genuinely separate exception branch from the `errors.ClientError`/
`errors.ServerError` this adapter already caught (both `errors.APIError`
subclasses). Confirmed by reading the SDK's actual source, not assumed:
`_load_json_from_response` is called on every streamed chunk
(`_api_client.py`), so any malformed chunk anywhere in a real stream
hits this path. Reproduced with a fake async iterator raising a real
`errors.UnknownApiResponseError` mid-stream; fixed with its own
dedicated `except errors.UnknownApiResponseError as e:` clause (a
separate `except`, not folded into the existing two, since it isn't a
subclass of what they already catch). This is a distinct, narrower fix
than the Anthropic/OpenAI catch-alls — it closes the one concrete gap
found, not a speculative "any future exception" case, since this
adapter's own module docstring already separately and honestly names a
different unhandled gap (network-level connection failures — no
documented `google-genai` equivalent to `APIConnectionError` was found)
that a broad `except errors.APIError` wouldn't have touched either.

3 new tests (2 for OpenAI mirroring the Anthropic fake-client tests, 1
for Gemini using `test_google_provider_streaming.py`'s existing fake
async-stream infra), 547 → 550 Python tests. **This closes the
"uncaught SDK exception" bug class across all three SDK-based real
adapters (Anthropic, OpenAI, Google) — Ollama's own equivalent
(malformed NDJSON, not an SDK exception) was already closed earlier.**

### Malformed tool-call-arguments JSON silently became an empty dict, with no signal anywhere — the one OpenAI-specific accumulation gap the interleaving test above didn't cover

The tool-call streaming shape named earlier in this chapter — OpenAI
scatters a tool call's `arguments` as string fragments across many
chunks, and the adapter accumulates and `json.loads()`s them itself
once the stream ends — has a failure mode the existing interleaving
test never exercised: what if the fully-accumulated string still isn't
valid JSON? Unescaped embedded quotes in a tool call's string
arguments are a real, documented GPT tool-calling failure mode,
independent of `max_tokens` truncation. The existing code caught
`json.JSONDecodeError` and silently substituted `{}` — no
`StreamErrorEvent`, unlike every other failure path in this same
function (rate limits, network errors, malformed SDK responses all
yield one). `AgentLoop` would then dispatch that corrupted, empty
arguments dict to whatever tool the model actually meant to call with
real arguments: a built-in tool's required-key access raises a
confusing bare `KeyError` with no way to trace it back to silently
dropped arguments; an MCP tool (`McpToolAdapter.run`) forwards `{}`
straight to the remote server with zero local validation, silently
executing the *wrong* action for any tool with optional or defaulted
parameters — genuinely undetectable corruption, not just a confusing
error message. Anthropic and Gemini don't share this gap: Anthropic's
SDK hands back an already-assembled message with no manual JSON
re-parse, and Gemini's SDK returns already-structured argument dicts
directly, no `json.loads()` on raw accumulated text at all — unique to
OpenAI's manual delta-accumulation path.

Confirmed live with a duck-typed fake stream delivering a tool call
whose accumulated arguments never form valid JSON. Fixed by treating
this the same as every other stream-level failure in this function:
`yield StreamErrorEvent(code="malformed_tool_arguments", ...,
retryable=True); return` instead of silently substituting `{}` and
continuing — `retryable=True` gives `AgentLoop`'s existing retry
mechanism (the same one a rate limit or network blip already gets) a
real chance to get a clean response on the next attempt, rather than
quietly corrupting the one it already has. The existing dedicated test
for this exact scenario had encoded the buggy behavior as its own
expectation (`arguments == {}`, "must degrade... not raise") — a
reasonable instinct half-satisfied: not raising was right, but silent
substitution was the wrong way to satisfy it. Rewritten to assert the
`StreamErrorEvent` instead, plus a new sibling test confirming ordinary
well-formed arguments still parse exactly as before. 1 test rewritten,
1 new, 728 → 729 Python tests.

### A `ToolResultBlock` carrying an image was silently dropped by all three real adapters — each fixed to match what its own SDK actually supports, not a uniform guess

A fresh sweep, checking each adapter's request-translation logic beyond
exception handling for the first time. All three built a tool result's
wire-format content with the identical shape: `"".join(c.text for c in
b.content if isinstance(c, TextBlock))` — silently discarding any block
that wasn't a `TextBlock`. `ToolResultBlock.content`'s own type comment
already names `ImageBlock` as an anticipated case, not a hypothetical
— a screenshot or image-generation tool returning its result this way
would have that image vanish with no error, on every adapter, the
moment any built-in tool actually returned one (none does yet, so this
was latent, not yet a live crash).

**This isn't a uniform gap with a uniform fix — the three real SDKs
genuinely disagree about whether a tool result can carry an image at
all, confirmed by reading each SDK's own type definitions directly,
not assumed:**

- **Anthropic** (`ToolResultBlockParam.content: Union[str,
  Iterable[Content]]`, where `Content` includes `ImageBlockParam`) —
  genuinely supports it. Fixed by building a content list mixing text
  and image parts whenever a result contains anything beyond plain
  text; a text-only result still sends a plain string exactly as
  before, so the overwhelmingly common case is unaffected.
- **Google Gemini** (`FunctionResponse.parts:
  list[FunctionResponsePart]`) — also genuinely supports it, via a
  separate `parts` field alongside the plain `response` dict. Fixed by
  attaching a `FunctionResponsePart.from_bytes(...)` for each image
  found.
- **OpenAI** (`ChatCompletionToolMessageParam.content: Union[str,
  Iterable[ChatCompletionContentPartTextParam]]`) — text only, no
  image variant exists in the type at all. Unlike the other two, this
  genuinely can't be fixed by sending the image along — there's no
  wire shape for it. Fixed the same way this project already treats
  every other untranslatable block (the top-level `else: raise
  ValueError(...)` a few lines above each of these fixes): raises a
  clear `ValueError` naming the unsupported block type, rather than
  silently completing the request with content missing.

**Verified the new tests are real:** reverted all three adapters and
watched each new test fail for the specific, correct reason — the
Anthropic/Gemini tests failed on a missing image in the translated
output, the OpenAI test failed with `Failed: DID NOT RAISE ValueError`
— before re-applying. All 27 pre-existing tests across the three
adapters' test files pass unchanged. 3 new tests, 574 → 577 Python
tests.

### The fourth real adapter had the identical gap — Ollama, never given the equivalent fix at the time

A much later fresh-eyes sweep pointed specifically at the two
providers never given their own dedicated round (Anthropic/OpenAI/
Google, Ollama's three siblings) found the same shape: `_to_ollama_
message`'s own `ToolResultBlock` branch had the identical plain
`"".join(c.text for c in b.content if isinstance(c, TextBlock))`, the
exact code the fix above closed for the other three adapters — just
never applied to this one. Confirmed live: a `ToolResultBlock`
carrying a `TextBlock` and an `ImageBlock` produced a wire message
with the image gone, no trace anywhere. Concretely reachable, not
latent the way it was for the other three at fix time: `McpToolAdapter.
run()` (`sarva.mcp_client`) already converts a real MCP server's
`ImageContent` (a standard MCP content type — screenshots, diagrams,
generated images) into a real `ImageBlock` inside the `ToolResultBlock`
it returns, so any real `sarva run --mcp-server ...` session using an
image-returning MCP tool, routed to Ollama (the project's own "free
& private," zero-config default), silently lost that image with the
model answering as if it had never received it.

Ollama's own `/api/chat` wire format has no per-tool-result content
shape at all, genuinely different from all three siblings above — but
unlike OpenAI's hard "no wire shape exists" case, Ollama's own message
schema already has a flat, message-level `images` array, the identical
one a top-level `ImageBlock` populates a few lines above this fix (see
"Ollama vision" above). So a tool-result image genuinely *can* be
sent, just via that same message-level array rather than nested inside
the tool result's own content the way Anthropic/Gemini do it. Fixed by
extracting a tool result's `ImageBlock`s into that array instead of
dropping them; any other, genuinely untranslatable block type inside a
tool result still raises, matching every other adapter's own
discipline for this exact case (and the top-level `else: raise` this
same file already has, a few lines below). Verified live across three
cases together: the image case (now sent via `images`, text content
unaffected), a text-only tool result (completely unaffected, matching
the pre-existing behavior exactly), and a genuinely untranslatable
block type inside a tool result (still raises `ValueError` naming it).
Verified by reverting and watching both new tests fail for the
specific, correct reason — the image test with a missing `images` key,
the unsupported-type test with `Failed: DID NOT RAISE ValueError` — the
identical failure signature the original three-adapter fix's own
revert-and-verify produced. 2 new tests, 745 → 747 total.

### Two parallel calls to the same tool collided onto one id in the Gemini adapter — found by giving tool-call id handling its own fresh-eyes sweep

`google.genai.types.FunctionCall.id` is documented `Optional[str]` on
the wire, and Gemini frequently leaves it unset. `google_provider.py`'s
own fallback for that case was `id or name` — harmless for a single
tool call, but two ordinary parallel calls to the *same* tool in one
turn (`get_weather("NYC")` and `get_weather("LA")`, a completely
ordinary agentic pattern, not contrived) both fell back to the
identical `id="get_weather"`. That collision reaches the wire, not
just this adapter's internal bookkeeping: `_to_gemini_content` echoes
`tool_call_id` straight into `FunctionResponse.id`, a field Gemini's
own docs say exists specifically so the model can match a response
back to the call that produced it — both responses would carry the
identical id, defeating that correlation for exactly the two calls
that most needed it disambiguated. Anthropic and OpenAI don't have
this gap: their SDKs always supply a real per-call id, so no fallback
exists on either adapter.

Confirmed live with the existing fake-`SimpleNamespace`-client pattern
(`test_google_provider_streaming.py`): two function-call parts with
`id=None` for the same tool name both produced `ToolCallBlock(id=
"get_weather", ...)`. Fixed with a per-response counter instead of the
name-only fallback — `f"{name}-{index}"`, incrementing only when
Gemini genuinely omitted an id — unique within a turn regardless of
how many calls share a name or how many distinct names appear, while a
real id Gemini *does* supply is still used exactly as given. Verified
live the two NYC/LA calls above now get distinct ids. Verified by
reverting and watching the new test fail with the literal old
collision reproducing itself (`'get_weather' != 'get_weather'` — both
sides identical). 2 new tests, 705 → 707 Python tests.

### Interleaved text and tool calls got silently reordered in the Gemini adapter — a sibling ordering bug, not the id-collision above

A different bug in the same `generate()` loop, found by a later
fresh-eyes sweep: `ToolCallBlock`/`ImageBlock` were already appended
into `blocks` in true chronological stream order as each part arrived,
but streamed text took a different path entirely — every text part was
accumulated into one running string (`text_acc`) across the *whole*
stream, then spliced into `blocks` exactly once, unconditionally
inserted at index 0, after the loop had already finished.

An ordinary sequential-tool-calling turn — reasoning text, a call, more
reasoning text, another call, a documented Gemini pattern and not
contrived — confirmed live before this fix: both text segments were
concatenated into a single `TextBlock` and hoisted ahead of *both* tool
calls, even though the second segment chronologically followed the
first call. The corruption is specific to the persisted `DoneEvent.
message`, not the live token-by-token stream — `TextDeltaEvent`/
`ToolCallEvent` still fire in correct order — but that persisted
message is exactly what `AgentLoop` appends to `transcript_out`/
`SessionStore` and re-sends as `history` via `_to_gemini_content` on
the next turn, so the saved/replayed record silently misrepresented
which reasoning text justified which call from the very first
multi-call turn.

Fixed by flushing any accumulated text into its own `TextBlock`, in
place, immediately before appending a tool call or image block — text
within one uninterrupted run still merges into a single block
(confirmed unaffected by a second new test), but a tool call or image
between two text runs no longer gets silently reordered around. The
trailing flush after the stream ends now appends rather than inserts
at the front, for the same reason. Verified live the NYC/LA turn above
now produces `[text, call, text, call]` in true order. Verified by
reverting and watching the new test fail with the literal old bug's
own shape: both text segments merged into one block, hoisted ahead of
both calls. 2 new tests, 748 → 750 Python tests.

### The identical ordering bug, never propagated to the Ollama adapter's own `generate()` loop

A later fresh-eyes sweep found the exact same shape one adapter over:
`OllamaProvider.generate()` accumulated every streamed text delta into
one running string across the *whole* NDJSON response, then spliced it
into the final message exactly once — unconditionally first, before
every tool call, regardless of when each tool call actually arrived
relative to the text. Tool calls themselves were already appended in
true chronological order, the identical asymmetry the Gemini fix above
closed, just never checked for in this sibling adapter.

Confirmed live before this fix: an ordinary sequential-tool-calling
turn against a real NDJSON stream shape (reasoning text, a call, more
reasoning text, another call — ordinary ReAct-style behavior for local
tool-using models served via Ollama, llama3.1/qwen2.5/mistral-nemo all
support this, not a contrived shape) produced one `TextBlock` with both
segments concatenated, hoisted ahead of both tool calls — the true
`[text, call, text, call]` order lost entirely. `AgentLoop.run()`
appends this exact, corrupted `Message` straight into
`transcript_out`/`SessionStore` with nothing stripped or corrected, so
the misleading merged/misordered record is persisted and resent as
history on every subsequent turn — the identical downstream
consequence the Gemini fix above already named.

Fixed with the same technique: flush any accumulated text into its own
`TextBlock`, in place, immediately before each tool call; the trailing
flush after the stream ends appends rather than inserts at the front.
Verified live the interleaved-turn repro now produces the true
`[text, call, text, call]` order. Verified by reverting and watching
the new test fail with the literal old bug's own shape: both text
segments merged into one block, hoisted ahead of both calls. 2 new
tests, 754 → 756 Python tests.

### The third real adapter with the identical gap, finally found in `openai_provider.py` after five rounds flagged as the likely place to check

`OllamaProvider`/`GoogleProvider`'s own ordering fixes were both found
and closed in earlier rounds; `openai_provider.py`'s own text-
accumulation logic was named as the natural next place to check every
round since, and kept coming back clean until a fresh-eyes sweep
finally confirmed the identical bug there too. `OpenAIProvider.
generate()` accumulated every streamed text delta into one running
string across the *whole* response, then spliced it into the final
message exactly once — unconditionally first, before every tool call —
while tool calls themselves were already assembled in true
chronological order relative to each other (Python dict insertion
order on `tool_call_parts`, keyed by the SDK's own per-call `index`).

Confirmed live: an ordinary sequential-tool-calling turn (reasoning
text, a call, more reasoning text, another call) produced one
`TextBlock` with both segments concatenated, hoisted ahead of both
tool calls — corrupting the persisted `Message` `AgentLoop` appends
straight to `transcript_out`/`SessionStore` and resends as history,
the identical downstream consequence both sibling fixes above already
named. Reachable on any OpenAI (or OpenAI-compatible — the adapter
just wraps `openai.AsyncOpenAI()`, and many self-hosted/third-party
endpoints speak this same SDK) tool-using model that emits explanatory
text between two sequential tool calls in one turn — the normal
sequential-tool-calling path, no malformed input needed.

**Genuinely trickier to fix here than in either sibling adapter**:
unlike Ollama (each tool call complete in one chunk) or Gemini (each
function-call part is a single, complete unit), OpenAI streams a tool
call's `arguments` string as fragments across *many* chunks sharing
one `index` — there's no single "this chunk completed a call" moment
to flush text against, since a call isn't a real `ToolCallBlock` until
the whole stream ends and every fragment across every chunk has been
assembled. Fixed instead by recording an *ordered* list of markers as
things first appear — a flushed text segment, or the first-seen index
of a tool call (its actual `ToolCallBlock` is still only built once,
at the end, from the same `tool_call_parts` accumulator this adapter
already had) — and replaying that order when assembling the final
block list, rather than reconstructing position from the accumulator
plus a single trailing text blob. Verified live the interleaved-turn
repro now produces the true `[text, call, text, call]` order, and the
pre-existing two-concurrent-tool-calls argument-reassembly test (the
one piece of genuinely novel logic this file's own module docstring
already calls out) still passes unchanged — the ordering fix doesn't
touch how arguments themselves accumulate, only where the resulting
blocks land. Verified by reverting and watching the new test fail with
the literal old bug's own shape: both text segments merged into one
block, hoisted ahead of both calls. 2 new tests, 766 → 768 Python
tests.

### A blocked Gemini generation was reported as a normal, successful, silently empty turn

A much later fresh-eyes sweep of `google_provider.py` checked
`_STOP_REASON_MAP`, the small dict that translates Gemini's own
`FinishReason` into this project's `StopReason`, against the real
installed SDK's enum rather than trusting the five members already
listed there. The real `types.FinishReason` has 17 members; only
`STOP`, `MAX_TOKENS`, `SAFETY`, `PROHIBITED_CONTENT`, and `BLOCKLIST`
were mapped. Everything else — including `RECITATION` (Gemini refusing
because the answer recites training-data text too closely, e.g. song
lyrics or well-known published text — a well-documented, ordinary
occurrence, not contrived), `SPII`, `MALFORMED_FUNCTION_CALL`,
`UNEXPECTED_TOOL_CALL`, `LANGUAGE`, and `OTHER` — fell through the
map's own `.get(..., StopReason.END_TURN)` default straight into the
*success* path.

Gemini sends no content parts on a blocked candidate, so the streaming
loop's existing `if not candidate.content or not candidate.content.
parts: continue` added no blocks either. Confirmed live with a
duck-typed fake client (no live API key needed, the same discipline
this file's other adapter-specific bugs were found and fixed with)
yielding `RECITATION`/`MALFORMED_FUNCTION_CALL`/`SPII` finish reasons:
each one produced a `DoneEvent(stop_reason=END_TURN, message=Message
(content=[]))` — a genuinely blocked generation reported to the
CLI/server caller as a normal, successful completion with an *empty*
message, silently masking the real failure. By contrast, the three
already-mapped block reasons correctly hit `StopReason.REFUSAL` →
`AgentState.FAILED` in `agent/loop.py`, proving the intended behavior
for a blocked generation is exactly what these others should get but
didn't.

Fixed two ways together: the newly-identified reasons are added
explicitly to `_STOP_REASON_MAP` (self-documenting, same as the
existing three), and the lookup's own default is changed from
`StopReason.END_TURN` to `StopReason.REFUSAL` — `STOP` is the only
finish reason that legitimately means success, and it's already
explicitly mapped, so any *other* value, known today or added by a
future SDK release (an image-out-specific variant like `IMAGE_SAFETY`,
say), now fails safe: a clean `REFUSAL`/`FAILED` state a caller can
see, never a silently "successful" empty response just because this
map hadn't named it yet. Verified live: the same repro now reports
`REFUSAL` for all three block reasons. Verified by reverting and
watching the new tests fail with the literal old bug's own shape:
`stop_reason == END_TURN` for both a known blocked reason and a
deliberately unrecognized one. 2 new tests, 786 → 788 Python tests.

### The identical stop-reason gap, never propagated to the Anthropic adapter's own `pause_turn`

A much later fresh-eyes sweep found the exact same bug class one
adapter over: `anthropic_provider.py`'s `_STOP_REASON_MAP` covered
`end_turn`/`tool_use`/`max_tokens`/`refusal`/`stop_sequence`, but the
real Anthropic Python SDK's `StopReason` type also includes
`pause_turn` — a real, documented, non-error state ("we paused a
long-running turn. You may provide the response back as-is in a
subsequent request to let the model continue," per the SDK's own
docstring), returned for long-running server-side tool use such as web
search or code execution. Left unmapped, it fell through the identical
`.get(..., StopReason.END_TURN)` default the Google fix above had
already closed, straight into the success path — confirmed live with a
duck-typed fake client producing a real `pause_turn` final message:
the CLI/server/agent loop reported it as a normal, complete answer,
silently dropping the unfinished long-running turn instead of
surfacing it as incomplete.

This project has no turn-resumption mechanism wired in yet, so there's
no StopReason value that accurately means "paused, resume me" — fixed
by mapping `pause_turn` to `REFUSAL` (the closest honest fit among
existing values: not a complete, successful answer, and `AgentLoop`
already turns it into a clean `FAILED` rather than crashing or
silently succeeding) and changing the map's own default from
`StopReason.END_TURN` to `StopReason.REFUSAL`, the identical
fail-safe-default change the Google fix made, so this adapter is now
protected against any future unmapped `stop_reason` too, not just the
one this sweep happened to find. Verified live: the same repro now
reports `REFUSAL` instead of `END_TURN`. Verified by reverting and
watching both new tests fail with the literal old bug's own shape:
`stop_reason == END_TURN` for both `pause_turn` and a deliberately
unrecognized future value. 2 new tests, 790 → 792 Python tests.

### The third instance of the same stop-reason gap, in the OpenAI adapter's own deprecated `function_call` value

A much later fresh-eyes sweep, following up directly on a lead named
but deliberately not acted on in the previous round (to keep one fix
per round), checked the third and last of the three real provider
adapters against its own SDK's finish-reason type the same way. The
real OpenAI SDK's `Choice.finish_reason`
(`openai.types.chat.chat_completion_chunk`) is `Literal["stop",
"length", "tool_calls", "content_filter", "function_call"]` —
`openai_provider.py`'s own `_STOP_REASON_MAP` covered the first four
but not `function_call`, the deprecated legacy single-function-calling
API's own stop reason — still a real, documented value the SDK's type
carries, not a hypothetical one.

Worse than the Google/Anthropic instances of this same bug: this
adapter only ever parses `delta.tool_calls` (the modern API), never
the legacy `delta.function_call` field, so a response using the old
API produces no `ToolCallBlock` at all — the function call itself is
silently unparsed, not just misreported. Confirmed live with a
duck-typed fake stream yielding `finish_reason="function_call"`: it
fell through the map's own `.get(finish_reason or "stop",
StopReason.END_TURN)` default straight into the success path,
reporting a genuinely unparsed function call as a normal, complete
answer.

Fixed identically to both sibling adapters: `function_call` mapped
explicitly to `REFUSAL`, and the lookup's own default changed from
`StopReason.END_TURN` to `StopReason.REFUSAL` — `stop` is the only
value that legitimately means success and it's already explicitly
mapped (the pre-existing `finish_reason or "stop"` fallback for a
`None` value is unchanged), so this adapter is now protected against
any future unmapped value too. This closes the loop across all three
real provider adapters — each has now independently needed this exact
fix for its own SDK's stop/finish-reason enum. Verified live: both
`function_call` and a deliberately unrecognized future value now
report `REFUSAL` instead of `END_TURN`. Verified by reverting and
watching both new tests fail with the literal old bug's own shape. 2
new tests, 794 → 796 Python tests.

### A fourth instance of the same gap, in the one adapter the "closes the loop across all three" claim above didn't count

A much later fresh-eyes sweep found the identical shape one adapter
further: `ollama_provider.py` never had a stop-reason mapping at
all — its local `done_reason` variable only ever moved away from its
`StopReason.END_TURN` default when a tool call was seen. Unlike the
three SDK-based adapters, Ollama has no vendor-typed enum to check
against (this project talks to its `/api/chat` endpoint over raw
`httpx`, not an official SDK), so this gap had gone unnoticed even
while the other three were each independently found and fixed —
`ollama_provider.py` simply wasn't a "real provider adapter" the
earlier chapter's own "all three" framing had in mind, even though
it's exactly as real a stop/finish-reason gap as the other three.

Ollama's real streaming `/api/chat` response includes a `done_reason`
field on its final chunk — `"stop"` for a normal end, `"length"` when
generation was cut off by hitting `num_predict`/the context window,
documented Ollama wire behavior — but nothing in this adapter ever
read it. Confirmed live with a mocked Ollama stream whose final chunk
says `"done_reason": "length"`: this adapter reported `StopReason.
END_TURN` anyway, silently indistinguishable from a genuinely complete
response — a caller branching on `StopReason.MAX_TOKENS` to retry with
a larger budget or warn the user never gets the chance to for any real
Ollama-served conversation that hits its own generation limit.

Fixed more narrowly than the three SDK-based adapters' own fixes,
deliberately: Ollama's `done_reason` has no closed, vendor-typed enum
to check exhaustively against — `"stop"`/`"length"` are the only two
values documented with confidence — so this fix maps only the
confirmed `"length"` case to `MAX_TOKENS`, leaving every other value
(known or not) on the existing `END_TURN` default rather than
introducing a new fail-safe-to-`REFUSAL` default with no verified
evidence backing it. Presence of a tool call still wins over the raw
`done_reason`, matching every sibling adapter's own rule — Ollama
reports `"stop"` as the `done_reason` even for a turn that ends in a
tool call, so checking `done_reason` before the tool-call check would
have silently misreported it. Verified live: the same repro now
reports `MAX_TOKENS`; an ordinary `"stop"` response and a tool-call
turn (with `done_reason="stop"`) both still report exactly what they
did before. Verified by reverting and watching the new test fail with
the literal old bug's own shape: `END_TURN` instead of `MAX_TOKENS`.
3 new tests, 811 → 814 Python tests.

### `redacted_thinking` blocks were silently dropped, breaking multi-turn tool use once thinking flagged its own reasoning

The signed-`thinking`-block round trip above was already hardened, with
its own dedicated end-to-end test proving a real two-turn conversation
survives intact. A fresh-eyes sweep applied the identical question one
layer deeper: does the adapter handle *every* content-block type
Anthropic's real SDK can return, not just the one already covered?
`anthropic.types.RedactedThinkingBlock` (`type="redacted_thinking"`,
`data: str`) is a normal, documented block the API returns whenever the
model's own safety classifier flags part of its reasoning — unrelated
to whether the user's own prompt is adversarial, and reachable any time
extended thinking is on (this adapter's own default for every
Anthropic call). `generate()`'s response-parsing loop had no branch for
it at all, so it matched none of the `if`/`elif`s and was silently
dropped — never surfaced as a block, never yielded.

Confirmed live with the exact same fake-SDK-client pattern the signed
round-trip test already uses, just swapping in a `redacted_thinking`
block ahead of a `tool_use` block (the ordinary "reason, then call a
tool" shape): the resulting `Message`, threaded unmodified into the
next turn's history the same way `AgentLoop` always does, started with
`tool_use` alone. Anthropic's extended-thinking + tool-use contract
requires the leading thinking/redacted-thinking block be replayed
verbatim on the follow-up request — the real API would reject a turn
missing it, not just silently ignore the loss.

Fixed by representing it as a `ThinkingBlock` too, reusing the exact
same `provider_data`-as-opaque-round-trip-storage mechanism the signed
case already established rather than inventing a new content-block
type: empty visible text (redaction's whole point — there is none), the
encrypted blob stored under a distinct `provider_data` key
(`redacted_data`) from the signed case's `signature`, so the outgoing
translation can tell the two apart and reconstruct the correct wire
shape (`{"type": "redacted_thinking", "data": ...}` vs. `{"type":
"thinking", "thinking": ..., "signature": ...}`) for each. Verified
live the redacted block now survives the identical two-turn round trip
the signed case's test already proves. Verified by reverting and
watching the new test fail exactly where the block should have been:
`content[0]` was `tool_use`, not `redacted_thinking` — the block simply
wasn't there. 1 new test, 709 → 710 Python tests.

### Two concurrent tool results silently fused into one misleading value in the Ollama adapter

A fresh-eyes sweep of `_to_ollama_message`'s own request-building side
(not response parsing, which every other recent adapter fix targeted)
found `text_parts` — accumulating a `TextBlock`'s text and, separately,
each `ToolResultBlock`'s text as the loop walks a message's content —
joined at the end with `"".join(text_parts)`, no separator at all.
Harmless for the overwhelmingly common single-block message. Not
harmless for the shape `AgentLoop` actually builds every time a model
turn issues more than one tool call: `Message(role="user",
content=list(results))`, one `ToolResultBlock` per concurrently-
dispatched call (`asyncio.gather` over the whole round) — ordinary use,
not an edge case. Confirmed live: two tool results, `"4"` and `"2"`,
glued into the single wire-format string `"42"` — not just losing which
result came from which tool (the accepted, already-documented
limitation directly above, from Ollama's chat API having no dedicated
tool-result role at all), but actively fusing two distinct values into
a different, misleading one the model on the other end has no way to
separate back out.

Fixed with a newline separator (`"\n".join(text_parts)`) instead of the
empty-string join — a strict improvement with zero regression for the
common single-block case, since joining a one-element list is
identical either way. Verified live the two results above now translate
to `"4\n2"`, not `"42"`. Verified by reverting and watching the new
test fail with the literal old bug's own fused value reproducing
itself (`'42' == '4\n2'` — false). 1 new test, 710 → 711 Python tests.

## The model registry: adding a model is a YAML edit, not a code change

`core/sarva/providers/data/models.yaml` is the one file that says which
models exist and what they can do:

```yaml
models:
  - id: claude-opus-4-8
    provider: anthropic
    display_name: Claude Opus 4.8
    capabilities:
      modalities_in: [text, image, document]
      modalities_out: [text]
      tool_use: true
      thinking: true
      context_window: 1000000
      max_output: 128000
    cost: { input_per_mtok: 5.0, output_per_mtok: 25.0 }
```

Each entry names a `provider` key (`anthropic`, `openai`, `google`,
`ollama`, `mock`) that maps to one of the five adapters above. This is
the literal mechanism behind "absorbing the next frontier model is a
one-entry registry change": the adapter code doesn't change at all,
because it was never written against a specific model — only against
the wire protocol its `provider` key names.

`core/sarva/providers/data/routing.yaml` is the *policy* layered on top
of that data — ordered candidate model ids per `TaskClass`
(`main`/`subtask`/`escalation`/`vision`/`audio`):

```yaml
routing:
  main: [claude-opus-4-8, "ollama/qwen3:8b", mock]
  subtask: [claude-haiku-4-5, "ollama/qwen3:8b", mock]
  escalation: [claude-fable-5, claude-opus-4-8, mock]
```

`Router.pick(task, needs, override)` walks a task class's candidate
list and returns the first model that's (a) registered, (b) supports
every modality the caller actually needs (`needs: set[Modality]`), and
(c) is *available* — present in the `available: set[str]` the caller
built from real runtime state (an API key set, a local Ollama instance
actually reachable). `mock` sits last in every list on purpose: it's
always available, so the CLI and the full test suite work with zero
configuration, and a broken or missing credential degrades gracefully
to a working (if unintelligent) offline model rather than a hard
failure. An explicit `override` bypasses all of this — a caller who
names a specific model gets exactly that model, no substitution.

**"No substitution" needed its own exception type to actually be
true.** `override`'s promise depends on a subtle distinction: an
unknown override must be a hard failure, never something
`AgentLoop.run()`'s modality-degradation fallback (see the agent-loop
chapter) could catch and quietly route around. Before this was
checked directly, `Registry.get()` raised a plain `KeyError` for a
missing id — and `KeyError` *is* a `LookupError` subclass in Python,
which is exactly the exception type that fallback catches. An invalid
`--model` would have been silently swallowed and replaced with
whatever model the degradation path happened to pick, the opposite of
what "no substitution" promises. Fixed with `UnknownModelError` — a
deliberately separate exception, not a `LookupError` subclass — that
`Router.pick()` raises for an unrecognized override, and that
`AgentLoop.run()` now catches in its own branch, ahead of (and
excluded from) the degradation-fallback logic: an immediate `FAILED`
with a clear `detail` message naming the bad id, every time, with
`degraders` configured or not.

**This was also the CLI's own last mile:** `model_override` has been a
real `AgentLoop.run()` parameter since T1, but neither `sarva chat`
nor `sarva run` ever exposed a way to set it — there was no way to pick
a model from the command line at all, confirmed by `sarva chat --help`
showing no `--model` flag before this. Both commands now have one;
wiring it in surfaced the `UnknownModelError` gap above (a real user
would hit a typo'd model id immediately, unlike this project's own
tests, which never exercised `override` with an invalid value before).
`_chat`/`_run` also now print every `StateChangedEvent.detail` a
FAILED run carries — previously silently dropped, visible only by
reading `.sarva/runs/<id>/transcript.jsonl` by hand — and exit nonzero
on any non-`DONE` terminal state, so a scripted `sarva chat ... ||
handle_it` can actually detect a failure instead of always seeing exit
code 0.

### The "zero config, always works" promise had one broken chain — `audio`

A fresh-eyes sweep of the routing data itself (not the `Router` code,
which is well-tested) found `routing.yaml`'s `audio: [mock]` chain —
the *only* entry in it — could never actually resolve: `mock`'s own
`models.yaml` entry didn't declare `audio` in its `modalities_in`
(only `text`, `image`, `document`). Confirmed live against the real
shipped YAML: `Router.pick(TaskClass.AUDIO, needs={Modality.AUDIO})`
raised `LookupError` even with `mock` available — the one guarantee
this file's own header comment makes ("mock is always last so the CLI
and test suite work with zero config") broken for exactly the task
class it exists to cover. `MockProvider.generate()` doesn't actually
inspect modality at all (it just echoes the last user message's text),
so there was no real technical limitation being papered over — this
was purely a registry-data omission. Fixed by adding `audio` (and
`video`, for the identical reason, ahead of any future `TaskClass.
VIDEO`) to `mock`'s declared `modalities_in`.

Currently latent in every shipped flow, not something an ordinary
`sarva run`/`sarva chat` user could hit today: `AgentLoop` only ever
constructs with `task_class=MAIN` or `SUBTASK`, so `TaskClass.AUDIO`
(and `ESCALATION`) are defined but never actually invoked by any real
CLI/server/subagent code path — a real audio attachment routes through
`AgentLoop`'s own `LookupError`-triggered degrader fallback instead,
which correctly falls back to the `MAIN` chain with `needs={TEXT}` and
already works. Fixed anyway, not deferred as unreachable the way the
quantization/`Budget` gaps are: unlike those, this fix is a one-line,
zero-risk data correction restoring a promise the config file makes
about itself, not a design decision requiring new validation logic for
a path nothing can currently reach. Verified live the fix resolves the
identical `Router.pick` call. Verified by reverting and watching the
new test fail with the literal old `LookupError` reproducing itself.
1 new test, 713 → 714 Python tests.

### Honestly named: no fabricated registry entries

`OpenAIProvider` and `GoogleProvider` are both real, complete, tested
adapters — but neither has a `models.yaml` entry naming a specific
OpenAI or Gemini model id with real pricing. That file's own header
states it's "re-validated at every release," and this project applies
the same no-fabrication discipline to data as to model output: without
a verified-current catalog of model ids and per-token pricing, adding
an entry would mean guessing, which this project doesn't do — the same
principle the multimodal degraders apply when they report only what's
objectively knowable rather than describing content they can't
actually see. Wiring a real model in is a one-line config change for
whoever has that data; the adapter code was the part that needed
writing.

### Image-out: the first adapter to actually produce a `ContentEvent`

`ModelCapabilities.modalities_out` has said `# v1: {TEXT}; image-out
models later` since this field was written, and `ContentEvent`'s own
docstring calls out "images from image-out models" — both naming
image generation as anticipated future work before any adapter
actually did it. `google_provider.py` closes that: an image-capable
Gemini model can return a response part with `inline_data` populated
(the same `Blob` shape used to *send* an image in) instead of, or
alongside, text — translated into `ImageBlock` + `ContentEvent`, the
first real producer of an event type that existed in the protocol all
along with nothing behind it. Same scoping discipline as the rest of
this chapter: no `models.yaml` entry claims a specific image-out
Gemini model id yet (no verified-current catalog of which variants
support it, or their pricing) — the wire-level translation is real and
tested, wiring a specific verified model in is separate, one-line
follow-up work.

### Video-in: native understanding, not just sampled frames

The design doc's own T5 roadmap line names "MCP client, video input" as
a still-open deliverable. Until now, a `VideoBlock` reaching any
provider had exactly one path: `VideoToTextDegrader` sampling up to 4
frames into `ImageBlock`s first (see the multimodal chapter) — real and
useful, but a lossy fallback, not native understanding. `google_provider.py`
now also translates `VideoBlock` directly, via the identical
`inline_data`/`Blob` shape already used for images — Gemini's own real,
native video understanding, sent as-is rather than pre-degraded.
The degrader stays exactly as useful as before for every other
provider, or for a caller who explicitly wants the frame-sampled
fallback; this is additive. Honestly scoped on size: inline `Blob` data
is base64-encoded into the request body, which Gemini's documented
limits cap around 20MB total — fine for short clips, but a real caller
with a long video needs Gemini's separate Files API (upload once,
reference by URI), named as real, deferred follow-up rather than
silently mishandled.

## Build it yourself

- Run `sarva models` to see the registry as loaded — which ids exist,
  and which are marked available given your current environment
  (API keys set, Ollama reachable).
- Read `core/sarva/providers/mock.py` — the simplest real `Provider`
  implementation, and the one every conformance test in this repo
  depends on. Try scripting a `ScriptedTurn` sequence and driving it
  through `AgentLoop` directly.
- Add a fifth backend. Pick any HTTP API that can hold a chat
  conversation, and write a `generate()` that translates its wire
  format into the five `ProviderEvent` types — you'll likely
  rediscover at least one of the friction points named above for
  yourself.
- Read `sarva.eval.harness.run_benchmark` and `sarva.distill.distill` —
  both short, single-purpose functions, and both proof that once a
  backend speaks `Provider`, it's immediately usable everywhere in the
  system that needs a model, with zero backend-specific code anywhere
  else.
