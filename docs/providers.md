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
  a wire format with no dedicated concept for it.

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
