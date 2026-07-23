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
  to translate.

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
