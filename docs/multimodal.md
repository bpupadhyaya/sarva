# Chapter 4 — Multimodality: the typed content model and graceful degradation

Chapter 3 covered how the agent loop drives a conversation. This
chapter is about what that conversation is actually *made of* —
`sarva.multimodal.content`, the typed vocabulary every layer in Sarva
speaks, and what happens when a model can't see everything a message
contains.

## One typed vocabulary, everywhere

Every input (text, an image, a PDF, audio, video), every model output
(text, thinking, tool calls), and every tool result is a frozen,
immutable `ContentBlock`:

```python
ContentBlock = TextBlock | ThinkingBlock | ImageBlock | AudioBlock \
             | VideoBlock | DocumentBlock | ToolCallBlock | ToolResultBlock
```

Nothing passes a raw provider dict across a module boundary — the
agent loop, every provider adapter, memory, and every skin all
construct and consume exactly these types. `TextBlock`/`ThinkingBlock`
are plain text. `ImageBlock`/`AudioBlock`/`VideoBlock`/`DocumentBlock`
all share a `_MediaBlock` base: exactly one of `data` (raw bytes),
`path` (local file), or `url` must be set — a Pydantic validator
enforces this, so a block with zero or two sources is a construction
error, not a runtime surprise three layers later.

Loading a media block's bytes is lazy and explicit. `block.resolve_bytes()`
handles `data`/`path` synchronously; a `url` source needs real network
I/O, which has no business happening synchronously inside the agent
loop's hot path, so it's `sarva.multimodal.fetch.resolve_media_bytes()`
that handles all three source types uniformly (streaming the download
with a real byte cap enforced from actual counted bytes, never trusted
`Content-Length`, and restricted to `http`/`https` schemes only).

## Degradation: never silently drop, or fail loudly instead

`Degrader` is the registry every "this model can't see modality X"
situation runs through:

```python
class Degrader(Protocol):
    source: Modality
    async def degrade(self, block: Any) -> list[Any]: ...
```

`degrade_message(msg, supported, degraders)` applies degraders
*recursively* — video → sampled image frames → (still unsupported) →
text — until every block in the message is something the target model
actually supports, or raises `UnsupportedModalityError` if no path
exists. This is a real, enforced guarantee at this layer: a block
either survives into something the model can see, or the caller gets a
loud, typed exception. Nothing in `degrade_message` itself silently
drops content.

Three real degraders exist, one per modality with no stdlib decoder of
its own worth trusting differently:

- **`ImageToTextDegrader`** — decodes real image bytes via Pillow and
  reports only objectively verifiable metadata (dimensions, format,
  byte size). Deliberately never attempts to describe what the image
  *shows* — that needs an actual vision-capable model call, a decision
  for the router/agent loop to make explicitly, not an implicit side
  effect of "degrade this content."
- **`AudioToTextDegrader`** — the stdlib `wave` module can decode
  exactly one real-world format (uncompressed WAV); every other format
  (the overwhelming majority of real audio) falls back to whatever the
  block itself declares (`media_type`, `duration_s` if set, and the
  always-knowable byte size).
- **`VideoToTextDegrader`** — the one with real decoding muscle: uses
  PyAV (statically-bundled decoder libraries, no system `ffmpeg`
  dependency) to sample up to 4 evenly-spaced real frames as
  `ImageBlock`s plus the real decoded duration. Undecodable bytes fall
  back to the same declared-metadata report the other two use.

`default_degraders()` wires all three into every real `AgentLoop` call
site (CLI, server) — but degradation itself is opt-in at the loop level
(`AgentLoop(degraders=...)`), not automatic: without it, a conversation
needing an unsupported modality still fails outright exactly as it did
before degraders existed. Nobody gets a lower-fidelity answer than they
explicitly asked for.

## An honest gap: `DocumentBlock` is typed but not processed

`DocumentBlock` (PDFs, docx, ...) exists in the type system —
`models.yaml` even marks `claude-opus-4-8` as supporting `document`
input — but there is no degrader for it, and no provider adapter
translates it. A `DocumentBlock` reaching `AnthropicProvider`,
`OpenAIProvider`, or `GoogleProvider` today is silently absent from the
translated request: each adapter's block-translation function is a
plain `if`/`elif` chain over the block types it knows how to handle,
with no `else` branch — an unrecognized block type (currently
`DocumentBlock`, and in practice `ThinkingBlock` too, on the second and
later turns of a real multi-turn conversation with an extended-thinking
model, since the agent loop appends the full assistant message —
thinking block included — back into history for the next turn) is
simply skipped rather than raised on.

This is a real, named exception to `degrade_message`'s own "never
silently drop" guarantee — that guarantee holds at the degradation
layer (a block that reaches a provider adapter is one `degrade_message`
already confirmed the target model's declared modality support
*should* cover), not at the lower-level wire-translation step inside
each adapter, which turned out to be a separate place the same
principle doesn't currently reach. Named here rather than left for a
reader to discover the hard way; not fixed in this entry, since making
every adapter raise loudly on an unhandled block type has a real
behavioral consequence for today's multi-turn thinking-model
conversations (they'd start raising on turn two instead of silently
continuing without the thinking content) that needs its own careful
pass, not a rushed side-fix.

## Build it yourself

- Read `tests/conformance/test_degraders.py` — the video degrader's
  tests synthesize real, tiny PyAV-encodable videos in the test itself
  rather than shipping fixture binaries.
  `test_video_frames_recursively_degrade_to_text_for_a_text_only_target`
  proves the full documented chain (video → sampled image frames → text)
  via `degrade_message`'s own recursion, not just
  `VideoToTextDegrader.degrade()` checked in isolation.
- Construct a `Message` with a `DocumentBlock` and run it through
  `AnthropicProvider`'s translation function directly — watch it come
  back with the document silently missing, the gap this chapter names.
- Try `sarva chat "..." --image path/to/photo.png` against a
  text-only-routed model with `degraders=default_degraders()` wired in
  (see `cli.py`) and watch the real fallback: route to a text-capable
  model, degrade the image into an honest metadata report, answer
  anyway instead of failing outright.
