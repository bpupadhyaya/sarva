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
- **`AudioToTextDegrader`** — attempts a real local transcription first
  via `sarva.audio.transcribe` (`faster-whisper`, the `sarva[audio]`
  extra — see the packaging chapter's "Local speech" section) when it's
  installed. Only when the extra is missing, or transcription genuinely
  fails on that specific audio, does it fall back to declared metadata:
  the stdlib `wave` module can decode exactly one real-world format
  (uncompressed WAV) for a real duration; every other format falls back
  to whatever the block itself declares (`media_type`, `duration_s` if
  set, and the always-knowable byte size). Never a fabricated
  transcript standing in for one that couldn't actually be produced.
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

## Where "never silently drop" stops, and how that boundary got closed

`degrade_message`'s "never silently drop" guarantee holds at the
degradation layer — a block that reaches a provider adapter is one
`degrade_message` already confirmed the target model's declared
modality support *should* cover. It turned out the *adapters
themselves* were a separate place that same principle didn't reach:
each block-translation function (`_to_anthropic_message`,
`_to_openai_messages`, `_to_gemini_content`) was a plain `if`/`elif`
chain over the block types it knew how to handle, with no `else`
branch — an unrecognized block type was simply skipped, not raised on.
`DocumentBlock` (PDFs, docx, ... — typed since T0, at the time with no
degrader and no adapter support at all) hit this every time; `ThinkingBlock`
hit it too, on the second and later turns of any real multi-turn
conversation with an extended-thinking model, since the agent loop
appends the full assistant message — thinking block included — back
into history for the next turn.

This is now closed, with the two cases handled differently on purpose:
every adapter's translation function has an explicit
`elif isinstance(b, ThinkingBlock): continue` — a **deliberate, named**
skip, since none of the three backends currently accept a
caller-supplied reasoning trace back on the next turn anyway (there's
nothing meaningful to round-trip yet) — followed by a catch-all
`else: raise ValueError(...)` for genuinely unhandled types. The
distinction matters: dropping a thinking trace the model can't use
anyway is harmless; silently omitting a document the user actually
attached and having the model answer as though it read it is a
materially misleading response, not a cosmetic gap — so that case
fails loudly instead. `DocumentBlock` now has a real degrader
(`DocumentToTextDegrader`, below) that converts it away before it would
ever reach this `else` branch through the normal opt-in degradation
path — the branch still exists and still raises for the residual case
of a `DocumentBlock` reaching an adapter directly (degradation skipped,
or a model whose registry entry claims document support no adapter
actually implements).

## The fourth degrader: `DocumentToTextDegrader`

The image/audio/video trio left the one modality named in `Degrader`'s
own motivating docstring example completely uncovered — confirmed
empty by grep before starting, not assumed. `DocumentToTextDegrader`
closes it with the same honesty principle as the other three: real
extracted text where a real extractor exists, never a fabricated
summary. `pypdf` (pure Python, the same "commodity substrate" tier as
Pillow/PyAV) gives real per-page PDF text extraction; plain-text-adjacent
media types (`text/plain`, `text/markdown`, `text/csv`, `text/html`,
`application/json`) need no library at all — a UTF-8 decode of the
block's own bytes *is* the real content. Extracted text is capped at
20,000 characters (the corpus pipeline's length-filter philosophy
applied here: an attached 300-page PDF shouldn't consume a target
model's whole context window on its own), and the degraded message
says honestly when and how much was cut.

**A scanned/image-only PDF (no embedded text layer) degrades the same
way a read error does** — both mean "nothing could be extracted," which
mirrors the audio degrader's own framing of an undecodable format as an
*expected* real case, not a bug to distinguish. `.docx` and other
binary office formats have no extractor yet — a second heavy dependency
isn't justified by one format the way `pypdf` is justified by PDF being
ubiquitous, so unsupported formats fall back to the same
declared-metadata-only report the other degraders use, a real, named,
deferred gap rather than an implicit one.

## Build it yourself

- Read `tests/conformance/test_degraders.py` — the video degrader's
  tests synthesize real, tiny PyAV-encodable videos in the test itself
  rather than shipping fixture binaries.
  `test_video_frames_recursively_degrade_to_text_for_a_text_only_target`
  proves the full documented chain (video → sampled image frames → text)
  via `degrade_message`'s own recursion, not just
  `VideoToTextDegrader.degrade()` checked in isolation.
- Construct a `Message` with a `DocumentBlock` and run it through
  `DocumentToTextDegrader().degrade(...)` directly — a real PDF
  produces its actual extracted text; garbage bytes or an unsupported
  format like `.docx` fall back honestly rather than raising.
- Then construct the same `DocumentBlock` and run it through any of the
  three adapters' translation functions *directly*, bypassing
  degradation — watch it raise `ValueError` naming exactly which block
  type it can't translate, instead of silently vanishing. The
  difference between these two paths is the whole point of this
  chapter's "where never silently drop stops" section above.
- Try `sarva chat "..." --image path/to/photo.png` against a
  text-only-routed model with `degraders=default_degraders()` wired in
  (see `cli.py`) and watch the real fallback: route to a text-capable
  model, degrade the image into an honest metadata report, answer
  anyway instead of failing outright.
