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

`resolve_media_bytes()`'s `url` path shares an SSRF guard
(`ensure_public_host`) with the agent loop's own `WebFetchTool` — every
fetch, and every redirect hop, is checked against `ipaddress.is_global`
before it runs, refusing loopback/private/link-local (which includes
cloud metadata endpoints) addresses. Not reachable through any current
input path in this codebase (no server endpoint or MCP tool result
constructs a `url`-sourced block from external input today, checked
directly before deciding this was still worth fixing rather than
skipping), but the type exists specifically to support url-sourced
media, and leaving this path unguarded while `WebFetchTool` got the fix
would have been real, avoidable inconsistency.

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
  back to the same declared-metadata report the other two use. Since
  the providers chapter's own "Video-in" section, this is no longer the
  only path for a `VideoBlock` — `google_provider.py` now sends genuine
  video directly to Gemini's native video understanding instead of
  degrading to frames first, when that adapter is the one in play. This
  degrader remains exactly as useful for every other provider, or a
  caller who wants the frame-sampled fallback explicitly.

**A real bug found by directly fuzzing a valid mp4** (random byte
flips across many seeded trials) and running the exact
`av.open`/`container.decode` call this degrader used to make
in-process: some fuzzed variants didn't raise a Python exception at
all — they killed the whole process outright with a real SIGBUS
(signal 10), a native memory fault inside PyAV/libavcodec that no
`try`/`except`, however broad, can catch. A user- or attacker-supplied
corrupted video attachment could crash the entire `sarva serve`/CLI
process, not just fail one degradation attempt. Fixed by moving the
actual decode into an isolated subprocess
(`sarva.multimodal.degraders._video_worker`, invoked via `python -m`):
a native crash there only kills that subprocess, and the parent treats
it exactly the same as an ordinary "couldn't decode this" failure — a
timeout kills and reaps the worker the same way `RunShellTool`'s own
timeout fix does, so a hung decode can't leave an orphaned process
running forever either. Verified this is a real, not hypothetical,
concern: reverting the fix and re-running the regression test (which
regenerates one of the confirmed-crashing fuzzed byte sequences
deterministically, not from a checked-in binary fixture) genuinely
killed the pytest process itself with a fatal-signal stack dump — the
strongest form of "verify the fix is necessary" this project's
revert-and-verify discipline has produced yet.

**A second real bug in the same worker, a genuine memory-exhaustion
DoS found by actually generating an ordinary, non-corrupted video and
measuring peak memory, not by fuzzing:** the worker used to do
`frames = list(container.decode(stream))` — every frame of the entire
video decoded and held in memory at once, before sampling just 4 of
them. Confirmed live: a completely ordinary 60-second, 1280×720, 30fps
mp4 (834 KB compressed, generated with plain `ffmpeg`, no corruption or
crafting at all) drove peak RSS to 2.6 GB — roughly 3000× the file's
own size. Unlike the SIGBUS bug above, no malicious or corrupted file
is needed; a completely ordinary video of modest file size — a 20-
minute screen recording is an entirely normal thing for a real user to
attach — scales this linearly toward tens of GB, exhausting host
memory well before the worker's own 30-second decode timeout even
fires (that timeout bounds wall-clock time, not memory, so it does
nothing to stop this). Fixed by sampling via `container.seek()` to each
target timestamp (evenly spaced across the stream's own duration) and
decoding forward only until the nearest actual frame there is found —
never materializing more than one frame at a time, and never decoding
further into the stream than the distance between keyframes (a video's
GOP structure, typically a few seconds' worth of frames, not the
video's total length). Confirmed live: peak RSS for the identical
60-second test video dropped from 2.6 GB to ~37 MB, landing on frames
at the exact requested timestamps, not merely close to them. Verified
with a real worker subprocess (the exact invocation this module itself
uses) against a real 3000-frame synthetic video: the reverted, old
list-based code measured ~549 MB peak RSS for that file; the new code
stays comfortably under 200 MB.

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

**A real bug found by actually building a PDF whose `/Contents` stream
is a genuine `FlateDecode` zlib bomb** (a small, highly-compressible
payload that decompresses to 100MB): `pypdf` has its own internal
decompression-bomb guard (`ZLIB_MAX_OUTPUT_LENGTH`), and the exception
it raises when a stream exceeds that limit, `LimitReachedError`, is a
direct sibling of `PdfReadError` under `PyPdfError` — **not** a
subclass of it, confirmed via the real class MRO, not assumed from the
name. The only `except` clause this degrader had (`(PdfReadError,
ValueError)`) never caught it, so a decompression-bomb PDF crashed with
a raw, uncaught `pypdf` exception instead of the documented "could not
be extracted" fallback — the exact same "tiny file declares/contains
something implausibly huge" DoS shape already fixed for
`ImageToTextDegrader`'s `DecompressionBombError`, just never checked
for documents, which hadn't been individually audited that way before.
Fixed by widening the except clause to `(PdfReadError, ValueError,
LimitReachedError)`. **Verified the new test is real:** reverted the
fix and watched it fail with the raw, uncaught `LimitReachedError`
before re-applying. All 7 pre-existing document-degrader tests pass
unchanged. 1 new test, 570 → 571 Python tests.

### Two degraders froze the whole event loop during ordinary use — the same shape already fixed once for the memory tools, found here by giving this module its own fresh-eyes sweep

`AudioToTextDegrader.degrade()` called `sarva.audio.transcribe()`
directly and synchronously — a blocking subprocess decode followed by
real, CPU-bound `faster-whisper` inference — with no `asyncio.
to_thread`, the identical mistake `NoteTool`/`remember`/`recall_memory`
were fixed for one sweep earlier (see the memory chapter). Confirmed
live: transcribing one ordinary ~45-word voice message froze the
*entire* event loop for the full real transcription time — a heartbeat
coroutine that should tick every 0.05s made **zero** ticks of progress
across several real seconds. `default_degraders()` wires this degrader
in unconditionally for both `sarva serve` and the CLI, and `AgentLoop`
reaches it via its ordinary fallback path whenever the routed model
can't accept audio directly — a single user's voice message freezes
every *other* concurrent user's turn too, for as long as transcription
takes (up to this module's own 10-minute cap for a long attachment).

The same fresh-eyes sweep, applying the identical lens one step
further, found `DocumentToTextDegrader`'s PDF path had a smaller but
real instance of the same shape: `_extract_pdf_text` (`pypdf` parsing +
per-page `extract_text()`) is also synchronous, CPU-bound work called
directly with no `asyncio.to_thread`. A 300-page PDF — not an extreme;
this chapter's own paragraph above already treats it as a plausible
real attachment size — took **0.52s** of real, measured wall-clock
extraction time, a genuine (if smaller and more size-dependent)
freeze, not a negligible one: unlike `ReadFileTool`/`WriteFileTool`'s
own file I/O (checked in an earlier round and found genuinely
negligible even at multi-gigabyte sizes, since realistic tool-call
argument sizes stay small), a PDF attached as a media block can
plausibly be multi-megabyte, and per-page text extraction is real CPU
work, not just I/O throughput.

Both fixed the same way: wrap the blocking call in `asyncio.to_thread`.
Verified live both fixes hold: the real transcription case now shows
the event loop ticking throughout its ~5-second real duration instead
of freezing solid; the 300-page PDF case now ticks throughout its
~0.5-second extraction instead of showing zero progress. Verified by
reverting and watching both new tests fail with the literal old bug's
own number — `0` ticks — reproducing itself. 2 new tests, 707 → 709
Python tests.

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
