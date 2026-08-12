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

### The `path` source had the identical unreachable-but-real gap the `url` source's SSRF guard was already fixed for — just for blocking I/O, not network safety

A much later fresh-eyes sweep found that `resolve_media_bytes()` fell
through to `block.resolve_bytes()` for a `path`-sourced block, which
does a real, synchronous `Path(self.path).read_bytes()` — the identical
"blocking I/O called directly from async code with no `asyncio.
to_thread`" class already found and fixed at 9+ other call sites across
this project (`ReadFileTool`, `SessionStore.load`/`.save`, `NoteTool`,
`SearchNotesTool`, `RememberTool`, `RecallMemoryTool`, and others).
`resolve_media_bytes()` is the one async entry point every provider
adapter and every multimodal degrader calls from inside their own
`async def` methods, so this sat directly on the hot path the module's
own opening paragraph above already names as the reason `url` needed
async handling in the first place.

Confirmed live with a simulated slow disk: zero heartbeat ticks landed
on the event loop for the whole duration of a single `path`-sourced
resolve — on a slow or network-mounted filesystem, every *other*
concurrent `/chat`/`/ws/chat` turn in a real `sarva serve` process
would freeze too, for as long as one media file read takes. Not
reachable through any current server/CLI input surface in this repo
(nothing here constructs a `path=`-sourced block today — `cli.py`'s own
`_load_image` deliberately pre-reads bytes and uses `data=` instead)
— the identical "not reachable today, but a real, documented,
first-class part of the public type, so leaving it unguarded would be
real, avoidable inconsistency the moment anything does wire a
`path`-sourced block up to real input" reasoning this module already
applied to the sibling `url` source above.

Fixed by dispatching the `path` branch through `asyncio.to_thread`,
leaving `data` (a plain in-memory attribute access, no I/O at all)
exactly as fast and untouched as before. Verified live: the same repro
now shows the event loop ticking throughout the call instead of
freezing solid. Verified by reverting and watching the new test fail
with the literal old bug's own shape — zero heartbeat ticks. 1 new
test, 801 → 802 Python tests.

### The `url` source also had the identical reachability gap `WebFetchTool`'s own sibling fix had just closed in the same round

A real bug found immediately after fixing `WebFetchTool`'s missing
`User-Agent` (`core/sarva/agent/tools.py`, see the tool-use chapter's
own entry): `fetch_bytes` here has no header set on its request either
— the exact same gap, in this module's own sibling url-fetching path.
Confirmed live with an ordinary, non-adversarial media URL (a
Wikimedia-hosted image, not a crafted target — the kind of URL a
url-sourced `ImageBlock` exists to support): a raw 403, because httpx's
own default `User-Agent` (`python-httpx/<version>`) is exactly what
real, legitimate sites — not just adversarial anti-bot ones — reject.
This module's own opening paragraphs above already state the principle
that both real url-fetching paths in this codebase must not drift out
of sync on what "safe to fetch" means; this is that same principle
applied to reachability, not just SSRF safety.

Fixed by passing the identical
`headers={"User-Agent": "Mozilla/5.0 (compatible; sarva-agent/1.0)"}`
`WebFetchTool`/`_duckduckgo_search` already send, on `fetch_bytes`'s own
`client.stream("GET", current_url)` call. Verified by reverting and
watching both a mocked-transport test (no real header reaches the
outgoing request) and a live test against the real Wikimedia URL fail
with the exact predicted shape, then restoring and confirming both pass
again. 2 new tests.

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

**A third real bug in the same worker, found much later by a
fresh-eyes sweep: a negative duration slipped past the NaN sentinel
this module's own docstring promises.** `_run()`'s sampling branch
(`if not duration_s or duration_s <= 0:`) already treats a negative
`duration_s` — not just `None` or `0` — as "no known duration" for
sampling purposes, but the output write-back only substituted the NaN
sentinel when `duration_s` was exactly `None`. A negative value (a
real, non-null float straight from a lightly-corrupted container's
`duration` field — precisely the "corrupted metadata, still-decodable
frames" threat model this whole worker exists to defend against, per
the SIGBUS bug above) sailed through unmodified into the field this
module's own docstring calls "NaN if no known duration," surfacing all
the way to the user/model as a nonsensical `"-5.0s"` instead of
`"unknown duration"` — the mirror image of an earlier, already-fixed
bug in `AudioToTextDegrader` (an `or`-fallback reporting a genuinely
known zero duration as unknown; here a genuinely unknown duration
reports as a bogus known one). Confirmed live with a faked container
whose stream reports `duration=-5.0`: the worker wrote the literal
`-5.0` into the duration field instead of NaN. Fixed by normalizing
`duration_s` to `None` inside that same branch, so the write-back
matches the branch's own sampling decision exactly. Verified by
reverting and watching the new test fail with the literal old value:
`-5.0` where NaN was expected.

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

### A much later fresh-eyes sweep found a third place "never silently drop" didn't reach: media nested inside a `ToolResultBlock`

`MODALITY_OF` hard-maps `"tool_result": Modality.TEXT` unconditionally
— `modality_of(some_tool_result_block)` always reports plain text, no
matter what's actually inside `ToolResultBlock.content`. Both
`_required_modalities()` (picks the turn's model, once, from the
initiating message) and `_degrade_block()`/`degrade_message()` (the
fallback that's supposed to convert unsupported media down to
something the picked model can use) operated purely on this top-level
value — a `ToolResultBlock` always registered as "already supported"
and was returned completely unmodified, never recursed into for
nested media. This directly contradicts this module's own opening
docstring ("Degradation is a registry of converters, applied
recursively... content is never silently dropped").

Reachable, not hypothetical: `mcp_client.py`'s `McpToolAdapter.run()`
builds exactly this shape today — a real MCP server's `ImageContent`
(screenshot, browser-automation, chart-generation tools all produce
this) becomes a genuine `ImageBlock` inside a `ToolResultBlock`.
`AgentLoop.run()` picks its model once, at the top of the turn, based
on the *initial* message only; a tool result produced mid-turn is
appended straight into `messages` with no re-pick and no
re-degradation ever attempted. Confirmed live: with a text-only router
(so no vision-capable model was ever in play) and a tool returning an
`ImageBlock` inside its `ToolResultBlock`, the turn ran to completion
silently — `state == DONE`, the raw image bytes sent straight to a
provider standing in for a model that, per its own registry entry,
cannot see them at all. Downstream this manifests two different ways
depending on the adapter: `OllamaProvider` silently puts the raw image
bytes on the wire with no error at all; `OpenAIProvider` raises an
uncaught `ValueError` from deep inside its own translation function,
turning what should be a clean, actionable failure into a confusing
internal-plumbing error instead.

Fixed with a new `required_modalities()` helper (recursing into
`ToolResultBlock.content`, used by `_required_modalities()` in place of
a bare top-level scan) and a matching fix in `_degrade_block()` (a
dedicated `ToolResultBlock` case that recurses into its own nested
content and rebuilds the block with the degraded result, instead of
short-circuiting on the block's own always-text top-level modality).
The agent loop now checks the newly-appended tool-result message
against what the run's already-picked model actually supports
immediately after appending it: with degraders configured, it degrades
just that message against the current model (mirroring the INIT-time
fallback's own logic); without them, or if degradation itself fails, the
turn fails cleanly with a specific, actionable reason instead of either
silently mis-sending unsupported content or crashing on an uncaught
adapter error. Verified live both paths: the same screenshot-tool
scenario now either completes normally (with a degrader configured, the
image degrades to a text description) or fails cleanly with a message
naming exactly what wasn't supported (without one). Verified by
reverting and watching the new test fail with the literal old bug's own
shape: `state == DONE` where `FAILED` was expected. 5 new tests, 757 →
762 Python tests.

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

### A real, genuinely known zero-second duration was reported as "unknown" — `or` is a truthiness fallback, not a None-check

A much later fresh-eyes sweep found a different bug in
`AudioToTextDegrader.degrade()`'s metadata fallback, one line below the
`asyncio.to_thread` fix above: `duration_s = _decode_wav_duration(raw)
or block.duration_s`. `_decode_wav_duration` legitimately returns
`0.0` for a real, valid WAV file with zero frames — a plausible real
artifact (a client that starts and immediately stops recording, or a
client bug that writes a valid WAV header with no sample data), not a
contrived edge case. `0.0` is falsy in Python, so `or` silently
discarded that correctly-decoded value and fell through to
`block.duration_s` instead, on both sides of the fallback: with no
declared duration, the message wrongly said "unknown duration" for a
duration that genuinely *was* known (zero); with a stale or simply
wrong caller-supplied `duration_s=42.0`, the message reported
`"42.0s"`, silently overriding the real, just-decoded `0.0s` with the
wrong value. This text block is the model's *only* signal about the
attachment on the text-only fallback path (`sarva[audio]` not
installed, or transcription failed/returned empty) — a wrong duration
here is a wrong fact fed straight into the model's own context, not
just a cosmetic display issue.

Confirmed live with a real, valid, zero-frame WAV built via the stdlib
`wave` module: `_decode_wav_duration` correctly returned `0.0`
directly, but the degrader's own output text said `"unknown
duration"` when nothing was declared, and `"42.0s"` when `42.0` was
declared — never the correct `"0.0s"` either way.

Fixed by replacing the `or` fallback with an explicit `is not None`
check: `_decode_wav_duration`'s result is used whenever it isn't
`None`, regardless of whether it happens to be `0.0`, and
`block.duration_s` is only consulted when the real decode itself
failed. Verified live: both cases above now correctly report
`"0.0s"`. Verified by reverting and watching both new tests fail with
the exact old bug's own shape — `"unknown duration"` and `"42.0s"`
respectively, instead of the correct `"0.0s"`. 2 new tests, 780 → 782
Python tests.

### A successful-but-empty transcription was reported as a transcription failure — the same truthiness shape, one branch up

A later fresh-eyes sweep found a third bug in the same method, one
branch above the duration fix: `if text: return [TextBlock(text=f"[Audio
transcript: {text}]")]` inside the `try` around `transcribe()`. `sarva.
audio.transcribe()` legitimately returns `""` whenever faster-whisper
finds no speech segments at all — silence, a blank or near-instant
voice memo, ambient noise, a music clip — a *successful* transcription
that correctly found nothing to say, not a failure. `if text:` sent
that down the exact same path as a genuine transcription exception, so
the degrader's output claimed *"the current model does not support
audio input, so its content could not be transcribed"* even though
transcription was attempted and succeeded. The identical "truthiness
treats a legitimate empty/zero value as absent/failed" shape as the
duration bug directly above, and as two other fixes elsewhere in this
project (`_decode_audio_isolated`'s stdout check, `sarva.config.
get_env()`'s env-var check) — reintroduced, unnoticed, one branch up in
this same method.

It's also the direct downstream continuation of the audio-decode
isolation fix (see the packaging doc's own per-bug narrative for
`_decode_audio_isolated`): that fix made a real zero-frame WAV decode
*successfully* to an empty sample array, so `transcribe()` correctly
returns `""` for it — but this `if text:` check still misreported that
as failure, undoing the user-facing benefit of that earlier fix for
the exact scenario it was meant to help.

Confirmed live: patching `transcribe` to return `""` (simulating a
real successful-but-silent transcription) for a valid WAV produced the
false "could not be transcribed" message.

Fixed by keying the branch off whether `transcribe()` raised
(`try`/`else`), not off the returned text's truthiness, and giving the
genuinely-empty case its own honest message — `"[Audio attached:
transcription found no speech]"` — instead of collapsing it into the
same message as "couldn't transcribe at all." Verified by reverting and
watching the new test fail with the exact old bug's own shape: the
false "could not be transcribed" message for a transcription that
actually succeeded.

**A quieter, pre-existing gap this also surfaced:** four existing tests
for the duration-decoding fallback path never forced `stt_extra_
installed()` to `False`, so on a machine with `sarva[audio]` installed
they were incidentally relying on real `transcribe()` returning `""`
for their synthetic tone/silence WAVs and the *old* buggy `if text:`
falling through to the metadata path by accident — an unintentional
coupling between two independent code paths, invisible until this fix
made the empty-transcription branch behave differently on purpose.
Fixed by having those four tests explicitly monkeypatch `stt_extra_
installed` to `False`, matching the pattern already used by `test_
audio_wired_into_degrade_message_end_to_end` for the same reason: they
mean to test duration decoding, not transcription, and should pass
identically regardless of whether the audio extra happens to be
installed in the environment running them. 1 new test, 852 → 853
Python tests.

### `asyncio.to_thread` fixed the event-loop freeze but left the whole agent turn unboundedly hangable — the one CPU-bound media call in this project missing an explicit ceiling

A much later hardening sweep, checking `DocumentToTextDegrader` against
the exact ceiling every OTHER CPU-bound media-processing call in this
project already has (`sarva.audio`'s `_DECODE_TIMEOUT_SECONDS`/
`_TTS_TIMEOUT_SECONDS`, the video degrader's own subprocess timeout),
found the one that had been missed: `_extract_pdf_text`'s
`asyncio.to_thread` wrapper (the fix two sections above) closes the
event-loop-freeze bug, but does nothing to bound how long the call
itself — and therefore the whole agent turn waiting on it — can take.

pypdf's own built-in guards (cyclic-page-reference detection,
`LimitReachedError` for a decompression bomb) both fire fast, confirmed
live against real crafted PDFs exercising each — but neither protects
against a legitimately valid, just extremely large PDF: a real,
ordinary attachment (a scanned archive, a programmatically generated
report) can genuinely run to tens of thousands of pages, and this
degrader always runs full per-page `extract_text()` to completion
*before* `_truncate` ever gets a chance to cut the output down.
Confirmed live with a scripted slow extraction: `degrade()` blocked
past a 5-second deadline with no recovery — and `AgentLoop`'s own
`Budget.max_wall_seconds` check can't help here either, since it only
fires *between* await points, never inside one still in flight, so a
genuinely stuck extraction would hang the entire turn indefinitely.

Fixed the same way every sibling call already is:
`asyncio.wait_for(asyncio.to_thread(_extract_pdf_text, raw), timeout=
_PDF_EXTRACT_TIMEOUT_SECONDS)` (30s), falling back to the same honest
"could not be extracted" message an unsupported format or a corrupt PDF
already gets. **One honest caveat, documented at the call site rather
than glossed over:** unlike the audio/video degraders' real subprocess
isolation, `asyncio.to_thread`'s underlying OS thread cannot actually
be killed on timeout — this fix stops the *agent turn* from hanging
(the real, user-facing symptom), not the abandoned worker thread
itself, the same "honest partial mitigation, not a magic bullet"
posture this project already applies elsewhere (e.g. the subagent-
cancellation spend-release comment in `agent/loop.py`).

Verified with a genuine revert-and-check: reverted the fix and watched
the new test's own outer 1.5-second safety-net `asyncio.wait_for` raise
`TimeoutError` itself — proof the inner call was still genuinely
hanging past that deadline — before re-applying. 1 new test, 914 → 915
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
