"""sarva.multimodal.degraders.audio — the second concrete Degrader.

Same honesty principle as `ImageToTextDegrader`: report only what's
actually known, never fabricate a transcript. But the failure-handling
tradeoff is deliberately different. Pillow reliably decodes nearly every
real-world image format, so `ImageToTextDegrader` treats undecodable
bytes as a genuine error. Real-world audio is overwhelmingly compressed
(MP3/AAC/OGG/M4A) — the stdlib `wave` module only handles uncompressed
WAV, and pulling in a heavier dependency (ffmpeg/pydub) isn't justified
for a metadata-only converter. So this degrader never raises on an
undecodable format: "not WAV" is the *expected* case for most real
audio, not an error. It falls back to whatever the block itself already
declares (`media_type`, `duration_s` if the caller set it, and the
actual byte size, which is always knowable) rather than treating
"couldn't decode" as exceptional.

**Real transcription, not just metadata, when `sarva[audio]` is
installed.** Until now this degrader never actually transcribed
anything — it always said "could not be transcribed," even though
nothing about the *architecture* prevented real transcription, only a
missing implementation. `sarva.audio.transcribe` (real local
`faster-whisper` STT, see that module's docstring) is now attempted
first; only when the extra isn't installed, or transcription genuinely
fails on this specific audio, does this degrader fall back to the
original honest metadata-only message — never a fabricated transcript
standing in for one that couldn't actually be produced.
"""

from __future__ import annotations

import asyncio
import io
import wave

from sarva.multimodal.content import AudioBlock, Modality, TextBlock
from sarva.multimodal.fetch import resolve_media_bytes


def _decode_wav_duration(raw: bytes) -> float | None:
    # Deliberately broad except: any failure here just means "not a WAV
    # stdlib can parse" (the expected case for most real audio), not a
    # bug to surface — the caller falls back to declared metadata either
    # way, so there's nothing more specific to distinguish or re-raise.
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except Exception:
        return None


class AudioToTextDegrader:
    source = Modality.AUDIO

    async def degrade(self, block: AudioBlock) -> list[TextBlock]:
        raw = await resolve_media_bytes(block)

        from sarva.audio import stt_extra_installed, transcribe

        if stt_extra_installed():
            # Broad except deliberately: a transcription failure on THIS
            # audio (corrupt bytes, an unsupported codec, a model
            # loading error) should degrade to the honest metadata
            # fallback below, not crash the whole agent turn -- the same
            # "never let a best-effort enrichment take down the request"
            # posture the rest of this degrader already has.
            #
            # A real bug found by giving this degrader the same
            # event-loop-freeze lens that had just found NoteTool/
            # remember/recall_memory blocking the whole process (see
            # sarva.memory.longterm/sarva.memory.vector): `transcribe()`
            # runs a blocking subprocess decode followed by real
            # CPU-bound faster-whisper inference, called directly here
            # with no `asyncio.to_thread` -- confirmed live, transcribing
            # one ordinary ~45-word voice message froze the entire event
            # loop for its full 9-second real transcription time (a
            # concurrent coroutine that should tick every 0.05s made
            # ZERO ticks of progress), meaning every OTHER user's
            # in-flight `/chat`/`/ws/chat` turn in a real `sarva serve`
            # process would freeze too for as long as this one
            # transcription takes -- up to this module's own 10-minute
            # cap for a long attachment, not milliseconds.
            try:
                text = await asyncio.to_thread(transcribe, raw)
                if text:
                    return [TextBlock(text=f"[Audio transcript: {text}]")]
            except Exception:
                pass

        # A real bug found by a fresh-eyes sweep: `or` is a truthiness
        # fallback, not a None-check -- `_decode_wav_duration` returns a
        # legitimate `0.0` for a valid but zero-frame WAV (an empty/
        # broken recording: a client that starts and immediately stops
        # recording, or a client bug that writes a valid WAV header with
        # no sample data -- a plausible real artifact, not contrived),
        # and `0.0` is falsy in Python. Confirmed live: a real,
        # zero-frame WAV correctly decoded to `duration_s=0.0`, but `or`
        # silently discarded that correct value and fell through to
        # `block.duration_s` instead -- with no declared duration, the
        # message wrongly said "unknown duration" for a duration that
        # genuinely WAS known (zero); with a stale/wrong declared
        # `duration_s=42.0`, the message reported "42.0s", silently
        # overriding the real, just-decoded `0.0s` with a wrong caller-
        # supplied value. This text block is the model's only signal
        # about the attachment on the text-only fallback path, so a
        # wrong duration is a wrong fact fed straight into its context.
        decoded_duration = _decode_wav_duration(raw)
        duration_s = decoded_duration if decoded_duration is not None else block.duration_s

        size_kb = len(raw) / 1024
        duration_text = f"{duration_s:.1f}s" if duration_s is not None else "unknown duration"
        text = (
            f"[Audio attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
            "The current model does not support audio input, so its content "
            "could not be transcribed.]"
        )
        return [TextBlock(text=text)]
